"""Draft state serialization for the live dashboard.

Converts in-memory draft objects into a JSON-serializable dict and provides
atomic file I/O to prevent the Flask thread from reading partial writes.

The module supports a **delta protocol** to avoid sending the full available-
player board (300+ rows) on every 2-second poll:

* ``serialize_board`` - one-time snapshot of the complete player board.
* ``serialize_state``  - lightweight state (pick info, roster, recs, balance)
  that includes a monotonic *version* counter.
* ``write_state`` / ``read_state`` - atomic JSON I/O.  ``write_state`` also
  writes a companion ``*_board.json`` file on the first call.
* ``read_delta`` - returns only the fields that changed between version *N*
  and the current version, or ``None`` when a full reload is required.
"""

import contextlib
import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import pandas as pd

from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.draft.recommender import Recommendation
from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import RATE_STATS, Category


class StateKey(StrEnum):
    """Canonical names of the dashboard-schema keys in ``draft_state.json``.

    Members serialize to their string value (StrEnum) so JSON I/O is
    transparent: ``state[StateKey.PICKS] = [...]`` is equivalent to
    ``state["picks"] = [...]`` but a typo on the enum form is caught
    statically by mypy/ruff. Mirrors the pattern in
    :class:`fantasy_baseball.data.cache_keys.CacheKey`.

    Scope: this enum covers the keys the dashboard writers
    (:mod:`draft_controller`) produce. The legacy CLI's
    :func:`serialize_state` writes a different schema (``current_pick``,
    ``recommendations``, ``balance``, ...) — those keys are bare strings
    because the legacy path is slated for removal once the dashboard
    fully replaces the CLI. Don't grow this enum to cover legacy keys.
    """

    VERSION = "version"
    KEEPERS = "keepers"
    PICKS = "picks"
    ON_THE_CLOCK = "on_the_clock"
    UNDO_STACK = "undo_stack"
    PROJECTED_STANDINGS_CACHE = "projected_standings_cache"


@dataclass
class Pick:
    """One drafted slot — live pick or pre-seeded keeper.

    ``pick_number`` is ``None`` for keepers (they do not consume a draft
    slot). ``undone`` is flipped when the pick is rolled back via undo and
    pushed onto the ``undo_stack``; the pick is removed from the live
    ``picks`` list, not marked in place.
    """

    pick_number: int | None
    round: int
    team: str
    player_id: str
    player_name: str
    position: str
    timestamp: float
    undone: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pick":
        return cls(**data)


# ---------------------------------------------------------------------------
# Module-level version counter (monotonically increasing, thread-safe)
# ---------------------------------------------------------------------------
_version_lock = threading.Lock()
_current_version: int = 0
_previous_state: dict[str, Any] | None = None


def _next_version() -> int:
    """Return the next version number (thread-safe)."""
    global _current_version
    with _version_lock:
        _current_version += 1
        return _current_version


def get_current_version() -> int:
    """Return the current version number without incrementing."""
    with _version_lock:
        return _current_version


# ---------------------------------------------------------------------------
# Board serialization (heavy, sent once)
# ---------------------------------------------------------------------------


def _serialize_player_stats(player: dict[str, Any], row: pd.Series) -> None:
    """Add stat fields to a player dict based on player type.

    Volume stats (AB/H, IP/ER/BB/H_allowed) are emitted alongside the
    displayed counting/rate stats so ``project_team_stats`` can recompute
    team rate stats (AVG/ERA/WHIP) from components.
    """
    if row["player_type"] == PlayerType.HITTER:
        player["r"] = int(row.get("r", 0))
        player["hr"] = int(row.get("hr", 0))
        player["rbi"] = int(row.get("rbi", 0))
        player["sb"] = int(row.get("sb", 0))
        player["avg"] = round(float(row.get("avg", 0)), 3)
        player["ab"] = float(row.get("ab", 0))
        player["h"] = float(row.get("h", 0))
    elif row["player_type"] == PlayerType.PITCHER:
        player["w"] = int(row.get("w", 0))
        player["k"] = int(row.get("k", 0))
        player["sv"] = int(row.get("sv", 0))
        player["era"] = round(float(row.get("era", 0)), 2)
        player["whip"] = round(float(row.get("whip", 0)), 2)
        player["ip"] = float(row.get("ip", 0))
        player["er"] = float(row.get("er", 0))
        player["bb"] = float(row.get("bb", 0))
        player["h_allowed"] = float(row.get("h_allowed", 0))


def serialize_board(board: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize the full draft board into a JSON-serializable list.

    This is meant to be fetched once by the client and cached locally.
    The client uses ``drafted_players`` from the poll state to determine
    which players are still available.
    """
    players = []
    for _, row in board.iterrows():
        adp_val = row.get("adp", None)
        sgp_val = row.get("total_sgp", None)
        player: dict[str, Any] = {
            "name": row["name"],
            "player_id": row.get("player_id", row["name"]),
            "positions": row["positions"]
            if isinstance(row["positions"], list)
            else [row["positions"]],
            "var": round(float(row["var"]), 1),
            # The projection CSVs use 999.0 as a sentinel for "no ADP
            # data" (e.g. NPB imports like Murakami who weren't in the
            # FanGraphs draft pool). Normalize to None so the dashboard
            # renders "—" instead of literal 999.
            "adp": round(float(adp_val), 1)
            if adp_val is not None
            and not pd.isna(adp_val)
            and adp_val != float("inf")
            and float(adp_val) < 999
            else None,
            "total_sgp": round(float(sgp_val), 2)
            if sgp_val is not None and not pd.isna(sgp_val)
            else None,
            "player_type": row["player_type"],
        }
        _serialize_player_stats(player, row)
        players.append(player)
    return players


# ---------------------------------------------------------------------------
# State serialization (lightweight, sent on every poll)
# ---------------------------------------------------------------------------


def serialize_state(
    tracker: DraftTracker,
    balance: CategoryBalance,
    board: pd.DataFrame,
    recommendations: list[Recommendation],
    filled_positions: dict[str, int],
    roster_slots: dict[str, int] | None = None,
    roster_by_position: dict[str, list[str]] | None = None,
    teams: dict[int, str] | None = None,
    num_keepers: int = 0,
    vona_scores: dict[str, float] | None = None,
    *,
    include_available: bool = True,
) -> dict[str, Any]:
    """Convert all draft objects into a JSON-serializable dict.

    Parameters
    ----------
    include_available:
        When *True* (the default), the ``available_players`` list is included
        for full backward compatibility.  Set to *False* when the client has
        already fetched the board via ``/api/board`` and only needs the delta
        protocol fields.
    """
    totals = balance.get_totals()
    rounded_totals: dict[Category, float] = {}
    for cat, val in totals.items():
        if val is None:
            rounded_totals[cat] = 0.0
        elif cat in RATE_STATS:
            rounded_totals[cat] = round(float(val), 3)
        else:
            rounded_totals[cat] = round(float(val))

    # Offset pick/round to account for keeper rounds so the UI shows
    # the overall draft pick number (e.g. pick 59 not pick 29).
    overall_pick = tracker.current_pick + num_keepers
    keeper_rounds = num_keepers // tracker.num_teams if tracker.num_teams else 0
    overall_round = tracker.current_round + keeper_rounds

    state: dict[str, Any] = {
        "version": get_current_version(),  # will be bumped by write_state
        "current_pick": overall_pick,
        "current_round": overall_round,
        "picking_team": tracker.picking_team,
        "picking_team_name": (teams or {}).get(
            tracker.picking_team, f"Team {tracker.picking_team}"
        ),
        "is_user_pick": tracker.is_user_pick,
        "picks_until_user_turn": tracker.picks_until_user_turn,
        "user_roster": list(tracker.user_roster),
        "user_roster_ids": list(tracker.user_roster_ids),
        "drafted_players": list(tracker.drafted_players),
        "drafted_ids": list(tracker.drafted_ids),
        "recommendations": [
            {
                "name": r.name,
                "var": round(float(r.var), 1),
                "best_position": r.best_position.value,
                "positions": [p.value for p in r.positions],
                "need_flag": bool(r.need_flag),
                "note": r.note,
            }
            for r in recommendations
        ],
        "balance": {
            "totals": {cat.value: v for cat, v in rounded_totals.items()},
            "warnings": balance.get_warnings(),
        },
        "filled_positions": dict(filled_positions),
        "roster_by_position": dict(roster_by_position) if roster_by_position else {},
    }

    if vona_scores is not None:
        # Round VONA to 1 decimal, keyed by player_id
        state["vona_scores"] = {pid: round(float(v), 1) for pid, v in vona_scores.items()}

    if roster_slots is not None:
        state["roster_slots"] = dict(roster_slots)

    if teams is not None:
        state["teams"] = {int(k): v for k, v in teams.items()}

    if include_available:
        available = board[~board["player_id"].isin(tracker.drafted_ids)]
        available_players = []
        for _, row in available.iterrows():
            player: dict[str, Any] = {
                "name": row["name"],
                "positions": row["positions"]
                if isinstance(row["positions"], list)
                else [row["positions"]],
                "var": round(float(row["var"]), 1),
                "player_type": row["player_type"],
            }
            _serialize_player_stats(player, row)
            available_players.append(player)
        state["available_players"] = available_players

    return state


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

# Keys that are always included in deltas so the client can render status.
_ALWAYS_INCLUDE = {"version"}

# Keys to compare for changes. Mixes the legacy CLI schema (current_pick,
# recommendations, balance, ...) and the dashboard schema (picks, keepers,
# on_the_clock, ...) because both writers produce state files this delta
# protocol serves. Without the dashboard keys here, polling clients see
# empty deltas after every pick and the UI freezes mid-draft.
_DELTA_KEYS = {
    # Legacy CLI fields (serialize_state)
    "current_pick",
    "current_round",
    "picking_team",
    "picking_team_name",
    "is_user_pick",
    "picks_until_user_turn",
    "user_roster",
    "drafted_players",
    "drafted_ids",
    "recommendations",
    "balance",
    "filled_positions",
    "roster_by_position",
    "projections",
    "vona_scores",
    # Dashboard fields (draft_controller.apply_pick / start_new_draft)
    StateKey.KEEPERS.value,
    StateKey.PICKS.value,
    StateKey.ON_THE_CLOCK.value,
    StateKey.UNDO_STACK.value,
    StateKey.PROJECTED_STANDINGS_CACHE.value,
}


def compute_delta(old_state: dict[str, Any] | None, new_state: dict[str, Any]) -> dict[str, Any]:
    """Return a dict with only the fields that differ between *old* and *new*.

    Always includes ``version``.  If *old_state* is ``None`` the full
    *new_state* is returned (minus ``available_players``).
    """
    delta: dict[str, Any] = {"version": new_state["version"]}

    if old_state is None:
        # No baseline -- send everything (excluding the heavy board list).
        for key in _DELTA_KEYS:
            if key in new_state:
                delta[key] = new_state[key]
        delta["full_state"] = True
        return delta

    for key in _DELTA_KEYS:
        new_val = new_state.get(key)
        old_val = old_state.get(key)
        if new_val != old_val:
            delta[key] = new_val

    return delta


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_state(state: dict[str, Any], path: Path) -> None:
    """Atomically write state dict to JSON (write tmp + rename).

    Writes to a temporary file in the same directory as ``path``, then
    renames.  On Windows ``os.replace`` is used which is atomic for
    same-volume renames.

    Also writes a companion delta file (``*_delta.json``) containing only
    the fields that changed since the last write.
    """
    global _previous_state

    # Bump the version for this write.
    version = _next_version()
    state["version"] = version

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Compute and write the delta file.
    delta = compute_delta(_previous_state, state)
    delta_path = path.with_name(path.stem + "_delta" + path.suffix)
    _atomic_write(delta, delta_path)

    # Write the full state file (for backward compat / initial loads).
    _atomic_write(state, path)

    # Store a copy of current state (excluding heavy available_players) for
    # the next delta computation.
    lightweight = {k: v for k, v in state.items() if k != "available_players"}
    _previous_state = lightweight


def write_board(board_data: list[dict[str, Any]], path: Path) -> None:
    """Atomically write the full board to a JSON file.

    Called once at draft start so the dashboard can fetch the board via
    ``/api/board``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(board_data, path)


def _atomic_write(data, path: Path) -> None:
    """Write *data* as JSON to *path* using atomic tmp+rename.

    On Windows, os.replace can fail with PermissionError if another
    process has the target file open.  Retry a few times with a short
    delay before giving up.
    """
    import time

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix="draft_state_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.1)
                else:
                    raise
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def read_state(path: Path) -> dict[str, Any]:
    """Read state dict from JSON. Returns empty dict on decode error or missing file."""
    try:
        with open(path) as f:
            return cast(dict[str, Any], json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def read_board(path: Path) -> list[dict[str, Any]]:
    """Read the board list from JSON. Returns empty list on error or missing file."""
    try:
        with open(path) as f:
            return cast(list[dict[str, Any]], json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def read_delta(path: Path) -> dict[str, Any]:
    """Read the most recent delta from JSON. Returns empty dict on error."""
    try:
        with open(path) as f:
            return cast(dict[str, Any], json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
