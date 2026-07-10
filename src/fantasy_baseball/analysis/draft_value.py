"""Draft-value metric: realized VAR vs draft-slot par expectation."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.config import LeagueConfig, load_config
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.redis_store import (
    get_game_log_totals,
)
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.draft.board import ScaleInputs, build_board_from_frames
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.rankings import rank_key
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip
from fantasy_baseball.utils.time_utils import compute_fraction_remaining, local_today
from fantasy_baseball.web.season_data import (
    read_cache_dict,
)

logger = logging.getLogger(__name__)

# Team volumes FROZEN at what the 2026 draft-day board was built with.
# The live DEFAULT_TEAM_AB/IP constants recalibrate over time (1450 -> 1300
# on 2026-07-05); this module's whole contract is reproducing the DRAFT-DAY
# scale so historical par curves and per-pick values stay comparable, the
# same reason it deliberately ignores sgp_denominators overrides. Do not
# point these at the live defaults.
_DRAFT_DAY_TEAM_AB = 5500
_DRAFT_DAY_TEAM_IP = 1450

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_BOARD = _REPO_ROOT / "data" / "draft_state_board.json"
_POSITIONS_JSON = _REPO_ROOT / "data" / "player_positions.json"
_CONFIG_DIR = _REPO_ROOT / "config"
_CONFIG = _CONFIG_DIR / "league.yaml"
_DRAFT_ORDER = _CONFIG_DIR / "draft_order.json"
_DRAFT_STATE = _REPO_ROOT / "data" / "draft_state.json"


def reproduce_draft_day_board(
    config: LeagueConfig | None = None,
) -> tuple[pd.DataFrame, ScaleInputs]:
    """Rebuild the preseason board from the Apr-1 projection CSVs -- pure, no DB/KV.

    blend_projections is deterministic over data/projections/{season_year}/*.csv;
    positions come from data/player_positions.json. The board core is the shared
    build_board_from_frames, so the scale (rates/floors/denoms/volumes) is identical
    to the real draft board's, just computed off the preserved draft-day CSVs. Pass a
    preloaded ``config`` to avoid re-reading league.yaml; ``None`` loads it.
    """
    if config is None:
        config = load_config(_CONFIG)
    # Preseason CSV dir tracks the configured season so this does not silently read a
    # prior year's projections after the season rolls over.
    preseason_csvs = _REPO_ROOT / "data" / "projections" / str(config.season_year)
    hitters, pitchers, _quality = blend_projections(
        preseason_csvs, config.projection_systems, config.projection_weights
    )
    positions = load_positions_cache(_POSITIONS_JSON)
    board, scale = build_board_from_frames(
        hitters,
        pitchers,
        positions,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
        team_ab=_DRAFT_DAY_TEAM_AB,
        team_ip=_DRAFT_DAY_TEAM_IP,
    )
    return board, scale


def frozen_drift_summary(
    board_df: pd.DataFrame,
    frozen_var: dict[str, float] | None = None,
    tol: float = 0.05,
) -> dict[str, float]:
    """SOFT cross-check vs the frozen draft-day board. Reports drift; never raises.

    The frozen board (draft_state_board.json) was built at draft time with
    possibly-since-churned code, so exact reproduction is not expected. A large
    systematic drift is worth surfacing (wrong config/vintage) but is not a stop.

    Pass ``frozen_var`` (the ``player_id -> VAR`` map from ``_frozen_var_by_player_id``)
    to reuse a load the caller already did; ``None`` self-loads the default frozen board.
    A missing/empty frozen board yields an empty summary (cross-check skipped, never raises).
    """
    empty: dict[str, float] = {"joined": 0, "over_tol": 0, "max": 0.0, "median": 0.0}
    if frozen_var is None:
        frozen_var = _frozen_var_by_player_id()
    if not frozen_var:
        return empty
    # Vectorized join (same board["player_id"].map(frozen_var) idiom the anchor uses) --
    # no per-row iterrows. Drift = |rebuilt VAR - frozen VAR| over the matched rows.
    mapped = board_df["player_id"].map(frozen_var)
    matched = mapped.notna()
    diffs = (board_df.loc[matched, "var"].astype(float) - mapped[matched]).abs()
    joined = int(matched.sum())
    over = int((diffs > tol).sum())
    summary: dict[str, float] = {
        "joined": joined,
        "over_tol": over,
        "max": float(diffs.max()) if joined else 0.0,
        "median": float(diffs.median()) if joined else 0.0,
    }
    if joined and over > 0.5 * joined:
        # INFO, not WARNING: the f=1 grade now anchors preseason_var and the projected par
        # curve to the FROZEN VAR, so rebuilt-VAR drift is EXPECTED (~100% by design) and no
        # longer affects the shipped grades -- warning every refresh would be alarm fatigue.
        # It still informs the rebuilt SCALE that drives the to-date (YTD) and luck sides.
        logger.info(
            "draft-value: rebuilt board VAR drifts from frozen draft_state_board.json "
            "(%d/%d players > %.2f VAR, max %.2f); expected -- VAR is anchored to the frozen "
            "board, so this only reflects churn in the rebuilt scale (YTD/luck sides).",
            over,
            joined,
            tol,
            summary["max"],
        )
    return summary


def _frozen_var_by_player_id(frozen_path: Path | str | None = None) -> dict[str, float]:
    """Authoritative draft-day VAR per ``player_id`` from the frozen board.

    ``draft_state_board.json`` is the board the league actually drafted against; its
    ``player_id`` is ``fg_id::player_type`` (e.g. ``"30279::pitcher"``), the SAME key
    ``board.py`` builds for the rebuilt board -- so the anchor join depends on that
    fg_id key format staying in lockstep between the two boards, not on mlbam. A missing,
    malformed, null-, non-numeric-, or non-finite-``var`` row is skipped, so a corrupt file
    yields an empty or partial dict; this LOADER itself never raises. Whether a small/empty
    result is safe to ignore or should fail loud is the ANCHOR's decision
    (``_anchor_board_var_to_frozen``), not this loader's.
    """
    frozen_path = Path(frozen_path) if frozen_path else _FROZEN_BOARD
    try:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("draft-value: no frozen VAR anchor (%s): %s", frozen_path, exc)
        return {}
    out: dict[str, float] = {}
    for row in frozen:
        pid, var = row.get("player_id"), row.get("var")
        if pid is None or var is None:
            continue
        try:
            v = float(var)
        except (TypeError, ValueError):
            continue  # corrupt non-numeric var: skip rather than raise (soft contract)
        if not math.isfinite(v):
            continue  # NaN/inf var: also corrupt -- a NaN anchor would read as unmatched
        out[pid] = v
    return out


def _anchor_board_var_to_frozen(board: pd.DataFrame, frozen_var: dict[str, float]) -> pd.DataFrame:
    """Replace the rebuilt board's VAR with the frozen draft-day VAR (join by player_id).

    The rebuilt board faithfully reproduces the draft-day SCALE (denoms/floors/volumes)
    and per-player projected lines, but its recomputed VAR drifts from the frozen board
    for ~100% of players (code/config churn since the freeze). Every f=1 quantity --
    ``preseason_var``, ``skill``, ``luck``, and the projected par curve -- must be the
    draft-day VAR the league actually drafted against, not the drifted rebuild, so anchor
    the board's ``var`` column to the frozen values here (before ``_board_index``, so the
    VAR tie-break also uses draft-day truth). A few rows the frozen board lacks (post-freeze
    additions) keep their rebuilt VAR (logged); but when the anchor CANNOT be meaningfully
    applied -- an empty frozen board, or a near-total key-format mismatch -- this raises
    rather than silently ship grades on the drifted rebuilt VAR (run_draft_value is wrapped
    by the refresh's try/except, which keeps the prior good cache). The rebuilt projected
    LINES still drive the to-date (f<1) rescale, which does not read ``var`` -- that is the
    only thing the rebuild is kept for.
    """
    if not frozen_var:
        raise RuntimeError(
            "draft-value: no frozen VAR to anchor to -- draft_state_board.json is missing, "
            "unreadable, or has no usable rows. Refusing to ship draft grades on the drifted "
            "rebuilt VAR (the refresh keeps the prior good cache); restore the frozen board."
        )
    board = board.copy()
    mapped = board["player_id"].map(frozen_var)
    unmatched = int(mapped.isna().sum())
    matched = len(board) - unmatched
    # A near-total mismatch means the two boards' player_id key formats have diverged:
    # board.py keys the whole board fg_id::type all-or-nothing, so one fg_id-less projection
    # row flips it to name::type while the frozen board stays fg_id::type, dropping the join
    # to ~0 matches. Silently keeping the drifted rebuilt VAR for everyone would defeat the
    # anchor invisibly, so fail loud -- run_draft_value is wrapped by the refresh's
    # try/except, which leaves the prior good draft-value cache untouched on a raise.
    if matched < 0.5 * len(board):
        raise RuntimeError(
            f"draft-value: frozen VAR anchor matched only {matched}/{len(board)} board rows -- "
            "the rebuilt and frozen board player_id key formats have likely diverged "
            "(expected fg_id::player_type on both). Refusing to ship draft grades on the "
            "drifted rebuilt VAR; check board.py fg_id keying vs draft_state_board.json."
        )
    if unmatched:
        logger.warning(
            "draft-value: %d/%d board rows have no frozen VAR anchor; keeping the rebuilt "
            "VAR for those (off-board or added since the freeze).",
            unmatched,
            len(board),
        )
    board["var"] = mapped.fillna(board["var"])
    return board


_COUNTING_HIT = ("r", "hr", "rbi", "sb", "ab")  # ab is volume (scales)
_COUNTING_PIT = ("w", "k", "sv", "ip")  # ip is volume (scales)


def _sgp(line: dict[str, Any], scale: ScaleInputs, team_ab: float, team_ip: float) -> float:
    """Shared player-SGP call: score ``line`` on the board scale's denoms/repl rates."""
    return calculate_player_sgp(
        pd.Series(line),
        denoms=scale.denoms,
        # Floor team volumes at 1: near f=0 (opening day) team_ab/team_ip scale toward 0
        # and int() truncates them to 0, which zeroes the rate-SGP denominator
        # (one_sgp_in_hits = denom * team_ab) and yields NaN. A real run (f>~0.001) is
        # unaffected; this only rescues the degenerate to-date scale from silent NaN.
        team_ab=max(1, int(team_ab)),
        team_ip=max(1, int(team_ip)),
        replacement_avg=scale.repl_rates["avg"],
        replacement_era=scale.repl_rates["era"],
        replacement_whip=scale.repl_rates["whip"],
    )


def _to_date_floors(scale: ScaleInputs, fraction: float) -> dict[str, float]:
    """Position floors on a to-date scale (NOT scale.replacement_levels * f).

    Floor SGP is NOT linear in f: its rate component is f-invariant while only the
    counting component scales. This is the SAME empirical-floor computation the board
    build uses, so delegate to the shared ``position_aware_replacement_levels`` with the
    board scale's denoms/rates/volumes and the elapsed ``fraction`` -- rather than
    re-encoding the counting-scaling recipe and the ``UTIL = max(hitter floors)`` rule
    here. At f=1 return the board's own floors unchanged (the cheap projected-side path).
    """
    if fraction == 1.0:
        return scale.replacement_levels
    return position_aware_replacement_levels(
        scale.denoms,
        scale.repl_rates,
        team_ab=scale.team_ab,
        team_ip=scale.team_ip,
        fraction=fraction,
    )


def score_var(
    line: dict[str, Any],
    positions: list[str],
    player_type: str,
    scale: ScaleInputs,
    fraction: float = 1.0,
    scale_counting: bool = True,
    floors: dict[str, float] | None = None,
) -> float:
    """Score a stat line into VAR on the board scale (projected or YTD-scaled).

    ``fraction < 1.0`` applies the YTD to-date scaling: the team volumes scale by
    ``fraction`` and the position floors are recomputed on the same to-date scale,
    while rates (AVG/ERA/WHIP) are held. ``player_type`` is ``"hitter"`` or
    ``"pitcher"``. The to-date floors default to ``_to_date_floors(scale, fraction)``
    (which early-returns the board floors at ``fraction==1.0``, so the projected path
    stays cheap); pass a prebuilt ``floors`` to reuse one dict across the many YTD calls
    at a fixed ``fraction`` instead of rebuilding it each call.

    ``scale_counting`` governs the LINE's counting stats. The EXPECTED / par side
    (a projected full-season line) keeps the default ``True`` so its counting stats
    scale by ``fraction`` to a to-date expectation. The ACTUAL-to-date side passes
    ``False``: those counting stats are already accumulated to date and must be used
    as-is -- per the spec, only the team denominators scale on the actual side.
    Double-scaling the actual line understates every player's to-date value.
    """
    # player_type must be "hitter"/"pitcher" (StrEnum-compatible) so calculate_player_sgp
    # dispatches (player.get("player_type") == PlayerType.HITTER/PITCHER, player_value.py:104,121).
    scaled: dict[str, Any] = dict(line)
    scaled["player_type"] = player_type
    counting = _COUNTING_HIT if player_type == "hitter" else _COUNTING_PIT
    if fraction != 1.0 and scale_counting:
        for k in counting:
            if scaled.get(k) is not None:
                scaled[k] = scaled[k] * fraction
    team_ab = scale.team_ab * fraction
    team_ip = scale.team_ip * fraction
    total_sgp = _sgp(scaled, scale, team_ab, team_ip)
    if floors is None:
        floors = _to_date_floors(scale, fraction)
    # calculate_var routes the pitcher floor by role (SP vs RP) at a mid-season IP cutoff,
    # but a starter/reliever ROLE is a full-season property. Pass a full-season-equivalent
    # ip as the explicit role_ip override so a real SP is graded vs the SP floor even
    # mid-season (its to-date ip ~90 would otherwise route to the RP floor, and the actual
    # and par sides could land on opposite sides of the cutoff). f in {0, 1} needs no
    # rescale.
    if player_type == "pitcher" and fraction not in (0.0, 1.0):
        # par line carries its pre-scale projected ip; the actual line extrapolates its
        # accumulated ip to full-season pace (ip / f).
        raw_ip = line.get("ip", 0.0)
        raw_ip = 0.0 if raw_ip is None else float(raw_ip)
        routing_ip = raw_ip if scale_counting else raw_ip / fraction
    else:
        raw_ip = scaled.get("ip", 0.0)
        routing_ip = 0.0 if raw_ip is None else float(raw_ip)
    series = pd.Series(
        {
            "total_sgp": total_sgp,
            "positions": list(positions),
            "player_type": player_type,
        }
    )
    return calculate_var(series, floors, role_ip=routing_ip)


@dataclass(frozen=True)
class DraftPick:
    slot: int | None  # 1..200 live-pick ordinal; None for keepers
    team: str
    player_name: str
    is_keeper: bool


def reconstruct_draft(
    config: LeagueConfig | None = None, state: dict[str, Any] | None = None
) -> list[DraftPick]:
    """Reconstruct (team, slot) per 2026 pick from draft_order + draft_state + keepers.

    ``draft_order.json`` ``rounds`` is the full 23x10 snake order; rounds 1-3 (the
    first 30 slots) are consumed by the 30 keepers, so the 200 live picks map to the
    remaining slots. ``drafted_players[0:30]`` are the keepers in league.yaml order;
    ``drafted_players[30:230]`` are the live picks in snake order. Trades are applied
    on the absolute ``[round-1][slot-1]`` cell before the keeper-round slots are
    skipped. Pass a pre-parsed ``state`` (draft_state.json) to avoid re-reading it when
    the caller already has it; ``None`` reads it here.
    """
    if config is None:
        config = load_config(_CONFIG)
    if state is None:
        state = json.loads(_DRAFT_STATE.read_text(encoding="utf-8"))
    order = json.loads(_DRAFT_ORDER.read_text(encoding="utf-8"))
    drafted: list[str] = state["drafted_players"]
    keeper_defs: list[dict[str, Any]] = config.keepers
    # load_config defaults keepers to [] on a missing/misspelled section (config.py),
    # so fail fast with a clear cause rather than a misleading downstream pick-count error.
    if not keeper_defs:
        raise ValueError(
            f"No keepers in {_CONFIG}: draft reconstruction requires the league.yaml "
            "'keepers:' section. Check the config is present and the key is spelled correctly."
        )

    picks: list[DraftPick] = []
    n_keep = len(keeper_defs)
    for i, kd in enumerate(keeper_defs):
        picks.append(DraftPick(None, kd["team"], drafted[i], True))

    # live picks: flatten ALL rounds, apply trades on the absolute cell, then SKIP the
    # first n_keep keeper-round slots and zip to live picks.
    rounds: list[list[str]] = [list(r) for r in order["rounds"]]
    for tr in order.get("trades", []):
        rounds[tr["round"] - 1][tr["slot"] - 1] = tr["to"]
    flat_teams = [team for rnd in rounds for team in rnd]
    live_teams = flat_teams[n_keep:]  # drop the 30 keeper-round slots (rounds 1-3)
    live = drafted[n_keep:]
    assert len(live) == len(live_teams), (
        f"draft reconstruction mismatch: {len(live)} live picks vs "
        f"{len(live_teams)} live slots (both should be 200)"
    )
    for slot, (name, team) in enumerate(zip(live, live_teams, strict=True), start=1):
        picks.append(DraftPick(slot, team, name, False))
    return picks


def validate_reconstruction(
    picks: list[DraftPick],
    known_team: str | None = None,
    known_roster: list[str] | None = None,
    config: LeagueConfig | None = None,
) -> list[str]:
    """Runtime gate: every keeper maps to its owning team, and (optionally) a known
    team's roster reconstructs as a superset. Returns a list of problems ([] == pass).
    """
    problems: list[str] = []
    if config is None:
        config = load_config(_CONFIG)
    keepers = config.keepers
    n_live = sum(1 for p in picks if not p.is_keeper)
    if n_live != 200:
        problems.append(f"non-keeper pick count {n_live} != 200")
    keeper_norm = {(normalize_name(k["name"]), k["team"]) for k in keepers}
    recon_keeper_norm = {(normalize_name(p.player_name), p.team) for p in picks if p.is_keeper}
    missing = keeper_norm - recon_keeper_norm
    if missing:
        problems.append(f"keeper mismatch (missing {len(missing)}): {sorted(missing)[:5]}")
    if known_team and known_roster:
        recon = {normalize_name(p.player_name) for p in picks if p.team == known_team}
        want = {normalize_name(n) for n in known_roster}
        if not want <= recon:
            problems.append(
                f"known-team {known_team} roster mismatch: missing {sorted(want - recon)[:5]}"
            )
    return problems


def _board_index(board: pd.DataFrame) -> dict[str, Any]:
    """Map ``name_normalized::player_type`` -> board row (VAR tie-break on collisions)."""
    idx: dict[str, Any] = {}
    for _, row in board.iterrows():
        key = rank_key(row["name"], row["player_type"])
        cur = idx.get(key)
        if cur is None:
            idx[key] = row
            continue
        # Same normalized-name + type collision -- distinct namesakes, e.g. the two
        # active 2026 Max Muncys. Keep the higher-VAR row and warn (mirrors the
        # namesake-collision convention in _insert_by_name); silently dropping one
        # would grade a drafted player as the other. Long-term fix: carry per-pick
        # Yahoo/mlbam ids through reconstruction.
        cur_var, new_var = float(cur["var"]), float(row["var"])
        logger.warning(
            "Board namesake collision on %s: keeping VAR=%.2f, dropping VAR=%.2f",
            key,
            max(cur_var, new_var),
            min(cur_var, new_var),
        )
        if new_var > cur_var:
            idx[key] = row
    return idx


@dataclass
class ParCurve:
    """Draft-slot par expectation: sorted on-board drafted VAR plus keeper flat par."""

    drafted_pars: list[float]
    keeper_par: float

    def par_for_slot(self, ordinal: int) -> float:
        # ordinal is 1-based among ON-BOARD drafted picks (sorted descending)
        return self.drafted_pars[ordinal - 1]


def _var_for_row(
    row: Any,
    scale: ScaleInputs | None,
    fraction: float,
    floors: dict[str, float] | None = None,
) -> float:
    """VAR for a board row: preseason VAR at f=1, else rescored on the to-date scale."""
    if fraction == 1.0:
        return float(row["var"])
    if scale is None:
        raise ValueError("scale is required when fraction != 1.0")
    ptype = str(row["player_type"])
    keys = (
        ("r", "hr", "rbi", "sb", "avg", "ab")
        if ptype == "hitter"
        else ("w", "k", "sv", "era", "whip", "ip")
    )
    line = {k: row[k] for k in keys}
    return score_var(line, list(row["positions"]), ptype, scale, fraction, floors=floors)


def build_par_curve(
    typed_picks: list[tuple[DraftPick, str]],
    bindex: dict[str, Any],
    fraction: float = 1.0,
    scale: ScaleInputs | None = None,
    floors: dict[str, float] | None = None,
) -> ParCurve:
    """Build the par curve from typed picks joined to the preseason board.

    Each pick is matched to its board row by its ASSIGNED ``name::player_type``, so a
    two-way player's drafted-pitcher pick contributes his PITCHER VAR (not his hitter
    VAR) -- consistent with how that pick is scored, and not double-counting the bat
    already credited to his keeper. On-board drafted players contribute their
    (optionally to-date rescored) preseason VAR to a descending par curve; off-board
    fliers are skipped so the curve shrinks. Keeper par is the flat mean of the keeper
    VARs. ``fraction < 1.0`` requires ``scale`` so ``_var_for_row`` can rescore to the
    to-date scale; pass ``floors`` (the to-date floors at that ``fraction``) to reuse
    one dict across all picks instead of rebuilding it per pick.
    """
    drafted_vars: list[float] = []
    keeper_vars: list[float] = []
    for pick, ptype in typed_picks:
        row = bindex.get(rank_key(pick.player_name, ptype))
        if row is None:
            continue  # off-board flier: excluded from par curve
        v = _var_for_row(row, scale, fraction, floors=floors)
        (keeper_vars if pick.is_keeper else drafted_vars).append(v)
    drafted_vars.sort(reverse=True)
    keeper_par = sum(keeper_vars) / len(keeper_vars) if keeper_vars else float("nan")
    return ParCurve(drafted_vars, keeper_par)


def _hit_line_from(rec: dict[str, Any]) -> dict[str, Any]:
    # A JSON-null on a KV round-trip can land ab=None; calculate_avg guards on
    # denom > 0, which raises TypeError against a literal None, so coerce None->0.
    ab = rec.get("ab", 0)
    if ab is None:
        ab = 0
    return {
        "r": rec.get("r", 0),
        "hr": rec.get("hr", 0),
        "rbi": rec.get("rbi", 0),
        "sb": rec.get("sb", 0),
        "ab": ab,
        "avg": calculate_avg(rec.get("h", 0), ab, default=0.0),
    }


def _pit_line_from(rec: dict[str, Any]) -> dict[str, Any]:
    # ip=None (JSON-null on a KV round-trip) would raise TypeError in the rate-stat
    # denom guards (denom > 0), so coerce None->0 before scoring.
    ip = rec.get("ip", 0.0)
    if ip is None:
        ip = 0.0
    return {
        "w": rec.get("w", 0),
        "k": rec.get("k", 0),
        "sv": rec.get("sv", 0),
        "ip": ip,
        "era": calculate_era(rec.get("er", 0), ip, default=0.0),
        "whip": calculate_whip(rec.get("bb", 0), rec.get("h_allowed", 0), ip, default=0.0),
    }


def _row_mlbam(row: Any) -> int | None:
    """Extract a real integer mlbam id from a board row or projection record.

    Board rows and projection records carry ``mlbam_id`` as a float that may be
    NaN (an id-less player). Returns the int id when present and finite, else
    ``None`` (also None for a ``None`` row, so callers can pass a missing row).
    """
    try:
        v = row["mlbam_id"]
    except (KeyError, IndexError, TypeError):
        return None
    if v is None or v != v:  # None or NaN (NaN != NaN)
        return None
    return int(v)


def _insert_by_name(
    by_name: dict[str, Any],
    name_mlbam: dict[str, int | None],
    key: str,
    line: dict[str, Any],
    volume_key: str,
    mlbam: int | None,
) -> None:
    """Insert ``line`` into ``by_name`` keeping the higher-volume namesake.

    When two different players normalize to the same ``name::player_type`` key,
    the real MLB player (more ``ab`` for hitters / ``ip`` for pitchers) wins over
    a scrub namesake, so the fallback name-join scores the intended player. The
    collision is logged (not swallowed), naming both mlbam ids. The mlbam-keyed
    map is the authoritative join; this name map is only a fallback for rows that
    lack an id on either side.
    """
    cur = by_name.get(key)
    if cur is None:
        by_name[key] = line
        name_mlbam[key] = mlbam
        return
    cur_vol = float(cur[volume_key])
    new_vol = float(line[volume_key])
    new_wins = new_vol > cur_vol
    logger.warning(
        "draft-value: namesake collision on %s (%s=%.1f); keeping mlbam=%s over "
        "mlbam=%s (kept %.1f vs dropped %.1f)",
        key,
        volume_key,
        max(cur_vol, new_vol),
        mlbam if new_wins else name_mlbam.get(key),
        name_mlbam.get(key) if new_wins else mlbam,
        max(cur_vol, new_vol),
        min(cur_vol, new_vol),
    )
    if new_wins:
        by_name[key] = line
        name_mlbam[key] = mlbam


def load_full_season_lines() -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Full-season projection lines, keyed by mlbam id AND by ``name::player_type``.

    Reads ``CacheKey.FULL_SEASON_PROJECTIONS`` from the KV store (Upstash on
    Render, SQLite locally). Each record carries a real ``mlbam_id`` and a
    ``name``. Returns ``(by_mlbam, by_name)``: ``by_mlbam`` is the authoritative
    join (immune to namesake collisions like the two Mason Millers), keyed by int
    mlbam id; ``by_name`` is a fallback keyed ``name_normalized::player_type`` that
    keeps the higher-volume namesake so the real MLB player wins over a scrub.
    Returns ``({}, {})`` when the KV store lacks the blob (unsynced local runtime).
    """
    payload = read_cache_dict(CacheKey.FULL_SEASON_PROJECTIONS) or {}
    by_mlbam: dict[tuple[int, str], dict[str, Any]] = {}
    by_name: dict[str, Any] = {}
    name_mlbam: dict[str, int | None] = {}
    for ptype, recs, builder, vol in (
        ("hitter", payload.get("hitters", []), _hit_line_from, "ab"),
        ("pitcher", payload.get("pitchers", []), _pit_line_from, "ip"),
    ):
        for rec in recs:
            line = builder(rec)
            mlbam = _row_mlbam(rec)
            if mlbam is not None:
                # Key by (mlbam, player_type): a two-way player (e.g. Ohtani) has a
                # hitter AND a pitcher record under ONE mlbam id; keying by id alone
                # would let one overwrite the other.
                by_mlbam[(mlbam, ptype)] = line
            name = rec.get("name") or ""
            if not name:
                continue  # nameless record: mlbam-keyed above, but no name fallback
            key = rank_key(name, ptype)
            _insert_by_name(by_name, name_mlbam, key, line, vol, mlbam)
    return by_mlbam, by_name


def load_actual_to_date_lines() -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    """Actual season-to-date lines, keyed by mlbam id AND by ``name::player_type``.

    Reads aggregated game-log totals via ``get_game_log_totals`` (keyed by string
    mlbam id, with a ``name`` field). Returns ``(by_mlbam, by_name)``: ``by_mlbam``
    is the authoritative int-mlbam join; ``by_name`` is a fallback keyed
    ``name_normalized::player_type`` keeping the higher-volume namesake. Returns
    ``({}, {})`` when the KV store has no game logs.
    """
    client = get_kv()
    by_mlbam: dict[tuple[int, str], dict[str, Any]] = {}
    by_name: dict[str, Any] = {}
    name_mlbam: dict[str, int | None] = {}
    for ptype, logs, builder, vol in (
        ("hitter", get_game_log_totals(client, "hitters"), _hit_line_from, "ab"),
        ("pitcher", get_game_log_totals(client, "pitchers"), _pit_line_from, "ip"),
    ):
        for mid_str, rec in logs.items():
            line = builder(rec)
            try:
                mlbam: int | None = int(mid_str)
            except (TypeError, ValueError):
                mlbam = None
            if mlbam is not None:
                by_mlbam[(mlbam, ptype)] = line
            name = rec.get("name") or ""
            if not name:
                continue
            key = rank_key(name, ptype)
            _insert_by_name(by_name, name_mlbam, key, line, vol, mlbam)
    return by_mlbam, by_name


@dataclass
class PlayerValue:
    """Per-player realized value with a projected skill/luck decomposition.

    ``value_proj`` subtracts the full-season ``baseline_proj``; ``value_ytd``
    subtracts the to-date ``baseline_ytd`` -- the two horizons have distinct pars
    and must never be conflated. ``skill``/``luck`` are only defined when the player
    was on the preseason board (``preseason_var is not None``); off-board fliers get
    ``None`` for both (value vs replacement is still computed).
    """

    team: str
    name: str
    player_type: str
    slot: int | None
    baseline_kind: str
    preseason_var: float | None
    est_var_proj: float | None
    est_var_ytd: float | None
    value_proj: float | None
    value_ytd: float | None
    skill: float | None
    luck: float | None


def compute_player_value(
    team: str,
    name: str,
    player_type: str,
    positions: list[str],
    baseline_proj: float,
    baseline_ytd: float,
    baseline_kind: str,
    preseason_var: float | None,
    full_line: dict[str, Any] | None,
    todate_line: dict[str, Any] | None,
    scale: ScaleInputs,
    fraction: float,
    slot: int | None = None,
    missing_line_est: float | None = None,
    floors_ytd: dict[str, float] | None = None,
) -> PlayerValue:
    """Score a player's projected and YTD VAR and decompose the projected value.

    ``full_line`` is scored at ``fraction=1.0`` (full-season projection); ``todate_line``
    is scored at the elapsed ``fraction`` (to-date scale). ``value_proj``/``value_ytd``
    subtract the full-season and to-date baselines respectively (never conflate them).
    ``skill``/``luck`` split the projected value only when both a preseason VAR and a
    projected estimate exist; otherwise both are ``None``.

    ``missing_line_est`` is the estimated VAR used when a stat line is absent. Default
    ``None`` yields ``None`` estimates (skip). Pass ``0.0`` to score a drafted/kept
    player who never played at replacement level, so value == ``0 - par == -par`` and a
    wasted pick is penalized rather than silently dropped. ``floors_ytd`` is the
    prebuilt to-date floors at ``fraction`` (reused across picks); ``None`` rebuilds it.
    """
    est_proj = (
        score_var(full_line, positions, player_type, scale, 1.0)
        if full_line is not None
        else missing_line_est
    )
    est_ytd = (
        # todate_line is the ACTUAL season-to-date accumulation: score it as-is
        # (scale_counting=False) so its counting stats are NOT re-scaled by f. Only
        # the team denominators + floors scale to date (spec: YTD to-date scaling).
        score_var(
            todate_line,
            positions,
            player_type,
            scale,
            fraction,
            scale_counting=False,
            floors=floors_ytd,
        )
        if todate_line is not None
        else missing_line_est
    )
    value_proj = (est_proj - baseline_proj) if est_proj is not None else None
    value_ytd = (est_ytd - baseline_ytd) if est_ytd is not None else None
    if preseason_var is not None and est_proj is not None:
        skill: float | None = preseason_var - baseline_proj
        luck: float | None = est_proj - preseason_var
    else:
        skill = luck = None
    return PlayerValue(
        team,
        name,
        player_type,
        slot,
        baseline_kind,
        preseason_var,
        est_proj,
        est_ytd,
        value_proj,
        value_ytd,
        skill,
        luck,
    )


def season_fraction(config: LeagueConfig | None = None) -> float:
    """League-wide elapsed fraction of the MLB season, computed purely from the date.

    Reads ``season_start``/``season_end`` from ``config/league.yaml`` and uses the
    user's local ``today`` (``local_today``), deriving the elapsed fraction as
    ``1 - compute_fraction_remaining(...)`` clamped to ``[0, 1]``. This is a
    date-based approximation of games played (it does NOT read any standings
    snapshot or per-team game count); it is the fallback source of the to-date
    scaling fraction ``f`` when the caller does not thread its own (RefreshRun does).
    """
    if config is None:
        config = load_config(_CONFIG)
    season_start = date.fromisoformat(config.season_start)
    season_end = date.fromisoformat(config.season_end)
    elapsed = 1.0 - compute_fraction_remaining(season_start, season_end, local_today())
    return max(0.0, min(1.0, elapsed))


@dataclass
class TeamRollup:
    """Per-team draft-value roll-up: sum, per-player average, and credited count.

    ``sum_value`` sums the chosen-horizon per-player values (``value_proj`` when
    ``horizon == "proj"``, else ``value_ytd``); ``avg_value`` divides by the number
    of credited players (``NaN`` when none are credited). ``credited_count`` is that
    number: every drafted pick + keeper credited to the team that drafted/kept the
    player, regardless of a later drop or trade.
    """

    team: str
    sum_value: float
    avg_value: float
    credited_count: int


def roll_up_team(
    team: str,
    player_values: list[PlayerValue],
    horizon: str = "proj",
) -> TeamRollup:
    """Roll a team's per-player values into sum, per-player average, and count.

    ``horizon`` picks ``value_proj`` (``"proj"``) or ``value_ytd`` (otherwise). Only
    finite values are credited (values can be ``0.0`` or negative, so filter on
    ``is not None``, never truthiness; ``NaN`` is also dropped so an unmatched keeper's
    NaN par cannot poison the whole team sum). ``avg_value`` is ``NaN`` when no player
    is credited. ``player_values`` covers every drafted pick + keeper credited to the
    team, so the sum, average, and credited count grade the full draft.
    """
    attr = "value_proj" if horizon == "proj" else "value_ytd"
    # Dropping NaN avoids poisoning the sum, but NaN means a player couldn't be graded
    # (e.g. keeper_par is NaN because no keeper matched the board) -- surface it rather
    # than silently grading over an incomplete roster.
    vals: list[float] = []
    dropped_nan = 0
    for pv in player_values:
        v = getattr(pv, attr)
        if v is None:
            continue
        if math.isnan(v):
            dropped_nan += 1
        else:
            vals.append(v)
    if dropped_nan:
        logger.warning(
            "draft-value: team %s has %d ungradeable player(s) (NaN %s -- unmatched "
            "keeper/board?) excluded from the roll-up; the grade covers the rest.",
            team,
            dropped_nan,
            attr,
        )
    n = len(vals)
    total = sum(vals)
    avg = total / n if n else float("nan")
    return TeamRollup(team, total, avg, n)


def _default_positions(player_type: str) -> list[str]:
    """Off-board default positions matching the board's default-attach convention."""
    return ["OF"] if player_type == "hitter" else ["SP"]


def _resolve_type(avail_types: set[str]) -> str:
    """Fallback type when notes/claims don't decide: the sole board type, else hitter.

    ``avail_types`` is drawn from {"hitter", "pitcher"}, so a 2-element set is always
    {hitter, pitcher} and an empty set is off-board -- both default to hitter.
    """
    if len(avail_types) == 1:
        return next(iter(avail_types))
    return "hitter"


def _assign_pick_types(
    picks: list[DraftPick],
    bindex: dict[str, Any],
    keepers: list[dict[str, Any]],
) -> list[tuple[DraftPick, str]]:
    """Assign a player_type to every pick, two-way aware.

    Draft-pick names are bare (no Yahoo " (Batter)"/" (Pitcher)" suffix), so a
    two-way player (e.g. Shohei Ohtani) appears as two picks under one name -- once
    as a keeper and once as a drafted pick. Iterating picks IN ORDER and tracking
    which board types each normalized name has already claimed lets the second pick
    take the remaining type: Ohtani's keeper claims "hitter" (his "batter only"
    keeper note), so the drafted Ohtani takes "pitcher".

    A KEEPER honors its league.yaml note ("batter only" -> hitter, "pitcher only" ->
    pitcher); otherwise it falls back to the sole board type (else hitter). A DRAFTED
    pick prefers an unclaimed board type (pitcher first, so a two-way second pick
    becomes pitcher), else the same sole-board-type-or-hitter fallback.

    KNOWN LIMITATION: picks carry only a name (draft_state has no per-pick mlbam id),
    so this cannot distinguish ONE two-way player from TWO DISTINCT same-name players
    of different types (e.g. a hitter and a pitcher both named "Will Smith"). Both are
    treated as a single two-way name and the board's two types are split across the
    picks by claim order, which mis-assigns type (and thus the joined stat line) for
    genuinely-distinct namesakes. Fixing this needs per-pick identity threaded through
    reconstruct_draft. Accepted for v1 (no such collision in the 2026 draft); revisit
    if the reconstruction ever gains mlbam ids.
    """
    avail: dict[str, set[str]] = {}
    for key in bindex:
        norm, ptype = key.rsplit("::", 1)
        avail.setdefault(norm, set()).add(ptype)
    keeper_notes: dict[str, str] = {}
    for k in keepers:
        note = k.get("note")
        if note:
            keeper_notes[normalize_name(k["name"])] = str(note).lower()

    claimed: dict[str, set[str]] = {}
    result: list[tuple[DraftPick, str]] = []
    for p in picks:
        norm = normalize_name(p.player_name)
        avail_types = avail.get(norm, set())
        if p.is_keeper:
            note = keeper_notes.get(norm, "")
            if "batter only" in note:
                ptype = "hitter"
            elif "pitcher only" in note:
                ptype = "pitcher"
            elif len(avail_types) > 1:
                # Two-way keeper (board carries both types) with no recognized
                # disambiguating note. Refuse to guess: a reworded or missing note
                # must fail loud, not silently fall through to _resolve_type and flip
                # the type, corrupting the par curve. Long-term fix: a structured
                # keeper player_type field in league.yaml validated at config load.
                raise ValueError(
                    f"Two-way keeper {p.player_name!r} needs a 'batter only' or "
                    f"'pitcher only' league.yaml note to disambiguate; got {note!r}"
                )
            else:
                ptype = _resolve_type(avail_types)
        else:  # drafted
            unclaimed = avail_types - claimed.get(norm, set())
            if unclaimed:
                ptype = "pitcher" if "pitcher" in unclaimed else next(iter(unclaimed))
            else:
                ptype = _resolve_type(avail_types)
        claimed.setdefault(norm, set()).add(ptype)
        result.append((p, ptype))
    return result


def run_draft_value(
    fraction: float | None = None,
    config: LeagueConfig | None = None,
) -> tuple[list[PlayerValue], list[TeamRollup]]:
    """Orchestrate the draft-value metric end-to-end against the synced KV store.

    Reproduces the preseason board (+ soft frozen drift check), anchors the f=1 VAR to
    the frozen draft-day board, reconstructs the draft and enforces the reconstruction
    gate, builds projected and to-date par curves, then scores EVERY drafted pick +
    keeper (30 keepers + 200 drafted) by full-season realized value vs par, crediting
    the team that drafted/kept the player -- regardless of a later drop or trade. Waivers
    stay out (deferred to the transaction analyzer). A drafted/kept player who never
    played is scored at replacement (value == -par), so a wasted pick is penalized.

    ``config`` and ``fraction`` are threaded from the caller (RefreshRun) so the whole
    refresh reads league.yaml once and shares ONE season fraction; ``None`` loads the
    config and derives the date-based fraction as a standalone fallback. Returns
    ``(player_values, team_rollups)``.
    """
    if config is None:
        config = load_config(_CONFIG)
    board, scale = reproduce_draft_day_board(config)
    # Load the frozen draft-day VAR once and feed both consumers (the soft drift check
    # and the anchor) instead of reading draft_state_board.json twice.
    frozen_var = _frozen_var_by_player_id()
    frozen_drift_summary(board, frozen_var=frozen_var)  # soft: logs (INFO) on drift, never raises
    # M7: anchor the f=1 VAR (preseason_var / skill / luck / projected par curve) to the
    # frozen draft-day board; the rebuilt board's VAR drifts, and only its lines+scale
    # are needed (for the to-date rescale). Must precede _board_index (VAR tie-break).
    board = _anchor_board_var_to_frozen(board, frozen_var)
    # Read draft_state.json once and share it: reconstruct_draft needs drafted_players,
    # the gate below needs user_roster.
    state = json.loads(_DRAFT_STATE.read_text(encoding="utf-8"))
    picks = reconstruct_draft(config, state=state)
    # enforce the reconstruction gate against the user's known roster (spec oracle 6b)
    user_roster: list[str] = state.get("user_roster") or []
    keepers = config.keepers
    keeper_team = {normalize_name(k["name"]): k["team"] for k in keepers}
    user_team = next(
        (keeper_team[normalize_name(n)] for n in user_roster if normalize_name(n) in keeper_team),
        None,
    )
    gate = validate_reconstruction(
        picks, known_team=user_team, known_roster=user_roster, config=config
    )
    if gate:
        raise RuntimeError(f"Draft reconstruction gate failed: {gate}")
    # Clamp to [0, 1]: season_fraction() already clamps, but a caller-threaded fraction
    # does not -- refresh_pipeline passes 1 - fraction_remaining, which goes NEGATIVE when a
    # refresh runs before season_start (compute_fraction_remaining clamps only its lower
    # bound). A negative f would scale every counting stat and floor negative -> garbage VAR.
    f = season_fraction(config) if fraction is None else fraction
    f = max(0.0, min(1.0, f))

    bindex = _board_index(board)
    # Assign each pick a player_type up front (two-way aware) so the par curve, the
    # slot index, and the scoring all join the board by the SAME name::player_type key.
    # Matching type-agnostically here would credit a two-way player's drafted-pitcher
    # pick with his hitter VAR, corrupting the drafted par curve for every slot.
    typed_picks = _assign_pick_types(picks, bindex, keepers)
    # The to-date floors are pure over (scale, f), so build them once and reuse across
    # the YTD par curve and every player's YTD score instead of rebuilding per call.
    floors_ytd = _to_date_floors(scale, f)
    par_proj = build_par_curve(typed_picks, bindex, fraction=1.0)
    par_ytd = build_par_curve(typed_picks, bindex, fraction=f, scale=scale, floors=floors_ytd)
    full_by_mlbam, full_by_name = load_full_season_lines()
    td_by_mlbam, td_by_name = load_actual_to_date_lines()

    # DRAFT-ORDER ordinal among on-board drafted picks, keyed by name::player_type.
    # par(slot) = drafted_pars[ordinal-1] (drafted_pars is VAR-sorted desc): the k-th
    # on-board drafted pick is measured against the k-th-best available VAR. This is the
    # spec's par(slot). Do NOT index by the player's OWN VAR rank -- that makes
    # par == own VAR, so skill == preseason_var - par == 0 for every drafted player.
    slot_by_key: dict[str, int] = {}
    onboard_ordinal = 0
    for pick, ptype in sorted(
        (tp for tp in typed_picks if not tp[0].is_keeper), key=lambda tp: tp[0].slot or 0
    ):
        key = rank_key(pick.player_name, ptype)
        if bindex.get(key) is None:
            continue  # off-board flier: excluded from par curve and slot indexing
        onboard_ordinal += 1
        slot_by_key[key] = onboard_ordinal

    # Score every pick (keepers + drafted), crediting pick.team.
    players: list[PlayerValue] = []
    for pick, ptype in typed_picks:
        key = rank_key(pick.player_name, ptype)
        row = bindex.get(key)  # may be None: off-board flier
        pre = float(row["var"]) if row is not None else None
        mlbam = _row_mlbam(row)
        positions = list(row["positions"]) if row is not None else _default_positions(ptype)
        # Join lines by mlbam id first (immune to namesake collisions like the two
        # Mason Millers), falling back to the name key only when the board row has
        # no id. Explicit None checks -- never `x or fallback` (a 0.0-valued line
        # dict is falsy but valid).
        full_line = full_by_mlbam.get((mlbam, ptype)) if mlbam is not None else None
        if full_line is None:
            full_line = full_by_name.get(key)
        todate_line = td_by_mlbam.get((mlbam, ptype)) if mlbam is not None else None
        if todate_line is None:
            todate_line = td_by_name.get(key)
        # draft-order ordinal among on-board drafted picks; also the compute slot arg
        # (slot == rank for drafted, None otherwise), so compute it once and reuse.
        rank = slot_by_key.get(key) if not pick.is_keeper else None
        if pick.is_keeper:
            base_proj, base_ytd = par_proj.keeper_par, par_ytd.keeper_par
        else:  # drafted
            base_proj = par_proj.par_for_slot(rank) if rank is not None else 0.0
            base_ytd = par_ytd.par_for_slot(rank) if rank is not None else 0.0
        pv = compute_player_value(
            team=pick.team,
            name=pick.player_name,
            player_type=ptype,
            positions=positions,
            baseline_proj=base_proj,
            baseline_ytd=base_ytd,
            baseline_kind=("keeper" if pick.is_keeper else "drafted"),
            preseason_var=pre,
            full_line=full_line,
            todate_line=todate_line,
            scale=scale,
            fraction=f,
            slot=rank,  # already None for keepers (see rank assignment above)
            # A drafted/kept player with NO stat line (never played) is scored at
            # replacement (0.0), so value == 0 - par == -par (a wasted pick is penalized).
            missing_line_est=0.0,
            floors_ytd=floors_ytd,  # prebuilt to-date floors, reused across all picks
        )
        players.append(pv)

    # group by team (PlayerValue carries .team) and roll up
    by_team: dict[str, list[PlayerValue]] = {}
    for pv in players:
        by_team.setdefault(pv.team, []).append(pv)
    teams = [roll_up_team(t, pvs) for t, pvs in sorted(by_team.items())]
    return players, teams


def _finite(x: float | None) -> float | None:
    """Map None/NaN/inf -> None so the payload survives strict JSON + Jinja."""
    return x if x is not None and math.isfinite(x) else None


def _rank(value: float | None) -> float:
    """Sort key that sinks non-finite/None values to the bottom of a descending sort.

    Sinks exactly what ``_finite`` nulls (None/NaN/inf), so a value that renders as a
    blank ``-`` cell also sorts last rather than by a raw ``inf`` key.
    """
    return value if value is not None and math.isfinite(value) else -math.inf


def build_draft_value_cache(players: list[PlayerValue], teams: list[TeamRollup]) -> dict[str, Any]:
    """Serialize run_draft_value() output into a JSON-safe, template-ready dict.

    Groups ``players`` by ``.team`` under each ``TeamRollup`` (teams sorted by
    ``avg_value`` desc with NaN sunk; players sorted by ``value_proj`` desc with
    None/NaN sunk). Every float field passes through ``_finite`` so no non-finite
    value reaches strict JSON or Jinja. Within each team, a ``name`` appearing
    under more than one ``player_type`` gets a ` (H)`/` (P)` ``display_name``
    suffix (two-way disambiguation); all other rows get ``display_name == name``.
    """
    by_team: dict[str, list[PlayerValue]] = {}
    for p in players:
        by_team.setdefault(p.team, []).append(p)

    out_teams: list[dict[str, Any]] = []
    for tr in sorted(teams, key=lambda t: _rank(t.avg_value), reverse=True):
        roster = by_team.get(tr.team, [])
        types_by_name: dict[str, set[str]] = {}
        for p in roster:
            types_by_name.setdefault(p.name, set()).add(str(p.player_type))
        out_players: list[dict[str, Any]] = []
        for p in sorted(roster, key=lambda p: _rank(p.value_proj), reverse=True):
            suffix = ""
            if len(types_by_name.get(p.name, ())) > 1:
                suffix = " (P)" if str(p.player_type) == "pitcher" else " (H)"
            out_players.append(
                {
                    "name": p.name,
                    "display_name": f"{p.name}{suffix}",
                    "player_type": str(p.player_type),
                    "kind": p.baseline_kind,
                    "slot": p.slot,
                    "preseason_var": _finite(p.preseason_var),
                    "est_var_proj": _finite(p.est_var_proj),
                    "value_proj": _finite(p.value_proj),
                    "value_ytd": _finite(p.value_ytd),
                    "skill": _finite(p.skill),
                    "luck": _finite(p.luck),
                }
            )
        out_teams.append(
            {
                "team": tr.team,
                "avg_value": _finite(tr.avg_value),
                "sum_value": _finite(tr.sum_value),
                "credited_count": tr.credited_count,
                "players": out_players,
            }
        )
    return {"horizon": "proj", "teams": out_teams}
