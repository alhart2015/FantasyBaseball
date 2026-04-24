"""Roster capacity + fill state for the draft recommender.

Bundles the two dicts that were previously passed around as a pair
(``filled_positions`` + ``roster_slots``) into a single dataclass with
Position-keyed maps and the derived queries the recommender needs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from fantasy_baseball.models.positions import (
    BENCH_SLOTS,
    IL_SLOTS,
    Position,
)
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot


@dataclass(frozen=True)
class RosterState:
    """Immutable snapshot of filled slots and league capacity.

    Both maps are Position-keyed. Use :meth:`from_dicts` to construct
    from the string-keyed dicts returned by ``get_filled_positions`` or
    read from ``config/league.yaml``.
    """

    filled: Mapping[Position, int]
    capacity: Mapping[Position, int]

    @classmethod
    def from_dicts(
        cls,
        filled: Mapping[str | Position, int],
        capacity: Mapping[str | Position, int],
    ) -> RosterState:
        return cls(
            filled=_coerce(filled),
            capacity=_coerce(capacity),
        )

    def open_slots(
        self,
        *,
        exclude: frozenset[Position] = IL_SLOTS,
    ) -> dict[Position, int]:
        """Slots with remaining capacity, mapped to the remaining count.

        ``exclude`` defaults to IL variants — you don't draft to IL, but
        BN is a valid destination so it stays in by default. Pass
        ``BENCH_SLOTS`` to get only starter slots.
        """
        out: dict[Position, int] = {}
        for pos, total in self.capacity.items():
            if pos in exclude:
                continue
            current = self.filled.get(pos, 0)
            if current < total:
                out[pos] = total - current
        return out

    def unfilled_starter_slots(self) -> set[Position]:
        """Starter slots (not BN/IL) that still have open capacity."""
        return set(self.open_slots(exclude=BENCH_SLOTS).keys())

    def any_slot_open_for(self, positions: Iterable[Position | str]) -> bool:
        """True if the player fits in any open (non-IL) slot."""
        open_ = self.open_slots()
        if not open_:
            return False
        pos_list = list(positions)
        return any(can_fill_slot(pos_list, slot) for slot in open_)


def _coerce(d: Mapping[str | Position, int]) -> dict[Position, int]:
    out: dict[Position, int] = {}
    for k, v in d.items():
        pos = k if isinstance(k, Position) else Position.parse(k)
        out[pos] = int(v)
    return out


# --- Slot-assignment helpers (moved from recommender.py 2026-04-24) ---


def compute_slot_scarcity_order(
    board: pd.DataFrame,
    roster_slots: dict[str, int] | None = None,
) -> list[str]:
    """Return roster slots ordered by positional scarcity (most scarce first).

    Scarcity = sum(SGP of all eligible players) / number of roster slots.
    Lower scarcity = fewer resources per slot = assign multi-eligible players
    here first so flex slots stay open for less flexible players.
    """
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS
    scarcity: dict[str, float] = {}
    for slot, n_slots in roster_slots.items():
        if slot in ("BN", "IL"):
            continue
        eligible = board[board["positions"].apply(lambda p, slot=slot: can_fill_slot(p, slot))]
        total_sgp = eligible["total_sgp"].sum() if "total_sgp" in eligible.columns else 0
        scarcity[slot] = total_sgp / n_slots if n_slots > 0 else float("inf")
    return sorted(scarcity.keys(), key=lambda s: scarcity[s])


def _board_content_hash(board: pd.DataFrame) -> str:
    pids = board["player_id"].astype(str).tolist() if "player_id" in board.columns else []
    vars_ = board["var"].round(2).astype(str).tolist() if "var" in board.columns else []
    h = hashlib.md5()
    for s in pids:
        h.update(s.encode())
    h.update(b"|")
    for s in vars_:
        h.update(s.encode())
    return h.hexdigest()


_scarcity_cache: dict[str, list[str]] = {}
_scarcity_cache_counters = {"hits": 0, "misses": 0}


def _scarcity_cache_stats() -> dict[str, int]:
    return dict(_scarcity_cache_counters)


def _scarcity_order_cached(
    board: pd.DataFrame,
    roster_slots: dict[str, int],
) -> list[str]:
    key = (
        _board_content_hash(board)
        + "|"
        + ",".join(f"{k}={v}" for k, v in sorted(roster_slots.items()))
    )
    if key in _scarcity_cache:
        _scarcity_cache_counters["hits"] += 1
        return _scarcity_cache[key]
    _scarcity_cache_counters["misses"] += 1
    _scarcity_cache.clear()
    order = compute_slot_scarcity_order(board, roster_slots)
    _scarcity_cache[key] = order
    return order


def _collect_roster_entries(
    user_roster_ids: list[str],
    board: pd.DataFrame,
    player_lookup: dict[str, pd.Series] | None = None,
) -> list[pd.Series]:
    """Look up board rows for the given roster player-ids."""
    if player_lookup is None:
        player_lookup = {row["player_id"]: row for _, row in board.iterrows()}
    players: list[pd.Series] = []
    for pid in user_roster_ids:
        if pid in player_lookup:
            players.append(player_lookup[pid])
            continue
        # Fallback: try name match for entries lacking a player_id prefix.
        prefix = pid.split("::")[0] if "::" in pid else pid
        if prefix.isdigit() or prefix.startswith("sa"):
            continue
        rows = board[board["name_normalized"] == normalize_name(prefix)]
        if not rows.empty:
            players.append(rows.iloc[0])
    return players


def get_filled_positions(
    user_roster_ids: list[str],
    board: pd.DataFrame,
    roster_slots: dict[str, int] | None = None,
    player_lookup: dict[str, pd.Series] | None = None,
) -> dict[str, int]:
    """Count how many of each roster slot the user has filled."""
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS

    capacity = {pos: count for pos, count in roster_slots.items() if pos != "IL"}
    filled: dict[str, int] = dict.fromkeys(capacity, 0)
    players = _collect_roster_entries(user_roster_ids, board, player_lookup)

    active_slots = {k: v for k, v in capacity.items() if k != "BN"}
    players.sort(key=lambda p: sum(1 for s in active_slots if can_fill_slot(p["positions"], s)))

    scarcity_order = _scarcity_order_cached(board, roster_slots)
    specific_slots = [s for s in scarcity_order if s not in ("IF", "UTIL")]
    flex_slots = [s for s in scarcity_order if s in ("IF", "UTIL")]

    for player in players:
        positions = player["positions"]
        assigned = False
        for slot in specific_slots:
            if filled[slot] < capacity[slot] and can_fill_slot(positions, slot):
                filled[slot] += 1
                assigned = True
                break
        if not assigned:
            for slot in flex_slots:
                if (
                    slot in active_slots
                    and filled[slot] < capacity[slot]
                    and can_fill_slot(positions, slot)
                ):
                    filled[slot] += 1
                    assigned = True
                    break
        if not assigned:
            filled["BN"] = filled.get("BN", 0) + 1
    return {pos: count for pos, count in filled.items() if count > 0}


def get_roster_by_position(
    user_roster_ids: list[str],
    board: pd.DataFrame,
    roster_slots: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    """Map roster slot -> list of player names for the user's roster."""
    if roster_slots is None:
        roster_slots = DEFAULT_ROSTER_SLOTS

    capacity = {pos: count for pos, count in roster_slots.items() if pos != "IL"}
    by_pos: dict[str, list[str]] = {pos: [] for pos in capacity}
    players = _collect_roster_entries(user_roster_ids, board)

    active_slots = {k: v for k, v in capacity.items() if k != "BN"}
    players.sort(key=lambda p: sum(1 for s in active_slots if can_fill_slot(p["positions"], s)))

    scarcity_order = _scarcity_order_cached(board, roster_slots)
    specific_slots = [s for s in scarcity_order if s not in ("IF", "UTIL")]
    flex_slots = [s for s in scarcity_order if s in ("IF", "UTIL")]

    for player in players:
        positions = player["positions"]
        assigned = False
        for slot in specific_slots:
            if len(by_pos[slot]) < capacity[slot] and can_fill_slot(positions, slot):
                by_pos[slot].append(player["name"])
                assigned = True
                break
        if not assigned:
            for slot in flex_slots:
                if (
                    slot in active_slots
                    and len(by_pos[slot]) < capacity[slot]
                    and can_fill_slot(positions, slot)
                ):
                    by_pos[slot].append(player["name"])
                    assigned = True
                    break
        if not assigned:
            by_pos.setdefault("BN", []).append(player["name"])
    return {pos: names for pos, names in by_pos.items() if names}
