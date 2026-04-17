"""Shared team lineup optimization.

Exposes two optimizers:
  - ``compute_team_wsgp``: legacy leverage-weighted SGP (used by waivers +
    roster audit).
  - ``compute_team_roto``: ERoto-max optimizer, returning a team total and
    both lineups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from fantasy_baseball.lineup.optimizer import (
    HitterAssignment,
    PitcherStarter,
    _TeamContext,
    _pitcher_active_slots,
    apply_lineup_to_roster,
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
    team_roto_total,
)
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import DEFAULT_ROSTER_SLOTS
from fantasy_baseball.utils.positions import can_fill_slot


def _build_hitter_slots_wsgp(roster_slots: dict[str, int]) -> list[str]:
    slots: list[str] = []
    for pos, count in roster_slots.items():
        if pos in ("P", "BN", "IL"):
            continue
        for _ in range(count):
            slots.append(pos)
    return slots


def _optimize_hitters_by_wsgp(
    hitters: list,
    leverage: dict[str, float],
    roster_slots: dict[str, int] | None = None,
) -> dict[str, str]:
    """Legacy wSGP Hungarian — retained only for `compute_team_wsgp` consumers."""
    if not hitters:
        return {}

    hitter_slots = (
        _build_hitter_slots_wsgp(roster_slots) if roster_slots
        else _build_hitter_slots_wsgp(DEFAULT_ROSTER_SLOTS)
    )
    n_players = len(hitters)
    n_slots = len(hitter_slots)

    values = [calculate_weighted_sgp(h.rest_of_season, leverage) for h in hitters]

    size = max(n_players, n_slots)
    cost = np.full((size, size), 1e9)
    for i, hitter in enumerate(hitters):
        positions = hitter.positions
        for j, slot in enumerate(hitter_slots):
            if can_fill_slot(positions, slot):
                cost[i][j] = -values[i]

    row_idx, col_idx = linear_sum_assignment(cost)

    lineup: dict[str, str] = {}
    assigned_slots: dict[str, int] = {}
    for r, c in zip(row_idx, col_idx):
        if r < n_players and c < n_slots and cost[r][c] < 1e8:
            slot = hitter_slots[c]
            player_name = hitters[r].name
            count = assigned_slots.get(slot, 0)
            slot_key = slot if count == 0 else f"{slot}_{count + 1}"
            assigned_slots[slot] = count + 1
            lineup[slot_key] = player_name
    return lineup


def _optimize_pitchers_by_wsgp(
    pitchers: list,
    leverage: dict[str, float],
    slots: int = 9,
) -> tuple[list[dict], list[dict]]:
    """Legacy wSGP ranking — retained only for `compute_team_wsgp` consumers."""
    scored = []
    for p in pitchers:
        wsgp = calculate_weighted_sgp(p.rest_of_season, leverage)
        scored.append({"name": p.name, "wsgp": wsgp, "player": p})
    scored.sort(key=lambda x: x["wsgp"], reverse=True)
    return scored[:slots], scored[slots:]


def compute_team_wsgp(
    roster: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
    denoms: dict[str, float] | None = None,
    player_wsgp: dict[str, float] | None = None,
) -> dict:
    """Run both optimizers and return total wSGP of assigned starters.

    Args:
        player_wsgp: Pre-computed name->wSGP lookup. If provided, skips recomputing
            wSGP for players already in the dict (only computes missing entries).

    Returns dict with:
        total_wsgp: float — sum of wSGP for players actually assigned to a slot
        hitter_lineup: dict[str, str] — slot -> player name from Hungarian optimizer
        pitcher_starters: list[dict] — pitcher starters from ranking
        player_wsgp: dict[str, float] — name -> wSGP lookup for all roster players
    """
    if denoms is None:
        denoms = get_sgp_denominators()

    hitters = [p for p in roster if p.player_type != PlayerType.PITCHER]
    pitchers = [p for p in roster if p.player_type == PlayerType.PITCHER]

    if player_wsgp is None:
        player_wsgp = {}
    else:
        player_wsgp = dict(player_wsgp)  # don't mutate caller's dict
    for p in roster:
        if p.name not in player_wsgp:
            player_wsgp[p.name] = calculate_weighted_sgp(p.rest_of_season, leverage, denoms=denoms)

    # Optimize hitters (Hungarian algorithm)
    hitter_lineup = _optimize_hitters_by_wsgp(hitters, leverage, roster_slots)

    # Optimize pitchers (simple ranking)
    p_slots = roster_slots.get("P", 9)
    pitcher_starters, _ = _optimize_pitchers_by_wsgp(pitchers, leverage, slots=p_slots)

    # Sum wSGP of assigned players only
    total = 0.0
    for name in hitter_lineup.values():
        total += player_wsgp.get(name, 0.0)
    for ps in pitcher_starters:
        total += player_wsgp.get(ps["name"], 0.0)

    return {
        "total_wsgp": total,
        "hitter_lineup": hitter_lineup,
        "pitcher_starters": pitcher_starters,
        "player_wsgp": player_wsgp,
    }


_SLOT_ORDER = ["C", "1B", "2B", "3B", "SS", "IF", "OF", "UTIL", "P", "BN"]


def build_lineup_summary(
    hitter_lineup: dict[str, str],
    pitcher_starters: list[dict],
    player_wsgp: dict[str, float],
    all_player_names: list[str],
) -> list[dict]:
    """Build a lineup summary list for display, sorted by slot order.

    Returns list of {"name", "slot", "wsgp"} dicts.
    Hitter slots have _N suffixes stripped. Unassigned players get slot="BN".
    Sorted by standard slot order (C, 1B, 2B, ... P, BN) so before/after
    lineups align visually.
    """
    slot_rank = {s: i for i, s in enumerate(_SLOT_ORDER)}
    summary = []
    assigned_names = set()

    for slot_key, name in hitter_lineup.items():
        display_slot = slot_key.split("_")[0]
        summary.append({
            "name": name,
            "slot": display_slot,
            "wsgp": round(player_wsgp.get(name, 0.0), 2),
        })
        assigned_names.add(name)

    for ps in pitcher_starters:
        name = ps["name"]
        summary.append({
            "name": name,
            "slot": "P",
            "wsgp": round(player_wsgp.get(name, 0.0), 2),
        })
        assigned_names.add(name)

    for name in all_player_names:
        if name not in assigned_names:
            summary.append({
                "name": name,
                "slot": "BN",
                "wsgp": round(player_wsgp.get(name, 0.0), 2),
            })

    summary.sort(key=lambda e: (slot_rank.get(e["slot"], 99), -e["wsgp"]))
    return summary


# ---------------------------------------------------------------------------
# ERoto-based team optimizer
# ---------------------------------------------------------------------------

@dataclass
class TeamRotoResult:
    total_roto: float
    hitter_lineup: list[HitterAssignment]
    pitcher_starters: list[PitcherStarter]
    pitcher_bench: list[Player]

    def to_dict(self) -> dict:
        return {
            "total_roto": round(self.total_roto, 2),
            "hitter_lineup": [a.to_dict() for a in self.hitter_lineup],
            "pitcher_starters": [s.to_dict() for s in self.pitcher_starters],
            "pitcher_bench": [p.name for p in self.pitcher_bench],
        }


def compute_team_roto(
    roster: list[Player],
    projected_standings: list[dict],
    team_name: str,
    roster_slots: dict[str, int],
    team_sds: dict[str, dict[str, float]] | None = None,
) -> TeamRotoResult:
    """Optimize both hitter and pitcher lineups by ERoto and return the team total."""
    hitters = [p for p in roster if p.player_type != PlayerType.PITCHER]
    pitchers = [p for p in roster if p.player_type == PlayerType.PITCHER]

    hitter_lineup = optimize_hitter_lineup(
        hitters=hitters, full_roster=roster,
        projected_standings=projected_standings, team_name=team_name,
        roster_slots=roster_slots, team_sds=team_sds,
    )
    p_slots = roster_slots.get("P", 9)
    pitcher_starters, pitcher_bench = optimize_pitcher_lineup(
        pitchers=pitchers, full_roster=roster,
        projected_standings=projected_standings, team_name=team_name,
        slots=p_slots, team_sds=team_sds,
    )

    # The two optimizers score independently; recompute once on the combined
    # lineup so the reported total reflects both sides together.
    active_slots = {a.name: a.slot for a in hitter_lineup}
    active_slots.update(_pitcher_active_slots([s.player for s in pitcher_starters]))
    bench_names = (
        {h.name for h in hitters} - {a.name for a in hitter_lineup}
        | {p.name for p in pitcher_bench}
    )
    hypothetical = apply_lineup_to_roster(roster, active_slots, bench_names)
    ctx = _TeamContext(roster, projected_standings, team_name, team_sds)
    total_roto = team_roto_total(hypothetical, ctx)

    return TeamRotoResult(
        total_roto=total_roto,
        hitter_lineup=hitter_lineup,
        pitcher_starters=pitcher_starters,
        pitcher_bench=pitcher_bench,
    )
