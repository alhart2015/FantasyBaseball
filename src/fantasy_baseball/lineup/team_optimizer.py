"""Shared team lineup optimization and wSGP computation.

Used by both the waiver scanner and the roster audit to evaluate
lineup configurations via the Hungarian algorithm (hitters) and
simple ranking (pitchers).
"""

from __future__ import annotations

from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators


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
            player_wsgp[p.name] = calculate_weighted_sgp(p.ros, leverage, denoms=denoms)

    # Optimize hitters (Hungarian algorithm)
    hitter_lineup = optimize_hitter_lineup(hitters, leverage, roster_slots)

    # Optimize pitchers (simple ranking)
    p_slots = roster_slots.get("P", 9)
    pitcher_starters, _ = optimize_pitcher_lineup(pitchers, leverage, slots=p_slots)

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
