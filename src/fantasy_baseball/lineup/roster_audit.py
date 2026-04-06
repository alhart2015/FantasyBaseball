"""Roster audit — evaluate every roster slot against the best available FA."""

from __future__ import annotations

from fantasy_baseball.lineup.waivers import (
    _compute_team_wsgp,
    _build_lineup_summary,
    evaluate_pickup,
)
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators


def audit_roster(
    roster: list[Player],
    free_agents: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
) -> list[dict]:
    """Evaluate every roster slot against the best available FA.

    For each roster player, finds the FA that produces the largest team
    wSGP gain when swapped in.  Returns an entry for every roster player,
    sorted by gap descending (biggest problems first).  Entries with no
    upgrade available have gap=0.0 and best_fa=None.
    """
    if not roster:
        return []

    denoms = get_sgp_denominators()

    # Baseline optimal lineup
    baseline = _compute_team_wsgp(roster, leverage, roster_slots, denoms=denoms)
    baseline_wsgp = baseline["total_wsgp"]
    baseline_summary = _build_lineup_summary(
        baseline["hitter_lineup"], baseline["pitcher_starters"],
        baseline["player_wsgp"], [p.name for p in roster],
    )

    # Map player name → assigned slot from baseline
    slot_lookup = {e["name"]: e["slot"] for e in baseline_summary}

    # Pre-compute FA wSGP
    fa_wsgp: dict[str, float] = {}
    for fa in free_agents:
        fa_wsgp[fa.name] = calculate_weighted_sgp(fa.ros, leverage, denoms=denoms)

    p_slots = roster_slots.get("P", 9)

    entries: list[dict] = []
    for player in roster:
        entry = {
            "player": player.name,
            "player_type": player.player_type.value,
            "positions": list(player.positions),
            "slot": slot_lookup.get(player.name, "BN"),
            "player_wsgp": round(baseline["player_wsgp"].get(player.name, 0.0), 2),
            "best_fa": None,
            "best_fa_type": None,
            "best_fa_positions": None,
            "best_fa_wsgp": None,
            "gap": 0.0,
            "categories": {},
        }

        best_gain = 0.0
        best_fa_player = None
        best_new_result = None

        # Pre-build wSGP dict without this player for swap simulation
        base_wsgp = {k: v for k, v in baseline["player_wsgp"].items()
                     if k != player.name}

        for fa in free_agents:
            # Quick skip: FA not better than this player individually
            if fa_wsgp.get(fa.name, 0) <= entry["player_wsgp"]:
                continue

            new_roster = [p for p in roster if p.name != player.name] + [fa]
            new_pitchers = [p for p in new_roster if p.player_type == PlayerType.PITCHER]

            # Pitcher count feasibility: a cross-type swap can't leave fewer
            # pitchers than required active pitcher slots.
            if player.player_type == PlayerType.PITCHER or fa.player_type == PlayerType.PITCHER:
                if len(new_pitchers) < p_slots:
                    continue

            swap_wsgp = dict(base_wsgp)
            swap_wsgp[fa.name] = fa_wsgp[fa.name]

            new_result = _compute_team_wsgp(
                new_roster, leverage, roster_slots,
                denoms=denoms, player_wsgp=swap_wsgp,
            )
            gain = round(new_result["total_wsgp"] - baseline_wsgp, 2)

            if gain > best_gain:
                best_gain = gain
                best_fa_player = fa
                best_new_result = new_result

        if best_fa_player and best_new_result:
            cat_result = evaluate_pickup(best_fa_player, player, leverage)
            entry["best_fa"] = best_fa_player.name
            entry["best_fa_type"] = best_fa_player.player_type.value
            entry["best_fa_positions"] = list(best_fa_player.positions)
            entry["best_fa_wsgp"] = round(fa_wsgp.get(best_fa_player.name, 0.0), 2)
            entry["gap"] = best_gain
            entry["categories"] = cat_result["categories"]

        entries.append(entry)

    entries.sort(key=lambda e: e["gap"], reverse=True)
    return entries
