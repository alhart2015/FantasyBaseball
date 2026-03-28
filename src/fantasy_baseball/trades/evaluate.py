"""Trade evaluation module.

Projects how team roto standings change when players are swapped, based on
rest-of-season (ROS) projections for the players involved.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.utils.positions import can_fill_slot

from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
    INVERSE_STATS as INVERSE_CATS,
)

COUNTING_CATS = ["R", "HR", "RBI", "SB", "W", "K", "SV"]

# Equal leverage weights for computing raw (unweighted) player value
EQUAL_LEVERAGE = {cat: 0.1 for cat in ALL_CATS}

# Maximum raw SGP gap between traded players. Prevents lopsided trades
# like Aaron Judge for Chris Sale that no human would accept.
# Applied to the roster data as-provided (should be recency-blended for
# best accuracy — injured players with 0 stats will have near-zero SGP).
MAX_SGP_GAP = 0.35

# Baseline estimates for AB and IP used to back out current totals
_TEAM_AB = 5500
_TEAM_IP = 1400


def compute_roto_points_by_cat(
    standings: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return per-category roto points for each team.

    Args:
        standings: list of {"name": str, "stats": {cat: float}} dicts.

    Returns:
        {team_name: {cat: points}} where points range from 1 (worst) to
        N (best) for N teams.  ERA and WHIP are inverse (lower is better).
    """
    n = len(standings)
    result: dict[str, dict[str, float]] = {t["name"]: {} for t in standings}

    # Default values for missing stats: 0 for counting, worst-case for rate
    _STAT_DEFAULTS = {"ERA": 99.0, "WHIP": 99.0}

    for cat in ALL_CATS:
        inverse = cat in INVERSE_CATS
        # Fill missing stats with defaults so all teams can be ranked
        for t in standings:
            stats = t.get("stats", {})
            if cat not in stats:
                stats[cat] = _STAT_DEFAULTS.get(cat, 0.0)
        # Sort: for inverse cats lowest value → rank n (best)
        ranked = sorted(standings, key=lambda t: t["stats"][cat], reverse=inverse)
        # Fractional tie-breaking: tied teams share the average of their ranks
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(ranked[j]["stats"][cat] - ranked[i]["stats"][cat]) < 1e-9:
                j += 1
            avg_rank = sum(range(i + 1, j + 1)) / (j - i)
            for k in range(i, j):
                result[ranked[k]["name"]][cat] = avg_rank
            i = j

    return result


def compute_roto_points(standings: list[dict[str, Any]]) -> dict[str, int]:
    """Return total roto points for each team.

    Args:
        standings: list of {"name": str, "stats": {cat: float}} dicts.

    Returns:
        {team_name: total_points}
    """
    by_cat = compute_roto_points_by_cat(standings)
    return {name: sum(cat_pts.values()) for name, cat_pts in by_cat.items()}


def _project_team_stats(
    current_stats: dict[str, float],
    loses_ros: dict[str, Any],
    gains_ros: dict[str, Any],
) -> dict[str, float]:
    """Project end-of-season team stats after trading a player away and gaining one.

    Args:
        current_stats: current team stats dict (all 10 categories).
        loses_ros: ROS projection for the player being traded away.
            Must include keys: R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP, ab, ip.
        gains_ros: ROS projection for the player being acquired.
            Same keys as loses_ros.

    Returns:
        Projected stats dict (same keys as current_stats).
    """
    projected = dict(current_stats)

    # --- Counting stats ---
    for cat in COUNTING_CATS:
        projected[cat] = current_stats[cat] - loses_ros[cat] + gains_ros[cat]

    # --- AVG: weighted by AB ---
    loses_ab = loses_ros["ab"]
    gains_ab = gains_ros["ab"]
    new_ab = _TEAM_AB - loses_ab + gains_ab
    if new_ab > 0:
        current_hits = current_stats["AVG"] * _TEAM_AB
        new_hits = current_hits - loses_ros["AVG"] * loses_ab + gains_ros["AVG"] * gains_ab
        projected["AVG"] = new_hits / new_ab
    else:
        projected["AVG"] = 0.0

    # --- ERA: convert to ER, adjust, recompute ---
    loses_ip = loses_ros["ip"]
    gains_ip = gains_ros["ip"]
    new_ip = _TEAM_IP - loses_ip + gains_ip

    if new_ip > 0:
        current_er = current_stats["ERA"] * _TEAM_IP / 9.0
        loses_er = loses_ros["ERA"] * loses_ip / 9.0
        gains_er = gains_ros["ERA"] * gains_ip / 9.0
        new_er = current_er - loses_er + gains_er
        projected["ERA"] = new_er * 9.0 / new_ip
    else:
        projected["ERA"] = 0.0

    # --- WHIP: total (BB+H), adjust, recompute ---
    if new_ip > 0:
        current_bh = current_stats["WHIP"] * _TEAM_IP
        loses_bh = loses_ros["WHIP"] * loses_ip
        gains_bh = gains_ros["WHIP"] * gains_ip
        new_bh = current_bh - loses_bh + gains_bh
        projected["WHIP"] = new_bh / new_ip
    else:
        projected["WHIP"] = 0.0

    return projected


def compute_trade_impact(
    standings: list[dict[str, Any]],
    hart_name: str,
    opp_name: str,
    hart_loses_ros: dict[str, Any],
    hart_gains_ros: dict[str, Any],
    opp_loses_ros: dict[str, Any],
    opp_gains_ros: dict[str, Any],
) -> dict[str, Any]:
    """Compute roto point impact of a proposed trade for both teams.

    Args:
        standings: current league standings (list of team dicts).
        hart_name: name of Hart's team in standings.
        opp_name: name of the trade partner's team.
        hart_loses_ros: ROS stats for the player Hart trades away.
        hart_gains_ros: ROS stats for the player Hart receives.
        opp_loses_ros: ROS stats for the player the opponent trades away.
        opp_gains_ros: ROS stats for the player the opponent receives.

    Returns:
        {
            "hart_delta": int,          # net roto point change for Hart
            "opp_delta": int,           # net roto point change for opponent
            "hart_cat_deltas": {cat: int},
            "opp_cat_deltas": {cat: int},
        }
    """
    # Baseline points
    baseline_by_cat = compute_roto_points_by_cat(standings)

    # Build projected standings
    projected_standings = []
    for team in standings:
        if team["name"] == hart_name:
            new_stats = _project_team_stats(
                team["stats"], hart_loses_ros, hart_gains_ros
            )
            projected_standings.append({"name": team["name"], "stats": new_stats})
        elif team["name"] == opp_name:
            new_stats = _project_team_stats(
                team["stats"], opp_loses_ros, opp_gains_ros
            )
            projected_standings.append({"name": team["name"], "stats": new_stats})
        else:
            projected_standings.append(team)

    projected_by_cat = compute_roto_points_by_cat(projected_standings)

    # Compute deltas
    hart_base = sum(baseline_by_cat[hart_name].values())
    hart_proj = sum(projected_by_cat[hart_name].values())
    opp_base = sum(baseline_by_cat[opp_name].values())
    opp_proj = sum(projected_by_cat[opp_name].values())

    hart_cat_deltas = {
        cat: projected_by_cat[hart_name][cat] - baseline_by_cat[hart_name][cat]
        for cat in ALL_CATS
    }
    opp_cat_deltas = {
        cat: projected_by_cat[opp_name][cat] - baseline_by_cat[opp_name][cat]
        for cat in ALL_CATS
    }

    return {
        "hart_delta": hart_proj - hart_base,
        "opp_delta": opp_proj - opp_base,
        "hart_cat_deltas": hart_cat_deltas,
        "opp_cat_deltas": opp_cat_deltas,
    }


def _player_ros_stats(player: dict) -> dict:
    """Extract ROS stats from a player dict for trade projection.

    Returns dict with R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP, ab, ip.
    """
    ptype = player.get("player_type", "hitter")
    if ptype == "hitter":
        return {
            "R": player.get("r", 0), "HR": player.get("hr", 0),
            "RBI": player.get("rbi", 0), "SB": player.get("sb", 0),
            "AVG": player.get("avg", 0),
            "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0,
            "ab": player.get("ab", 0), "ip": 0,
        }
    else:
        return {
            "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
            "W": player.get("w", 0), "K": player.get("k", 0),
            "SV": player.get("sv", 0), "ERA": player.get("era", 0),
            "WHIP": player.get("whip", 0),
            "ab": 0, "ip": player.get("ip", 0),
        }


def _find_player_by_name(name: str, roster: list[dict]) -> dict | None:
    """Find a player in a roster by name (case-insensitive)."""
    name_lower = name.lower()
    for p in roster:
        if p.get("name", "").lower() == name_lower:
            return p
    return None


def _can_roster_without(roster: list[dict], remove: dict, add: dict,
                        roster_slots: dict) -> bool:
    """Check if a roster remains legal after swapping one player.

    Simple check: the incoming player must be able to fill at least one
    non-bench active slot.
    """
    positions = add.get("positions", [])
    for slot in roster_slots:
        if slot in ("BN", "IL"):
            continue
        if can_fill_slot(positions, slot):
            return True
    return False


def find_trades(
    hart_name: str,
    hart_roster: list[dict],
    opp_rosters: dict[str, list[dict]],
    standings: list[dict],
    leverage_by_team: dict[str, dict],
    roster_slots: dict[str, int],
    max_results: int = 5,
) -> list[dict]:
    """Find and rank the best 1-for-1 trades for Hart.

    Evaluates every possible swap between Hart and each opponent.
    Filters to trades where both sides gain wSGP (opponent can break even).
    Rejects lopsided trades where players' SGP differs too much.
    Ranks by Hart's projected roto point gain.

    The roster data should reflect the best-available projections (ideally
    recency-blended) so that injured players show near-zero value and the
    fairness check naturally filters them out.

    Returns list of trade dicts with: send, receive, opponent, hart_delta,
    opp_delta, hart_cat_deltas, opp_cat_deltas, hart_wsgp_gain, opp_wsgp_gain,
    send_positions, receive_positions.
    """
    hart_leverage = leverage_by_team.get(hart_name, {})
    proposals = []

    for opp_name, opp_roster in opp_rosters.items():
        opp_leverage = leverage_by_team.get(opp_name, {})

        for hart_player in hart_roster:
            hart_p_series = pd.Series(hart_player)
            hart_wsgp = calculate_weighted_sgp(hart_p_series, hart_leverage)

            for opp_player in opp_roster:
                # Roster legality
                if not _can_roster_without(hart_roster, hart_player, opp_player, roster_slots):
                    continue
                if not _can_roster_without(opp_roster, opp_player, hart_player, roster_slots):
                    continue

                opp_p_series = pd.Series(opp_player)

                # Fairness guardrail: reject lopsided trades where raw
                # player values are too far apart. Uses the roster data
                # as-provided (should be recency-blended so injured/inactive
                # players show near-zero value).
                hart_raw = calculate_weighted_sgp(hart_p_series, EQUAL_LEVERAGE)
                opp_raw = calculate_weighted_sgp(opp_p_series, EQUAL_LEVERAGE)
                if abs(hart_raw - opp_raw) > MAX_SGP_GAP:
                    continue

                # wSGP from each side's perspective
                gain_wsgp = calculate_weighted_sgp(opp_p_series, hart_leverage)
                hart_wsgp_gain = gain_wsgp - hart_wsgp

                opp_current_wsgp = calculate_weighted_sgp(opp_p_series, opp_leverage)
                opp_gain_wsgp = calculate_weighted_sgp(hart_p_series, opp_leverage)
                opp_wsgp_gain = opp_gain_wsgp - opp_current_wsgp

                # Both sides must benefit
                if hart_wsgp_gain <= 0 or opp_wsgp_gain < 0:
                    continue

                # Roto point impact
                hart_loses = _player_ros_stats(hart_player)
                hart_gains = _player_ros_stats(opp_player)
                opp_loses = _player_ros_stats(opp_player)
                opp_gains = _player_ros_stats(hart_player)

                impact = compute_trade_impact(
                    standings, hart_name, opp_name,
                    hart_loses, hart_gains, opp_loses, opp_gains,
                )

                proposals.append({
                    "send": hart_player["name"],
                    "send_positions": hart_player.get("positions", []),
                    "receive": opp_player["name"],
                    "receive_positions": opp_player.get("positions", []),
                    "opponent": opp_name,
                    "hart_delta": impact["hart_delta"],
                    "opp_delta": impact["opp_delta"],
                    "hart_cat_deltas": impact["hart_cat_deltas"],
                    "opp_cat_deltas": impact["opp_cat_deltas"],
                    "hart_wsgp_gain": round(hart_wsgp_gain, 2),
                    "opp_wsgp_gain": round(opp_wsgp_gain, 2),
                })

    proposals.sort(key=lambda t: (t["hart_delta"], t["opp_delta"]), reverse=True)
    return proposals[:max_results]
