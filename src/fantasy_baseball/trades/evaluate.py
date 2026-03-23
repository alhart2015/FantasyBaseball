"""Trade evaluation module.

Projects how team roto standings change when players are swapped, based on
rest-of-season (ROS) projections for the players involved.
"""

from __future__ import annotations

from typing import Any

COUNTING_CATS = ["R", "HR", "RBI", "SB", "W", "K", "SV"]
INVERSE_CATS = {"ERA", "WHIP"}  # lower is better
ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]

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
    result: dict[str, dict[str, int]] = {t["name"]: {} for t in standings}

    for cat in ALL_CATS:
        inverse = cat in INVERSE_CATS
        # Sort: for inverse cats lowest value → rank n (best)
        ranked = sorted(standings, key=lambda t: t["stats"][cat], reverse=inverse)
        for rank, team in enumerate(ranked, start=1):
            result[team["name"]][cat] = rank

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
