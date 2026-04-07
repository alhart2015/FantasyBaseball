"""Trade evaluation module.

Projects how team roto standings change when players are swapped, based on
rest-of-season (ROS) projections for the players involved.
"""

from __future__ import annotations

from typing import Any

from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.sgp.rankings import rank_key_from_positions
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot

from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
    INVERSE_STATS as INVERSE_CATS,
)
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era

COUNTING_CATS = ["R", "HR", "RBI", "SB", "W", "K", "SV"]

# Maximum ranking gap for perception-based filtering. A trade is accepted
# when send_rank - receive_rank <= MAX_RANK_GAP (the player we send can
# be up to this many spots worse-ranked than the player we receive).
MAX_RANK_GAP = 5

# Baseline estimates for AB and IP used to back out current totals
_TEAM_AB = 5500
_TEAM_IP = 1450


def compute_roto_points_by_cat(
    standings: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Return per-category roto points for each team.

    Delegates to ``scoring.score_roto`` for the actual ranking logic.

    Args:
        standings: list of {"name": str, "stats": {cat: float}} dicts.

    Returns:
        {team_name: {cat: points}} where points range from 1 (worst) to
        N (best) for N teams.  ERA and WHIP are inverse (lower is better).
    """
    _STAT_DEFAULTS = {"ERA": 99.0, "WHIP": 99.0}

    # Fill missing stats with defaults so all teams can be ranked
    for t in standings:
        stats = t.get("stats", {})
        for cat in ALL_CATS:
            if cat not in stats:
                stats[cat] = _STAT_DEFAULTS.get(cat, 0.0)

    # Convert to score_roto input format and call canonical implementation
    all_stats = {t["name"]: t["stats"] for t in standings}
    roto = score_roto(all_stats)

    # Convert "R_pts" keys back to bare "R" keys (drop "total")
    return {
        name: {cat: pts[f"{cat}_pts"] for cat in ALL_CATS}
        for name, pts in roto.items()
    }


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
    current_hits = current_stats["AVG"] * _TEAM_AB
    new_hits = current_hits - loses_ros["AVG"] * loses_ab + gains_ros["AVG"] * gains_ab
    projected["AVG"] = calculate_avg(new_hits, new_ab, default=0.0)

    # --- ERA: convert to ER, adjust, recompute ---
    loses_ip = loses_ros["ip"]
    gains_ip = gains_ros["ip"]
    new_ip = _TEAM_IP - loses_ip + gains_ip

    current_er = current_stats["ERA"] * _TEAM_IP / 9.0
    loses_er = loses_ros["ERA"] * loses_ip / 9.0
    gains_er = gains_ros["ERA"] * gains_ip / 9.0
    new_er = current_er - loses_er + gains_er
    projected["ERA"] = calculate_era(new_er, new_ip, default=0.0)

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
    projected_standings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute roto point impact of a proposed trade for both teams.

    Uses projected end-of-season standings as the baseline when available.
    The trade delta (swap ROS stats) is applied to the projected stats, so
    both baseline and post-trade numbers represent end-of-season totals.

    Falls back to current standings when projected_standings is not provided.

    Args:
        standings: current league standings (list of team dicts).
        hart_name: name of Hart's team in standings.
        opp_name: name of the trade partner's team.
        hart_loses_ros: ROS stats for the player Hart trades away.
        hart_gains_ros: ROS stats for the player Hart receives.
        opp_loses_ros: ROS stats for the player the opponent trades away.
        opp_gains_ros: ROS stats for the player the opponent receives.
        projected_standings: projected end-of-season standings for all teams.
            When provided, used as baseline instead of current standings.

    Returns:
        {
            "hart_delta": int,          # net roto point change for Hart
            "opp_delta": int,           # net roto point change for opponent
            "hart_cat_deltas": {cat: int},
            "opp_cat_deltas": {cat: int},
        }
    """
    baseline = projected_standings if projected_standings is not None else standings

    # Baseline points (from projected end-of-season or current standings)
    baseline_by_cat = compute_roto_points_by_cat(baseline)

    # Apply trade swap to the baseline
    post_trade = []
    for team in baseline:
        if team["name"] == hart_name:
            new_stats = _project_team_stats(
                team["stats"], hart_loses_ros, hart_gains_ros
            )
            post_trade.append({"name": team["name"], "stats": new_stats})
        elif team["name"] == opp_name:
            new_stats = _project_team_stats(
                team["stats"], opp_loses_ros, opp_gains_ros
            )
            post_trade.append({"name": team["name"], "stats": new_stats})
        else:
            post_trade.append(team)

    post_trade_by_cat = compute_roto_points_by_cat(post_trade)

    # Compute deltas
    hart_base = sum(baseline_by_cat[hart_name].values())
    hart_proj = sum(post_trade_by_cat[hart_name].values())
    opp_base = sum(baseline_by_cat[opp_name].values())
    opp_proj = sum(post_trade_by_cat[opp_name].values())

    hart_cat_deltas = {
        cat: post_trade_by_cat[hart_name][cat] - baseline_by_cat[hart_name][cat]
        for cat in ALL_CATS
    }
    opp_cat_deltas = {
        cat: post_trade_by_cat[opp_name][cat] - baseline_by_cat[opp_name][cat]
        for cat in ALL_CATS
    }

    return {
        "hart_delta": hart_proj - hart_base,
        "opp_delta": opp_proj - opp_base,
        "hart_cat_deltas": hart_cat_deltas,
        "opp_cat_deltas": opp_cat_deltas,
    }


def _player_ros_stats(player: Player) -> dict:
    """Extract ROS stats from a Player for trade projection."""
    ros = player.ros
    if ros is None:
        return {cat: 0 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP", "ab", "ip"]}
    if player.player_type == PlayerType.HITTER:
        return {
            "R": ros.r, "HR": ros.hr, "RBI": ros.rbi, "SB": ros.sb,
            "AVG": ros.avg,
            "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0,
            "ab": ros.ab, "ip": 0,
        }
    else:
        return {
            "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
            "W": ros.w, "K": ros.k, "SV": ros.sv,
            "ERA": ros.era, "WHIP": ros.whip,
            "ab": 0, "ip": ros.ip,
        }


def _find_player_by_name(name: str, roster: list[Player]) -> Player | None:
    """Find a player in a roster by normalized name (accent-safe)."""
    target = normalize_name(name)
    for p in roster:
        if normalize_name(p.name) == target:
            return p
    return None


def _can_roster_without(roster: list[Player], remove: Player, add: Player,
                        roster_slots: dict) -> bool:
    """Check if a roster remains legal after swapping one player.

    Simple check: the incoming player must be able to fill at least one
    non-bench active slot.
    """
    for slot in roster_slots:
        if slot in ("BN", "IL"):
            continue
        if can_fill_slot(add.positions, slot):
            return True
    return False


def find_trades(
    hart_name: str,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    standings: list[dict],
    leverage_by_team: dict[str, dict],
    roster_slots: dict[str, int],
    rankings: dict[str, int],
    max_results: int = 5,
    projected_standings: list[dict] | None = None,
) -> list[dict]:
    """Find and rank the best 1-for-1 trades for Hart.

    Uses a perception-based approach: filters to trades where the player
    sent is similarly ranked to the player received (looks fair to the
    opponent), then ranks by Hart's wSGP gain (biggest hidden value first).

    Args:
        hart_name: Hart's team name in standings.
        hart_roster: Hart's roster as Player objects.
        opp_rosters: {opponent_name: [Player]} for each opponent.
        standings: current league standings.
        leverage_by_team: {team_name: {cat: weight}} leverage weights.
        roster_slots: league roster slot configuration.
        rankings: {rank_key: int} unweighted SGP ROS rankings.
        max_results: maximum number of trade proposals to return.
        projected_standings: optional projected end-of-season standings.

    Returns list of trade dicts with: send, receive, opponent, hart_delta,
    opp_delta, hart_cat_deltas, opp_cat_deltas, hart_wsgp_gain,
    send_positions, receive_positions, send_rank, receive_rank.
    """
    hart_leverage = leverage_by_team.get(hart_name, {})
    proposals = []

    for opp_name, opp_roster in opp_rosters.items():
        for hart_player in hart_roster:
            send_rank = rankings.get(
                rank_key_from_positions(hart_player.name, hart_player.positions))
            if send_rank is None:
                continue

            hart_wsgp = calculate_weighted_sgp(hart_player.ros, hart_leverage)

            for opp_player in opp_roster:
                receive_rank = rankings.get(
                    rank_key_from_positions(opp_player.name, opp_player.positions))
                if receive_rank is None:
                    continue

                # Roster legality
                if not _can_roster_without(hart_roster, hart_player, opp_player, roster_slots):
                    continue
                if not _can_roster_without(opp_roster, opp_player, hart_player, roster_slots):
                    continue

                # Ranking proximity: looks fair to the opponent
                rank_gap = send_rank - receive_rank
                if rank_gap > MAX_RANK_GAP:
                    continue

                # wSGP gain for Hart
                gain_wsgp = calculate_weighted_sgp(opp_player.ros, hart_leverage)
                hart_wsgp_gain = gain_wsgp - hart_wsgp

                if hart_wsgp_gain <= 0:
                    continue

                # Roto point impact
                hart_ros = _player_ros_stats(hart_player)
                opp_ros = _player_ros_stats(opp_player)

                impact = compute_trade_impact(
                    standings, hart_name, opp_name,
                    hart_ros, opp_ros, opp_ros, hart_ros,
                    projected_standings=projected_standings,
                )

                # Reject trades that hurt our overall roto standing
                if impact["hart_delta"] < 0:
                    continue

                proposals.append({
                    "send": hart_player.name,
                    "send_positions": hart_player.positions,
                    "receive": opp_player.name,
                    "receive_positions": opp_player.positions,
                    "opponent": opp_name,
                    "hart_delta": impact["hart_delta"],
                    "opp_delta": impact["opp_delta"],
                    "hart_cat_deltas": impact["hart_cat_deltas"],
                    "opp_cat_deltas": impact["opp_cat_deltas"],
                    "hart_wsgp_gain": round(hart_wsgp_gain, 2),
                    "send_rank": send_rank,
                    "receive_rank": receive_rank,
                })

    # Sort: biggest roto gain first, then wSGP gain, then rank generosity
    proposals.sort(
        key=lambda t: (-t["hart_delta"], -t["hart_wsgp_gain"], t["send_rank"] - t["receive_rank"]),
    )
    return proposals[:max_results]
