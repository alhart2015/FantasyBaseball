"""Trade evaluation module.

Projects how team roto standings change when players are swapped, based on
rest-of-season (ROS) projections for the players involved.
"""

from __future__ import annotations

from typing import Any, TypedDict

from fantasy_baseball.models.player import HitterStats, Player
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.sgp.rankings import rank_key_from_positions
from fantasy_baseball.utils.constants import ALL_CATEGORIES as ALL_CATS
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_fill_slot
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era

COUNTING_CATS = ["R", "HR", "RBI", "SB", "W", "K", "SV"]

# Maximum ranking gap for perception-based filtering. A trade is accepted
# when send_rank - receive_rank <= MAX_RANK_GAP (the player we send can
# be up to this many spots worse-ranked than the player we receive).
MAX_RANK_GAP = 5

# Baseline estimates for AB and IP used to back out current totals
_TEAM_AB = 5500
_TEAM_IP = 1450


class OpponentGroup(TypedDict):
    """One opponent's trade-candidate group, as returned by search_trades_away."""

    opponent: str
    candidates: list[dict[str, Any]]


def compute_roto_points_by_cat(
    standings: list[dict[str, Any]],
    *,
    team_sds: dict[str, dict[Category, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Return per-category roto points for each team.

    Delegates to ``scoring.score_roto`` for the actual ranking logic.

    Args:
        standings: list of {"name": str, "stats": {cat: float}} dicts.
        team_sds: optional per-team per-category standard deviations.
            When provided, ``score_roto`` returns fractional ERoto points
            (EV under Gaussian pairwise win-probabilities) instead of
            hard integer ranks.

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

    all_stats = {t["name"]: t["stats"] for t in standings}
    roto = score_roto(all_stats, team_sds=team_sds)

    return {name: {cat: pts[f"{cat}_pts"] for cat in ALL_CATS} for name, pts in roto.items()}


def compute_roto_points(standings: list[dict[str, Any]]) -> dict[str, float]:
    """Return total roto points for each team.

    Args:
        standings: list of {"name": str, "stats": {cat: float}} dicts.

    Returns:
        {team_name: total_points}
    """
    by_cat = compute_roto_points_by_cat(standings)
    return {name: sum(cat_pts.values()) for name, cat_pts in by_cat.items()}


def apply_swap_delta(
    current_stats: dict[str, float],
    loses_ros: dict[str, Any],
    gains_ros: dict[str, Any],
) -> dict[str, float]:
    """Project end-of-season team stats after swapping one player for another.

    Applies a delta to cached projected standings — the single source of
    truth for projected team stats (built once during the refresh pipeline
    via ``project_team_stats``).  All code that needs a "what-if" swap
    scenario (comparisons, trade evaluation, waiver search) should use
    this function rather than recomputing standings from scratch.

    Args:
        current_stats: current team stats dict (all 10 categories).
        loses_ros: ROS projection for the player being dropped/traded away.
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
    *,
    team_sds: dict[str, dict[Category, float]] | None = None,
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
        team_sds: optional per-team per-category standard deviations.
            When provided, deltas are fractional ERoto points rather than
            integer rank changes.

    Returns:
        {
            "hart_delta": float,        # net roto point change for Hart
            "opp_delta": float,         # net roto point change for opponent
            "hart_cat_deltas": {cat: float},
            "opp_cat_deltas": {cat: float},
        }
    """
    baseline = projected_standings if projected_standings is not None else standings

    # Baseline points (from projected end-of-season or current standings)
    baseline_by_cat = compute_roto_points_by_cat(baseline, team_sds=team_sds)

    # Apply trade swap to the baseline
    post_trade = []
    for team in baseline:
        if team["name"] == hart_name:
            new_stats = apply_swap_delta(team["stats"], hart_loses_ros, hart_gains_ros)
            post_trade.append({"name": team["name"], "stats": new_stats})
        elif team["name"] == opp_name:
            new_stats = apply_swap_delta(team["stats"], opp_loses_ros, opp_gains_ros)
            post_trade.append({"name": team["name"], "stats": new_stats})
        else:
            post_trade.append(team)

    post_trade_by_cat = compute_roto_points_by_cat(post_trade, team_sds=team_sds)

    # Compute deltas
    hart_base = sum(baseline_by_cat[hart_name].values())
    hart_proj = sum(post_trade_by_cat[hart_name].values())
    opp_base = sum(baseline_by_cat[opp_name].values())
    opp_proj = sum(post_trade_by_cat[opp_name].values())

    hart_cat_deltas = {
        cat: post_trade_by_cat[hart_name][cat] - baseline_by_cat[hart_name][cat] for cat in ALL_CATS
    }
    opp_cat_deltas = {
        cat: post_trade_by_cat[opp_name][cat] - baseline_by_cat[opp_name][cat] for cat in ALL_CATS
    }

    return {
        "hart_delta": hart_proj - hart_base,
        "opp_delta": opp_proj - opp_base,
        "hart_cat_deltas": hart_cat_deltas,
        "opp_cat_deltas": opp_cat_deltas,
    }


def player_rest_of_season_stats(player: Player) -> dict:
    """Extract ROS stats from a Player for swap projection.

    Returns a flat dict with keys R, HR, RBI, SB, AVG, W, K, SV, ERA,
    WHIP, ab, ip — the format expected by :func:`apply_swap_delta`.
    """
    ros = player.rest_of_season
    if ros is None:
        return {
            cat: 0
            for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP", "ab", "ip"]
        }
    if isinstance(ros, HitterStats):
        return {
            "R": ros.r,
            "HR": ros.hr,
            "RBI": ros.rbi,
            "SB": ros.sb,
            "AVG": ros.avg,
            "W": 0,
            "K": 0,
            "SV": 0,
            "ERA": 0,
            "WHIP": 0,
            "ab": ros.ab,
            "ip": 0,
        }
    else:
        return {
            "R": 0,
            "HR": 0,
            "RBI": 0,
            "SB": 0,
            "AVG": 0,
            "W": ros.w,
            "K": ros.k,
            "SV": ros.sv,
            "ERA": ros.era,
            "WHIP": ros.whip,
            "ab": 0,
            "ip": ros.ip,
        }


def aggregate_player_stats(players: list[Player]) -> dict:
    """Aggregate ROS stats across multiple players into one dict.

    Returns the same shape as :func:`player_rest_of_season_stats`. Counting
    stats sum; rate stats are weighted (AVG by AB, ERA/WHIP by IP). An
    empty list returns all zeros.

    This lets multi-player trades call :func:`apply_swap_delta` exactly
    once per team with combined loses/gains stats.
    """
    total = {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AVG": 0.0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 0.0,
        "WHIP": 0.0,
        "ab": 0,
        "ip": 0,
    }
    if not players:
        return total

    total_hits = 0.0
    total_er = 0.0
    total_bh = 0.0

    for p in players:
        s = player_rest_of_season_stats(p)
        for cat in ("R", "HR", "RBI", "SB", "W", "K", "SV"):
            total[cat] += s[cat]
        total["ab"] += s["ab"]
        total["ip"] += s["ip"]
        total_hits += s["AVG"] * s["ab"]
        total_er += s["ERA"] * s["ip"] / 9.0
        total_bh += s["WHIP"] * s["ip"]

    if total["ab"] > 0:
        total["AVG"] = total_hits / total["ab"]
    if total["ip"] > 0:
        total["ERA"] = 9.0 * total_er / total["ip"]
        total["WHIP"] = total_bh / total["ip"]
    return total


def find_player_by_name(name: str, roster: list[Player]) -> Player | None:
    """Find a player in a roster by normalized name (accent-safe)."""
    target = normalize_name(name)
    for p in roster:
        if normalize_name(p.name) == target:
            return p
    return None


def _can_roster_without(
    roster: list[Player], remove: Player, add: Player, roster_slots: dict
) -> bool:
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


def search_trades_away(
    player_name: str,
    hart_name: str,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    standings: list[dict],
    leverage_by_team: dict[str, dict],
    roster_slots: dict[str, int],
    rankings: dict[str, int],
    projected_standings: list[dict] | None = None,
    *,
    team_sds: dict[str, dict[Category, float]] | None = None,
) -> list[OpponentGroup]:
    """Find trade candidates for a player the user wants to trade away.

    Searches all opponent rosters for players the user could receive in
    exchange. Results are grouped by opponent and sorted alphabetically
    by opponent name. Candidates within each group are sorted by the
    user's projected roto-point gain (hart_delta).

    Args:
        player_name: name of the player to trade away (on user's roster).
        hart_name: user's team name in standings.
        hart_roster: user's roster as Player objects.
        opp_rosters: {opponent_name: [Player]} for each opponent.
        standings: current league standings.
        leverage_by_team: {team_name: {cat: weight}} leverage weights.
        roster_slots: league roster slot configuration.
        rankings: {rank_key: int} unweighted SGP ROS rankings.
        projected_standings: optional projected end-of-season standings.
        team_sds: optional per-team per-category standard deviations for
            fractional ERoto scoring (passed through to compute_trade_impact).

    Returns:
        List of opponent groups:
        [{"opponent": str, "candidates": [...]}, ...]
        Groups sorted alphabetically by opponent name.
        Candidates sorted by hart_delta descending within each group.
    """
    hart_player = find_player_by_name(player_name, hart_roster)
    if hart_player is None:
        return []

    send_rank = rankings.get(rank_key_from_positions(hart_player.name, hart_player.positions))
    if send_rank is None:
        return []

    grouped: dict[str, list[dict]] = {}

    for opp_name, opp_roster in opp_rosters.items():
        for opp_player in opp_roster:
            receive_rank = rankings.get(
                rank_key_from_positions(opp_player.name, opp_player.positions)
            )
            if receive_rank is None:
                continue

            if not _can_roster_without(hart_roster, hart_player, opp_player, roster_slots):
                continue
            if not _can_roster_without(opp_roster, opp_player, hart_player, roster_slots):
                continue

            rank_gap = send_rank - receive_rank
            if rank_gap > MAX_RANK_GAP:
                continue

            hart_ros = player_rest_of_season_stats(hart_player)
            opp_ros = player_rest_of_season_stats(opp_player)

            impact = compute_trade_impact(
                standings,
                hart_name,
                opp_name,
                hart_ros,
                opp_ros,
                opp_ros,
                hart_ros,
                projected_standings=projected_standings,
                team_sds=team_sds,
            )

            if impact["hart_delta"] < 0:
                continue

            grouped.setdefault(opp_name, []).append(
                {
                    "send": hart_player.name,
                    "send_positions": hart_player.positions,
                    "send_rank": send_rank,
                    "receive": opp_player.name,
                    "receive_positions": opp_player.positions,
                    "receive_rank": receive_rank,
                    "hart_delta": impact["hart_delta"],
                    "opp_delta": impact["opp_delta"],
                    "hart_cat_deltas": impact["hart_cat_deltas"],
                    "opp_cat_deltas": impact["opp_cat_deltas"],
                }
            )

    # Sort candidates within each group by hart_delta descending
    for candidates in grouped.values():
        candidates.sort(key=lambda c: -c["hart_delta"])

    # Build result sorted alphabetically by opponent name
    results: list[OpponentGroup] = [
        {"opponent": opp_name, "candidates": candidates}
        for opp_name, candidates in sorted(grouped.items())
    ]
    return results


def search_trades_for(
    player_name: str,
    hart_name: str,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    standings: list[dict],
    leverage_by_team: dict[str, dict],
    roster_slots: dict[str, int],
    rankings: dict[str, int],
    projected_standings: list[dict] | None = None,
    *,
    team_sds: dict[str, dict[Category, float]] | None = None,
) -> list[dict]:
    """Find trade offers the user can make to acquire a specific opponent player.

    Searches the user's roster for players they could send that pass
    the rank proximity filter and project a non-negative roto-point gain.

    Args:
        player_name: name of the player to acquire (on an opponent's roster).
        hart_name: user's team name in standings.
        hart_roster: user's roster as Player objects.
        opp_rosters: {opponent_name: [Player]} for each opponent.
        standings: current league standings.
        leverage_by_team: {team_name: {cat: weight}} leverage weights.
        roster_slots: league roster slot configuration.
        rankings: {rank_key: int} unweighted SGP ROS rankings.
        projected_standings: optional projected end-of-season standings.
        team_sds: optional per-team per-category standard deviations for
            fractional ERoto scoring (passed through to compute_trade_impact).

    Returns:
        List with a single opponent group (or empty if player not found):
        [{"opponent": str, "candidates": [...]}]
        Candidates sorted by hart_delta descending.
    """
    # Find which opponent owns the target player
    target_player = None
    target_opp = None
    for opp_name, opp_roster in opp_rosters.items():
        found = find_player_by_name(player_name, opp_roster)
        if found is not None:
            target_player = found
            target_opp = opp_name
            break

    if target_player is None or target_opp is None:
        return []

    receive_rank = rankings.get(
        rank_key_from_positions(target_player.name, target_player.positions)
    )
    if receive_rank is None:
        return []

    opp_roster = opp_rosters[target_opp]

    candidates = []
    for hart_player in hart_roster:
        send_rank = rankings.get(rank_key_from_positions(hart_player.name, hart_player.positions))
        if send_rank is None:
            continue

        if not _can_roster_without(hart_roster, hart_player, target_player, roster_slots):
            continue
        if not _can_roster_without(opp_roster, target_player, hart_player, roster_slots):
            continue

        rank_gap = send_rank - receive_rank
        if rank_gap > MAX_RANK_GAP:
            continue

        hart_ros = player_rest_of_season_stats(hart_player)
        target_ros = player_rest_of_season_stats(target_player)

        impact = compute_trade_impact(
            standings,
            hart_name,
            target_opp,
            hart_ros,
            target_ros,
            target_ros,
            hart_ros,
            projected_standings=projected_standings,
            team_sds=team_sds,
        )

        if impact["hart_delta"] < 0:
            continue

        candidates.append(
            {
                "send": hart_player.name,
                "send_positions": hart_player.positions,
                "send_rank": send_rank,
                "receive": target_player.name,
                "receive_positions": target_player.positions,
                "receive_rank": receive_rank,
                "hart_delta": impact["hart_delta"],
                "opp_delta": impact["opp_delta"],
                "hart_cat_deltas": impact["hart_cat_deltas"],
                "opp_cat_deltas": impact["opp_cat_deltas"],
            }
        )

    candidates.sort(key=lambda c: -c["hart_delta"])

    if not candidates:
        return []

    return [{"opponent": target_opp, "candidates": candidates}]
