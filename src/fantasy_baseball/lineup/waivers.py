import pandas as pd
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import (
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
    DEFAULT_TEAM_AB,
    DEFAULT_TEAM_IP,
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
)


def evaluate_pickup(
    add_player: pd.Series,
    drop_player: pd.Series,
    leverage: dict[str, float],
) -> dict:
    """Evaluate the SGP gain of adding one player and dropping another.

    Returns:
        Dict with add, drop, sgp_gain, and per-category breakdown.
    """
    add_wsgp = calculate_weighted_sgp(add_player, leverage)
    drop_wsgp = calculate_weighted_sgp(drop_player, leverage)

    denoms = get_sgp_denominators()
    categories = {}
    for stat, col in _get_stat_cols(add_player):
        add_val = _category_sgp(add_player, stat, col, denoms)
        drop_val = _category_sgp(drop_player, stat, col, denoms)
        weight = leverage.get(stat, 0)
        categories[stat] = (add_val - drop_val) * weight

    return {
        "add": add_player["name"],
        "drop": drop_player["name"],
        "sgp_gain": add_wsgp - drop_wsgp,
        "categories": categories,
    }


def scan_waivers(
    roster: list[pd.Series],
    free_agents: list[pd.Series],
    leverage: dict[str, float],
    max_results: int = 5,
) -> list[dict]:
    """Scan free agents and rank add/drop recommendations.

    For each free agent, finds the weakest roster player they could replace
    (same position type: hitter vs pitcher) and evaluates the swap.
    Returns only positive-gain recommendations, sorted best-first.

    Args:
        roster: List of player stat Series (must have 'positions' and 'player_type').
        free_agents: List of free agent stat Series.
        leverage: Category leverage weights.
        max_results: Maximum number of recommendations to return.

    Returns:
        List of evaluate_pickup result dicts, sorted by sgp_gain descending.
    """
    if not free_agents or not roster:
        return []

    # Pre-compute wSGP for all roster players
    roster_scores = []
    for p in roster:
        wsgp = calculate_weighted_sgp(p, leverage)
        roster_scores.append({"player": p, "wsgp": wsgp})

    recommendations = []
    seen_pairs: set[tuple[str, str]] = set()

    for fa in free_agents:
        fa_type = fa.get("player_type", "hitter")

        # Find worst roster player of same type
        same_type = [rs for rs in roster_scores if rs["player"].get("player_type") == fa_type]
        if not same_type:
            continue

        worst = min(same_type, key=lambda x: x["wsgp"])
        pair_key = (fa["name"], worst["player"]["name"])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        result = evaluate_pickup(fa, worst["player"], leverage)
        if result["sgp_gain"] > 0:
            recommendations.append(result)

    recommendations.sort(key=lambda x: x["sgp_gain"], reverse=True)
    return recommendations[:max_results]


def _get_stat_cols(player: pd.Series) -> list[tuple[str, str]]:
    """Get relevant stat/column pairs for a player's type."""
    if player.get("player_type") == "hitter":
        return [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb"), ("AVG", "avg")]
    elif player.get("player_type") == "pitcher":
        return [("W", "w"), ("K", "k"), ("SV", "sv"), ("ERA", "era"), ("WHIP", "whip")]
    return []


def _category_sgp(player: pd.Series, stat: str, col: str, denoms: dict) -> float:
    """Calculate raw SGP for a single category."""
    if stat in ("AVG",):
        return calculate_hitting_rate_sgp(
            player_avg=player.get("avg", 0),
            player_ab=int(player.get("ab", 0)),
            replacement_avg=REPLACEMENT_AVG,
            sgp_denominator=denoms["AVG"],
            team_ab=DEFAULT_TEAM_AB,
        )
    elif stat in ("ERA",):
        ip = player.get("ip", 0)
        if ip > 0:
            return calculate_pitching_rate_sgp(
                player_rate=player.get("era", 0), player_ip=ip,
                replacement_rate=REPLACEMENT_ERA,
                sgp_denominator=denoms["ERA"],
                team_ip=DEFAULT_TEAM_IP, innings_divisor=9,
            )
        return 0.0
    elif stat in ("WHIP",):
        ip = player.get("ip", 0)
        if ip > 0:
            return calculate_pitching_rate_sgp(
                player_rate=player.get("whip", 0), player_ip=ip,
                replacement_rate=REPLACEMENT_WHIP,
                sgp_denominator=denoms["WHIP"],
                team_ip=DEFAULT_TEAM_IP, innings_divisor=1,
            )
        return 0.0
    else:
        return calculate_counting_sgp(player.get(col, 0), denoms[stat])
