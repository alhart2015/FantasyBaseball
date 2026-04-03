import pandas as pd
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
from fantasy_baseball.models.player import HitterStats, PitcherStats


def calculate_weighted_sgp(
    player: "HitterStats | PitcherStats | pd.Series",
    leverage: dict[str, float],
    denoms: dict[str, float] | None = None,
) -> float:
    """Calculate leverage-weighted SGP for a player.

    Like calculate_player_sgp but each category's contribution is
    multiplied by the leverage weight.
    """
    if denoms is None:
        denoms = get_sgp_denominators()

    total = 0.0

    if isinstance(player, HitterStats):
        for stat, val in [("R", player.r), ("HR", player.hr), ("RBI", player.rbi), ("SB", player.sb)]:
            weight = leverage.get(stat, 0)
            if weight > 0:
                sgp = calculate_counting_sgp(val, denoms[stat])
                total += sgp * weight

        weight_avg = leverage.get("AVG", 0)
        if weight_avg > 0:
            sgp = calculate_hitting_rate_sgp(
                player_avg=player.avg,
                player_ab=int(player.ab),
                replacement_avg=REPLACEMENT_AVG,
                sgp_denominator=denoms["AVG"],
                team_ab=DEFAULT_TEAM_AB,
            )
            total += sgp * weight_avg

    elif isinstance(player, PitcherStats):
        for stat, val in [("W", player.w), ("K", player.k), ("SV", player.sv)]:
            weight = leverage.get(stat, 0)
            if weight > 0:
                sgp = calculate_counting_sgp(val, denoms[stat])
                total += sgp * weight

        if player.ip > 0:
            weight_era = leverage.get("ERA", 0)
            if weight_era > 0:
                sgp = calculate_pitching_rate_sgp(
                    player_rate=player.era, player_ip=player.ip,
                    replacement_rate=REPLACEMENT_ERA,
                    sgp_denominator=denoms["ERA"],
                    team_ip=DEFAULT_TEAM_IP, innings_divisor=9,
                )
                total += sgp * weight_era

            weight_whip = leverage.get("WHIP", 0)
            if weight_whip > 0:
                sgp = calculate_pitching_rate_sgp(
                    player_rate=player.whip, player_ip=player.ip,
                    replacement_rate=REPLACEMENT_WHIP,
                    sgp_denominator=denoms["WHIP"],
                    team_ip=DEFAULT_TEAM_IP, innings_divisor=1,
                )
                total += sgp * weight_whip

    elif player.get("player_type") == "hitter":
        for stat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            weight = leverage.get(stat, 0)
            if weight > 0:
                sgp = calculate_counting_sgp(player.get(col, 0), denoms[stat])
                total += sgp * weight

        weight_avg = leverage.get("AVG", 0)
        if weight_avg > 0:
            sgp = calculate_hitting_rate_sgp(
                player_avg=player.get("avg", 0),
                player_ab=int(player.get("ab", 0)),
                replacement_avg=REPLACEMENT_AVG,
                sgp_denominator=denoms["AVG"],
                team_ab=DEFAULT_TEAM_AB,
            )
            total += sgp * weight_avg

    elif player.get("player_type") == "pitcher":
        for stat, col in [("W", "w"), ("K", "k"), ("SV", "sv")]:
            weight = leverage.get(stat, 0)
            if weight > 0:
                sgp = calculate_counting_sgp(player.get(col, 0), denoms[stat])
                total += sgp * weight

        ip = player.get("ip", 0)
        if ip > 0:
            weight_era = leverage.get("ERA", 0)
            if weight_era > 0:
                sgp = calculate_pitching_rate_sgp(
                    player_rate=player.get("era", 0), player_ip=ip,
                    replacement_rate=REPLACEMENT_ERA,
                    sgp_denominator=denoms["ERA"],
                    team_ip=DEFAULT_TEAM_IP, innings_divisor=9,
                )
                total += sgp * weight_era

            weight_whip = leverage.get("WHIP", 0)
            if weight_whip > 0:
                sgp = calculate_pitching_rate_sgp(
                    player_rate=player.get("whip", 0), player_ip=ip,
                    replacement_rate=REPLACEMENT_WHIP,
                    sgp_denominator=denoms["WHIP"],
                    team_ip=DEFAULT_TEAM_IP, innings_divisor=1,
                )
                total += sgp * weight_whip

    return total
