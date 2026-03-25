import pandas as pd
from fantasy_baseball.utils.constants import DEFAULT_SGP_DENOMINATORS, safe_float as _safe
from .denominators import get_sgp_denominators

DEFAULT_TEAM_AB: int = 5500
DEFAULT_TEAM_IP: int = 1400
REPLACEMENT_AVG: float = 0.250
REPLACEMENT_ERA: float = 4.50
REPLACEMENT_WHIP: float = 1.35


def calculate_counting_sgp(stat_value: float, sgp_denominator: float) -> float:
    """SGP = stat_value / sgp_denominator"""
    return stat_value / sgp_denominator


def calculate_hitting_rate_sgp(
    player_avg: float, player_ab: int, replacement_avg: float,
    sgp_denominator: float, team_ab: int,
) -> float:
    """SGP for AVG using marginal hits approach."""
    marginal_hits = (player_avg - replacement_avg) * player_ab
    one_sgp_in_hits = sgp_denominator * team_ab
    return marginal_hits / one_sgp_in_hits


def calculate_pitching_rate_sgp(
    player_rate: float, player_ip: float, replacement_rate: float,
    sgp_denominator: float, team_ip: float, innings_divisor: float,
) -> float:
    """SGP for ERA/WHIP using marginal value. Positive = better than replacement."""
    marginal = (replacement_rate - player_rate) * player_ip / innings_divisor
    one_sgp = sgp_denominator * team_ip / innings_divisor
    return marginal / one_sgp


def calculate_player_sgp(
    player: pd.Series,
    denoms: dict[str, float] | None = None,
    team_ab: int = DEFAULT_TEAM_AB,
    team_ip: int = DEFAULT_TEAM_IP,
    replacement_avg: float = REPLACEMENT_AVG,
    replacement_era: float = REPLACEMENT_ERA,
    replacement_whip: float = REPLACEMENT_WHIP,
) -> float:
    """Calculate total SGP for a player across all relevant categories."""
    if denoms is None:
        denoms = get_sgp_denominators()

    total_sgp = 0.0

    if player.get("player_type") == "hitter":
        for stat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            val = _safe(player.get(col, 0))
            total_sgp += calculate_counting_sgp(val, denoms[stat])
        total_sgp += calculate_hitting_rate_sgp(
            player_avg=_safe(player.get("avg", 0)),
            player_ab=int(_safe(player.get("ab", 0))),
            replacement_avg=replacement_avg,
            sgp_denominator=denoms["AVG"],
            team_ab=team_ab,
        )

    elif player.get("player_type") == "pitcher":
        for stat, col in [("W", "w"), ("K", "k"), ("SV", "sv")]:
            val = _safe(player.get(col, 0))
            total_sgp += calculate_counting_sgp(val, denoms[stat])
        ip = _safe(player.get("ip", 0))
        if ip > 0:
            total_sgp += calculate_pitching_rate_sgp(
                player_rate=_safe(player.get("era", 0)), player_ip=ip,
                replacement_rate=replacement_era,
                sgp_denominator=denoms["ERA"], team_ip=team_ip, innings_divisor=9,
            )
            total_sgp += calculate_pitching_rate_sgp(
                player_rate=_safe(player.get("whip", 0)), player_ip=ip,
                replacement_rate=replacement_whip,
                sgp_denominator=denoms["WHIP"], team_ip=team_ip, innings_divisor=1,
            )

    return total_sgp
