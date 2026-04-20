import pandas as pd

from fantasy_baseball.models.player import HitterStats, PitcherStats, PlayerType
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.utils.constants import safe_float as _safe

from .denominators import get_sgp_denominators

DEFAULT_TEAM_AB: int = 5500
DEFAULT_TEAM_IP: int = 1450
REPLACEMENT_AVG: float = 0.250
REPLACEMENT_ERA: float = 4.50
REPLACEMENT_WHIP: float = 1.35


def calculate_counting_sgp(stat_value: float, sgp_denominator: float) -> float:
    """SGP = stat_value / sgp_denominator"""
    return stat_value / sgp_denominator


def calculate_hitting_rate_sgp(
    player_avg: float,
    player_ab: int,
    replacement_avg: float,
    sgp_denominator: float,
    team_ab: int,
) -> float:
    """SGP for AVG using marginal hits approach."""
    marginal_hits = (player_avg - replacement_avg) * player_ab
    one_sgp_in_hits = sgp_denominator * team_ab
    return marginal_hits / one_sgp_in_hits


def calculate_pitching_rate_sgp(
    player_rate: float,
    player_ip: float,
    replacement_rate: float,
    sgp_denominator: float,
    team_ip: float,
    innings_divisor: float,
) -> float:
    """SGP for ERA/WHIP using marginal value. Positive = better than replacement."""
    marginal = (replacement_rate - player_rate) * player_ip / innings_divisor
    one_sgp = sgp_denominator * team_ip / innings_divisor
    return marginal / one_sgp


def calculate_player_sgp(
    player: "HitterStats | PitcherStats | pd.Series",
    denoms: dict[Category, float] | None = None,
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

    if isinstance(player, HitterStats):
        for cat, val in [
            (Category.R, player.r),
            (Category.HR, player.hr),
            (Category.RBI, player.rbi),
            (Category.SB, player.sb),
        ]:
            total_sgp += calculate_counting_sgp(val, denoms[cat])
        total_sgp += calculate_hitting_rate_sgp(
            player_avg=player.avg,
            player_ab=int(player.ab),
            replacement_avg=replacement_avg,
            sgp_denominator=denoms[Category.AVG],
            team_ab=team_ab,
        )

    elif isinstance(player, PitcherStats):
        for cat, val in [
            (Category.W, player.w),
            (Category.K, player.k),
            (Category.SV, player.sv),
        ]:
            total_sgp += calculate_counting_sgp(val, denoms[cat])
        if player.ip > 0:
            total_sgp += calculate_pitching_rate_sgp(
                player_rate=player.era,
                player_ip=player.ip,
                replacement_rate=replacement_era,
                sgp_denominator=denoms[Category.ERA],
                team_ip=team_ip,
                innings_divisor=9,
            )
            total_sgp += calculate_pitching_rate_sgp(
                player_rate=player.whip,
                player_ip=player.ip,
                replacement_rate=replacement_whip,
                sgp_denominator=denoms[Category.WHIP],
                team_ip=team_ip,
                innings_divisor=1,
            )

    elif player.get("player_type") == PlayerType.HITTER:
        for cat, col in [
            (Category.R, "r"),
            (Category.HR, "hr"),
            (Category.RBI, "rbi"),
            (Category.SB, "sb"),
        ]:
            val = _safe(player.get(col, 0))
            total_sgp += calculate_counting_sgp(val, denoms[cat])
        total_sgp += calculate_hitting_rate_sgp(
            player_avg=_safe(player.get("avg", 0)),
            player_ab=int(_safe(player.get("ab", 0))),
            replacement_avg=replacement_avg,
            sgp_denominator=denoms[Category.AVG],
            team_ab=team_ab,
        )

    elif player.get("player_type") == PlayerType.PITCHER:
        for cat, col in [
            (Category.W, "w"),
            (Category.K, "k"),
            (Category.SV, "sv"),
        ]:
            val = _safe(player.get(col, 0))
            total_sgp += calculate_counting_sgp(val, denoms[cat])
        ip = _safe(player.get("ip", 0))
        if ip > 0:
            total_sgp += calculate_pitching_rate_sgp(
                player_rate=_safe(player.get("era", 0)),
                player_ip=ip,
                replacement_rate=replacement_era,
                sgp_denominator=denoms[Category.ERA],
                team_ip=team_ip,
                innings_divisor=9,
            )
            total_sgp += calculate_pitching_rate_sgp(
                player_rate=_safe(player.get("whip", 0)),
                player_ip=ip,
                replacement_rate=replacement_whip,
                sgp_denominator=denoms[Category.WHIP],
                team_ip=team_ip,
                innings_divisor=1,
            )

    else:
        ptype = player.get("player_type") if hasattr(player, "get") else type(player).__name__
        raise ValueError(f"Unknown player_type: {ptype!r}")

    return total_sgp
