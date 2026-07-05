from typing import Any

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import (
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
    calculate_player_sgp,
)
from fantasy_baseball.utils.constants import (
    DEFAULT_TEAM_AB,
    DEFAULT_TEAM_IP,
    REPLACEMENT_BY_POSITION,
    STARTERS_PER_POSITION,
    Category,
)
from fantasy_baseball.utils.positions import is_hitter
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


def find_replacement_players(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Find the marginal replacement-level player at each position.

    The marginal player at a position is the (N+1)-th best by ``total_sgp``
    when N starter slots exist there (or the worst eligible if the pool is
    shallower than demand). Returns the actual player row (as a dict) so
    downstream code can swap a candidate against the replacement player's
    full stat line -- e.g. the ERoto draft recommender, which compares
    ``score_roto(team_with_candidate)`` to ``score_roto(team_with_replacement)``.

    Note: this demand-based floor is no longer the live VAR replacement
    source -- the board and recommender use
    :func:`position_aware_replacement_levels` (static empirical waiver
    lines). This function survives as the row-returning primitive for the
    recs swap path.

    Skips ``IF`` (handled via UTIL/positional fallback in ``calculate_var``).
    For ``UTIL``, uses the marginal hitter at depth
    ``positional_hitter_starters + util_starters`` over the full hitter pool.
    """
    if starters_per_position is None:
        starters_per_position = dict(STARTERS_PER_POSITION)

    out: dict[str, dict[str, Any]] = {}

    for position, num_starters in starters_per_position.items():
        if position in ("IF", "UTIL"):
            continue

        eligible = _get_eligible_players(player_pool, position)
        eligible = eligible.sort_values("total_sgp", ascending=False).reset_index(drop=True)
        if eligible.empty:
            continue
        idx = num_starters if len(eligible) > num_starters else len(eligible) - 1
        out[position] = eligible.iloc[idx].to_dict()

    util_starters = starters_per_position.get("UTIL", 0)
    if util_starters > 0:
        all_hitters = (
            player_pool[player_pool["positions"].apply(is_hitter)]
            .sort_values("total_sgp", ascending=False)
            .reset_index(drop=True)
        )
        positional_hitter_slots = sum(
            n for pos, n in starters_per_position.items() if pos not in ("P", "IF", "UTIL")
        )
        total_hitter_starters = positional_hitter_slots + util_starters
        if not all_hitters.empty:
            idx = (
                total_hitter_starters
                if len(all_hitters) > total_hitter_starters
                else len(all_hitters) - 1
            )
            out["UTIL"] = all_hitters.iloc[idx].to_dict()

    return out


def _get_eligible_players(pool: pd.DataFrame, position: str) -> pd.DataFrame:
    if position == "P":
        return pool[pool["positions"].apply(lambda pos: any(p in ("P", "SP", "RP") for p in pos))]
    if position == "OF":
        return pool[pool["positions"].apply(lambda pos: "OF" in pos)]
    return pool[pool["positions"].apply(lambda pos: position in pos)]


def calculate_replacement_rates(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
) -> dict[str, float]:
    """Derive replacement-level rate stats from the player pool.

    Averages ERA/WHIP/AVG across a band of ±5 players around the
    replacement threshold to smooth noise from any single player.
    Falls back to hardcoded defaults when the pool is empty.
    """
    if starters_per_position is None:
        starters_per_position = dict(STARTERS_PER_POSITION)

    rates: dict[str, float] = {}

    # Pitcher replacement rates
    num_p_starters = starters_per_position.get("P", 90)
    pitchers = _get_eligible_players(player_pool, "P")
    pitchers = pitchers.sort_values("total_sgp", ascending=False).reset_index(drop=True)

    if len(pitchers) > num_p_starters:
        lo = max(0, num_p_starters - 5)
        hi = min(len(pitchers), num_p_starters + 6)
        band = pitchers.iloc[lo:hi]
        band = band[band["ip"] > 0]
        if not band.empty:
            total_er = band["er"].sum()
            total_ip = band["ip"].sum()
            total_bb = band["bb"].sum()
            total_ha = band["h_allowed"].sum()
            rates["era"] = calculate_era(total_er, total_ip)
            rates["whip"] = calculate_whip(total_bb, total_ha, total_ip)
        else:
            rates["era"] = REPLACEMENT_ERA
            rates["whip"] = REPLACEMENT_WHIP
    else:
        rates["era"] = REPLACEMENT_ERA
        rates["whip"] = REPLACEMENT_WHIP

    # Hitter replacement rates
    all_hitters = (
        player_pool[player_pool["positions"].apply(is_hitter)]
        .sort_values("total_sgp", ascending=False)
        .reset_index(drop=True)
    )

    positional_hitter_slots = sum(
        n for pos, n in starters_per_position.items() if pos not in ("P", "IF", "UTIL")
    )
    util_slots = starters_per_position.get("UTIL", 0)
    total_hitter_starters = positional_hitter_slots + util_slots

    if len(all_hitters) > total_hitter_starters:
        lo = max(0, total_hitter_starters - 5)
        hi = min(len(all_hitters), total_hitter_starters + 6)
        band = all_hitters.iloc[lo:hi]
        band = band[band["ab"] > 0]
        if not band.empty:
            rates["avg"] = calculate_avg(band["h"].sum(), band["ab"].sum())
        else:
            rates["avg"] = REPLACEMENT_AVG
    else:
        rates["avg"] = REPLACEMENT_AVG

    return rates


# Hitter positions for which REPLACEMENT_BY_POSITION carries a waiver line.
_EMPIRICAL_HITTER_POSITIONS = ("C", "1B", "2B", "3B", "SS", "OF")


def _empirical_floor_sgp(
    position: str,
    denoms: dict[Category, float],
    replacement_avg: float,
    team_ab: int = DEFAULT_TEAM_AB,
) -> float:
    """SGP of a position's empirical waiver line (REPLACEMENT_BY_POSITION).

    Built and scored through the same ``calculate_player_sgp`` path as real
    players so the floor and player values land on one scale.
    """
    line = REPLACEMENT_BY_POSITION[position]
    row = pd.Series(
        {
            "player_type": PlayerType.HITTER,
            "r": line["r"],
            "hr": line["hr"],
            "rbi": line["rbi"],
            "sb": line["sb"],
            "ab": line["ab"],
            "avg": calculate_avg(line["h"], line["ab"]),
        }
    )
    return calculate_player_sgp(
        row, denoms=denoms, replacement_avg=replacement_avg, team_ab=team_ab
    )


def _empirical_pitcher_floor(
    position: str,
    denoms: dict[Category, float],
    replacement_era: float,
    replacement_whip: float,
    team_ip: int = DEFAULT_TEAM_IP,
) -> float:
    """SGP of an empirical SP/RP waiver line (REPLACEMENT_BY_POSITION).

    SP and RP are kept separate so a closer's saves net against the RP line's
    free SV while a starter's strikeouts net against the SP line's deep K --
    a single unified-"P" floor cannot do both correctly.
    """
    line = REPLACEMENT_BY_POSITION[position]
    ip = line["ip"]
    row = pd.Series(
        {
            "player_type": PlayerType.PITCHER,
            "w": line["w"],
            "k": line["k"],
            "sv": line["sv"],
            "ip": ip,
            "era": calculate_era(line["er"], ip),
            "whip": calculate_whip(line["bb"], line["h_allowed"], ip),
        }
    )
    return calculate_player_sgp(
        row,
        denoms=denoms,
        replacement_era=replacement_era,
        replacement_whip=replacement_whip,
        team_ip=team_ip,
    )


def position_aware_replacement_levels(
    denoms: dict[Category, float] | None = None,
    repl_rates: dict[str, float] | None = None,
    team_ab: int = DEFAULT_TEAM_AB,
    team_ip: int = DEFAULT_TEAM_IP,
) -> dict[str, float]:
    """Empirical waiver replacement floors per position.

    A pure function of ``denoms`` + the AVG/ERA/WHIP rate baselines: every
    floor is the SGP of an empirical ``REPLACEMENT_BY_POSITION`` waiver line,
    independent of the live draft pool. ``calculate_var`` routes pitchers to
    the SP/RP floor by role and hitters to their position floor; ``UTIL`` is the
    best (highest-SGP) hitter floor, used as the fallback for DH-only bats.
    """
    if denoms is None:
        denoms = get_sgp_denominators()
    rates = repl_rates or {}
    replacement_avg = rates.get("avg", REPLACEMENT_AVG)
    replacement_era = rates.get("era", REPLACEMENT_ERA)
    replacement_whip = rates.get("whip", REPLACEMENT_WHIP)

    levels: dict[str, float] = {
        pos: _empirical_floor_sgp(pos, denoms, replacement_avg, team_ab=team_ab)
        for pos in _EMPIRICAL_HITTER_POSITIONS
    }
    levels["UTIL"] = max(levels.values())
    levels["SP"] = _empirical_pitcher_floor(
        "SP", denoms, replacement_era, replacement_whip, team_ip=team_ip
    )
    levels["RP"] = _empirical_pitcher_floor(
        "RP", denoms, replacement_era, replacement_whip, team_ip=team_ip
    )
    return levels
