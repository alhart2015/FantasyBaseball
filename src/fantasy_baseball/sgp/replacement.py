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
    REPLACEMENT_BY_POSITION,
    STARTERS_PER_POSITION,
    Category,
)
from fantasy_baseball.utils.positions import is_hitter
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


def calculate_replacement_levels(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
) -> dict[str, float]:
    """Calculate replacement-level SGP for each position."""
    if starters_per_position is None:
        starters_per_position = dict(STARTERS_PER_POSITION)

    replacement_levels: dict[str, float] = {}

    for position, num_starters in starters_per_position.items():
        if position in ("IF", "UTIL"):
            continue

        eligible = _get_eligible_players(player_pool, position)
        eligible = eligible.sort_values("total_sgp", ascending=False).reset_index(drop=True)

        if len(eligible) > num_starters:
            replacement_levels[position] = eligible.iloc[num_starters]["total_sgp"]
        elif len(eligible) > 0:
            replacement_levels[position] = eligible.iloc[-1]["total_sgp"]
        else:
            replacement_levels[position] = 0.0

    # Calculate UTIL replacement level from the full hitter pool.
    # UTIL slots are filled by the best remaining hitters after all
    # positional starter slots are accounted for, so the replacement
    # level is the SGP of the marginal hitter at that combined depth.
    util_starters = starters_per_position.get("UTIL", 0)
    if util_starters > 0:
        all_hitters = (
            player_pool[player_pool["positions"].apply(is_hitter)]
            .sort_values("total_sgp", ascending=False)
            .reset_index(drop=True)
        )

        # Total hitter starters across all positional + UTIL slots
        positional_hitter_slots = sum(
            n for pos, n in starters_per_position.items() if pos not in ("P", "IF", "UTIL")
        )
        total_hitter_starters = positional_hitter_slots + util_starters

        if len(all_hitters) > total_hitter_starters:
            replacement_levels["UTIL"] = all_hitters.iloc[total_hitter_starters]["total_sgp"]
        elif len(all_hitters) > 0:
            replacement_levels["UTIL"] = all_hitters.iloc[-1]["total_sgp"]
        else:
            replacement_levels["UTIL"] = 0.0

    return replacement_levels


def find_replacement_players(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Find the marginal replacement-level player at each position.

    Mirrors :func:`calculate_replacement_levels` demand math but returns
    the actual player row (as a dict) instead of just the SGP threshold.
    Use this when downstream code needs to swap a candidate against the
    replacement player's full stat line — e.g. the ERoto draft
    recommender, which compares ``score_roto(team_with_candidate)`` to
    ``score_roto(team_with_replacement)``.

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
            "avg": line["h"] / line["ab"] if line["ab"] else 0.0,
        }
    )
    return calculate_player_sgp(row, denoms=denoms, replacement_avg=replacement_avg)


def _empirical_pitcher_floor(
    position: str,
    denoms: dict[Category, float],
    replacement_era: float,
    replacement_whip: float,
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
            "era": 9 * line["er"] / ip if ip else 0.0,
            "whip": (line["bb"] + line["h_allowed"]) / ip if ip else 0.0,
        }
    )
    return calculate_player_sgp(
        row, denoms=denoms, replacement_era=replacement_era, replacement_whip=replacement_whip
    )


def position_aware_replacement_levels(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
    denoms: dict[Category, float] | None = None,
    repl_rates: dict[str, float] | None = None,
) -> dict[str, float]:
    """Replacement levels with empirical waiver floors per position.

    Starts from the demand-based :func:`calculate_replacement_levels` (whose
    unified "P" floor is kept as a fallback), then overrides each hitter
    position + UTIL with its empirical waiver-line SGP and adds separate
    empirical "SP"/"RP" floors. :func:`calculate_var` routes pitchers to the
    SP/RP floor by role and hitters to their position floor.
    """
    levels = calculate_replacement_levels(player_pool, starters_per_position)

    if denoms is None:
        denoms = get_sgp_denominators()
    rates = repl_rates or {}
    replacement_avg = rates.get("avg", REPLACEMENT_AVG)
    replacement_era = rates.get("era", REPLACEMENT_ERA)
    replacement_whip = rates.get("whip", REPLACEMENT_WHIP)

    empirical: dict[str, float] = {}
    for pos in _EMPIRICAL_HITTER_POSITIONS:
        if pos in levels and pos in REPLACEMENT_BY_POSITION:
            empirical[pos] = _empirical_floor_sgp(pos, denoms, replacement_avg)

    levels.update(empirical)
    if "UTIL" in levels and empirical:
        levels["UTIL"] = max(empirical.values())

    levels["SP"] = _empirical_pitcher_floor("SP", denoms, replacement_era, replacement_whip)
    levels["RP"] = _empirical_pitcher_floor("RP", denoms, replacement_era, replacement_whip)

    return levels
