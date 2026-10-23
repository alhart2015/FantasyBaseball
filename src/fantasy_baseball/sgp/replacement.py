from typing import Any

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import SgpOverrides, get_sgp_denominators
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


# Hitter positions for which REPLACEMENT_BY_POSITION carries a waiver line.
_EMPIRICAL_HITTER_POSITIONS = ("C", "1B", "2B", "3B", "SS", "OF")


def _empirical_floor_sgp(
    position: str,
    denoms: dict[Category, float],
    replacement_avg: float,
    team_ab: int = DEFAULT_TEAM_AB,
    fraction: float = 1.0,
) -> float:
    """SGP of a position's empirical waiver line (REPLACEMENT_BY_POSITION).

    Built and scored through the same ``calculate_player_sgp`` path as real
    players so the floor and player values land on one scale. ``fraction < 1.0``
    yields the to-date floor: the counting stats and the team AB volume scale by
    ``fraction`` while the AVG rate is HELD. Floor SGP is not linear in ``fraction``
    (only its counting component scales), so this is what to-date player scoring nets
    against -- not ``full_floor * fraction``. Team AB is floored at 1 so the degenerate
    near-f=0 scale cannot truncate the rate-SGP denominator to 0 (silent NaN).
    """
    line = REPLACEMENT_BY_POSITION[position]
    row = pd.Series(
        {
            "player_type": PlayerType.HITTER,
            "r": line["r"] * fraction,
            "hr": line["hr"] * fraction,
            "rbi": line["rbi"] * fraction,
            "sb": line["sb"] * fraction,
            "ab": line["ab"] * fraction,
            "avg": calculate_avg(line["h"], line["ab"]),
        }
    )
    return calculate_player_sgp(
        row,
        denoms=denoms,
        replacement_avg=replacement_avg,
        team_ab=max(1, int(team_ab * fraction)),
    )


def _empirical_pitcher_floor(
    position: str,
    denoms: dict[Category, float],
    replacement_era: float,
    replacement_whip: float,
    team_ip: int = DEFAULT_TEAM_IP,
    fraction: float = 1.0,
) -> float:
    """SGP of an empirical SP/RP waiver line (REPLACEMENT_BY_POSITION).

    SP and RP are kept separate so a closer's saves net against the RP line's
    free SV while a starter's strikeouts net against the SP line's deep K --
    a single unified-"P" floor cannot do both correctly. ``fraction < 1.0`` yields
    the to-date floor: the counting stats and the team IP volume scale by ``fraction``
    while the ERA/WHIP rates are HELD (see ``_empirical_floor_sgp``). Team IP is
    floored at 1 to keep the near-f=0 scale out of a zero rate-SGP denominator.
    """
    line = REPLACEMENT_BY_POSITION[position]
    ip = line["ip"]
    row = pd.Series(
        {
            "player_type": PlayerType.PITCHER,
            "w": line["w"] * fraction,
            "k": line["k"] * fraction,
            "sv": line["sv"] * fraction,
            "ip": ip * fraction,
            "era": calculate_era(line["er"], ip),
            "whip": calculate_whip(line["bb"], line["h_allowed"], ip),
        }
    )
    return calculate_player_sgp(
        row,
        denoms=denoms,
        replacement_era=replacement_era,
        replacement_whip=replacement_whip,
        team_ip=max(1, int(team_ip * fraction)),
    )


def position_aware_replacement_levels(
    denoms: dict[Category, float] | None = None,
    repl_rates: dict[str, float] | None = None,
    team_ab: int = DEFAULT_TEAM_AB,
    team_ip: int = DEFAULT_TEAM_IP,
    fraction: float = 1.0,
) -> dict[str, float]:
    """Empirical waiver replacement floors per position.

    A pure function of ``denoms`` + the AVG/ERA/WHIP rate baselines: every
    floor is the SGP of an empirical ``REPLACEMENT_BY_POSITION`` waiver line,
    independent of the live draft pool. ``calculate_var`` routes pitchers to
    the SP/RP floor by role and hitters to their position floor; ``UTIL`` is the
    best (highest-SGP) hitter floor, used as the fallback for DH-only bats.

    ``fraction < 1.0`` returns the to-date floors used by the draft-value metric's
    YTD path: counting stats and team volumes scale by ``fraction`` while rates are
    held (floor SGP is not linear in ``fraction``). At ``fraction == 1.0`` this is
    the full-season board floor, so the draft-value module delegates here instead of
    re-encoding the scaling recipe and the ``UTIL = max(hitter floors)`` rule.
    """
    if denoms is None:
        denoms = get_sgp_denominators()
    rates = repl_rates or {}
    replacement_avg = rates.get("avg", REPLACEMENT_AVG)
    replacement_era = rates.get("era", REPLACEMENT_ERA)
    replacement_whip = rates.get("whip", REPLACEMENT_WHIP)

    levels: dict[str, float] = {
        pos: _empirical_floor_sgp(pos, denoms, replacement_avg, team_ab=team_ab, fraction=fraction)
        for pos in _EMPIRICAL_HITTER_POSITIONS
    }
    levels["UTIL"] = max(levels.values())
    levels["SP"] = _empirical_pitcher_floor(
        "SP", denoms, replacement_era, replacement_whip, team_ip=team_ip, fraction=fraction
    )
    levels["RP"] = _empirical_pitcher_floor(
        "RP", denoms, replacement_era, replacement_whip, team_ip=team_ip, fraction=fraction
    )
    return levels


def league_replacement_levels(sgp_overrides: SgpOverrides | None) -> dict[str, float]:
    """The league's replacement floors: THE source for every live model.

    The measured waiver lines (``REPLACEMENT_BY_POSITION``) scored on the fixed
    AVG / ERA / WHIP baselines (``REPLACEMENT_RATES``, #521), the league's SGP
    denominators (pass ``config.sgp_overrides``; ``None`` only where there is no
    league config) and the default team volumes. The draft board, the recommender,
    the trajectory boards, the keeper report and the value bars all read their
    floors here, so none can assemble the inputs differently. The one deliberate
    exception is draft-value, which rebuilds a past draft day's floors from that
    day's frozen scale (``position_aware_replacement_levels`` with its inputs).
    """
    return position_aware_replacement_levels(get_sgp_denominators(sgp_overrides))
