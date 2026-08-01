"""Trade-for-future-pick calculator.

Two separate views of trading a current player for a next-year draft pick:
the this-year ROS Monte Carlo impact (win% / top-3% / per-category) and the
next-year marginal value of the extra pick (VAR at the post-keeper-round draft
ordinal). See docs/superpowers/specs/2026-07-31-trade-pick-calculator-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from fantasy_baseball.analysis.draft_value import ParCurve, projected_par_curve
from fantasy_baseball.config import LeagueConfig


def keeper_rounds_for(config: LeagueConfig) -> int:
    """Number of keeper rounds = keepers per team = len(keepers) // num_teams.

    Requires an even split (every team keeps the same number, which is what
    "the first K rounds are keeper rounds" means). A non-divisible count breaks
    the nominal-to-drafted-round mapping, so fail loud rather than truncate.
    """
    nk = len(config.keepers)
    nt = config.num_teams
    if nt <= 0 or nk == 0 or nk % nt != 0:
        raise ValueError(
            f"Cannot derive keeper rounds: {nk} keepers / {nt} teams is not an "
            "even split. The nominal-to-drafted-round mapping needs a fixed "
            "keeper-rounds count."
        )
    return nk // nt


def pick_ordinal_range(
    nominal_round: int,
    keeper_rounds: int,
    num_teams: int,
    curve_len: int,
    pick_slot: str = "round",
) -> tuple[int, int]:
    """1-based inclusive par-curve ordinal range for a nominal pick round.

    drafted_round = nominal_round - keeper_rounds; the drafted round spans
    ordinals [(drafted_round-1)*num_teams + 1, drafted_round*num_teams] in the
    VAR-sorted par curve. ``pick_slot`` narrows to the top ("early"), middle
    ("mid"), or bottom ("late") third of that span. The upper bound is clamped
    to ``curve_len``; a range entirely beyond the curve is an error.
    """
    drafted_round = nominal_round - keeper_rounds
    if drafted_round < 1:
        raise ValueError(
            f"Round {nominal_round} is a keeper round; drafted picks start at "
            f"round {keeper_rounds + 1}."
        )
    lo = (drafted_round - 1) * num_teams + 1
    hi = drafted_round * num_teams
    if lo > curve_len:
        raise ValueError(
            f"drafted round {drafted_round} (ordinals {lo}-{hi}) is beyond the "
            f"par curve ({curve_len} picks)."
        )
    hi = min(hi, curve_len)
    if pick_slot == "round":
        return lo, hi
    if pick_slot not in ("early", "mid", "late"):
        raise ValueError("pick_slot must be one of: round, early, mid, late")
    span = hi - lo + 1
    third = max(1, span // 3)
    if pick_slot == "early":
        return lo, lo + third - 1
    if pick_slot == "late":
        return hi - third + 1, hi
    # mid: the middle third, clamped to stay non-empty inside [lo, hi]
    mlo = min(lo + third, hi)
    mhi = max(hi - third, mlo)
    return mlo, mhi


@dataclass(frozen=True)
class NextYearValue:
    nominal_round: int
    keeper_rounds: int
    drafted_round: int
    pick_slot: str
    expected_var: float
    early_var: float
    keeper_par: float
    ordinal_lo: int
    ordinal_hi: int


def _mean_over(par: ParCurve, lo: int, hi: int) -> float:
    vals = [par.par_for_slot(k) for k in range(lo, hi + 1)]
    return sum(vals) / len(vals)


def pick_value(
    par: ParCurve,
    nominal_round: int,
    keeper_rounds: int,
    num_teams: int,
    pick_slot: str = "round",
) -> NextYearValue:
    """Expected pick VAR = mean par over the (narrowed) drafted-round ordinals.

    Also reports an "early in the round" VAR (top third) for context and the
    keeper-average par (``par.keeper_par``, may be NaN).
    """
    curve_len = len(par.drafted_pars)
    lo, hi = pick_ordinal_range(nominal_round, keeper_rounds, num_teams, curve_len, pick_slot)
    elo, ehi = pick_ordinal_range(nominal_round, keeper_rounds, num_teams, curve_len, "early")
    return NextYearValue(
        nominal_round=nominal_round,
        keeper_rounds=keeper_rounds,
        drafted_round=nominal_round - keeper_rounds,
        pick_slot=pick_slot,
        expected_var=_mean_over(par, lo, hi),
        early_var=_mean_over(par, elo, ehi),
        keeper_par=par.keeper_par,
        ordinal_lo=lo,
        ordinal_hi=hi,
    )


def next_year_value(
    config: LeagueConfig, nominal_round: int, pick_slot: str = "round"
) -> NextYearValue:
    """Build the projected par curve and value a nominal-round pick on it."""
    par = projected_par_curve(config)
    return pick_value(par, nominal_round, keeper_rounds_for(config), config.num_teams, pick_slot)
