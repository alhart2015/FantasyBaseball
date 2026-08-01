import pytest

from fantasy_baseball.analysis.draft_value import ParCurve
from fantasy_baseball.analysis.trade_pick import (
    NextYearValue,
    pick_ordinal_range,
    pick_value,
)


def _curve(n=200):
    # Strictly descending so mean-over-a-range is order-sensitive and testable.
    return ParCurve(drafted_pars=[float(n - i) for i in range(n)], keeper_par=18.0)


def test_ordinal_range_round_2_is_11_to_20():
    # nominal 5, 3 keeper rounds, 10 teams -> drafted round 2 -> ordinals 11..20
    assert pick_ordinal_range(5, 3, 10, 200) == (11, 20)


def test_ordinal_range_first_drafted_round():
    assert pick_ordinal_range(4, 3, 10, 200) == (1, 10)


def test_ordinal_range_keeper_round_rejected():
    with pytest.raises(ValueError, match="keeper round"):
        pick_ordinal_range(3, 3, 10, 200)


def test_ordinal_range_beyond_curve_rejected():
    with pytest.raises(ValueError, match="beyond the par curve"):
        pick_ordinal_range(60, 3, 10, 200)  # drafted round 57 -> lo far past 200


def test_ordinal_range_clamps_upper_bound():
    # drafted round 20 -> ordinals 191..200; a 195-long curve clamps hi to 195.
    assert pick_ordinal_range(23, 3, 10, 195) == (191, 195)


def test_ordinal_range_early_mid_late_partition_the_round():
    lo_e, hi_e = pick_ordinal_range(5, 3, 10, 200, "early")
    lo_m, hi_m = pick_ordinal_range(5, 3, 10, 200, "mid")
    lo_l, hi_l = pick_ordinal_range(5, 3, 10, 200, "late")
    assert lo_e == 11 and hi_e < 20  # early starts at the round's top
    assert hi_l == 20 and lo_l > 11  # late ends at the round's bottom
    assert lo_e <= lo_m <= lo_l and hi_e <= hi_m <= hi_l


def test_pick_value_round_average_and_early_higher():
    par = _curve()
    nv = pick_value(par, 5, 3, 10, "round")
    assert isinstance(nv, NextYearValue)
    assert nv.drafted_round == 2
    assert nv.ordinal_lo == 11 and nv.ordinal_hi == 20
    # mean of par_for_slot(11..20) = mean of drafted_pars[10..19] = mean(190..181) = 185.5
    assert nv.expected_var == pytest.approx(185.5)
    # early third (higher VAR) exceeds the full-round average on a descending curve
    assert nv.early_var > nv.expected_var
    assert nv.keeper_par == 18.0
