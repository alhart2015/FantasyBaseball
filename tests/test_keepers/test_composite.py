import math

import pandas as pd
import pytest

from fantasy_baseball.keepers.composite import (
    FITTED_WEIGHTS,
    LOWER_IS_BETTER,
    SKILL_COLUMNS,
    composite,
    percentile,
    regression_gap,
    skill_percentile,
)


def test_percentile_is_zero_to_one_and_ordered():
    out = percentile(pd.Series([10.0, 20.0, 30.0, 40.0]))
    assert out.tolist() == [0.25, 0.5, 0.75, 1.0]


def test_percentile_flips_when_lower_is_better():
    values = pd.Series([2.0, 4.0, 6.0, 8.0])
    assert percentile(values, higher_is_better=False).tolist() == [1.0, 0.75, 0.5, 0.25]


def test_flipped_percentile_spans_the_same_range_as_an_unflipped_one():
    """`1 - rank` would be off by 1/n, holding era_minus and fip below the four
    unflipped pitcher stats they are averaged against."""
    rising = pd.Series([1.0, 2.0, 3.0, 4.0])
    falling = pd.Series([4.0, 3.0, 2.0, 1.0])
    up = percentile(rising)
    down = percentile(falling, higher_is_better=False)
    assert up.tolist() == down.tolist()
    assert up.max() == 1.0 and down.max() == 1.0


def test_percentile_keeps_nan_rather_than_sorting_it_last():
    """A missing observation must not read as the worst one."""
    out = percentile(pd.Series([1.0, float("nan"), 3.0]))
    assert math.isnan(out.iloc[1])
    assert out.iloc[0] < out.iloc[2]


def test_era_minus_and_fip_are_the_flipped_pitcher_stats():
    """A lower ERA- is a better pitcher; the rest are higher-is-better."""
    assert {"era_minus", "fip"} == LOWER_IS_BETTER
    assert set(SKILL_COLUMNS["pitcher"]) > LOWER_IS_BETTER
    assert not LOWER_IS_BETTER & set(SKILL_COLUMNS["hitter"])


def test_skill_percentile_ranks_the_better_pitcher_higher():
    frame = pd.DataFrame(
        {
            "era_minus": [70.0, 130.0],  # lower better
            "fip": [3.0, 5.0],  # lower better
            "k_pct": [30.0, 18.0],
            "swstr_pct": [14.0, 8.0],
            "whiff_pct": [30.0, 18.0],
            "csw_pct": [32.0, 24.0],
        }
    )
    out = skill_percentile(frame, "pitcher")
    assert out.iloc[0] > out.iloc[1]
    assert out.iloc[0] == pytest.approx(1.0)


def test_skill_percentile_averages_over_what_is_present():
    """A hitter missing barrel rate is averaged over his other stats, not dropped."""
    frame = pd.DataFrame(
        {
            "barrel_pct": [12.0, float("nan")],
            "barrel_pa_pct": [5.0, 4.0],
            "xwoba": [0.360, 0.330],
            "xba": [0.270, 0.250],
            "wrc_plus": [140.0, 110.0],
        }
    )
    out = skill_percentile(frame, "hitter")
    assert not out.isna().any()
    assert out.iloc[0] > out.iloc[1]


def test_skill_percentile_raises_on_a_renamed_column():
    frame = pd.DataFrame({col: [1.0] for col in SKILL_COLUMNS["hitter"]}).rename(
        columns={"xwoba": "expected_woba"}
    )
    with pytest.raises(KeyError, match="xwoba"):
        skill_percentile(frame, "hitter")


def test_composite_is_a_weighted_blend_on_the_same_scale():
    one = pd.Series([1.0])
    assert composite(one, one, one, "hitter").iloc[0] == pytest.approx(1.0)
    zero = pd.Series([0.0])
    assert composite(zero, zero, zero, "pitcher").iloc[0] == pytest.approx(0.0)


def test_composite_normalizes_weights_that_do_not_sum_to_one():
    value, skill, age = pd.Series([1.0]), pd.Series([0.0]), pd.Series([0.0])
    doubled = composite(value, skill, age, "hitter", weights=(2.0, 0.0, 0.0))
    assert doubled.iloc[0] == pytest.approx(1.0)


def test_composite_rejects_degenerate_weights():
    s = pd.Series([0.5])
    with pytest.raises(ValueError, match="positive"):
        composite(s, s, s, "hitter", weights=(0.0, 0.0, 0.0))


def test_age_lowers_an_older_players_composite():
    """Age is the only family where the two players differ."""
    value = skill = pd.Series([0.9, 0.9])
    age = pd.Series([1.0, 0.0])  # already a percentile: younger is 1.0
    out = composite(value, skill, age, "hitter")
    assert out.iloc[0] > out.iloc[1]


def test_fitted_weights_favor_value_and_include_every_family():
    """Pins the shape the backtest found -- value dominant, age a small
    adjustment -- so a future edit that inverts it has to be deliberate."""
    for kind, (w_value, w_skill, w_age) in FITTED_WEIGHTS.items():
        assert w_value > w_skill >= 0.0, kind
        assert w_value > w_age >= 0.0, kind
        assert sum((w_value, w_skill, w_age)) == pytest.approx(1.0), kind


def test_regression_gap_is_positive_when_production_outran_skills():
    gap = regression_gap(pd.Series([0.95, 0.40]), pd.Series([0.55, 0.80]))
    assert gap.iloc[0] == pytest.approx(0.40)  # sell high
    assert gap.iloc[1] == pytest.approx(-0.40)  # buy low
