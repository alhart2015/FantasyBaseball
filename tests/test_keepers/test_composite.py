import math

import pandas as pd
import pytest

from fantasy_baseball.keepers.composite import (
    FAMILIES,
    FITTED_WEIGHTS,
    FUTURE_BLEND,
    LOWER_IS_BETTER,
    SKILL_COLUMNS,
    batted_ball,
    composite,
    future_percentile,
    luck,
    percentile,
    skill_percentile,
)


def _fam(skill, luck_, future, age):
    return {
        "skill": pd.Series(skill),
        "luck": pd.Series(luck_),
        "future": pd.Series(future),
        "age": pd.Series(age),
    }


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
    assert composite(_fam([1.0], [1.0], [1.0], [1.0]), "hitter").iloc[0] == pytest.approx(1.0)
    assert composite(_fam([0.0], [0.0], [0.0], [0.0]), "pitcher").iloc[0] == pytest.approx(0.0)


def test_composite_normalizes_weights_that_do_not_sum_to_one():
    out = composite(_fam([1.0], [0.0], [0.0], [0.0]), "hitter", weights=(2.0, 0.0, 0.0, 0.0))
    assert out.iloc[0] == pytest.approx(1.0)


def test_composite_rejects_an_unknown_family():
    with pytest.raises(KeyError, match="peripherals"):
        composite({"peripherals": pd.Series([0.5])}, "hitter")


def test_composite_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="no weighted family"):
        composite(_fam([0.5], [0.0], [0.0], [0.0]), "hitter", weights=(0, 0, 0, 0))


def test_a_missing_family_drops_out_of_the_denominator():
    """A pool with no out-year projection must stay comparable, not be silently
    scaled down as though every player projected to zero. With every supplied
    family at the same level the blend must equal that level either way -- which
    only holds if the absent family's weight leaves the denominator too."""
    supplied = {family: pd.Series([0.8]) for family in FAMILIES["hitter"]}
    full = composite(supplied, "hitter").iloc[0]
    partial = composite({k: v for k, v in supplied.items() if k != "future"}, "hitter").iloc[0]
    assert full == pytest.approx(0.8)
    assert partial == pytest.approx(0.8)


def test_a_missing_family_is_not_treated_as_zero():
    """The failure this guards: keeping the weight but supplying nothing, which
    would drag every player toward zero and quietly compress the whole pool."""
    supplied = {"skill": pd.Series([0.8]), "luck": pd.Series([0.0]), "age": pd.Series([0.8])}
    partial = composite(supplied, "hitter").iloc[0]
    as_zero = composite({**supplied, "future": pd.Series([0.0])}, "hitter").iloc[0]
    assert partial > as_zero


def test_luck_is_value_minus_skill():
    assert luck(pd.Series([0.95, 0.40]), pd.Series([0.55, 0.80])).tolist() == pytest.approx(
        [0.40, -0.40]
    )


def test_luck_carries_a_positive_weight_for_every_position():
    """Not what the name suggests, and load-bearing: the gap also encodes playing
    time, and forcing it negative collapses the fit (rho 0.65 -> 0.13)."""
    for kind, weights in FITTED_WEIGHTS.items():
        assert weights[FAMILIES[kind].index("luck")] > 0, kind


def test_future_blend_favors_the_nearer_year_and_sums_to_one():
    near, far = FUTURE_BLEND
    assert near > far
    assert near + far == pytest.approx(1.0)


def test_future_percentile_weights_the_nearer_year_more():
    """Two players, mirrored projections: the one better in the NEAR year wins."""
    near = pd.Series([10.0, 5.0])
    far = pd.Series([5.0, 10.0])
    out = future_percentile(near, far)
    assert out.iloc[0] > out.iloc[1]


def test_future_percentile_falls_back_to_the_near_year_when_far_is_missing():
    """A player the 2028 file does not cover keeps his 2027 standing rather than
    being dragged toward zero."""
    near = pd.Series([10.0, 8.0, 6.0])
    both = future_percentile(near, pd.Series([10.0, 8.0, 6.0]))
    partial = future_percentile(near, pd.Series([10.0, float("nan"), 6.0]))
    assert partial.tolist() == pytest.approx(both.tolist())


def test_fitted_weights_have_one_entry_per_family_and_lead_with_skill():
    for kind, weights in FITTED_WEIGHTS.items():
        assert len(weights) == len(FAMILIES[kind]), kind
        assert weights[FAMILIES[kind].index("skill")] == max(weights), kind
        assert weights[FAMILIES[kind].index("future")] > 0, kind


def test_composite_honors_an_explicit_family_order():
    fams = {"skill": pd.Series([1.0, 0.0]), "pt": pd.Series([0.0, 1.0])}
    out = composite(fams, "hitter", weights=(1.0, 1.0), family_order=("skill", "pt"))
    assert out.tolist() == pytest.approx([0.5, 0.5])


def test_composite_rejects_a_family_outside_the_known_universe():
    with pytest.raises(KeyError, match="peripherals"):
        composite({"peripherals": pd.Series([0.5])}, "hitter", family_order=("skill",))


def test_batted_ball_is_avg_over_xba_for_hitters():
    frame = pd.DataFrame({"avg": [0.278, 0.240], "xba": [0.242, 0.250]})
    out = batted_ball(frame, "hitter")
    assert out.tolist() == pytest.approx([0.036, -0.010])


def test_batted_ball_is_fip_minus_era_for_pitchers():
    """ERA below FIP means the pitcher outran his peripherals -- luckier, higher."""
    frame = pd.DataFrame({"fip": [4.20, 3.50], "era": [3.10, 3.60]})
    out = batted_ball(frame, "pitcher")
    assert out.tolist() == pytest.approx([1.10, -0.10])


def test_batted_ball_keeps_nan_when_an_input_is_missing():
    frame = pd.DataFrame({"avg": [0.278, float("nan")], "xba": [0.242, 0.250]})
    out = batted_ball(frame, "hitter")
    assert out.iloc[0] == pytest.approx(0.036)
    assert math.isnan(out.iloc[1])
