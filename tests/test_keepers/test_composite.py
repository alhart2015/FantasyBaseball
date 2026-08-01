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
    speed,
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
    out = composite(
        _fam([1.0], [0.0], [0.0], [0.0]),
        "hitter",
        weights=(2.0, 0.0, 0.0, 0.0),
        family_order=("skill", "luck", "future", "age"),
    )
    assert out.iloc[0] == pytest.approx(1.0)


def test_composite_rejects_an_unknown_family():
    with pytest.raises(KeyError, match="peripherals"):
        composite({"peripherals": pd.Series([0.5])}, "hitter")


def test_composite_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="no weighted family"):
        composite(
            _fam([0.5], [0.0], [0.0], [0.0]),
            "hitter",
            weights=(0, 0, 0, 0),
            family_order=("skill", "luck", "future", "age"),
        )


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


def test_an_all_nan_family_drops_out_like_a_missing_one():
    """If a family is all-NaN (e.g. xba unavailable, so batted_ball is all-NaN), it
    must drop from the denominator like a missing family, not poison the whole blend
    to NaN via `fillna(series.mean())` where the mean is itself NaN."""
    supplied = {family: pd.Series([0.8, 0.8]) for family in FAMILIES["hitter"]}
    supplied["batted_ball"] = pd.Series([float("nan"), float("nan")])
    out = composite(supplied, "hitter")
    assert not out.isna().any()
    assert out.iloc[0] == pytest.approx(0.8)


def test_luck_is_value_minus_skill():
    assert luck(pd.Series([0.95, 0.40]), pd.Series([0.55, 0.80])).tolist() == pytest.approx(
        [0.40, -0.40]
    )


def test_no_residual_family_ships():
    """#288 removed `luck` because it was a RESIDUAL, not a measurement: its largest
    component was playing time (+0.66) and only its third was actual batted-ball luck
    (+0.25), so its famous positive weight was paying for durability and steals under a
    false name. Those have their own families now. Reintroducing a
    whatever-is-left-over term would silently start paying for lineup context again --
    R/RBI rate, a team property that does not travel with a traded player."""
    for kind, families in FAMILIES.items():
        assert "luck" not in families, kind


def test_luck_survives_as_a_diagnostic_even_though_it_does_not_ship():
    """`--backtest` bakes the residual parameterizations off against the shipped one and
    `--study` prints what the residual was made of. Deleting the function would make the
    composite docstring's central argument unreproducible, which is how it drifts."""
    value = pd.Series([0.9, 0.5, 0.2])
    skill = pd.Series([0.5, 0.5, 0.5])
    assert luck(value, skill).tolist() == pytest.approx([0.4, 0.0, -0.3])


def test_batted_ball_carries_a_negative_weight_for_every_position():
    """The entire point of the family: it claws the batted-ball half back out of the
    positive `luck` weight. A regression to a positive or zero weight -- a copy-paste
    from the luck slot, or an 'all weights should be positive' cleanup -- would revert
    #277's demotion of the everyday-plus-lucky bats with every other test still green,
    since future>0 and the family/weight alignment all continue to hold. The hitter
    weight is a deliberate override of the grid, which zeroes this family once
    `durability` absorbs the volume signal; see the composite docstring for the cost
    (0.0092 rho, inside the noise band) and the reason."""
    for kind, weights in FITTED_WEIGHTS.items():
        assert weights[FAMILIES[kind].index("batted_ball")] < 0, kind


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


def test_fitted_weights_align_with_their_families_and_pin_skill_at_one():
    """`skill` is the unit every other family is measured in -- `_FAMILY_GRID` pins it
    at 1.0 and grid-searches the rest against it -- so a value other than 1.0 means the
    constants drifted from the fit that produced them.

    It is deliberately NOT asserted to be the LARGEST weight any more. Hitter
    `durability` fits above it (1.2), which is #288's central finding rather than a
    bug: playing time predicts next-year playing time at 0.607 against skill's 0.464,
    so availability repeats better than talent does. Capping durability at 1.0 to keep
    skill on top was measured and costs 0.0012 rho -- unmeasurable against a 0.0126
    noise band -- so this asserts what the grid actually guarantees instead of an
    aesthetic the data does not support."""
    for kind, weights in FITTED_WEIGHTS.items():
        assert len(weights) == len(FAMILIES[kind]), kind
        assert weights[FAMILIES[kind].index("skill")] == 1.0, kind
        assert weights[FAMILIES[kind].index("future")] > 0, kind


def test_composite_honors_an_explicit_family_order():
    fams = {"skill": pd.Series([1.0, 0.0]), "pt": pd.Series([0.0, 1.0])}
    out = composite(fams, "hitter", weights=(1.0, 1.0), family_order=("skill", "pt"))
    assert out.tolist() == pytest.approx([0.5, 0.5])


def test_composite_rejects_a_family_outside_the_known_universe():
    with pytest.raises(KeyError, match="peripherals"):
        composite({"peripherals": pd.Series([0.5])}, "hitter", family_order=("skill",))


def test_composite_rejects_an_unknown_name_in_family_order():
    """The 'typo not a silent no-op' promise covers `family_order`, not only the
    families dict: an unknown name there would `.get()` to None and silently drop its
    weighted slot, shifting every downstream rank with nothing raised."""
    with pytest.raises(KeyError, match="sklll"):
        composite(
            {"skill": pd.Series([0.5])},
            "hitter",
            weights=(1.0, 1.0),
            family_order=("skill", "sklll"),
        )


def test_composite_blends_an_empty_pool_to_an_empty_result():
    """An empty pool (0 rows) has nothing to fail on and must blend to an empty result,
    not raise: callers build intentional empty sub-pools (an early-season board, a
    `--study` truncation) and skip an empty return. #277's all-NaN drop would otherwise
    drop every vacuously-all-NaN family and hit the 'no weighted family' guard."""
    empty = {family: pd.Series([], dtype=float) for family in FAMILIES["hitter"]}
    assert composite(empty, "hitter").empty
    assert composite(empty, "hitter", strict=True).empty  # strict callers skip empties too


def test_strict_raises_on_an_all_nan_family_instead_of_dropping_it():
    """`strict` is the fit/backtest's guard: their blended number is persisted as
    constants or decides the shipped model, so a family silently dropped for missing
    data must fail loud -- while the live board (strict=False) keeps dropping it so an
    xba or ZiPS outage cannot take the board down."""
    supplied = {family: pd.Series([0.8, 0.8]) for family in FAMILIES["hitter"]}
    supplied["batted_ball"] = pd.Series([float("nan"), float("nan")])
    assert not composite(supplied, "hitter").isna().any()  # default: drops, stays valid
    with pytest.raises(ValueError, match="batted_ball"):
        composite(supplied, "hitter", strict=True)


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


def test_speed_is_per_opportunity_not_a_raw_total():
    """The family exists because the peripherals carry no speed input. It has to be a
    RATE: a full-time player with the same steal rate as a part-timer is not faster,
    and a raw total would just re-import the playing-time signal `durability` owns."""
    frame = pd.DataFrame({"sb_sgp": [10.0, 5.0], "pt": [600.0, 300.0]})
    assert speed(frame).tolist() == pytest.approx([10 / 600, 5 / 300])


def test_speed_returns_nan_for_zero_playing_time_rather_than_inf():
    """An inf would survive `percentile` and pin that player to the top of the family,
    where NaN is mean-filled to neutral. A 0-PT row is undefined speed, not elite."""
    frame = pd.DataFrame({"sb_sgp": [1.0, 0.0], "pt": [500.0, 0.0]})
    out = speed(frame)
    assert math.isnan(out.iloc[1])
    assert not math.isinf(out.iloc[1])


def test_speed_names_the_column_it_is_missing():
    with pytest.raises(KeyError, match="sb_sgp"):
        speed(pd.DataFrame({"pt": [500.0]}))
