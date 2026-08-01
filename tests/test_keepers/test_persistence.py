from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.keepers.persistence import (
    HITTER_COUNTING,
    PITCHER_COUNTING,
    Share,
    apply_share,
    evaluate_shares,
    fit_counting_share,
    fit_share,
    gap,
    rmse,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_gap_is_observed_minus_projected() -> None:
    assert list(gap(_series([5.0, 3.0]), _series([4.0, 4.0]))) == [1.0, -1.0]


def test_gap_propagates_nan_from_either_side() -> None:
    out = gap(_series([5.0, np.nan]), _series([np.nan, 4.0]))
    assert out.isna().all()


def test_fit_share_recovers_a_planted_slope_and_intercept() -> None:
    # gap_next = 0.25 + 0.40 * gap_now, exactly.
    now = _series([-3.0, -1.0, 0.0, 2.0, 4.0, 7.0])
    nxt = 0.25 + 0.40 * now
    fit = fit_share(now, nxt, column="hr_pa")
    assert fit.share == pytest.approx(0.40)
    assert fit.intercept == pytest.approx(0.25)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.n == 6


def test_fit_share_weights_shift_the_slope_toward_the_heavy_rows() -> None:
    """A weight is not cosmetic: it must actually move the estimate.

    Two clean sub-populations with different slopes. Weighting almost entirely onto
    the steep one has to pull the pooled slope toward it, which is the whole reason
    the caller passes playing time.
    """
    now = _series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    nxt = _series([0.2, 0.4, 0.6, 0.9, 1.8, 2.7])  # slopes 0.2 and 0.9
    flat = fit_share(now, nxt, column="c").share
    steep = fit_share(now, nxt, column="c", weights=_series([1, 1, 1, 100, 100, 100])).share
    assert steep > flat
    assert steep == pytest.approx(0.9, abs=0.05)


def test_fit_share_drops_pairs_with_nan_or_nonpositive_weight() -> None:
    now = _series([1.0, 2.0, 3.0, np.nan, 5.0])
    nxt = _series([0.5, 1.0, 1.5, 2.0, np.nan])
    fit = fit_share(now, nxt, column="c", weights=_series([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert fit.n == 3
    zeroed = fit_share(
        _series([1.0, 2.0, 3.0, 9.0]),
        _series([0.5, 1.0, 1.5, 99.0]),
        column="c",
        weights=_series([1.0, 1.0, 1.0, 0.0]),
    )
    # The zero-weight outlier must not bend the line at all.
    assert zeroed.n == 3
    assert zeroed.share == pytest.approx(0.5)


def test_fit_share_treats_infinities_as_missing() -> None:
    # safe_ratio yields NaN, but a caller-built gap can still produce inf; it must not
    # poison the fit into nan.
    fit = fit_share(
        _series([1.0, 2.0, 3.0, np.inf]),
        _series([0.5, 1.0, 1.5, 4.0]),
        column="c",
    )
    assert fit.n == 3
    assert fit.share == pytest.approx(0.5)


def test_fit_share_on_a_degenerate_regressor_reports_no_signal() -> None:
    """Every gap identical: there is no slope to estimate, and dividing by a zero
    spread would emit nan/inf that reads downstream as a real fit."""
    fit = fit_share(_series([2.0, 2.0, 2.0, 2.0]), _series([1.0, 3.0, 2.0, 2.0]), column="c")
    assert fit.share == 0.0
    assert fit.stderr == float("inf")
    assert not fit.separable_from_zero
    assert fit.intercept == pytest.approx(2.0)


def test_fit_share_raises_below_three_pairs() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_share(_series([1.0, 2.0]), _series([1.0, 2.0]), column="hr_pa")


def test_separable_from_zero_is_a_two_stderr_test() -> None:
    assert Share("c", share=0.30, intercept=0.0, n=99, r2=0.1, stderr=0.10).separable_from_zero
    assert not Share("c", share=0.15, intercept=0.0, n=99, r2=0.1, stderr=0.10).separable_from_zero


def test_apply_share_is_projection_plus_drift_plus_shared_gap() -> None:
    fit = Share("c", share=0.5, intercept=0.1, n=9, r2=0.2, stderr=0.01)
    out = apply_share(_series([10.0, 20.0]), _series([4.0, -6.0]), fit)
    assert list(out) == pytest.approx([10.0 + 0.1 + 2.0, 20.0 + 0.1 - 3.0])


def test_apply_share_passes_projection_through_where_the_gap_is_unobserved() -> None:
    """An unobserved player is not evidence of a zero gap -- but he must still come out
    as a usable number, namely the projection plus the population drift."""
    fit = Share("c", share=0.5, intercept=0.1, n=9, r2=0.2, stderr=0.01)
    out = apply_share(_series([10.0, 20.0]), _series([np.nan, 4.0]), fit)
    assert out.iloc[0] == pytest.approx(10.1)
    assert out.iloc[1] == pytest.approx(22.1)


def test_rmse_is_weighted_and_ignores_missing_rows() -> None:
    assert rmse(_series([1.0, 2.0]), _series([2.0, 4.0])) == pytest.approx(np.sqrt(2.5))
    assert rmse(_series([1.0, np.nan]), _series([2.0, 99.0])) == pytest.approx(1.0)
    # Weighting onto the exact row drives the error to zero.
    weighted = rmse(_series([1.0, 5.0]), _series([1.0, 9.0]), _series([1.0, 0.0]))
    assert weighted == pytest.approx(0.0)


def test_rmse_of_an_empty_overlap_is_nan_not_zero() -> None:
    assert np.isnan(rmse(_series([np.nan]), _series([1.0])))


def test_evaluate_shares_brackets_the_fit_with_both_endpoints() -> None:
    projected = _series([1.0, 2.0, 3.0, 4.0])
    gap_now = _series([0.4, -0.2, 0.6, -0.8])
    fit = Share("c", share=0.5, intercept=0.0, n=4, r2=0.3, stderr=0.05)
    truth = projected + 0.5 * gap_now  # the fitted share is exactly right here

    scores = evaluate_shares(projected, gap_now, truth, fit)
    assert set(scores) == {"s0", "s1", "fitted"}
    assert scores["fitted"] == pytest.approx(0.0)
    assert scores["s0"] > 0 and scores["s1"] > 0


def test_evaluate_shares_endpoints_ignore_the_fitted_share() -> None:
    """s0/s1 must be the true endpoints -- if they leaked `fit.share` the comparison
    would be against the fit rather than against 0 and 1."""
    projected = _series([1.0, 2.0, 3.0])
    gap_now = _series([1.0, 1.0, 1.0])
    fit = Share("c", share=0.7, intercept=0.0, n=3, r2=0.1, stderr=0.01)
    scores = evaluate_shares(projected, gap_now, projected, fit)
    assert scores["s0"] == pytest.approx(0.0)  # truth == projection, so s=0 is perfect
    assert scores["s1"] == pytest.approx(1.0)  # s=1 adds the whole unit gap


def test_fit_counting_share_measures_the_same_gap_on_raw_counts() -> None:
    proj = _series([20.0, 30.0, 40.0, 50.0])
    obs = _series([26.0, 30.0, 34.0, 62.0])
    nxt = proj + 0.5 * (obs - proj)
    fit = fit_counting_share(proj, obs, nxt, column="HR")
    assert fit.share == pytest.approx(0.5)
    assert fit.column == "HR"


def test_counting_maps_cover_the_five_by_five_categories() -> None:
    # Guards against a rename silently dropping a scored category from the report.
    assert set(HITTER_COUNTING) == {"R", "HR", "RBI", "SB"}
    assert set(PITCHER_COUNTING) == {"W", "SV", "SO"}
    assert set(HITTER_COUNTING.values()) == {"r_pa", "hr_pa", "rbi_pa", "sb_pa"}
    assert set(PITCHER_COUNTING.values()) == {"w_ip", "sv_ip", "k_ip"}
