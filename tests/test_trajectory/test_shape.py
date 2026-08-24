from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.shape import (
    MIN_EFFECTIVE_ROWS,
    _bootstrap_predictions,
    _weighted_least_squares,
    build_history,
    prepare,
    shape_trajectory,
)


def _panel(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """(mlbam_id, season, age, sgp) rows."""
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])


def _linear_population(coef_current: float, coef_prior: float, n: int = 240) -> pd.DataFrame:
    """A population whose next season is EXACTLY intercept-free a*current + b*prior, so the
    fit has a known right answer to recover."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        prior = float(rng.uniform(5, 25))
        current = float(rng.uniform(5, 25))
        rows.append((i, 2010, 27, prior))
        rows.append((i, 2011, 28, current))
        rows.append((i, 2012, 29, coef_current * current + coef_prior * prior))
    return _panel(rows)


def test_build_history_censors_a_prior_before_the_panel_begins() -> None:
    # The 2010 season's prior is 2009, outside the panel: dropped, not scored as 0.
    frame = build_history(_panel([(1, 2010, 25, 13.0), (1, 2011, 26, 11.0)]))
    assert list(frame["season"]) == [2011]
    assert frame.iloc[0]["prior"] == pytest.approx(13.0)


def test_build_history_scores_a_missing_year_as_zero() -> None:
    # He was in the league in 2010 and out in 2011, so his 2012 prior is a real 0.
    frame = build_history(_panel([(1, 2010, 25, 13.0), (1, 2012, 27, 9.0)]))
    assert list(frame["season"]) == [2012]
    assert frame.iloc[0]["prior"] == pytest.approx(0.0)


def test_the_fit_recovers_a_known_relationship() -> None:
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    _, anchors = shape_trajectory(
        panel, kind="hitter", age=28, sgp=15.0, prior_sgp=15.0, horizons=(1,), prior_window=50.0
    )
    assert anchors[0].on_current == pytest.approx(0.4, abs=0.02)
    assert anchors[0].on_prior == pytest.approx(0.5, abs=0.02)
    assert anchors[0].intercept == pytest.approx(0.0, abs=0.3)


def test_the_prediction_uses_both_anchors() -> None:
    # Two players at the same current level and different peaks must not get the same
    # forecast -- that is the entire point of the mode.
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    low, _ = shape_trajectory(
        panel, kind="hitter", age=28, sgp=12.0, prior_sgp=8.0, horizons=(1,), prior_window=50.0
    )
    high, _ = shape_trajectory(
        panel, kind="hitter", age=28, sgp=12.0, prior_sgp=22.0, horizons=(1,), prior_window=50.0
    )
    assert high.path[0].mean > low.path[0].mean + 5


def test_a_nearby_age_still_contributes_instead_of_being_discarded() -> None:
    # Level matching requires an exact age. Here every fitting row is age 28 and the
    # query is 27: with a window it still fits, which is what recovers the cohort.
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    traj, anchors = shape_trajectory(
        panel, kind="hitter", age=27, sgp=15.0, prior_sgp=15.0, horizons=(1,), prior_window=50.0
    )
    assert anchors[0].n_fit > 0
    assert not np.isnan(traj.path[0].mean)


def test_age_weight_falls_off_with_distance() -> None:
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    near = shape_trajectory(
        panel, kind="hitter", age=28, sgp=15.0, prior_sgp=15.0, horizons=(1,), prior_window=50.0
    )[1][0]
    far = shape_trajectory(
        panel, kind="hitter", age=30, sgp=15.0, prior_sgp=15.0, horizons=(1,), prior_window=50.0
    )[1][0]
    assert far.n_effective < near.n_effective


def test_a_query_beyond_every_kernel_yields_no_fit() -> None:
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    traj, anchors = shape_trajectory(
        panel, kind="hitter", age=45, sgp=15.0, prior_sgp=15.0, horizons=(1,), prior_window=1.0
    )
    assert anchors[0].n_fit == 0
    assert np.isnan(traj.path[0].mean)
    assert traj.path[0].n == 0


def test_an_unobservable_horizon_is_reported_empty_not_fitted() -> None:
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    traj, anchors = shape_trajectory(
        panel,
        kind="hitter",
        age=28,
        sgp=15.0,
        prior_sgp=15.0,
        horizons=(1, 5),
        prior_window=50.0,
        last_complete_season=2012,
    )
    assert traj.path[0].n > 0
    assert traj.path[1].n == 0
    assert np.isnan(anchors[1].on_current)
    assert traj.total == pytest.approx(traj.path[0].mean)


def test_the_mode_is_labelled_so_render_cannot_confuse_it_with_comps() -> None:
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    traj, _ = shape_trajectory(
        panel, kind="hitter", age=28, sgp=15.0, prior_sgp=15.0, horizons=(1,), prior_window=50.0
    )
    assert traj.mode == "shape"
    assert traj.prior_sgp == pytest.approx(15.0)


def test_the_band_is_empirical_not_a_gaussian_multiple_of_spread() -> None:
    """`spread` is one number, so reading a band off it assumes the residuals are normal.
    Measured out of sample they are not: for pitchers at +3 the standardized error has
    kurtosis 2.48 -- FLATTER than a bell, mass pushed symmetrically onto both shoulders --
    so +/-1 spread holds 59% of players where a Gaussian reading promises 68%. `p10`/`p90`
    are read off the weighted residual distribution instead, so the band carries whatever
    shape and skew the comps actually had."""
    # NOISY on purpose: `_linear_population` is exactly linear, so its residuals are
    # zero-width and every quantile collapses onto the point estimate. A band only means
    # something where the comps scatter.
    rng = np.random.default_rng(1)
    rows = []
    for i in range(400):
        prior = float(rng.uniform(8.0, 22.0))
        current = float(rng.uniform(8.0, 22.0))
        forward = 0.4 * current + 0.5 * prior + float(rng.normal(0, 3.0))
        rows += [(i, 2010, 27, prior), (i, 2011, 28, current), (i, 2012, 29, forward)]
    traj, _ = shape_trajectory(
        _panel(rows),
        kind="hitter",
        age=28,
        sgp=15.0,
        prior_sgp=15.0,
        horizons=(1,),
        prior_window=50.0,
    )
    point = traj.path[0]
    assert point.p10 < point.median < point.p90
    assert point.p10 < point.mean < point.p90
    # On a symmetric normal population the empirical band should land near the Gaussian
    # one -- the point is that it is FREE to differ, not that it always does.
    assert (point.p90 - point.p10) == pytest.approx(2 * 1.2816 * point.spread, rel=0.15)


def test_the_band_always_contains_its_own_point_estimate() -> None:
    """The band is reweighted toward the query's current season while the point estimate
    comes off the fit weights, so the two were centred on different distributions -- and
    where the reweighted one sat below zero BOTH quantiles landed under `mean`, printing
    an interval that excluded the number it was drawn around. 135 of 4020 horizon-rows on
    the live board did this, mostly low-VAR veterans where the replacement floor compresses
    everything. The quantiles are now measured relative to the reweighted MEDIAN, so
    q10 <= q50 <= q90 makes the containment structural.

    A COHORT MOSTLY BELOW REPLACEMENT is what triggers it, which is why the real failures
    were low-VAR veterans. The panel itself piles a mass of careers at exactly 0 (the
    `max(forward, 0.0)` below), so against a floor of 8 most comps sit near the bottom of
    the shifted response and their residuals against the prediction are nearly all
    negative -- then even the 90th-percentile residual sits below zero, dragging `p90`
    under `mean`. (The pile used to come from the response clamp instead; #331 removed
    that, and the fixture now has to supply the skew itself, which it does.)
    A smoothly-curved or purely linear population does NOT reproduce it; two earlier
    fixtures passed against the broken code before this one was found."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(600):
        prior = float(rng.uniform(0.0, 14.0))
        current = float(rng.uniform(0.0, 14.0))
        forward = 0.45 * current + 0.2 * prior + float(rng.normal(0, 1.5))
        rows += [(i, 2010, 33, prior), (i, 2011, 34, current), (i, 2012, 35, max(forward, 0.0))]
    panel = _panel(rows)

    checked = 0
    for current in (3.0, 5.0, 7.0, 9.0, 11.0, 13.0):
        for prior in (2.0, 6.0, 10.0):
            # Floor of 8 against a cohort topping out near 8 -- most comps land at 0.
            traj, _ = shape_trajectory(
                panel,
                kind="hitter",
                age=34,
                sgp=current,
                prior_sgp=prior,
                horizons=(1,),
                replacement=8.0,
                slot="UTIL",
            )
            for point in traj.observable:
                checked += 1
                assert point.p10 <= point.mean <= point.p90, (
                    f"current={current} prior={prior}: "
                    f"mean {point.mean:.2f} outside {point.p10:.2f}..{point.p90:.2f}"
                )
    assert checked > 10, "swept too few fittable queries to mean anything"


def test_an_asymmetric_residual_distribution_gives_an_asymmetric_band() -> None:
    """The whole reason for reading quantiles rather than scaling `spread`: a population
    that mostly holds steady and occasionally collapses has a long LEFT tail, and a
    symmetric band would overstate the upside and understate the downside by the same
    wrong amount."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        prior = float(rng.uniform(12.0, 18.0))
        current = float(rng.uniform(12.0, 18.0))
        # 85% land near 15; 15% collapse to nearly nothing.
        forward = 1.0 if rng.random() < 0.15 else 15.0 + float(rng.normal(0, 0.5))
        rows += [(i, 2010, 27, prior), (i, 2011, 28, current), (i, 2012, 29, forward)]
    traj, _ = shape_trajectory(
        _panel(rows),
        kind="hitter",
        age=28,
        sgp=15.0,
        prior_sgp=15.0,
        horizons=(1,),
        prior_window=50.0,
    )
    point = traj.path[0]
    below = point.median - point.p10
    above = point.p90 - point.median
    assert below > 3 * above, f"expected a long left tail, got -{below:.2f} / +{above:.2f}"


def test_the_band_fallback_is_recorded_on_the_horizon_it_happened_at() -> None:
    """A fallback at +2 must not mark +1 unreliable.

    One sweep at the longest horizon serves every shorter range (#321), so a board for
    2027 alone reads its points off a fit that also went out to 2031. While the flag was
    a single latched bool on the trajectory, that board inherited a fallback from a year
    it was not showing -- and the flag is the one telling the reader the band is the
    trustworthy part.

    The panel puts a cohort at the query's own level ONLY in the last season that has a
    +1 outcome, so the band has real weight near the query at +1 and none at +2.
    """
    rng = np.random.default_rng(0)
    rows = []
    # Carries the fit but no band weight: 15 SGP from the query, outside CURRENT_WINDOW.
    for i in range(120):
        level = float(rng.uniform(8.0, 12.0))
        for offset, season in enumerate(range(2010, 2016)):
            rows.append((i, season, 25 + offset, max(level + float(rng.normal(0, 1.5)), 0.0)))
    for j in range(30):
        rows.append((900 + j, 2014, 28, 25.0 + float(rng.normal(0, 1.0))))
        rows.append((900 + j, 2015, 29, 25.0 + float(rng.normal(0, 1.0))))

    traj, _ = shape_trajectory(
        _panel(rows),
        kind="hitter",
        age=28,
        sgp=25.0,
        prior_sgp=24.0,
        horizons=(1, 2),
        prior_window=50.0,
    )

    near, far = traj.path
    # Both fitted, so neither assertion below passes on an empty point.
    assert near.n_effective > MIN_EFFECTIVE_ROWS and far.n_effective > MIN_EFFECTIVE_ROWS
    assert not near.band_fell_back
    assert far.band_fell_back
    assert traj.band_fell_back == any(p.band_fell_back for p in traj.path)


def test_the_bootstrap_is_reproducible() -> None:
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    kw = {"kind": "hitter", "age": 28, "sgp": 15.0, "prior_sgp": 15.0, "horizons": (1,)}
    first = shape_trajectory(panel, **kw)[0].path[0].se
    assert first == shape_trajectory(panel, **kw)[0].path[0].se


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prior_window": 0.0}, "prior_window"),
        ({"age_window": 0}, "age_window"),
        ({"horizons": ()}, "horizons"),
        # `std(ddof=1)` on fewer than two draws is a NaN and a RuntimeWarning, which
        # reaches the caller as a silently missing SE rather than a refused argument.
        ({"bootstrap_draws": 0}, "bootstrap_draws"),
        ({"bootstrap_draws": 1}, "bootstrap_draws"),
    ],
)
def test_rejects_impossible_settings(kwargs: dict, match: str) -> None:
    panel = _linear_population(coef_current=0.4, coef_prior=0.5)
    with pytest.raises(ValueError, match=match):
        shape_trajectory(panel, kind="hitter", age=28, sgp=15.0, prior_sgp=15.0, **kwargs)


# --- the batch entry point (#311) ---------------------------------------------------
#
# The whole contract is "same answer, less work", so these assert EQUALITY against the
# panel path rather than checking the batch path is self-consistent. A batch API that
# quietly returns different numbers is worse than a slow one.


def _mixed_panel() -> pd.DataFrame:
    """A multi-season population plus the awkward cases -- a split season, a gap year,
    and a prior that predates the panel -- so the batch path has to reproduce the
    censoring rules and not just the arithmetic.

    Six seasons deep on purpose: on a panel that only supports horizon 1, every
    horizon-2 assertion below would pass on a pair of NaNs.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        level = float(rng.uniform(6.0, 24.0))
        for offset, season in enumerate(range(2010, 2016)):
            rows.append((i, season, 25 + offset, max(level + float(rng.normal(0, 2.5)), 0.0)))
    rows += [
        (900, 2010, 27, 8.0),
        (900, 2011, 28, 6.0),  # split season: two rows for one player-year
        (900, 2011, 28, 5.0),
        (900, 2012, 29, 14.0),
        (900, 2013, 30, 11.0),
        (901, 2010, 26, 12.0),
        (901, 2012, 28, 9.0),  # gap year: his 2011 is a real 0, not a censored unknown
        (901, 2013, 29, 10.0),
    ]
    return _panel(rows)


@pytest.mark.parametrize("horizons", [(1,), (1, 2), (2,)])
def test_prepared_state_gives_the_same_answer_as_the_panel(horizons: tuple[int, ...]) -> None:
    panel = _mixed_panel()
    kw = {"kind": "hitter", "age": 28, "sgp": 15.0, "prior_sgp": 15.0, "prior_window": 50.0}
    direct, direct_anchors = shape_trajectory(panel, horizons=horizons, **kw)
    prepared, prepared_anchors = shape_trajectory(
        prepare(panel, kind="hitter", horizons=(1, 2)), horizons=horizons, **kw
    )

    # Nothing below may pass by comparing NaN to NaN.
    assert all(np.isfinite(p.mean) and np.isfinite(p.se) for p in direct.path)

    assert (prepared.n_comps, prepared.seasons) == (direct.n_comps, direct.seasons)
    assert prepared.mean_start == direct.mean_start
    assert prepared.mean_prior == direct.mean_prior
    assert prepared_anchors == direct_anchors
    for got, want in zip(prepared.path, direct.path, strict=True):
        # Exact, not approx: the batch path reorders no arithmetic, it only stops
        # redoing it. `se` included -- same rng, same draws, same order.
        assert got == want


def test_prepared_state_refuses_a_query_from_the_other_pool() -> None:
    """`kind` is otherwise a pure label -- it lands on `Trajectory.kind` for `render` and
    is never checked against the panel. That was safe while every caller loaded the panel
    and named the pool in one expression, but the whole point of `prepare` is hoisting the
    panel out of the loop, and a board is mixed hitters and pitchers. One `prepare` above
    that loop would fit every pitcher on hitter seasons and print it under
    `kind='pitcher'` with a plausible `n_comps` and no warning."""
    prepared = prepare(_mixed_panel(), kind="hitter", horizons=(1,))
    with pytest.raises(ValueError, match="pitcher"):
        shape_trajectory(prepared, kind="pitcher", age=28, sgp=15.0, prior_sgp=15.0, horizons=(1,))


def test_a_repeated_horizon_is_not_fitted_twice() -> None:
    """`prepare` normalizes with `sorted(set(...))` and `shape_trajectory` used to sort
    without deduping, so `(1, 1, 2)` passed the prepared-state check on set arithmetic and
    then fitted h1 twice -- two identical `PathPoint`s and `Anchors`, the bootstrap run
    twice, and h1 counted twice in the `total` the caller reads."""
    panel = _mixed_panel()
    kw = {"kind": "hitter", "age": 28, "sgp": 15.0, "prior_sgp": 15.0, "prior_window": 50.0}
    once, once_anchors = shape_trajectory(panel, horizons=(1, 2), **kw)
    twice, twice_anchors = shape_trajectory(panel, horizons=(1, 1, 2), **kw)

    assert [p.horizon for p in twice.path] == [1, 2]
    assert twice_anchors == once_anchors
    assert twice.total == pytest.approx(once.total)


def test_a_prepared_state_can_be_cached_on() -> None:
    """A frozen dataclass over ndarrays derives an `__eq__` that raises on the ambiguous
    truth value of an array and a `__hash__` that raises on unhashable ndarrays -- both
    on the natural use, an `lru_cache`d scoring helper keyed by the prepared state."""
    prepared = prepare(_mixed_panel(), kind="hitter", horizons=(1,))
    assert hash(prepared) == hash(prepared)
    assert prepared == prepared
    assert prepared != prepare(_mixed_panel(), kind="hitter", horizons=(1,))
    assert len({prepared, prepared}) == 1


def test_prepared_state_refuses_a_horizon_it_has_no_forward_values_for() -> None:
    """Silently returning an empty path here would read as "no comps for this player"."""
    panel = _mixed_panel()
    with pytest.raises(ValueError, match="horizons"):
        shape_trajectory(
            prepare(panel, kind="hitter", horizons=(1,)),
            kind="hitter",
            age=28,
            sgp=15.0,
            prior_sgp=15.0,
            horizons=(2,),
        )


def test_a_prepared_state_honours_a_lower_cutoff_without_a_rebuild() -> None:
    """`prepare` never uses `last` -- it carries it, builds `forward` for every row, and
    leaves all censoring to the query. So an as-of-season sweep can reuse one state across
    cutoffs instead of re-running `build_history` and a full reindex per cutoff, which is
    the exact work `prepare` exists to hoist."""
    panel = _mixed_panel()
    kw = {"kind": "hitter", "age": 28, "sgp": 15.0, "prior_sgp": 15.0, "prior_window": 50.0}
    prepared = prepare(panel, kind="hitter", horizons=(1,))

    reused, _ = shape_trajectory(prepared, horizons=(1,), last_complete_season=2012, **kw)
    rebuilt, _ = shape_trajectory(panel, horizons=(1,), last_complete_season=2012, **kw)

    assert reused.n_comps == rebuilt.n_comps
    assert reused.path[0] == rebuilt.path[0]
    # And the cutoff genuinely bit, rather than both silently using the panel maximum.
    assert reused.n_comps < shape_trajectory(prepared, horizons=(1,), **kw)[0].n_comps


def test_prepared_state_refuses_a_cutoff_past_what_it_was_built_for() -> None:
    """The unsafe direction. `forward` was looked up against the panel, so a season past
    `prepared.last` came back missing and was recorded as the 0 that means "out of the
    league" -- raising the cutoff would reinterpret "not yet played" as "did not play"."""
    panel = _mixed_panel()
    with pytest.raises(ValueError, match="last_complete_season"):
        shape_trajectory(
            prepare(panel, kind="hitter", horizons=(1,), last_complete_season=2012),
            kind="hitter",
            age=28,
            sgp=15.0,
            prior_sgp=15.0,
            horizons=(1,),
            last_complete_season=2014,
        )


def test_the_batched_bootstrap_matches_a_refit_per_draw() -> None:
    """The vectorized bootstrap must draw the same rows in the same order as the loop it
    replaced -- only the solver changed, so the two agree to floating-point noise rather
    than to bootstrap noise."""
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(10.0, 4.0, (n, 2))
    y = rng.normal(9.0, 5.0, n)
    w = rng.uniform(0.01, 1.0, n)
    query = np.array([1.0, 12.0, 18.0])

    # ONE generator across all draws, exactly as the old loop consumed it.
    reference_rng = np.random.default_rng(5)
    loop = np.empty(300)
    for i in range(300):
        pick = reference_rng.integers(0, n, n)
        loop[i] = query @ _weighted_least_squares(x[pick], y[pick], w[pick])

    batched = _bootstrap_predictions(x, y, w, query, np.random.default_rng(5), 300)
    assert batched == pytest.approx(loop, rel=1e-9)


def test_the_bootstrap_answer_does_not_depend_on_the_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch is a vectorization width, and the memory budget narrows it on a wide
    fit. Indices are drawn in the same order at any width, so a fit that happens to be
    memory-bound must not get a different SE from one that is not."""
    rng = np.random.default_rng(3)
    n = 150
    x = rng.normal(10.0, 4.0, (n, 2))
    y = rng.normal(9.0, 5.0, n)
    w = rng.uniform(0.01, 1.0, n)
    query = np.array([1.0, 12.0, 18.0])

    wide = _bootstrap_predictions(x, y, w, query, np.random.default_rng(9), 400)
    # Small enough to force a batch of 1 -- the pathological end of the budget.
    monkeypatch.setattr("fantasy_baseball.trajectory.shape.BOOTSTRAP_BYTES", 1)
    narrow = _bootstrap_predictions(x, y, w, query, np.random.default_rng(9), 400)
    assert narrow == pytest.approx(wide, rel=1e-12)


def test_the_batched_bootstrap_survives_a_rank_deficient_draw() -> None:
    """Two anchors can be collinear -- every comp's current season exactly half his prior,
    say -- and then three parameters are not identified. `lstsq` answered that with the
    least-norm solution and a plain `solve` raises `LinAlgError` on it, so the batched form
    has to keep the old behaviour rather than fail the query."""
    rng = np.random.default_rng(2)
    n = 60
    current = rng.uniform(5.0, 25.0, n)
    x = np.column_stack([current, 2.0 * current])  # prior is exactly 2 * current: rank 2
    y = rng.normal(9.0, 5.0, n)
    w = rng.uniform(0.2, 1.0, n)
    query = np.array([1.0, 12.0, 24.0])

    batched = _bootstrap_predictions(x, y, w, query, np.random.default_rng(4), 40)
    assert np.isfinite(batched).all()

    reference_rng = np.random.default_rng(4)
    loop = np.empty(40)
    for i in range(40):
        pick = reference_rng.integers(0, n, n)
        loop[i] = query @ _weighted_least_squares(x[pick], y[pick], w[pick])
    assert batched == pytest.approx(loop, rel=1e-6)


# --------------------------------------------------------------------------
# `Prepared.back` -- the backward career window (#358)
# --------------------------------------------------------------------------


def _back_panel() -> pd.DataFrame:
    """One player with a HOLE in his career, which is the whole point of these tests.

    Ages 24-28 with age 26 never played. `build_history` drops the first season (no
    observable prior), so the anchor rows are ages 25, 27 and 28.
    """
    return pd.DataFrame(
        {
            "mlbam_id": [1, 1, 1, 1],
            "season": [2010, 2011, 2013, 2014],
            "age": [24, 25, 27, 28],
            "sgp": [5.0, 6.0, 8.0, 9.0],
        }
    )


def test_prepare_builds_no_backward_window_unless_asked() -> None:
    """OPT-IN. `back` is read by one consumer (`career_comps`, via the push script) while
    `prepare` is called by `sweep_pool`, the CLI, and `tune_shape_windows` inside its
    tuning loop. Building it by default charged all of them N full-history reindexes for
    state they never touch."""
    prepared = prepare(_back_panel(), kind="hitter", horizons=(1,))
    assert prepared.back == {} and prepared.lookback == 0


def test_a_season_he_did_not_play_is_nan_in_back_and_zero_in_forward() -> None:
    """THE ASYMMETRY, asserted against `prepare` rather than against a hand-built
    fixture. Both matchers read the same panel through the same reindex, and the ONLY
    thing separating them is the `nan_to_num` on one side -- so nothing but this test
    fails if `back` is ever "tidied up" to match `forward`.

    Scoring an unplayed year as 0 on the backward side is exactly what made an injured
    star match a replacement-level journeyman (#357, #358).
    """
    prepared = prepare(_back_panel(), kind="hitter", horizons=(1,), lookback=3)
    at_28 = int(np.flatnonzero(prepared.age == 28.0)[0])

    # Age 28 looking back: 27 played (8.0), 26 did NOT, 28 is himself.
    assert prepared.back[0][at_28] == 9.0
    assert prepared.back[1][at_28] == 8.0
    assert np.isnan(prepared.back[2][at_28]), "age 26 was never played -- absent, not 0.0"

    # The same hole on the FORWARD side is a real 0.0, deliberately: absence there is
    # the outcome, and filling it is what keeps a comp set from being all survivors.
    at_25 = int(np.flatnonzero(prepared.age == 25.0)[0])
    assert prepared.forward[1][at_25] == 0.0, "age 26 forward from 25 -- a real zero"


def test_back_offsets_count_backwards_not_forwards() -> None:
    """A sign flip here is silent: every RMSE still computes, and the comps returned are
    matched on the years AFTER the anchor instead of the years before -- which is the
    forward matcher this replaced, wearing the new name."""
    prepared = prepare(_back_panel(), kind="hitter", horizons=(1,), lookback=2)
    at_27 = int(np.flatnonzero(prepared.age == 27.0)[0])
    assert prepared.back[1][at_27] != 9.0, "offset 1 is not age 28"
    assert np.isnan(prepared.back[1][at_27]), "offset 1 from age 27 is age 26, unplayed"


def test_back_offset_zero_is_the_rows_own_season() -> None:
    """`back[0]` is `current` aliased, not a second reindex of the same key. Asserting
    the VALUES rather than identity so the aliasing stays an implementation choice."""
    prepared = prepare(_back_panel(), kind="hitter", horizons=(1,), lookback=4)
    assert np.array_equal(prepared.back[0], prepared.current)


def test_a_negative_lookback_is_refused() -> None:
    with pytest.raises(ValueError, match="lookback must not be negative"):
        prepare(_back_panel(), kind="hitter", horizons=(1,), lookback=-1)
