from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.shape import (
    _bootstrap_predictions,
    _weighted_least_squares,
    build_history,
    prepare,
    shape_trajectory,
)


def _panel(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """(mlbam_id, season, age, sgp) rows."""
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])


def _linear_population(coef_down: float, coef_peak: float, n: int = 240) -> pd.DataFrame:
    """A population whose next season is EXACTLY intercept-free a*down + b*peak, so the
    fit has a known right answer to recover."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        peak = float(rng.uniform(5, 25))
        down = float(rng.uniform(5, 25))
        rows.append((i, 2010, 27, peak))
        rows.append((i, 2011, 28, down))
        rows.append((i, 2012, 29, coef_down * down + coef_peak * peak))
    return _panel(rows)


def test_build_history_censors_a_prior_before_the_panel_begins() -> None:
    # The 2010 season's prior is 2009, outside the panel: dropped, not scored as 0.
    frame = build_history(_panel([(1, 2010, 25, 13.0), (1, 2011, 26, 11.0)]))
    assert list(frame["season"]) == [2011]
    assert frame.iloc[0]["peak"] == pytest.approx(13.0)


def test_build_history_scores_a_missing_year_as_zero() -> None:
    # He was in the league in 2010 and out in 2011, so his 2012 prior is a real 0.
    frame = build_history(_panel([(1, 2010, 25, 13.0), (1, 2012, 27, 9.0)]))
    assert list(frame["season"]) == [2012]
    assert frame.iloc[0]["peak"] == pytest.approx(0.0)


def test_the_fit_recovers_a_known_relationship() -> None:
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    _, anchors = shape_trajectory(
        panel, kind="hitter", age=28, sgp=15.0, peak=15.0, horizons=(1,), peak_band=50.0
    )
    assert anchors[0].on_down == pytest.approx(0.4, abs=0.02)
    assert anchors[0].on_peak == pytest.approx(0.5, abs=0.02)
    assert anchors[0].intercept == pytest.approx(0.0, abs=0.3)


def test_the_prediction_uses_both_anchors() -> None:
    # Two players at the same current level and different peaks must not get the same
    # forecast -- that is the entire point of the mode.
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    low, _ = shape_trajectory(
        panel, kind="hitter", age=28, sgp=12.0, peak=8.0, horizons=(1,), peak_band=50.0
    )
    high, _ = shape_trajectory(
        panel, kind="hitter", age=28, sgp=12.0, peak=22.0, horizons=(1,), peak_band=50.0
    )
    assert high.path[0].mean > low.path[0].mean + 5


def test_a_nearby_age_still_contributes_instead_of_being_discarded() -> None:
    # Level matching requires an exact age. Here every fitting row is age 28 and the
    # query is 27: with a window it still fits, which is what recovers the cohort.
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    traj, anchors = shape_trajectory(
        panel, kind="hitter", age=27, sgp=15.0, peak=15.0, horizons=(1,), peak_band=50.0
    )
    assert anchors[0].n_fit > 0
    assert not np.isnan(traj.path[0].mean)


def test_age_weight_falls_off_with_distance() -> None:
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    near = shape_trajectory(
        panel, kind="hitter", age=28, sgp=15.0, peak=15.0, horizons=(1,), peak_band=50.0
    )[1][0]
    far = shape_trajectory(
        panel, kind="hitter", age=30, sgp=15.0, peak=15.0, horizons=(1,), peak_band=50.0
    )[1][0]
    assert far.n_effective < near.n_effective


def test_a_query_beyond_every_kernel_yields_no_fit() -> None:
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    traj, anchors = shape_trajectory(
        panel, kind="hitter", age=45, sgp=15.0, peak=15.0, horizons=(1,), peak_band=1.0
    )
    assert anchors[0].n_fit == 0
    assert np.isnan(traj.path[0].mean)
    assert traj.path[0].n == 0


def test_an_unobservable_horizon_is_reported_empty_not_fitted() -> None:
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    traj, anchors = shape_trajectory(
        panel,
        kind="hitter",
        age=28,
        sgp=15.0,
        peak=15.0,
        horizons=(1, 5),
        peak_band=50.0,
        last_complete_season=2012,
    )
    assert traj.path[0].n > 0
    assert traj.path[1].n == 0
    assert np.isnan(anchors[1].on_down)
    assert traj.total == pytest.approx(traj.path[0].mean)


def test_the_mode_is_labelled_so_render_cannot_confuse_it_with_comps() -> None:
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    traj, _ = shape_trajectory(
        panel, kind="hitter", age=28, sgp=15.0, peak=15.0, horizons=(1,), peak_band=50.0
    )
    assert traj.mode == "shape"
    assert traj.prior_sgp == pytest.approx(15.0)


def test_the_bootstrap_is_reproducible() -> None:
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    kw = {"kind": "hitter", "age": 28, "sgp": 15.0, "peak": 15.0, "horizons": (1,)}
    first = shape_trajectory(panel, **kw)[0].path[0].se
    assert first == shape_trajectory(panel, **kw)[0].path[0].se


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"peak_band": 0.0}, "peak_band"),
        ({"age_window": 0}, "age_window"),
        ({"horizons": ()}, "horizons"),
        # `std(ddof=1)` on fewer than two draws is a NaN and a RuntimeWarning, which
        # reaches the caller as a silently missing SE rather than a refused argument.
        ({"bootstrap_draws": 0}, "bootstrap_draws"),
        ({"bootstrap_draws": 1}, "bootstrap_draws"),
    ],
)
def test_rejects_impossible_settings(kwargs: dict, match: str) -> None:
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    with pytest.raises(ValueError, match=match):
        shape_trajectory(panel, kind="hitter", age=28, sgp=15.0, peak=15.0, **kwargs)


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
    kw = {"kind": "hitter", "age": 28, "sgp": 15.0, "peak": 15.0, "peak_band": 50.0}
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
        shape_trajectory(prepared, kind="pitcher", age=28, sgp=15.0, peak=15.0, horizons=(1,))


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
            peak=15.0,
            horizons=(2,),
        )


def test_prepared_state_refuses_a_contradictory_last_season() -> None:
    """`prepare` has already censored on its own `last`; honouring a different one here
    would apply two different cutoffs to the same query."""
    panel = _mixed_panel()
    with pytest.raises(ValueError, match="last_complete_season"):
        shape_trajectory(
            prepare(panel, kind="hitter", horizons=(1,), last_complete_season=2012),
            kind="hitter",
            age=28,
            sgp=15.0,
            peak=15.0,
            horizons=(1,),
            last_complete_season=2011,
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
    """Two anchors can be collinear -- every comp down exactly half his peak, say -- and
    then three parameters are not identified. `lstsq` answered that with the least-norm
    solution and a plain `solve` raises `LinAlgError` on it, so the batched form has to
    keep the old behaviour rather than fail the query."""
    rng = np.random.default_rng(2)
    n = 60
    down = rng.uniform(5.0, 25.0, n)
    x = np.column_stack([down, 2.0 * down])  # peak is exactly 2 * down: rank 2, not 3
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
