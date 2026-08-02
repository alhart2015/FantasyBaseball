from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.shape import build_history, shape_trajectory


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
    ],
)
def test_rejects_impossible_settings(kwargs: dict, match: str) -> None:
    panel = _linear_population(coef_down=0.4, coef_peak=0.5)
    with pytest.raises(ValueError, match=match):
        shape_trajectory(panel, kind="hitter", age=28, sgp=15.0, peak=15.0, **kwargs)
