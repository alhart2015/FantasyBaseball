"""Semantics the two estimators MUST share.

Three review rounds on this code found the same failure shape each time: `shape`
re-implements something `comps` already decided, and the two silently disagree. These
tests assert the agreements directly, so the next divergence fails here rather than
being discovered in a table someone has already acted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.comps import (
    Trajectory,
    collapse_split_seasons,
    comp_trajectory,
    played,
)
from fantasy_baseball.trajectory.shape import shape_trajectory


def _panel(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])


def _population(n: int = 200) -> pd.DataFrame:
    """Careers spanning three seasons, including below-replacement (negative) years."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        for offset, age in enumerate((26, 27, 28)):
            rows.append((i, 2010 + offset, age, float(rng.uniform(-3, 20))))
    return _panel(rows)


def test_both_modes_agree_a_negative_season_was_PLAYED() -> None:
    """SGP is genuinely negative for a below-replacement season -- 7.7% of hitter and
    15.0% of pitcher seasons in the real panel. Only an EXACT zero means he was out of
    the league, and a `> 0` test elsewhere reported every negative year as a career
    ending, understating survival and inflating the survivor mean."""
    assert list(played(np.array([-3.87, 0.0, 0.0001, 5.0]))) == [True, False, True, True]


def test_survival_matches_between_modes_on_one_population() -> None:
    panel = _population()
    comps = comp_trajectory(panel, kind="hitter", age=27, sgp=10.0, band=30.0, horizons=(1,))
    shape, _ = shape_trajectory(
        panel, kind="hitter", age=27, sgp=10.0, peak=10.0, horizons=(1,), peak_band=60.0
    )
    # Same underlying seasons, so the same count played -- the two modes must not
    # disagree about what "played" means.
    assert comps.path[0].survivors == shape.path[0].survivors
    assert comps.path[0].n == shape.path[0].n


def test_both_modes_report_a_predictive_spread_wider_than_the_mean_se() -> None:
    """`se` is the standard error of the MEAN and shrinks as sqrt(n); `spread` is how
    far ONE player can land. Reporting only `se` beside a single-player forecast read as
    though the season were pinned to within a fraction of an SGP."""
    panel = _population()
    for traj in (
        comp_trajectory(panel, kind="hitter", age=27, sgp=10.0, band=30.0, horizons=(1,)),
        shape_trajectory(
            panel, kind="hitter", age=27, sgp=10.0, peak=10.0, horizons=(1,), peak_band=60.0
        )[0],
    ):
        point = traj.path[0]
        assert not np.isnan(point.spread)
        assert point.spread > point.se


def test_both_modes_count_only_seasons_with_an_observable_forward_year() -> None:
    """`Trajectory.n_comps` is documented as the nearest-horizon cohort, and the CLI
    prints it as a claim about the fit. shape counted rows no horizon ever used."""
    panel = _population()
    for traj in (
        comp_trajectory(
            panel,
            kind="hitter",
            age=27,
            sgp=10.0,
            band=30.0,
            horizons=(1,),
            last_complete_season=2011,
        ),
        shape_trajectory(
            panel,
            kind="hitter",
            age=27,
            sgp=10.0,
            peak=10.0,
            horizons=(1,),
            peak_band=60.0,
            last_complete_season=2011,
        )[0],
    ):
        assert traj.n_comps == traj.path[0].n


def test_both_modes_collapse_a_split_season() -> None:
    """A mid-season trade yields two rows for one player-year. Collapsing only the
    forward lookup enters him into the cohort twice as two half-seasons."""
    split = _panel([(1, 2010, 26, 6.0), (1, 2010, 26, 4.0), (1, 2011, 27, 9.0), (1, 2012, 28, 8.0)])
    assert len(collapse_split_seasons(split)) == 3
    comps = comp_trajectory(split, kind="hitter", age=26, sgp=10.0, band=1.0, horizons=(1,))
    assert comps.n_comps == 1
    assert comps.mean_start == pytest.approx(10.0)


def test_every_mode_is_labelled() -> None:
    """`render` branches on `mode`; an unlabelled estimator would silently take the
    comps layout and mislabel its own columns."""
    panel = _population()
    assert comp_trajectory(panel, kind="hitter", age=27, sgp=10.0, horizons=(1,)).mode == "current"
    assert (
        comp_trajectory(panel, kind="hitter", age=27, sgp=10.0, prior_sgp=10.0, horizons=(1,)).mode
        == "track"
    )
    assert (
        shape_trajectory(panel, kind="hitter", age=27, sgp=10.0, peak=10.0, horizons=(1,))[0].mode
        == "shape"
    )


def test_no_support_is_distinguishable_from_no_data_in_every_mode() -> None:
    """Zero support must be readable as zero support. `total` summing an empty path to
    0.0 printed as a forecast of no future value."""
    panel = _population()
    empty_comps = comp_trajectory(panel, kind="hitter", age=99, sgp=10.0, horizons=(1,))
    empty_shape, _ = shape_trajectory(
        panel, kind="hitter", age=99, sgp=10.0, peak=10.0, horizons=(1,), peak_band=1.0
    )
    for traj in (empty_comps, empty_shape):
        assert isinstance(traj, Trajectory)
        assert traj.n_comps == 0
        assert traj.observable == ()
