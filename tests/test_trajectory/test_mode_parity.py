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
        panel, kind="hitter", age=27, sgp=10.0, prior_sgp=10.0, horizons=(1,), prior_window=60.0
    )
    # Same underlying seasons, so the same count played -- the two modes must not
    # disagree about what "played" means.
    assert comps.path[0].survivors == shape.path[0].survivors
    assert comps.path[0].n == shape.path[0].n


def _known_sigma_population(sigma: float, n: int = 400) -> pd.DataFrame:
    """Next season is exactly 0.4*current + 0.5*prior plus noise of a KNOWN sigma, so a
    correct predictive spread has a value to be checked against."""
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        prior = float(rng.uniform(5, 25))
        current = float(rng.uniform(5, 25))
        rows.append((i, 2010, 26, prior))
        rows.append((i, 2011, 27, current))
        rows.append((i, 2012, 28, 0.4 * current + 0.5 * prior + float(rng.normal(0, sigma))))
    return _panel(rows)


def test_shape_spread_recovers_the_generating_sigma() -> None:
    """`spread > se` is a TAUTOLOGY in shape mode -- spread is sqrt(residual_var + se^2)
    with residual_var >= 0 -- so asserting it locks nothing. This binds instead: on a
    population whose noise sigma is known, the reported spread must recover it. It fails
    if the residual variance is dropped, mis-weighted, or the degrees-of-freedom
    correction is removed."""
    sigma = 3.0
    traj, _ = shape_trajectory(
        _known_sigma_population(sigma),
        kind="hitter",
        age=27,
        sgp=15.0,
        prior_sgp=15.0,
        horizons=(1,),
        prior_window=60.0,
    )
    point = traj.path[0]
    assert point.spread == pytest.approx(sigma, rel=0.15)
    # And it must dwarf the SE of the mean, which shrinks as sqrt(n).
    assert point.spread > 5 * point.se


def test_comps_spread_recovers_the_comp_to_comp_sd() -> None:
    """The comps half of the same contract: `spread` is the SD of the forward values,
    not a rescaled SE."""
    forward = [2.0, 6.0, 10.0, 14.0]
    panel = _panel(
        [(i, 2010, 27, 10.0) for i in range(4)] + [(i, 2011, 28, v) for i, v in enumerate(forward)]
    )
    traj = comp_trajectory(panel, kind="hitter", age=27, sgp=10.0, band=1.0, horizons=(1,))
    assert traj.path[0].spread == pytest.approx(float(np.std(forward, ddof=1)))


def test_thin_support_is_visible_as_an_effective_size_in_both_modes() -> None:
    """A raw row count overstates support wherever kernels taper, which is why the
    thin-support gate reads `n_effective`. In comps every row counts once, so the two
    agree; in shape the effective size must be strictly smaller."""
    panel = _population()
    comps = comp_trajectory(panel, kind="hitter", age=27, sgp=10.0, band=30.0, horizons=(1,))
    assert comps.path[0].n_effective == pytest.approx(comps.path[0].n)

    shape, anchors = shape_trajectory(
        panel, kind="hitter", age=27, sgp=10.0, prior_sgp=10.0, horizons=(1,), prior_window=8.0
    )
    assert shape.path[0].n_effective == pytest.approx(anchors[0].n_effective)
    assert shape.path[0].n_effective < shape.path[0].n


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
            prior_sgp=10.0,
            horizons=(1,),
            prior_window=60.0,
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


@pytest.mark.parametrize("draws", [0, 1])
def test_both_modes_refuse_an_unusable_bootstrap_count(draws: int) -> None:
    """`std(ddof=1)` over fewer than two draws is a NaN and a RuntimeWarning, so the SE
    goes missing silently and `spread` quietly drops its SE term. `_bootstrap_se` guards
    the COMP count, not the draw count, so it never covered this -- and for a while shape
    mode raised while comps mode degraded silently on the identical argument."""
    panel = _population()
    with pytest.raises(ValueError, match="bootstrap_draws"):
        comp_trajectory(
            panel, kind="hitter", age=27, sgp=10.0, band=5.0, horizons=(1,), bootstrap_draws=draws
        )
    with pytest.raises(ValueError, match="bootstrap_draws"):
        shape_trajectory(
            panel,
            kind="hitter",
            age=27,
            sgp=10.0,
            prior_sgp=10.0,
            horizons=(1,),
            bootstrap_draws=draws,
        )


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
        shape_trajectory(panel, kind="hitter", age=27, sgp=10.0, prior_sgp=10.0, horizons=(1,))[
            0
        ].mode
        == "shape"
    )


def test_no_support_is_distinguishable_from_no_data_in_every_mode() -> None:
    """Zero support must be readable as zero support. `total` summing an empty path to
    0.0 printed as a forecast of no future value."""
    panel = _population()
    empty_comps = comp_trajectory(panel, kind="hitter", age=99, sgp=10.0, horizons=(1,))
    empty_shape, _ = shape_trajectory(
        panel, kind="hitter", age=99, sgp=10.0, prior_sgp=10.0, horizons=(1,), prior_window=1.0
    )
    for traj in (empty_comps, empty_shape):
        assert isinstance(traj, Trajectory)
        assert traj.n_comps == 0
        assert traj.observable == ()
