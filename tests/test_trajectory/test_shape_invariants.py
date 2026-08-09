"""Invariants `shape` must hold, inherited from the retired mode-parity suite.

This began as "semantics the two estimators MUST share" -- three review rounds had each
found `shape` re-implementing something `comps` had already decided and the two silently
disagreeing. #325 retired the comp matchers, so there is no second estimator left to
disagree with.

The assertions were triaged rather than deleted with their partner. Every one that
stated a property of the panel, the censoring or the VAR scale is kept here as a
shape-only invariant, because those properties are what the parity tests were really
protecting -- agreement was the mechanism, not the point. Two were dropped and both are
named at the bottom of this file with the reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.model import Trajectory, collapse_split_seasons, played
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


def test_survival_counts_only_seasons_actually_played() -> None:
    """`played` keys on an exact 0.0 meaning "out of the league"; a negative season is a
    real season. Was a cross-mode agreement; now a statement about shape alone."""
    panel = _population()
    traj, _ = shape_trajectory(
        panel, kind="hitter", age=27, sgp=10.0, prior_sgp=10.0, horizons=(1,), prior_window=60.0
    )
    assert traj.path[0].survivors <= traj.path[0].n
    assert traj.path[0].survival == pytest.approx(traj.path[0].survivors / traj.path[0].n)


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


def test_thin_support_is_visible_as_an_effective_size() -> None:
    """A raw row count overstates support wherever kernels taper, which is why the
    thin-support gate reads `n_effective`. In shape it must be strictly smaller than n."""
    panel = _population()
    traj, anchors = shape_trajectory(
        panel, kind="hitter", age=27, sgp=10.0, prior_sgp=10.0, horizons=(1,), prior_window=8.0
    )
    assert traj.path[0].n_effective == pytest.approx(anchors[0].n_effective)
    assert traj.path[0].n_effective < traj.path[0].n


def test_only_seasons_with_an_observable_forward_year_are_counted() -> None:
    """`Trajectory.n_comps` is documented as the nearest-horizon cohort and the CLI
    prints it as a claim about the fit. shape once counted rows no horizon ever used."""
    panel = _population()
    traj, _ = shape_trajectory(
        panel,
        kind="hitter",
        age=27,
        sgp=10.0,
        prior_sgp=10.0,
        horizons=(1,),
        prior_window=60.0,
        last_complete_season=2011,
    )
    assert traj.n_comps == traj.path[0].n


def test_a_split_season_is_collapsed() -> None:
    """A mid-season trade yields two rows for one player-year. Collapsing only the
    forward lookup enters him into the cohort twice as two half-seasons.

    Asserted on `collapse_split_seasons` directly plus shape's own use of it: the
    helper is now in `model` and is the single definition both the estimator and the
    outcome side of any backtest read."""
    split = _panel([(1, 2010, 26, 6.0), (1, 2010, 26, 4.0), (1, 2011, 27, 9.0), (1, 2012, 28, 8.0)])
    assert len(collapse_split_seasons(split)) == 3
    collapsed = collapse_split_seasons(split)
    assert float(collapsed.loc[collapsed["season"] == 2010, "sgp"].iloc[0]) == pytest.approx(10.0)


@pytest.mark.parametrize("draws", [0, 1])
def test_an_unusable_bootstrap_count_is_refused(draws: int) -> None:
    """`std(ddof=1)` over fewer than two draws is a NaN and a RuntimeWarning, so the SE
    goes missing silently and `spread` quietly drops its SE term. shape raises rather
    than degrading."""
    panel = _population()
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


def test_var_is_the_raw_line_minus_the_floor_unclamped() -> None:
    """One definition of VAR: raw minus the floor, clamped nowhere.

    Before #331 shape FITTED `max(forward - floor, 0)`, and a clamped fit is not a
    shifted one -- it moves the slope, so VAR was not even a fixed offset from the raw
    fit, which reordered players sharing a slot. If a future change restores the clamp
    this fails.

    This is the assertion four deleted tests in `test_value.py` were reaching for
    through the retired comp matcher; it lives here because this fixture has the
    predictor variance shape needs and a three-comp band does not.
    """
    panel = _population()
    floor = 9.0
    query = {"kind": "hitter", "age": 27, "sgp": 10.0, "horizons": (1,)}
    raw = shape_trajectory(panel, prior_sgp=10.0, prior_window=60.0, **query)[0]
    var = shape_trajectory(
        panel, prior_sgp=10.0, prior_window=60.0, replacement=floor, slot="C", **query
    )[0]

    for field in ("mean", "median", "p10", "p90", "mean_if_survived"):
        assert getattr(var.path[0], field) == pytest.approx(getattr(raw.path[0], field) - floor)
    # Widths are differences and must NOT move with the floor.
    for field in ("se", "spread"):
        assert getattr(var.path[0], field) == pytest.approx(getattr(raw.path[0], field))
    assert (var.scale, var.slot, var.floor) == ("var", "C", floor)
    assert (raw.scale, raw.slot, raw.floor) == ("sgp", None, 0.0)


def test_the_trajectory_is_labelled_shape() -> None:
    """`Trajectory.mode` rides on the object so a printed table cannot misattribute its
    own numbers. One estimator remains, and it still has to say so."""
    panel = _population()
    traj, _ = shape_trajectory(
        panel, kind="hitter", age=27, sgp=10.0, prior_sgp=10.0, horizons=(1,)
    )
    assert traj.mode == "shape"


def test_no_support_is_distinguishable_from_no_data() -> None:
    """Zero support must be readable as zero support. `total` summing an empty path to
    0.0 printed as a forecast of no future value."""
    panel = _population()
    traj, _ = shape_trajectory(
        panel, kind="hitter", age=99, sgp=10.0, prior_sgp=10.0, horizons=(1,), prior_window=1.0
    )
    assert isinstance(traj, Trajectory)
    assert traj.n_comps == 0
    assert traj.observable == ()


# DROPPED with the comp matchers, both deliberately:
#
#   test_comps_spread_recovers_the_comp_to_comp_sd -- asserted that comps' `spread`
#   recovers the comp-to-comp standard deviation. That is a property of averaging a
#   cohort. shape fits a model and its spread is predictive, which
#   test_shape_spread_recovers_the_generating_sigma already pins against a KNOWN sigma.
#
#   the cross-mode halves of the survival, effective-size, forward-year, split-season
#   and bootstrap-guard tests -- agreement was the mechanism those used to catch a
#   divergence, not the property being protected. Each property is kept above as a
#   statement about shape alone.
