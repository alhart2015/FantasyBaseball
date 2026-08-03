"""The methodology guards in the window tuner (#310).

The full sweep is a measurement and does not belong in the suite -- it takes minutes and
its answer is data, not behaviour. What is asserted here is the part that decides whether
the answer means anything: that the query player really is absent from the panel he is
scored against, that a grid point cannot be scored on a kinder population than its
neighbours, and that a player cannot appear on both sides of a cross-validation split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.shape import AGE_WINDOW, PRIOR_WINDOW, shape_trajectory
from scripts.tune_shape_windows import (
    _folds,
    _n_queries,
    complete_cases,
    error_table,
    plateau,
    score_grid,
)


def _rows(records: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        records,
        columns=[
            "mlbam_id",
            "season",
            "horizon",
            "age_window",
            "prior_window",
            "predicted",
            "actual",
        ],
    )


def _panel(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """(mlbam_id, season, age, sgp) rows, matching `tests/test_trajectory/test_shape.py`."""
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])


def _population(n: int = 200) -> list[tuple[int, int, int, float]]:
    """Careers whose next season is a clean 0.5x the current one."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        prior, current = float(rng.uniform(8, 16)), float(rng.uniform(8, 16))
        rows += [(i, 2010, 27, prior), (i, 2011, 28, current), (i, 2012, 29, 0.5 * current)]
    return rows


def _saboteur(mlbam_id: int, seasons: int = 60) -> list[tuple[int, int, int, float]]:
    """One id's worth of seasons that pull the fit hard the other way.

    A flat 12.0 forever, against a population that halves -- and enough of them, sitting
    dead centre of both kernels, that leaving him in visibly moves the answer. Without that
    the held-out and not-held-out fits would agree and the test would pass vacuously.

    One row per season, ending at the population's last year: a duplicate `(id, season)`
    would be collapsed as a split season and a later one would push `last` past the
    population's forward values, censoring the very rows the fit needs.
    """
    return [(mlbam_id, 2012 - s, 28, 12.0) for s in range(seasons)]


def test_the_query_player_is_absent_from_the_panel_he_is_scored_against() -> None:
    """The claim the whole measurement rests on. If his own seasons stayed in, an estimator
    that fits a model would be matching him to himself and every RMSE here would flatter it.
    """
    saboteur_id = 9999
    panel = _panel(_population() + _saboteur(saboteur_id))
    queries = pd.DataFrame(
        [{"mlbam_id": saboteur_id, "season": 2011, "age": 28, "current": 12.0, "prior": 12.0}]
    )
    scored = score_grid(
        panel,
        queries,
        kind="hitter",
        horizons=(1,),
        age_windows=(AGE_WINDOW,),
        prior_windows=(PRIOR_WINDOW,),
    )

    without_him = panel[panel["mlbam_id"] != saboteur_id]
    expected, _ = shape_trajectory(
        without_him,
        kind="hitter",
        age=28,
        sgp=12.0,
        prior_sgp=12.0,
        horizons=(1,),
        last_complete_season=int(panel["season"].max()),
        bootstrap_draws=2,
    )
    assert scored["predicted"].iloc[0] == pytest.approx(expected.path[0].mean)

    # And the exclusion has to BITE, or the assertion above is vacuous.
    with_him, _ = shape_trajectory(
        panel,
        kind="hitter",
        age=28,
        sgp=12.0,
        prior_sgp=12.0,
        horizons=(1,),
        last_complete_season=int(panel["season"].max()),
        bootstrap_draws=2,
    )
    assert with_him.path[0].mean != pytest.approx(expected.path[0].mean)


def test_the_pool_is_stamped_on_every_row_so_a_reanalysis_cannot_mislabel_it() -> None:
    """`--from-csv` refuses a pool mismatch, which it can only do if the sweep recorded one."""
    scored = score_grid(
        _panel(_population()),
        pd.DataFrame([{"mlbam_id": 0, "season": 2011, "age": 28, "current": 12.0, "prior": 12.0}]),
        kind="pitcher",
        horizons=(1,),
        age_windows=(AGE_WINDOW,),
        prior_windows=(PRIOR_WINDOW,),
    )
    assert set(scored["pool"]) == {"pitcher"}


def test_a_query_one_grid_point_refused_is_dropped_from_all_of_them() -> None:
    """The gate that matters. A tight kernel starves `MIN_EFFECTIVE_ROWS` and refuses the
    thin queries -- which are the hard ones -- so scoring each setting on whatever it
    answered would hand the narrowest window the easiest rows and read as accuracy."""
    df = _rows(
        [
            (1, 2015, 1, 1, 4.0, np.nan, 10.0),  # the tight window refused this query
            (1, 2015, 1, 4, 8.0, 9.0, 10.0),
            (2, 2015, 1, 1, 4.0, 5.0, 6.0),
            (2, 2015, 1, 4, 8.0, 5.5, 6.0),
        ]
    )
    complete, kept = complete_cases(df, n_grid=2)
    assert list(complete["mlbam_id"].unique()) == [2]
    assert kept == 0.5


def test_complete_cases_reports_no_survivors_rather_than_an_empty_argmin() -> None:
    df = _rows([(1, 2015, 1, 1, 4.0, np.nan, 10.0), (1, 2015, 1, 4, 8.0, 9.0, 10.0)])
    complete, kept = complete_cases(df, n_grid=2)
    assert complete.empty
    assert kept == 0.0


def test_the_same_season_at_two_horizons_is_two_independent_cases() -> None:
    """A season answered at +1 but refused at +3 must keep its +1, not lose the season."""
    df = _rows(
        [
            (1, 2015, 1, 1, 4.0, 8.0, 10.0),
            (1, 2015, 1, 4, 8.0, 9.0, 10.0),
            (1, 2015, 3, 1, 4.0, np.nan, 7.0),
            (1, 2015, 3, 4, 8.0, 6.0, 7.0),
        ]
    )
    complete, _ = complete_cases(df, n_grid=2)
    assert set(complete["horizon"]) == {1}
    assert len(complete) == 2


def test_error_table_is_rmse_per_grid_point() -> None:
    df = _rows(
        [
            (1, 2015, 1, 2, 8.0, 10.0, 7.0),  # error -3
            (2, 2015, 1, 2, 8.0, 4.0, 8.0),  # error +4
            (1, 2015, 1, 3, 8.0, 7.0, 7.0),
            (2, 2015, 1, 3, 8.0, 8.0, 8.0),
        ]
    )
    table = error_table(df)
    assert table.loc[8.0, 2] == np.sqrt((9 + 16) / 2)
    assert table.loc[8.0, 3] == 0.0


def test_a_career_never_spans_two_folds() -> None:
    """Cut by player, never by row. A player's seasons are correlated, so splitting them
    would leak him into his own training set and flatter whatever the fold selected."""
    ids = np.array([100, 100, 100, 101, 101, 102])
    assert len(set(_folds(ids[:3], 5))) == 1
    assert len(set(_folds(ids[3:5], 5))) == 1


def _two_cell_grid(alternative_error: float) -> pd.DataFrame:
    """Thirty players, each one season: the default off by 1.0, the alternative by more."""
    rows = []
    for player in range(30):
        rows.append((player, 2015, 1, AGE_WINDOW, PRIOR_WINDOW, 1.0, 0.0))
        rows.append((player, 2015, 1, AGE_WINDOW + 1, PRIOR_WINDOW, alternative_error, 0.0))
    return _rows(rows)


def test_a_setting_that_never_differs_from_the_default_reads_as_a_coin_flip() -> None:
    """The degenerate tie. A bare `>` scores an identical cell 0.00 -- which on this table
    reads as 'beat the default in every draw', the exact opposite of what happened."""
    _, confidence = plateau(_two_cell_grid(1.0), draws=200)
    assert confidence.loc[PRIOR_WINDOW, AGE_WINDOW + 1] == 0.5


def test_the_default_cell_is_blank_rather_than_a_comparison_with_itself() -> None:
    _, confidence = plateau(_two_cell_grid(1.0), draws=200)
    assert np.isnan(confidence.loc[PRIOR_WINDOW, AGE_WINDOW])


def test_the_plateau_bootstrap_gives_the_same_answer_at_any_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch is a memory budget, not a modelling choice -- draws are consumed in the
    same order at any width, so narrowing it must not move a number. Mirrors
    `test_shape.py`'s BOOTSTRAP_BYTES invariance test, for the same reason."""
    df = _two_cell_grid(2.0)
    wide_delta, wide_confidence = plateau(df, draws=64)
    monkeypatch.setattr("scripts.tune_shape_windows.PLATEAU_BYTES", 1)
    narrow_delta, narrow_confidence = plateau(df, draws=64)
    pd.testing.assert_frame_equal(wide_delta, narrow_delta)
    pd.testing.assert_frame_equal(wide_confidence, narrow_confidence)


def test_a_clearly_worse_setting_is_called_worse_in_every_draw() -> None:
    delta, confidence = plateau(_two_cell_grid(3.0), draws=200)
    assert delta.loc[PRIOR_WINDOW, AGE_WINDOW + 1] == 2.0
    assert confidence.loc[PRIOR_WINDOW, AGE_WINDOW + 1] == 1.0
    assert delta.loc[PRIOR_WINDOW, AGE_WINDOW] == 0.0


def test_n_counts_query_horizons_not_grid_expanded_rows() -> None:
    """Two queries at nine grid points is n=2, not n=18 -- the printed n is the sample the
    RMSE rests on, and the grid-expanded count overstates it by the size of the grid."""
    df = _rows(
        [(1, 2015, 1, aw, 8.0, 5.0, 6.0) for aw in (1, 2, 3)]
        + [(2, 2016, 1, aw, 8.0, 5.0, 6.0) for aw in (1, 2, 3)]
    )
    assert _n_queries(df) == 2
