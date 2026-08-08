from __future__ import annotations

import numpy as np

from fantasy_baseball.trajectory.comp_paths import closest_paths
from fantasy_baseball.trajectory.shape import Prepared

HORIZONS = (1, 2, 3, 4, 5)


def _prepared(rows: list[tuple[int, int, int, list[float]]], last: int = 2020) -> Prepared:
    """Build a Prepared straight from (mlbam_id, season, age, forward path) rows.

    Hand-built rather than swept from a panel: `closest_paths` reads seven arrays and
    nothing else, so a fixture that goes through `prepare()` would test `prepare()`.
    """
    return Prepared(
        kind="hitter",
        horizons=HORIZONS,
        last=last,
        age=np.array([r[2] for r in rows], dtype=float),
        current=np.zeros(len(rows)),
        prior=np.zeros(len(rows)),
        season=np.array([r[1] for r in rows]),
        mlbam_id=np.array([r[0] for r in rows]),
        forward={h: np.array([r[3][h - 1] for r in rows], dtype=float) for h in HORIZONS},
    )


def test_comps_come_back_ordered_by_rmse() -> None:
    """Closest first -- that ordering is the whole product."""
    target = [10.0, 10.0, 10.0, 10.0, 10.0]
    prepared = _prepared(
        [
            (1, 2010, 25, [12.0] * 5),  # off by 2.0
            (2, 2011, 25, [10.5] * 5),  # off by 0.5  <- closest
            (3, 2012, 25, [7.0] * 5),  # off by 3.0
        ]
    )
    got = closest_paths(prepared, target, age=25, n=3)
    assert [c.mlbam_id for c in got] == [2, 1, 3]
    assert got[0].rmse < got[1].rmse < got[2].rmse
    assert got[0].path == (10.5, 10.5, 10.5, 10.5, 10.5)


def test_a_near_age_row_is_not_a_candidate() -> None:
    """EXACT age, so the comp's +1..+5 lands on the query's projected ages. A
    26-year-old's next five years are a different five years on the x-axis."""
    target = [10.0] * 5
    prepared = _prepared(
        [
            (1, 2010, 26, [10.0] * 5),  # perfect match, WRONG age
            (2, 2011, 25, [14.0] * 5),  # bad match, right age
        ]
    )
    got = closest_paths(prepared, target, age=25, n=5)
    assert [c.mlbam_id for c in got] == [2]


def test_a_short_path_cannot_win_by_having_less_to_match() -> None:
    """The recency rule. A row whose season+5 runs past `last` has unrealized years,
    and scoring it on the two that exist would beat a five-year match for free."""
    target = [10.0] * 5
    prepared = _prepared(
        [
            (1, 2019, 25, [10.0, 10.0, 0.0, 0.0, 0.0]),  # only +1 and +2 realized
            (2, 2010, 25, [11.0] * 5),  # all five realized
        ],
        last=2021,
    )
    got = closest_paths(prepared, target, age=25, n=5)
    assert [c.mlbam_id for c in got] == [2], "the 2019 row has no realized +3..+5"


def test_ties_break_deterministically_regardless_of_row_order() -> None:
    """Two identical paths must not swap between reads -- the arbitrary-ordering
    defect `index_rosters` was fixed for in 06bf2646, one module over."""
    target = [10.0] * 5
    rows = [
        (7, 2011, 25, [12.0] * 5),
        (3, 2010, 25, [12.0] * 5),
    ]
    forward = [c.mlbam_id for c in closest_paths(_prepared(rows), target, age=25, n=2)]
    reverse = [c.mlbam_id for c in closest_paths(_prepared(rows[::-1]), target, age=25, n=2)]
    assert forward == [3, 7], "tie breaks on mlbam_id ascending"
    assert forward == reverse


def test_no_candidates_returns_empty_rather_than_raising() -> None:
    prepared = _prepared([(1, 2010, 30, [10.0] * 5)])
    assert closest_paths(prepared, [10.0] * 5, age=25, n=5) == []


def test_n_larger_than_the_candidate_pool_returns_what_exists() -> None:
    prepared = _prepared([(1, 2010, 25, [10.0] * 5)])
    assert len(closest_paths(prepared, [10.0] * 5, age=25, n=10)) == 1
