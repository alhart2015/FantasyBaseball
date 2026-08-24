from __future__ import annotations

import numpy as np
import pytest

from fantasy_baseball.trajectory.career_comps import (
    MIN_OVERLAP,
    closest_careers,
    required_overlap,
)
from fantasy_baseball.trajectory.shape import Prepared

HORIZONS = (1, 2, 3, 4, 5)
LOOKBACK = 8


def _prepared(
    rows: list[tuple[int, int, int, dict[int, float], list[float]]],
    last: int = 2020,
) -> Prepared:
    """Build a Prepared straight from (mlbam_id, season, age, career, forward) rows.

    `career` is the row's realized SGP by AGE OFFSET -- {0: this season, 1: the year
    before, ...} -- and an absent offset becomes the NaN `Prepared.back` uses for "no
    season there". Hand-built rather than swept from a panel: `closest_careers` reads
    eight arrays and nothing else, so a fixture going through `prepare()` would be
    testing `prepare()`.
    """
    return Prepared(
        kind="hitter",
        horizons=HORIZONS,
        last=last,
        age=np.array([r[2] for r in rows], dtype=float),
        current=np.array([r[3].get(0, np.nan) for r in rows], dtype=float),
        prior=np.array([r[3].get(1, np.nan) for r in rows], dtype=float),
        season=np.array([r[1] for r in rows]),
        mlbam_id=np.array([r[0] for r in rows]),
        forward={h: np.array([r[4][h - 1] for r in rows], dtype=float) for h in HORIZONS},
        back={
            k: np.array([r[3].get(k, np.nan) for r in rows], dtype=float) for k in range(LOOKBACK)
        },
        lookback=LOOKBACK,
    )


def _flat(value: float, ages: int = LOOKBACK) -> dict[int, float]:
    """A career that sat at `value` for `ages` consecutive years, by offset."""
    return {k: value for k in range(ages)}


def _career(value: float, age: int = 25, ages: int = LOOKBACK) -> dict[int, float]:
    """The same, keyed by AGE -- the shape a subject is passed in."""
    return {age - k: value for k in range(ages)}


def test_comps_are_ordered_by_distance_on_the_realized_career() -> None:
    """Closest first, and closest means closest on what he DID -- the forward path is
    the payload, never the selector. Here the ordering is the exact reverse of what
    matching on the forward path would give, so a regression to forward matching fails
    this rather than merely looking different."""
    prepared = _prepared(
        [
            (1, 2010, 25, _flat(12.0), [99.0] * 5),  # career off by 2.0, wild future
            (2, 2011, 25, _flat(10.5), [50.0] * 5),  # career off by 0.5  <- closest
            (3, 2012, 25, _flat(7.0), [10.0] * 5),  # career off by 3.0, future matches
        ]
    )
    got = closest_careers(prepared, _career(10.0), age=25, n=3)
    assert [c.mlbam_id for c in got] == [2, 1, 3]
    assert got[0].rmse < got[1].rmse < got[2].rmse
    assert got[0].path == (50.0, 50.0, 50.0, 50.0, 50.0)
    assert got[0].overlap == LOOKBACK


def test_a_near_age_row_is_not_a_candidate() -> None:
    """EXACT age, so the comp's +1..+5 lands on the subject's projected ages. A
    26-year-old's next five years are a different five years on the x-axis."""
    prepared = _prepared(
        [
            (1, 2010, 26, _flat(10.0), [10.0] * 5),  # perfect career, WRONG age
            (2, 2011, 25, _flat(14.0), [10.0] * 5),  # poor career, right age
        ]
    )
    assert [c.mlbam_id for c in closest_careers(prepared, _career(10.0), age=25, n=5)] == [2]


def test_a_censored_row_is_not_a_candidate() -> None:
    """The recency rule, unchanged from the forward matcher. A row whose season+5 runs
    past `last` has years that have not happened yet, and a comp we cannot follow
    forward answers nothing -- which is a statement about the SEASON, because a real
    0.0 in `forward` reads identically to an unrealized one."""
    prepared = _prepared(
        [
            (1, 2019, 25, _flat(10.0), [10.0, 10.0, 0.0, 0.0, 0.0]),  # perfect, unfinished
            (2, 2010, 25, _flat(14.0), [11.0] * 5),  # poor, fully realized
        ],
        last=2021,
    )
    got = closest_careers(prepared, _career(10.0), age=25, n=5)
    assert [c.mlbam_id for c in got] == [2], "the 2019 row has no realized +3..+5"


def test_a_player_who_left_the_league_is_kept_with_his_zeros() -> None:
    """SURVIVORSHIP. Requiring a played season at every horizon would drop exactly the
    players whose careers ended, leaving a comp set that is all survivors and a spread
    that is optimistic by construction. His 0.0 years are what happened to him."""
    prepared = _prepared([(1, 2010, 25, _flat(10.0), [9.0, 4.0, 0.0, 0.0, 0.0])])
    got = closest_careers(prepared, _career(10.0), age=25, n=5)
    assert [c.mlbam_id for c in got] == [1]
    assert got[0].path == (9.0, 4.0, 0.0, 0.0, 0.0)


def test_a_year_he_did_not_play_is_dropped_from_the_match_not_scored_as_zero() -> None:
    """THE ALVAREZ CASE (#357/#358). Two candidates identical in every year they both
    played; one simply missed a season. Scoring that hole as a 0 would push him far
    down the list -- an injured star made to look like a replacement-level journeyman.
    He is dropped from the error instead, and the miss shows up as a lower `overlap`."""
    intact = _flat(10.0)
    interrupted = {k: v for k, v in intact.items() if k != 3}
    prepared = _prepared(
        [
            (1, 2010, 25, intact, [10.0] * 5),
            (2, 2011, 25, interrupted, [10.0] * 5),
        ]
    )
    got = {c.mlbam_id: c for c in closest_careers(prepared, _career(10.0), age=25, n=5)}
    assert got[2].rmse == pytest.approx(got[1].rmse), "the missing year cost him nothing"
    assert (got[1].overlap, got[2].overlap) == (LOOKBACK, LOOKBACK - 1)


def test_a_candidate_without_enough_shared_ages_is_refused() -> None:
    """`required_overlap` is a floor on how much career the match actually saw. A
    candidate with two years in common can score a beautiful RMSE on them and tell you
    nothing about the seven-year shape you asked about."""
    prepared = _prepared(
        [
            (1, 2010, 25, {0: 10.0, 1: 10.0}, [10.0] * 5),  # 2 shared ages, perfect
            (2, 2011, 25, _flat(13.0), [10.0] * 5),  # all 8, poor
        ]
    )
    assert [c.mlbam_id for c in closest_careers(prepared, _career(10.0), age=25, n=5)] == [2]


def test_a_young_subject_is_matched_on_what_he_has() -> None:
    """The overlap floor is a FRACTION of the subject's own ages, not a flat count.
    A flat 6-of-8 would refuse every 23-year-old -- who is precisely the ambiguous
    keeper call this board exists to answer."""
    prepared = _prepared([(1, 2010, 23, {0: 10.0, 1: 9.0, 2: 8.0}, [11.0] * 5)])
    got = closest_careers(prepared, {23: 10.0, 22: 9.0, 21: 8.0}, age=23, n=5)
    assert [c.mlbam_id for c in got] == [1]
    assert got[0].overlap == 3


def test_required_overlap_scales_with_the_subject_but_never_below_the_floor() -> None:
    assert required_overlap(8) == 6, "the >= 6 of 8 the #358 prototype matched Alvarez on"
    assert required_overlap(3) == 3
    assert required_overlap(2) == MIN_OVERLAP
    assert required_overlap(1) == MIN_OVERLAP, "a floor he cannot buy his way under"


def test_a_subject_with_one_season_gets_no_comps_rather_than_a_level_match() -> None:
    """A debut player has no career to match on, and one shared age is level matching --
    the question #325 retired -- wearing a career comp's heading.

    This is what a clamped floor hides. Clamping `required_overlap` to what the subject
    has gave 100% comp coverage across both 2026 pools at every age from 20 up, which
    looks like the feature working and is really the floor switching itself off at
    exactly the ages a reader cannot sanity-check.
    """
    prepared = _prepared([(1, 2010, 25, _flat(10.0), [10.0] * 5)])
    assert closest_careers(prepared, {25: 10.0}, age=25, n=5) == []


def test_the_subject_is_not_his_own_comp() -> None:
    """He matches himself at RMSE 0.00 and it renders perfectly. The forward-
    observability mask usually removes him first, but only because his anchor season
    happens to be the newest one -- a fact about the panel, not a rule."""
    prepared = _prepared(
        [
            (7, 2010, 25, _flat(10.0), [10.0] * 5),
            (8, 2011, 25, _flat(11.0), [10.0] * 5),
        ]
    )
    got = closest_careers(prepared, _career(10.0), age=25, n=5, exclude_id=7)
    assert [c.mlbam_id for c in got] == [8]


def test_ties_break_deterministically_regardless_of_row_order() -> None:
    """Two identical careers must not swap between reads -- the arbitrary-ordering
    defect `index_rosters` was fixed for in 06bf2646, one module over."""
    rows = [
        (7, 2011, 25, _flat(12.0), [12.0] * 5),
        (3, 2010, 25, _flat(12.0), [12.0] * 5),
    ]
    forward = [c.mlbam_id for c in closest_careers(_prepared(rows), _career(10.0), age=25, n=2)]
    reverse = [
        c.mlbam_id for c in closest_careers(_prepared(rows[::-1]), _career(10.0), age=25, n=2)
    ]
    assert forward == [3, 7], "tie breaks on mlbam_id ascending"
    assert forward == reverse


def test_ages_outside_the_lookback_window_are_ignored() -> None:
    """A caller may pass a whole career; only the window is matched on. Without the
    slice a 35-year-old's teenage seasons would be compared against `back` offsets that
    do not exist and raise."""
    prepared = _prepared([(1, 2010, 25, _flat(10.0), [10.0] * 5)])
    whole = {age: 10.0 for age in range(18, 26)} | {age: -99.0 for age in range(14, 18)}
    got = closest_careers(prepared, whole, age=25, n=5)
    assert got[0].rmse == pytest.approx(0.0)
    assert got[0].overlap == LOOKBACK


def test_no_candidates_returns_empty_rather_than_raising() -> None:
    prepared = _prepared([(1, 2010, 30, _flat(10.0), [10.0] * 5)])
    assert closest_careers(prepared, _career(10.0), age=25, n=5) == []


def test_a_career_missing_the_anchor_age_is_a_caller_bug() -> None:
    """Every candidate is selected at exactly this age, so with no value there the
    match has nothing to anchor to. The push script supplies it from the anchored base
    season, which `complete` does not carry -- silently returning [] would hide that
    wiring coming undone."""
    prepared = _prepared([(1, 2010, 25, _flat(10.0), [10.0] * 5)])
    with pytest.raises(ValueError, match="anchor age 25"):
        closest_careers(prepared, {24: 10.0, 23: 10.0}, age=25, n=5)


def test_n_larger_than_the_candidate_pool_returns_what_exists() -> None:
    prepared = _prepared([(1, 2010, 25, _flat(10.0), [10.0] * 5)])
    assert len(closest_careers(prepared, _career(10.0), age=25, n=10)) == 1
