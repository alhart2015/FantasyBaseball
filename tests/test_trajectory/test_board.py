from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.trajectory.board import board_inputs
from fantasy_baseball.trajectory.shape import shape_trajectory

LEVELS = {"C": 7.70, "1B": 9.15, "2B": 9.45, "3B": 9.27, "SS": 9.51, "OF": 9.96, "UTIL": 9.96}
PITCHER_LEVELS = {"SP": 9.29, "RP": 7.42}
NAMES = pd.Series({1: "Alpha", 2: "Bravo", 3: "Charlie", 900: "Two Way"})


def _panel(rows: list[tuple], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _hitters(rows: list[tuple], games: float = 113.0) -> pd.DataFrame:
    """(mlbam_id, season, age, sgp, partial_season).

    `pa` and `games` are what `season_elapsed_fraction` dates the season from -- the
    busiest player's games over a full schedule -- and it refuses a frame without `pa`
    precisely so a pitcher panel can never be used for it. 113 of 162 is ~70% elapsed.
    """
    frame = _panel(rows, ["mlbam_id", "season", "age", "sgp", "partial_season"])
    return frame.assign(pa=500.0, games=games)


def _one(rows: list, pid: int):
    return next(r for r in rows if r.mlbam_id == pid)


def test_a_split_season_enters_once_at_its_full_total() -> None:
    """A mid-season trade gives one player two rows. Left uncollapsed he enters the board
    twice, each half compared against other players' whole seasons."""
    panel = _hitters(
        [(1, 2025, 26, 12.0, False), (1, 2026, 27, 5.0, False), (1, 2026, 27, 6.0, False)]
    )
    rows = board_inputs(panel, kind="hitter", names=NAMES, replacement_levels=LEVELS)
    assert len(rows) == 1
    assert rows[0].sgp == pytest.approx(11.0)


def test_an_in_progress_season_is_paced_to_a_full_year() -> None:
    """Two thirds of a season is not a season. Comparing the fragment against full years
    would rank every current player below where he belongs."""
    panel = _hitters([(1, 2026, 27, 8.0, True), (2, 2026, 27, 8.0, False)])
    rows = board_inputs(panel, kind="hitter", names=NAMES, replacement_levels=LEVELS)
    partial, complete = _one(rows, 1), _one(rows, 2)
    assert partial.sgp > complete.sgp
    assert complete.sgp == pytest.approx(8.0)


def test_a_missing_prior_season_is_a_real_zero() -> None:
    """He was not in the league. For a young player that is the normal case, and it is
    the same convention the forward path uses -- not missing data to be imputed."""
    panel = _hitters([(1, 2026, 22, 13.0, False)])
    rows = board_inputs(panel, kind="hitter", names=NAMES, replacement_levels=LEVELS)
    assert rows[0].prior_sgp == pytest.approx(0.0)


def test_an_unknown_slot_falls_back_to_the_highest_floor() -> None:
    """A missing eligibility lookup must only ever UNDERSTATE a player. UTIL is the most
    expensive floor, so netting against it cannot invent value."""
    panel = _hitters([(1, 2026, 27, 14.0, False)])
    rows = board_inputs(
        panel, kind="hitter", names=NAMES, replacement_levels=LEVELS, eligibility={}
    )
    assert (rows[0].slot, rows[0].floor) == ("UTIL", 9.96)


def test_a_multi_eligible_hitter_is_priced_at_his_scarcest_slot() -> None:
    panel = _hitters([(1, 2026, 27, 14.0, False)])
    rows = board_inputs(
        panel,
        kind="hitter",
        names=NAMES,
        replacement_levels=LEVELS,
        eligibility={1: frozenset({"C", "1B", "OF"})},
    )
    assert (rows[0].slot, rows[0].floor) == ("C", 7.70)


def _pitchers(rows: list[tuple]) -> pd.DataFrame:
    """(mlbam_id, season, age, sgp, partial_season, starts, games)."""
    return _panel(rows, ["mlbam_id", "season", "age", "sgp", "partial_season", "starts", "games"])


def test_the_split_season_rule_matches_the_shared_one() -> None:
    """`_collapse` carries columns `collapse_split_seasons` drops, so it cannot call it --
    but the `sgp` rule must stay the same rule. Asserted rather than commented, since a
    second definition of "a split season is summed" is exactly how the two estimators
    drifted apart before."""
    from fantasy_baseball.trajectory.board import _collapse
    from fantasy_baseball.trajectory.comps import collapse_split_seasons

    panel = _hitters(
        [(1, 2026, 27, 5.0, True), (1, 2026, 27, 6.0, False), (2, 2026, 25, 9.0, False)]
    )
    mine = _collapse(panel).set_index(["mlbam_id", "season"])["sgp"]
    shared = collapse_split_seasons(panel).set_index(["mlbam_id", "season"])["sgp"]
    pd.testing.assert_series_equal(mine.sort_index(), shared.sort_index())


def test_a_split_season_that_is_partly_in_progress_stays_partial() -> None:
    """Half a finished season plus half an in-progress one is an in-progress season.
    Losing the flag would compare a fragment against full years."""
    from fantasy_baseball.trajectory.board import _collapse

    panel = _hitters([(1, 2026, 27, 5.0, False), (1, 2026, 27, 4.0, True)])
    assert bool(_collapse(panel).iloc[0]["partial_season"])


def test_pacing_a_pitcher_board_refuses_to_date_the_season_off_its_own_panel() -> None:
    """In the pitcher panel `games` counts appearances, not team games, so the elapsed
    fraction comes out near half the truth and roughly DOUBLES every projected pace.
    Refused at the boundary rather than deep inside `season_elapsed_fraction`."""
    panel = _pitchers([(1, 2026, 30, 6.0, True, 10.0, 12.0)])
    with pytest.raises(ValueError, match="HITTER panel"):
        board_inputs(panel, kind="pitcher", names=NAMES, replacement_levels=PITCHER_LEVELS)


def test_a_pitchers_role_comes_from_a_settled_season() -> None:
    """A starter back from the IL with two September relief outings is not a reliever,
    but `starts / games` on that fragment says he is -- and the pace adjustment applied
    to his SGP was never applied to the role read. Getting this wrong moves him 1.87 SGP
    a year between the SP and RP floors."""
    panel = _pitchers(
        [
            (1, 2025, 29, 15.0, False, 30.0, 30.0),  # unambiguous starter
            (1, 2026, 30, 3.0, True, 0.0, 2.0),  # two relief outings back from the IL
        ]
    )
    rows = board_inputs(
        panel,
        kind="pitcher",
        names=NAMES,
        replacement_levels=PITCHER_LEVELS,
        calendar=_hitters([(99, 2026, 27, 10.0, True)]),
    )
    assert (rows[0].slot, rows[0].floor) == ("SP", 9.29)


def test_a_genuine_reliever_still_prices_as_one() -> None:
    panel = _pitchers([(1, 2025, 29, 9.0, False, 0.0, 60.0), (1, 2026, 30, 9.0, False, 0.0, 62.0)])
    rows = board_inputs(panel, kind="pitcher", names=NAMES, replacement_levels=PITCHER_LEVELS)
    assert (rows[0].slot, rows[0].floor) == ("RP", 7.42)


def test_a_pitcher_who_has_never_settled_still_gets_a_slot() -> None:
    """No season clears the appearance floor. He must still be priced -- falling through
    to no slot at all would drop him off the board silently."""
    panel = _pitchers([(1, 2026, 24, 6.0, True, 4.0, 5.0)])
    rows = board_inputs(
        panel,
        kind="pitcher",
        names=NAMES,
        replacement_levels=PITCHER_LEVELS,
        calendar=_hitters([(99, 2026, 27, 10.0, True)]),
    )
    assert rows[0].slot in {"SP", "RP"}


def test_only_players_with_a_line_in_the_scored_season_appear() -> None:
    """A retired player still has panel rows. Ranking him against active players would
    put a career that ended in 2019 on a 2026 keeper board."""
    panel = _hitters([(1, 2019, 30, 18.0, False), (2, 2026, 27, 11.0, False)])
    rows = board_inputs(panel, kind="hitter", names=NAMES, replacement_levels=LEVELS)
    assert [r.mlbam_id for r in rows] == [2]


# --- the extrapolation guard --------------------------------------------------------


def _cohort(down_range: tuple[float, float], n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        peak = float(rng.uniform(*down_range))
        down = float(rng.uniform(*down_range))
        rows += [(i, 2010, 27, peak), (i, 2011, 28, down), (i, 2012, 29, down * 0.8)]
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])


def test_local_support_is_high_when_the_query_sits_inside_its_cohort() -> None:
    traj, _ = shape_trajectory(
        _cohort((10.0, 20.0)), kind="hitter", age=28, sgp=15.0, peak=15.0, horizons=(1,)
    )
    assert traj.local_support > 0.25


def test_local_support_collapses_when_the_query_outruns_its_cohort() -> None:
    """The failure the board exists to guard: `peak` is kernel-weighted but `down` is a
    bare regressor, so a player whose current season far outruns his prior is matched to
    a cohort he sits outside and then priced by extrapolating their fitted line. The
    band comes out NARROW at the same time, because it is that cohort's scatter --
    confident and wrong together, which is what makes it worth refusing."""
    panel = _cohort((0.0, 6.0))
    inside, _ = shape_trajectory(
        panel, kind="hitter", age=28, sgp=3.0, peak=3.0, horizons=(1,), peak_band=8.0
    )
    outside, _ = shape_trajectory(
        panel, kind="hitter", age=28, sgp=16.0, peak=3.0, horizons=(1,), peak_band=8.0
    )
    assert inside.local_support > 0.25
    assert outside.local_support < 0.10
    # And the giveaway: the extrapolated query reports a NARROWER band, not a wider one.
    assert (outside.path[0].p90 - outside.path[0].p10) <= (
        inside.path[0].p90 - inside.path[0].p10
    ) * 1.5
