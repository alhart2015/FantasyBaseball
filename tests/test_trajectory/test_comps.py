from __future__ import annotations

import pandas as pd
import pytest

from fantasy_baseball.trajectory.comps import comp_trajectory


def _panel(rows: list[tuple[int, int, int, float]]) -> pd.DataFrame:
    """(mlbam_id, season, age, sgp) rows."""
    return pd.DataFrame(rows, columns=["mlbam_id", "season", "age", "sgp"])


def _career(mlbam_id: int, start_season: int, start_age: int, sgps: list[float]):
    return [
        (mlbam_id, start_season + i, start_age + i, sgp)
        for i, sgp in enumerate(sgps)
        if sgp is not None
    ]


def test_matches_only_within_the_band_and_at_the_right_age() -> None:
    panel = _panel(
        _career(1, 2010, 25, [13.0, 9.0])  # in band
        + _career(2, 2010, 25, [20.0, 18.0])  # out of band
        + _career(3, 2010, 24, [13.0, 9.0])  # wrong age
    )
    traj = comp_trajectory(panel, kind="hitter", age=25, sgp=13.0, band=2.5, horizons=(1,))
    assert traj.n_comps == 1
    assert list(traj.comps["mlbam_id"]) == [1]


def test_a_player_who_leaves_the_league_scores_zero() -> None:
    # He is worth zero to a roster slot. Dropping him instead is the single largest
    # bias available here.
    panel = _panel(_career(1, 2010, 25, [13.0, 10.0]) + _career(2, 2010, 25, [13.0]))
    traj = comp_trajectory(panel, kind="hitter", age=25, sgp=13.0, horizons=(1,))
    point = traj.path[0]
    assert point.mean == pytest.approx(5.0)  # (10 + 0) / 2
    assert point.survivors == 1
    assert point.n == 2
    assert point.mean_if_survived == pytest.approx(10.0)
    assert point.survival == pytest.approx(0.5)


def test_a_gap_year_scores_zero_but_the_player_can_return() -> None:
    panel = _panel(_career(1, 2010, 25, [13.0, None, 8.0]))
    traj = comp_trajectory(panel, kind="hitter", age=25, sgp=13.0, horizons=(1, 2))
    assert traj.path[0].mean == pytest.approx(0.0)
    assert traj.path[1].mean == pytest.approx(8.0)


def test_comps_whose_horizon_is_not_yet_observable_are_dropped_not_zeroed() -> None:
    # A 2024 age-25 season has no age-26 to look at yet. Zero-filling it would score
    # "has not happened yet" as "career over".
    panel = _panel(_career(1, 2010, 25, [13.0, 10.0]) + _career(2, 2024, 25, [13.0]))
    traj = comp_trajectory(
        panel, kind="hitter", age=25, sgp=13.0, horizons=(1,), last_complete_season=2024
    )
    assert traj.n_comps == 1
    assert traj.path[0].mean == pytest.approx(10.0)


def test_a_horizon_landing_exactly_on_the_last_complete_season_is_observable() -> None:
    # 2024 + 1 == 2025, a finished season. He is absent from it because he did not
    # play, not because we cannot see it yet -- so he scores 0 rather than dropping out.
    panel = _panel(_career(1, 2024, 25, [13.0]))
    traj = comp_trajectory(
        panel, kind="hitter", age=25, sgp=13.0, horizons=(1,), last_complete_season=2025
    )
    assert traj.n_comps == 1
    assert traj.path[0].mean == pytest.approx(0.0)
    assert traj.path[0].survivors == 0


def test_observability_is_judged_per_horizon_not_by_the_longest() -> None:
    # This comp's h1 (2021) is observable and its h2 (2022) is not. It must count
    # toward h1 and be absent from h2 -- judging it by the longest horizon would
    # discard a real year-one observation.
    panel = _panel(_career(1, 2020, 25, [13.0, 10.0, 9.0]))
    traj = comp_trajectory(
        panel, kind="hitter", age=25, sgp=13.0, horizons=(1, 2), last_complete_season=2021
    )
    assert traj.n_comps == 1
    assert traj.path[0].n == 1
    assert traj.path[0].mean == pytest.approx(10.0)
    assert traj.path[1].n == 0  # 2022 has not been played


def test_a_comp_with_no_observable_horizon_at_all_is_excluded() -> None:
    panel = _panel(_career(1, 2021, 25, [13.0, 10.0]))
    traj = comp_trajectory(
        panel, kind="hitter", age=25, sgp=13.0, horizons=(1, 2), last_complete_season=2021
    )
    assert traj.n_comps == 0


def test_n_falls_as_the_horizon_grows() -> None:
    # The module docstring's claim, asserted: an older comp supports both horizons, a
    # newer one only the first.
    panel = _panel(_career(1, 2018, 25, [13.0, 10.0, 8.0]) + _career(2, 2020, 25, [13.0, 9.0]))
    traj = comp_trajectory(
        panel, kind="hitter", age=25, sgp=13.0, horizons=(1, 2), last_complete_season=2021
    )
    assert [p.n for p in traj.path] == [2, 1]


def test_reports_the_age_at_each_horizon() -> None:
    panel = _panel(_career(1, 2010, 25, [13.0, 10.0, 9.0]))
    traj = comp_trajectory(panel, kind="hitter", age=25, sgp=13.0, horizons=(1, 2))
    assert [p.age for p in traj.path] == [26, 27]


def test_total_sums_the_expected_path() -> None:
    panel = _panel(_career(1, 2010, 25, [13.0, 10.0, 9.0]))
    traj = comp_trajectory(panel, kind="hitter", age=25, sgp=13.0, horizons=(1, 2))
    assert traj.total == pytest.approx(19.0)
    assert len(traj.observable) == 2


def test_an_unobservable_horizon_does_not_poison_the_total() -> None:
    # h2 is beyond the panel, so it carries no estimate. Summing it as NaN would
    # destroy the headline number for the year we CAN see.
    panel = _panel(_career(1, 2020, 25, [13.0, 10.0]))
    traj = comp_trajectory(
        panel, kind="hitter", age=25, sgp=13.0, horizons=(1, 2), last_complete_season=2021
    )
    assert traj.path[1].n == 0
    assert traj.total == pytest.approx(10.0)
    assert [p.horizon for p in traj.observable] == [1]


def test_mean_start_exposes_a_band_that_pulls_the_comps_low() -> None:
    # The SGP distribution thins at the top, so a wide band admits more players from
    # below the query than above it. The path then answers an easier question.
    panel = _panel(
        _career(1, 2010, 25, [10.0, 5.0])
        + _career(2, 2010, 25, [11.0, 5.0])
        + _career(3, 2010, 25, [13.0, 5.0])
    )
    traj = comp_trajectory(panel, kind="hitter", age=25, sgp=13.0, band=4.0, horizons=(1,))
    assert traj.n_comps == 3
    assert traj.mean_start == pytest.approx(11.333, abs=1e-3)


def test_a_split_season_is_summed_not_silently_truncated() -> None:
    # Two rows for one (player, season) -- a mid-season trade -- would otherwise make
    # the forward lookup return a Series instead of a float.
    panel = _panel([(1, 2010, 25, 13.0), (1, 2011, 26, 6.0), (1, 2011, 26, 4.0)])
    traj = comp_trajectory(panel, kind="hitter", age=25, sgp=13.0, horizons=(1,))
    assert traj.path[0].mean == pytest.approx(10.0)


def test_a_split_season_is_one_comp_not_two() -> None:
    # Collapsing only the forward lookup summed his future correctly while still
    # entering him into the cohort twice as two half-seasons -- inflating n, double
    # weighting him, and dragging mean_start below the band.
    panel = _panel([(1, 2010, 25, 8.0), (1, 2010, 25, 5.0), (1, 2011, 26, 9.0)])
    traj = comp_trajectory(panel, kind="hitter", age=25, sgp=13.0, band=2.5, horizons=(1,))
    assert traj.n_comps == 1
    assert traj.mean_start == pytest.approx(13.0)
    assert traj.path[0].mean == pytest.approx(9.0)


def test_no_comps_is_reported_rather_than_raising() -> None:
    traj = comp_trajectory(
        _panel(_career(1, 2010, 30, [13.0, 9.0])),
        kind="hitter",
        age=25,
        sgp=13.0,
        horizons=(1,),
    )
    assert traj.n_comps == 0
    assert traj.seasons is None
    assert traj.path[0].n == 0


def test_standard_error_shrinks_as_comps_are_added() -> None:
    few = _panel(_career(1, 2010, 25, [13.0, 4.0]) + _career(2, 2010, 25, [13.0, 16.0]))
    many = _panel(
        [row for i in range(1, 21) for row in _career(i, 2010, 25, [13.0, 4.0 if i % 2 else 16.0])]
    )
    a = comp_trajectory(few, kind="hitter", age=25, sgp=13.0, horizons=(1,)).path[0]
    b = comp_trajectory(many, kind="hitter", age=25, sgp=13.0, horizons=(1,)).path[0]
    assert b.se < a.se


def test_the_bootstrap_is_reproducible() -> None:
    panel = _panel(
        [row for i in range(1, 21) for row in _career(i, 2010, 25, [13.0, 4.0 if i % 2 else 16.0])]
    )
    kwargs = {"kind": "hitter", "age": 25, "sgp": 13.0, "horizons": (1,)}
    assert (
        comp_trajectory(panel, **kwargs).path[0].se == comp_trajectory(panel, **kwargs).path[0].se
    )


@pytest.mark.parametrize("band", [0.0, -1.0])
def test_rejects_a_non_positive_band(band: float) -> None:
    with pytest.raises(ValueError, match="band"):
        comp_trajectory(
            _panel(_career(1, 2010, 25, [13.0])), kind="hitter", age=25, sgp=13.0, band=band
        )


def test_rejects_an_empty_horizon() -> None:
    with pytest.raises(ValueError, match="horizons"):
        comp_trajectory(
            _panel(_career(1, 2010, 25, [13.0])), kind="hitter", age=25, sgp=13.0, horizons=()
        )
