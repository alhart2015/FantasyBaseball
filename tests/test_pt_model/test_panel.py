"""Panel-shaping tests (#291).

The load-bearing assertions here are about ABSENCE: a missed season must never
arrive downstream as an observed zero, and the rows that exist must be the ones a
lagged-PA feature can legitimately read.
"""

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pt_model.panel import HITTER_PANEL_COLUMNS, build_hitter_panel


def _season(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    """Raw-shaped hitting frame: (mlbam_id, pa, games) -> API column names."""
    return pd.DataFrame(
        {
            "player.id": [r[0] for r in rows],
            "stat.plateAppearances": [r[1] for r in rows],
            "stat.gamesPlayed": [r[2] for r in rows],
            "stat.atBats": [r[1] * 0.9 for r in rows],
            "stat.hits": [r[1] * 0.25 for r in rows],
            "stat.runs": [r[1] * 0.13 for r in rows],
            "stat.homeRuns": [r[1] * 0.04 for r in rows],
            "stat.rbi": [r[1] * 0.12 for r in rows],
            "stat.stolenBases": [r[1] * 0.02 for r in rows],
        }
    )


def _people(rows: list[tuple[int, str, str, str]]) -> pd.DataFrame:
    """(id, birthDate, primaryPosition, mlbDebutDate)."""
    return pd.DataFrame(
        {
            "id": [r[0] for r in rows],
            "birthDate": [r[1] for r in rows],
            "primaryPosition.abbreviation": [r[2] for r in rows],
            "mlbDebutDate": [r[3] for r in rows],
        }
    )


PEOPLE = _people(
    [
        (1, "1990-03-01", "SS", "2011-04-05"),
        (2, "1988-08-15", "C", "2009-06-01"),
    ]
)


def test_missed_interior_season_is_nan_not_zero():
    """The whole point of the panel: an absent season is unobserved, not a zero."""
    seasons = {
        2011: _season([(1, 600, 150)]),
        2012: _season([]),  # player 1 missed the year entirely
        2013: _season([(1, 580, 145)]),
    }
    panel = build_hitter_panel(seasons, PEOPLE)

    gap = panel.loc[panel["season"] == 2012].iloc[0]
    assert not gap["observed"]
    assert pd.isna(gap["pa"])
    assert pd.isna(gap["games"])
    assert pd.isna(gap["hr_pa"])
    # The row must still exist -- it is what makes 2013's lag readable as "missed".
    assert list(panel["season"]) == [2011, 2012, 2013]
    assert (panel["pa"] == 0).sum() == 0


def test_span_stops_at_first_and_last_observed_season():
    """No pre-debut rows, and no trailing rows for a career that ended mid-window."""
    seasons = {
        2011: _season([(1, 600, 150)]),
        2012: _season([(1, 600, 150), (2, 400, 120)]),
        2013: _season([(2, 410, 121)]),
        2014: _season([(2, 420, 122)]),
    }
    panel = build_hitter_panel(seasons, PEOPLE)

    p1 = panel.loc[panel["mlbam_id"] == 1, "season"]
    p2 = panel.loc[panel["mlbam_id"] == 2, "season"]
    assert list(p1) == [2011, 2012]  # retired after 2012: no 2013/2014 rows
    assert list(p2) == [2012, 2013, 2014]  # debuted 2012 here: no 2011 row


def test_age_uses_the_june_30_convention():
    """Pinned against the API's own `stat.age`, which is age on June 30 (verified
    1000/1000 for 2015). A birthday just after the reference date must not round up."""
    people = _people(
        [
            (10, "1990-06-29", "1B", "2011-04-05"),  # birthday BEFORE Jun 30
            (11, "1990-06-30", "1B", "2011-04-05"),  # birthday ON Jun 30
            (12, "1990-07-01", "1B", "2011-04-05"),  # birthday AFTER Jun 30
        ]
    )
    seasons = {2015: _season([(10, 500, 140), (11, 500, 140), (12, 500, 140)])}
    panel = build_hitter_panel(seasons, people).set_index("mlbam_id")

    assert panel.loc[10, "age"] == 25
    assert panel.loc[11, "age"] == 25
    assert panel.loc[12, "age"] == 24


def test_unobserved_rows_still_carry_age_and_covariates():
    """Age is derived from the birth date, so the rows with no stat block -- the ones
    the model most needs to reason about -- are not missing their main covariate."""
    seasons = {
        2011: _season([(1, 600, 150)]),
        2012: _season([]),
        2013: _season([(1, 580, 145)]),
    }
    panel = build_hitter_panel(seasons, PEOPLE).set_index("season")

    assert panel.loc[2012, "age"] == 22
    assert panel.loc[2012, "primary_position"] == "SS"
    assert panel.loc[2012, "seasons_since_debut"] == 1


def test_history_truncated_flags_careers_older_than_the_window():
    """Player 2 debuted in 2009, before this window, so his earliest rows have prior
    seasons the panel cannot see. Player 1 debuted inside it."""
    seasons = {2011: _season([(1, 600, 150), (2, 400, 120)])}
    panel = build_hitter_panel(seasons, PEOPLE).set_index("mlbam_id")

    assert not panel.loc[1, "history_truncated"]
    assert panel.loc[2, "history_truncated"]


def test_partial_season_is_flagged_and_unknown_year_raises():
    seasons = {2025: _season([(1, 600, 150)]), 2026: _season([(1, 300, 75)])}
    panel = build_hitter_panel(seasons, PEOPLE, partial_seasons=[2026])
    assert list(panel.loc[panel["partial_season"], "season"]) == [2026]

    with pytest.raises(ValueError, match="partial_seasons not present"):
        build_hitter_panel(seasons, PEOPLE, partial_seasons=[2027])


def test_duplicate_player_in_one_season_raises():
    """Multi-team seasons arrive pre-aggregated. If that ever stops being true,
    keeping both rows would double the player's weight in every downstream fit."""
    seasons = {2011: _season([(1, 300, 80), (1, 300, 80)])}
    with pytest.raises(ValueError, match="duplicate mlbam ids"):
        build_hitter_panel(seasons, PEOPLE)


def test_player_missing_from_people_keeps_nan_covariates():
    seasons = {2011: _season([(1, 600, 150), (99, 500, 140)])}
    panel = build_hitter_panel(seasons, PEOPLE).set_index("mlbam_id")

    assert panel.loc[99, "observed"]
    assert panel.loc[99, "pa"] == 500
    assert pd.isna(panel.loc[99, "age"])
    assert pd.isna(panel.loc[99, "primary_position"])


def test_schema_and_ordering_are_stable():
    seasons = {
        2011: _season([(2, 400, 120)]),
        2012: _season([(1, 600, 150), (2, 410, 121)]),
    }
    panel = build_hitter_panel(seasons, PEOPLE)

    assert tuple(panel.columns) == HITTER_PANEL_COLUMNS
    assert list(zip(panel["mlbam_id"], panel["season"], strict=True)) == [
        (1, 2012),
        (2, 2011),
        (2, 2012),
    ]


def test_pitcher_batters_are_flagged_not_dropped():
    """Pre-2022 the hitting leaderboard is ~half pitchers taking a PA or two. Keeping
    them un-flagged would drag any PA fit toward zero; dropping them on position would
    also drop a two-way player's real hitting season. So: flag, and let the model
    choose."""
    people = _people(
        [
            (1, "1990-03-01", "SS", "2011-04-05"),
            (5, "1990-03-01", "P", "2011-04-05"),
            (6, "1994-07-05", "TWP", "2018-03-29"),  # two-way
        ]
    )
    seasons = {2011: _season([(1, 600, 150), (5, 2, 30), (6, 500, 140)])}
    panel = build_hitter_panel(seasons, people).set_index("mlbam_id")

    assert not panel.loc[1, "is_pitcher"]
    assert panel.loc[5, "is_pitcher"]
    assert not panel.loc[6, "is_pitcher"]  # two-way keeps his hitting season
    assert len(panel) == 3


def test_unknown_position_leaves_is_pitcher_nan():
    """A player absent from `people` is unknown, not known-to-be-a-hitter."""
    seasons = {2011: _season([(99, 500, 140)])}
    panel = build_hitter_panel(seasons, PEOPLE).set_index("mlbam_id")
    assert pd.isna(panel.loc[99, "is_pitcher"])


def test_shortened_season_carries_its_schedule():
    """2020 was 60 games. Without this column a model reads that year as every player
    in the league getting hurt at once."""
    seasons = {2019: _season([(1, 600, 150)]), 2020: _season([(1, 220, 55)])}
    panel = build_hitter_panel(seasons, PEOPLE).set_index("season")

    assert panel.loc[2019, "scheduled_games"] == 162
    assert panel.loc[2020, "scheduled_games"] == 60


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no season frames"):
        build_hitter_panel({}, PEOPLE)


def test_zero_pa_observed_row_is_distinct_from_absence():
    """A player who appeared but recorded 0 PA (pinch-runner, defensive sub) is an
    OBSERVATION of zero, and must not be conflated with a season that never happened."""
    seasons = {
        2011: _season([(1, 600, 150)]),
        2012: _season([(1, 0, 4)]),
        2013: _season([(1, 580, 145)]),
    }
    panel = build_hitter_panel(seasons, PEOPLE).set_index("season")

    assert panel.loc[2012, "observed"]
    assert panel.loc[2012, "pa"] == 0
    assert panel.loc[2012, "games"] == 4
    # Rates are undefined on a 0-PA denominator -- NaN, not 0.0.
    assert np.isnan(panel.loc[2012, "hr_pa"])
