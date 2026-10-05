from datetime import date, timedelta
from itertools import pairwise

import pandas as pd
import pytest

from fantasy_baseball.pitch_data.milb import (
    MILB_LEVELS,
    fetch_milb_season,
    milb_season_path,
    milb_week_path,
    week_windows,
)
from fantasy_baseball.pitch_data.store import connect

FIRST, LAST = date(2025, 3, 27), date(2025, 4, 20)  # a short "season" keeps tests small
DONE = date(2025, 12, 1)  # long after the season: everything settled


def _split(player, pa, sport):
    return {
        "season": "2025",
        "player": {"id": player, "fullName": f"P{player}"},
        "stat": {"plateAppearances": pa, "hits": pa // 4},
        "league": {"id": 100 + sport},
        "sport": {"id": sport},
    }


class FakeApi:
    """Player 1 bats 10 PA a day in AAA (sportId 11) from opening day through ``LAST``;
    player 2 bats 5 PA a day in AA on April 1-10. Other levels have nobody. ``short``
    drops player 1 from one window, like a short page the API didn't flag."""

    def __init__(self, short=None):
        self.calls = []
        self.short = short

    def _days(self, start, end, first, last):
        lo, hi = max(start, first), min(end, last)
        return max((hi - lo).days + 1, 0)

    def __call__(self, params):
        self.calls.append(params)
        if params["stats"] == "season":
            start, end = date(2025, 1, 1), date(2025, 12, 31)
        else:
            start = date.fromisoformat(params["startDate"])
            end = date.fromisoformat(params["endDate"])
        out = []
        if params["sportId"] == 11 and (start, end) != self.short:
            days = self._days(start, end, FIRST, LAST)
            if days:
                out.append(_split(1, 10 * days, 11))
        if params["sportId"] == 12:
            days = self._days(start, end, date(2025, 4, 1), date(2025, 4, 10))
            if days:
                out.append(_split(2, 5 * days, 12))
        return out


def test_week_windows_tile_the_year_around_the_as_of_grid():
    windows = week_windows(FIRST, LAST, 7)
    assert windows[0] == (date(2025, 1, 1), date(2025, 3, 26))
    assert windows[-1] == (date(2025, 4, 17), date(2025, 12, 31))
    # One window per as-of date, each starting on it: a table row's "season so far" is
    # exactly the windows that end before its as-of date.
    assert [w[0] for w in windows[1:]] == [FIRST + timedelta(days=7 * k) for k in range(4)]
    for (_, end), (start, _) in pairwise(windows):
        assert start == end + timedelta(days=1)  # no gaps, no overlaps


def test_fetch_writes_season_and_weekly_lines(tmp_path):
    api = FakeApi()
    result = fetch_milb_season(tmp_path, 2025, FIRST, LAST, step_days=7, today=DONE, fetch=api)
    windows = week_windows(FIRST, LAST, 7)
    assert result == {"written": 1 + len(windows), "final": 0, "failed": 0}
    season = pd.read_parquet(milb_season_path(tmp_path, 2025))
    assert set(season["sport_id"]) == {11, 12}
    assert season["season"].tolist() == [2025, 2025]  # the API's string season, as an int
    assert season.set_index("player.id")["stat.plateAppearances"].to_dict() == {1: 250, 2: 50}
    # Every level is asked for, the season and each window.
    assert {c["sportId"] for c in api.calls} == set(MILB_LEVELS)
    ranged = [c for c in api.calls if c["stats"] == "byDateRange"]
    assert len(ranged) == len(windows) * len(MILB_LEVELS)
    # The pre-season window has no games but is still written, so it can become final.
    pre = pd.read_parquet(milb_week_path(tmp_path, 2025, windows[0]))
    assert pre.empty and "player.id" in pre.columns

    conn = connect(tmp_path)
    total = conn.execute('SELECT sum("stat.plateAppearances") FROM milb_weekly').fetchone()[0]
    assert total == 300
    assert conn.execute("SELECT DISTINCT season FROM milb_season").fetchall() == [(2025,)]


def test_a_second_run_skips_final_files(tmp_path):
    fetch_milb_season(tmp_path, 2025, FIRST, LAST, step_days=7, today=DONE, fetch=FakeApi())
    api = FakeApi()
    result = fetch_milb_season(tmp_path, 2025, FIRST, LAST, step_days=7, today=DONE, fetch=api)
    assert result["written"] == 0 and api.calls == []


def test_only_windows_that_have_started_are_fetched(tmp_path):
    today = date(2025, 4, 5)
    api = FakeApi()
    result = fetch_milb_season(tmp_path, 2025, FIRST, LAST, step_days=7, today=today, fetch=api)
    # Season file, the pre-season window, and the windows from Mar 27 and Apr 3.
    assert result["written"] == 4
    assert not milb_week_path(tmp_path, 2025, (date(2025, 4, 10), date(2025, 4, 16))).exists()


def _game_log(missing_day=None):
    """Player 1's AAA game log: one 10-PA game a day, opening day through ``LAST``."""

    def log(player, season, sport_id):
        assert (player, season, sport_id) == (1, 2025, 11)
        days = [FIRST + timedelta(days=k) for k in range((LAST - FIRST).days + 1)]
        return [
            {"date": d.isoformat(), "stat": {"plateAppearances": 10, "hits": 2, "age": 24}}
            for d in days
            if d != missing_day
        ]

    return log


def test_a_window_the_api_dropped_is_rebuilt_from_the_game_log(tmp_path):
    windows = week_windows(FIRST, LAST, 7)
    fetch_milb_season(
        tmp_path,
        2025,
        FIRST,
        LAST,
        step_days=7,
        today=DONE,
        fetch=FakeApi(short=windows[2]),
        game_log=_game_log(),
    )
    weekly = connect(tmp_path).execute("SELECT * FROM milb_weekly").df()
    mine = weekly[weekly["player.id"] == 1]
    # Every window of player 1 now comes from his game log: 7 games x 10 PA in a full one.
    assert mine["from_game_log"].all()
    assert mine.groupby("window_start")["stat.plateAppearances"].sum().tolist() == [70, 70, 70, 40]
    assert mine["stat.hits"].sum() == 50
    # Columns that aren't counts come from his season line, not a sum of games.
    assert set(mine["league.id"]) == {111} and "stat.age" not in mine.columns
    # Player 2 was fine and keeps his API rows.
    assert not weekly.loc[weekly["player.id"] == 2, "from_game_log"].any()
    assert weekly["stat.plateAppearances"].sum() == 300


def test_weekly_lines_that_still_dont_add_up_fail_the_fetch(tmp_path):
    short = week_windows(FIRST, LAST, 7)[2]
    with pytest.raises(ValueError, match="don't add up"):
        fetch_milb_season(
            tmp_path,
            2025,
            FIRST,
            LAST,
            step_days=7,
            today=DONE,
            fetch=FakeApi(short=short),
            game_log=_game_log(missing_day=date(2025, 4, 12)),
        )


def test_no_minor_league_season_in_2020(tmp_path):
    api = FakeApi()
    result = fetch_milb_season(tmp_path, 2020, FIRST, LAST, step_days=7, today=DONE, fetch=api)
    assert result == {"written": 0, "final": 0, "failed": 0} and api.calls == []


def test_a_season_with_no_lines_at_all_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no lines"):
        fetch_milb_season(tmp_path, 2025, FIRST, LAST, step_days=7, today=DONE, fetch=lambda p: [])
