from datetime import date

import pandas as pd
import pytest

from fantasy_baseball.pitch_data import store
from fantasy_baseball.pitch_data.store import (
    connect,
    fetch_lineups_season,
    fetch_pitches_season,
    fetch_sprint_speed_season,
    final_game_pks_by_date,
    is_settled,
    lineup_path,
    lineup_rows,
    pitch_chunk_path,
    season_window,
    sprint_speed_path,
)


def _game(pk, d, game_type="R", state="Final", coded="F"):
    return {
        "gamePk": pk,
        "officialDate": d,
        "gameType": game_type,
        "status": {"abstractGameState": state, "codedGameState": coded},
    }


def _pitches(pks, d="2025-04-01"):
    return pd.DataFrame(
        {
            "game_pk": pks,
            "game_date": [d] * len(pks),
            "game_type": ["R"] * len(pks),
            "type": ["X"] * len(pks),
            "launch_speed": [100.0] * len(pks),
        }
    )


def test_is_settled_needs_a_margin_after_end():
    today = date(2025, 5, 10)
    assert is_settled(date(2025, 5, 7), today)
    assert not is_settled(date(2025, 5, 8), today)
    assert not is_settled(date(2025, 5, 10), today)


def test_season_window_ignores_non_regular_games():
    games = [
        _game(1, "2025-02-25", game_type="S"),
        _game(2, "2025-03-27"),
        _game(3, "2025-09-28"),
        _game(4, "2025-10-20", game_type="W"),
    ]
    assert season_window(games) == (date(2025, 3, 27), date(2025, 9, 28))


def test_final_game_pks_drops_unfinished_and_suspended_games():
    games = [
        _game(1, "2025-04-01"),
        _game(2, "2025-04-01", state="Preview"),
        _game(3, "2025-04-01"),  # suspended ...
        _game(3, "2025-04-02"),  # ... and resumed under the same gamePk
        _game(4, "2025-04-02"),
    ]
    assert final_game_pks_by_date(games) == {
        date(2025, 4, 1): {1},
        date(2025, 4, 2): {4},
    }


def test_cancelled_and_postponed_games_are_not_expected():
    # The schedule marks both "Final" at the abstract level; neither has pitches.
    games = [
        _game(1, "2025-04-01"),
        _game(2, "2025-04-01", coded="C"),  # cancelled
        _game(3, "2025-04-01", coded="D"),  # postponed ...
        _game(3, "2025-04-03"),  # ... and made up
    ]
    assert final_game_pks_by_date(games) == {
        date(2025, 4, 1): {1},
        date(2025, 4, 3): {3},
    }


def test_pitches_retry_a_failed_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RETRY_SLEEP_SECONDS", 0)
    games = [_game(10, "2025-04-01")]
    attempts = []

    def flaky(start, end):
        attempts.append(start)
        if len(attempts) == 1:
            raise ValueError("Error tokenizing data")
        return _pitches([10])

    result = fetch_pitches_season(tmp_path, 2025, date(2025, 6, 1), games=games, fetch=flaky)
    assert result["written"] == 1
    assert len(attempts) == 2


def test_pitches_that_always_fail_count_as_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RETRY_SLEEP_SECONDS", 0)
    games = [_game(10, "2025-04-01")]

    def broken(start, end):
        raise ValueError("Error tokenizing data")

    result = fetch_pitches_season(tmp_path, 2025, date(2025, 6, 1), games=games, fetch=broken)
    assert result == {"written": 0, "skipped": 0, "incomplete": 1, "rows": 0}


def test_pitches_writes_complete_chunks_and_skips_settled_ones(tmp_path):
    games = [_game(10, "2025-04-01"), _game(11, "2025-04-09")]
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        pks = [10] if start == date(2025, 4, 1) else [11]
        return _pitches(pks, start.isoformat())

    today = date(2025, 6, 1)
    first = fetch_pitches_season(tmp_path, 2025, today, games=games, fetch=fetch)
    assert first["written"] == 2
    assert pitch_chunk_path(tmp_path, 2025, date(2025, 4, 1), date(2025, 4, 7)).exists()
    assert pitch_chunk_path(tmp_path, 2025, date(2025, 4, 8), date(2025, 4, 9)).exists()

    calls.clear()
    second = fetch_pitches_season(tmp_path, 2025, today, games=games, fetch=fetch)
    assert second == {"written": 0, "skipped": 2, "incomplete": 0, "rows": 0}
    assert calls == []


def test_pitches_refetches_a_chunk_that_is_not_settled(tmp_path):
    games = [_game(10, "2025-04-01")]
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        return _pitches([10])

    today = date(2025, 4, 2)
    fetch_pitches_season(tmp_path, 2025, today, games=games, fetch=fetch)
    fetch_pitches_season(tmp_path, 2025, today, games=games, fetch=fetch)
    assert len(calls) == 2


def test_pitches_short_chunk_is_not_written(tmp_path):
    games = [_game(10, "2025-04-01"), _game(12, "2025-04-02")]

    def fetch(start, end):
        return _pitches([10])  # game 12 missing from Savant's answer

    result = fetch_pitches_season(tmp_path, 2025, date(2025, 6, 1), games=games, fetch=fetch)
    assert result["incomplete"] == 1
    assert result["written"] == 0
    assert not list(tmp_path.rglob("*.parquet"))


def test_pitches_empty_answer_for_scheduled_games_is_incomplete(tmp_path):
    games = [_game(10, "2025-04-01")]
    result = fetch_pitches_season(
        tmp_path, 2025, date(2025, 6, 1), games=games, fetch=lambda s, e: pd.DataFrame()
    )
    assert result["incomplete"] == 1


def test_lineup_rows_parses_batting_order():
    box = {
        "teams": {
            "home": {
                "team": {"id": 144},
                "players": {
                    "ID1": {
                        "person": {"id": 1},
                        "battingOrder": "300",
                        "position": {"abbreviation": "2B"},
                    },
                    "ID2": {
                        "person": {"id": 2},
                        "battingOrder": "301",
                        "position": {"abbreviation": "PH"},
                    },
                    "ID3": {"person": {"id": 3}, "position": {"abbreviation": "P"}},
                },
            },
            "away": {"team": {"id": 136}, "players": {}},
        }
    }
    rows = lineup_rows(box, 99, "2025-04-01")
    assert [(r["player_id"], r["lineup_spot"], r["sub_index"]) for r in rows] == [
        (1, 3, 0),
        (2, 3, 1),
    ]
    assert all(r["team_id"] == 144 and r["is_home"] for r in rows)


def _box(player_id):
    return {
        "teams": {
            "home": {
                "team": {"id": 1},
                "players": {"x": {"person": {"id": player_id}, "battingOrder": "100"}},
            },
            "away": {"team": {"id": 2}, "players": {}},
        }
    }


def test_lineups_fetch_each_game_once_and_skip_when_final(tmp_path):
    games = [
        _game(5, "2025-04-01"),
        _game(5, "2025-04-02"),  # suspended game listed twice
        _game(6, "2025-09-28"),
        _game(7, "2025-09-28", state="Preview"),
        _game(8, "2025-09-28", coded="C"),  # cancelled: no box score to fetch
    ]
    fetched = []

    def fetch_boxscore(pk):
        fetched.append(pk)
        return _box(pk * 10)

    today = date(2025, 10, 15)
    n = fetch_lineups_season(tmp_path, 2025, today, games=games, fetch_boxscore=fetch_boxscore)
    assert n == 2
    assert sorted(fetched) == [5, 6]
    df = pd.read_parquet(lineup_path(tmp_path, 2025))
    assert df.set_index("game_pk").loc[5, "game_date"] == "2025-04-02"

    assert (
        fetch_lineups_season(tmp_path, 2025, today, games=games, fetch_boxscore=fetch_boxscore)
        is None
    )


def test_lineups_failure_writes_nothing(tmp_path):
    games = [_game(5, "2025-04-01")]

    def boom(pk):
        raise RuntimeError("api down")

    with pytest.raises(RuntimeError):
        fetch_lineups_season(tmp_path, 2025, date(2025, 10, 15), games=games, fetch_boxscore=boom)
    assert not lineup_path(tmp_path, 2025).exists()


def test_sprint_speed_adds_season_and_skips_when_final(tmp_path):
    games = [_game(1, "2025-09-28")]
    calls = []

    def fetch(season):
        calls.append(season)
        return pd.DataFrame({"player_id": [1, 2], "sprint_speed": [28.1, 30.2]})

    today = date(2025, 11, 1)
    assert fetch_sprint_speed_season(tmp_path, 2025, today, games=games, fetch=fetch) == 2
    assert set(pd.read_parquet(sprint_speed_path(tmp_path, 2025))["season"]) == {2025}
    assert fetch_sprint_speed_season(tmp_path, 2025, today, games=games, fetch=fetch) is None
    assert calls == [2025]


def test_sprint_speed_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        fetch_sprint_speed_season(
            tmp_path, 2025, date(2025, 11, 1), games=[], fetch=lambda s: pd.DataFrame()
        )


def test_connect_stacks_seasons_with_drifting_columns(tmp_path):
    old = _pitches([1])
    new = _pitches([2]).assign(bat_speed=[72.5])
    for season, df in ((2016, old), (2024, new)):
        path = pitch_chunk_path(tmp_path, season, date(season, 4, 1), date(season, 4, 7))
        path.parent.mkdir(parents=True)
        df.to_parquet(path, index=False)

    conn = connect(tmp_path)
    rows = conn.execute("SELECT season, bat_speed FROM pitches ORDER BY season").fetchall()
    assert rows == [(2016, None), (2024, 72.5)]
    views = {r[0] for r in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()}
    assert "lineups" not in views
