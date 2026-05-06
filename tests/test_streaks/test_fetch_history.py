"""Tests for the fetch_season orchestrator."""

from datetime import date
from unittest.mock import patch

import duckdb
import pytest

from fantasy_baseball.streaks.data.fetch_history import fetch_season
from fantasy_baseball.streaks.data.schema import init_schema


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _stub_qualified():
    return [
        {"player_id": 660271, "name": "Mike Trout", "team": "LAA", "pa": 162},
        {"player_id": 545361, "name": "Other Hitter", "team": "BOS", "pa": 200},
    ]


def _stub_game_logs(player_id, name, team, season, **_kwargs):
    return [
        {
            "player_id": player_id,
            "name": name,
            "team": team,
            "season": season,
            "date": "2024-04-01",
            "pa": 4,
            "ab": 3,
            "h": 1,
            "hr": 1,
            "r": 1,
            "rbi": 2,
            "sb": 0,
            "bb": 1,
            "k": 1,
        }
    ]


def _stub_statcast(start, end, **_kwargs):
    return [
        {
            "player_id": 660271,
            "date": date(2024, 4, 1),
            "pa_index": 1,
            "event": "single",
            "launch_speed": 95.0,
            "launch_angle": 10.0,
            "estimated_woba_using_speedangle": 0.4,
            "barrel": False,
        }
    ]


def test_fetch_season_loads_game_logs_and_statcast(conn):
    with (
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_qualified_hitters",
            return_value=_stub_qualified(),
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_hitter_season_game_logs",
            side_effect=_stub_game_logs,
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_statcast_pa_for_date_range",
            side_effect=_stub_statcast,
        ),
    ):
        summary = fetch_season(season=2024, conn=conn)

    games = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    statcast = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert games == 2  # one row each for Trout and Other Hitter
    assert statcast == 1
    assert summary["players_fetched"] == 2
    assert summary["game_log_rows"] == 2
    assert summary["statcast_rows"] == 1


def test_fetch_season_skips_already_loaded_players(conn):
    # Pre-populate Trout
    conn.execute(
        """
        INSERT INTO hitter_games VALUES
        (660271, 'Mike Trout', 'LAA', 2024, '2024-03-28', 4, 3, 1, 1, 1, 2, 0, 1, 1)
        """
    )

    fetch_calls: list[int] = []

    def _record_calls(player_id, name, team, season, **_kwargs):
        fetch_calls.append(player_id)
        return _stub_game_logs(player_id, name, team, season)

    with (
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_qualified_hitters",
            return_value=_stub_qualified(),
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_hitter_season_game_logs",
            side_effect=_record_calls,
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_statcast_pa_for_date_range",
            side_effect=_stub_statcast,
        ),
    ):
        fetch_season(season=2024, conn=conn)

    # Only the second hitter should have been fetched (660271 was already loaded)
    assert fetch_calls == [545361]


def test_fetch_season_uses_correct_date_range_for_statcast(conn):
    captured: dict[str, date] = {}

    def _capture_dates(start, end, **_kwargs):
        captured["start"] = start
        captured["end"] = end
        return []

    with (
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_qualified_hitters",
            return_value=_stub_qualified(),
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_hitter_season_game_logs",
            side_effect=_stub_game_logs,
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_statcast_pa_for_date_range",
            side_effect=_capture_dates,
        ),
    ):
        fetch_season(season=2024, conn=conn)

    # Statcast range should span 3/15 .. 11/15 of the season year (covers all of MLB regular season + playoffs)
    assert captured["start"] == date(2024, 3, 15)
    assert captured["end"] == date(2024, 11, 15)


def test_fetch_season_continues_after_per_player_failure(conn):
    """If one player's game-log fetch raises, the loop should log and continue."""

    def _raises_for_trout(player_id, name, team, season, **_kwargs):
        if player_id == 660271:
            raise RuntimeError("simulated 404 from MLB Stats API")
        return _stub_game_logs(player_id, name, team, season)

    with (
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_qualified_hitters",
            return_value=_stub_qualified(),
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_hitter_season_game_logs",
            side_effect=_raises_for_trout,
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_statcast_pa_for_date_range",
            side_effect=_stub_statcast,
        ),
    ):
        summary = fetch_season(season=2024, conn=conn)

    # Other Hitter still loaded; Trout silently skipped via exception handler
    games = conn.execute("SELECT player_id FROM hitter_games").fetchall()
    player_ids = {row[0] for row in games}
    assert player_ids == {545361}
    assert summary["players_fetched"] == 2  # both attempted
    assert summary["game_log_rows"] == 1  # only one succeeded
