"""Tests for streaks-specific game log parsing and per-season fetch."""

from datetime import date
from unittest.mock import Mock, patch

from fantasy_baseball.streaks.data.game_logs import (
    fetch_hitter_season_game_logs,
    parse_hitter_game_log_full,
)
from fantasy_baseball.streaks.models import HitterGame


def _split(date="2024-04-01", **stat_overrides):
    stat = {
        "plateAppearances": 4,
        "atBats": 3,
        "hits": 1,
        "homeRuns": 1,
        "runs": 1,
        "rbi": 2,
        "stolenBases": 0,
        "baseOnBalls": 1,
        "strikeOuts": 1,
    }
    stat.update(stat_overrides)
    return {"date": date, "stat": stat}


def test_parse_hitter_game_log_full_extracts_all_columns():
    row = parse_hitter_game_log_full(
        _split(),
        player_id=660271,
        name="Mike Trout",
        team="LAA",
        season=2024,
    )
    assert row == HitterGame(
        player_id=660271,
        name="Mike Trout",
        team="LAA",
        season=2024,
        date=date(2024, 4, 1),
        pa=4,
        ab=3,
        h=1,
        hr=1,
        r=1,
        rbi=2,
        sb=0,
        bb=1,
        k=1,
    )


def test_parse_hitter_game_log_full_defaults_missing_stats_to_zero():
    row = parse_hitter_game_log_full(
        {"date": "2024-04-01", "stat": {}},
        player_id=1,
        name="X",
        team=None,
        season=2024,
    )
    assert row.pa == 0
    assert row.bb == 0


def test_fetch_hitter_season_game_logs_returns_one_row_per_split():
    fake_resp = Mock()
    fake_resp.raise_for_status = Mock()
    fake_resp.json = Mock(
        return_value={
            "stats": [
                {
                    "splits": [
                        _split(date="2024-04-01"),
                        _split(date="2024-04-02", homeRuns=0),
                    ]
                }
            ]
        }
    )
    with patch("fantasy_baseball.streaks.data.game_logs.requests.get", return_value=fake_resp):
        rows = fetch_hitter_season_game_logs(
            player_id=660271, name="Mike Trout", team="LAA", season=2024
        )
    assert len(rows) == 2
    assert rows[0].date == date(2024, 4, 1)
    assert rows[0].hr == 1
    assert rows[1].hr == 0
    assert all(r.player_id == 660271 for r in rows)
    assert all(r.season == 2024 for r in rows)


def test_fetch_hitter_season_game_logs_handles_empty_splits():
    fake_resp = Mock()
    fake_resp.raise_for_status = Mock()
    fake_resp.json = Mock(return_value={"stats": [{"splits": []}]})
    with patch("fantasy_baseball.streaks.data.game_logs.requests.get", return_value=fake_resp):
        rows = fetch_hitter_season_game_logs(player_id=1, name="X", team=None, season=2024)
    assert rows == []
