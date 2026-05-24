"""Tests for game_log_totals + season_progress helpers."""

import json

import pytest

from fantasy_baseball.data import redis_store


def test_get_game_log_totals_empty(fake_redis):
    assert redis_store.get_game_log_totals(fake_redis, "hitters") == {}
    assert redis_store.get_game_log_totals(fake_redis, "pitchers") == {}


def test_set_and_get_game_log_totals_hitters(fake_redis):
    totals = {"660271": {"pa": 80, "ab": 68, "h": 20, "r": 15, "hr": 4, "rbi": 12, "sb": 3}}
    redis_store.set_game_log_totals(fake_redis, "hitters", totals)
    assert redis_store.get_game_log_totals(fake_redis, "hitters") == totals


def test_set_and_get_game_log_totals_pitchers(fake_redis):
    totals = {"594798": {"ip": 22.0, "er": 7, "bb": 5, "h_allowed": 16, "k": 28, "w": 2, "sv": 0}}
    redis_store.set_game_log_totals(fake_redis, "pitchers", totals)
    assert redis_store.get_game_log_totals(fake_redis, "pitchers") == totals


def test_set_game_log_totals_rejects_bad_type(fake_redis):
    with pytest.raises(ValueError, match="player_type must be"):
        redis_store.set_game_log_totals(fake_redis, "goalies", {})


def test_get_game_log_totals_ignores_corrupt_json(fake_redis):
    fake_redis.set("game_log_totals:hitters", "not valid json")
    assert redis_store.get_game_log_totals(fake_redis, "hitters") == {}


def test_get_game_log_totals_returns_empty_when_client_none():
    assert redis_store.get_game_log_totals(None, "hitters") == {}


def test_season_progress_empty(fake_redis):
    assert redis_store.get_season_progress(fake_redis) == {
        "games_elapsed": 0,
        "total": 162,
        "as_of": None,
    }


def test_set_and_get_season_progress(fake_redis):
    redis_store.set_season_progress(fake_redis, games_elapsed=18, total=162, as_of="2026-04-15")
    assert redis_store.get_season_progress(fake_redis) == {
        "games_elapsed": 18,
        "total": 162,
        "as_of": "2026-04-15",
    }


def test_set_season_progress_defaults(fake_redis):
    redis_store.set_season_progress(fake_redis, games_elapsed=5)
    assert redis_store.get_season_progress(fake_redis) == {
        "games_elapsed": 5,
        "total": 162,
        "as_of": None,
    }


def test_season_progress_ignores_corrupt_json(fake_redis):
    fake_redis.set("season_progress", "not valid")
    assert redis_store.get_season_progress(fake_redis) == {
        "games_elapsed": 0,
        "total": 162,
        "as_of": None,
    }


def test_get_season_progress_coerces_non_str_as_of_to_none(fake_redis):
    """Redis payloads with non-str, non-None as_of values coerce to None."""
    fake_redis.set(
        "season_progress",
        json.dumps({"games_elapsed": 10, "total": 162, "as_of": 12345}),
    )
    result = redis_store.get_season_progress(fake_redis)
    assert result["as_of"] is None
    assert result["games_elapsed"] == 10
    assert result["total"] == 162


def test_season_progress_returns_default_when_client_none():
    assert redis_store.get_season_progress(None) == {
        "games_elapsed": 0,
        "total": 162,
        "as_of": None,
    }


def test_set_season_progress_is_noop_when_client_none():
    # Must not raise
    assert redis_store.set_season_progress(None, games_elapsed=10) is None


def test_player_game_log_roundtrip(fake_redis):
    payload = {
        "name": "Shohei Ohtani",
        "games": [
            {
                "gamePk": 776213,
                "gameNumber": 1,
                "date": "2025-09-23",
                "pa": 4,
                "ab": 3,
                "h": 0,
                "r": 1,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
            }
        ],
    }
    redis_store.set_player_game_log(fake_redis, 2026, "660271", "hitting", payload)
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting") == payload


def test_player_game_log_missing_returns_none(fake_redis):
    assert redis_store.get_player_game_log(fake_redis, 2026, "999", "pitching") is None


def test_player_game_log_rejects_bad_group(fake_redis):
    with pytest.raises(ValueError, match="group must be"):
        redis_store.set_player_game_log(fake_redis, 2026, "1", "fielding", {})


def test_player_game_log_corrupt_returns_none(fake_redis):
    fake_redis.set("game_logs:2026:1:hitting", "not json")
    assert redis_store.get_player_game_log(fake_redis, 2026, "1", "hitting") is None


def test_watermark_roundtrip(fake_redis):
    assert redis_store.get_game_logs_watermark(fake_redis, 2026) is None
    redis_store.set_game_logs_watermark(fake_redis, 2026, "2026-05-24T13:00:00+00:00")
    assert redis_store.get_game_logs_watermark(fake_redis, 2026) == "2026-05-24T13:00:00+00:00"


def test_player_positions_roundtrip(fake_redis):
    assert redis_store.get_player_positions(fake_redis, 2026) == {}
    redis_store.set_player_positions(fake_redis, 2026, {"660271": "Y", "543037": "1"})
    assert redis_store.get_player_positions(fake_redis, 2026) == {"660271": "Y", "543037": "1"}


def test_game_log_dates_dedup_and_sort(fake_redis):
    redis_store.set_game_log_dates(fake_redis, 2026, ["2026-04-02", "2026-04-01", "2026-04-02"])
    assert redis_store.get_game_log_dates(fake_redis, 2026) == ["2026-04-01", "2026-04-02"]


def test_game_log_helpers_noop_on_none_client():
    assert redis_store.get_player_game_log(None, 2026, "1", "hitting") is None
    assert redis_store.set_player_game_log(None, 2026, "1", "hitting", {}) is None
    assert redis_store.get_game_logs_watermark(None, 2026) is None
    assert redis_store.get_player_positions(None, 2026) == {}
    assert redis_store.get_game_log_dates(None, 2026) == []
