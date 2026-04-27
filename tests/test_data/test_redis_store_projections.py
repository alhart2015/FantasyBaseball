"""Tests for blended_projections helpers."""

from fantasy_baseball.data import redis_store

HITTER_ROW = {
    "fg_id": "sa3001523",
    "name": "Juan Soto",
    "team": "NYY",
    "player_type": "hitter",
    "pa": 650.0,
    "ab": 540.0,
    "h": 148.0,
    "r": 110.0,
    "hr": 34.0,
    "rbi": 100.0,
    "sb": 8.0,
    "avg": 0.274,
}
PITCHER_ROW = {
    "fg_id": "sa3001524",
    "name": "Gerrit Cole",
    "team": "NYY",
    "player_type": "pitcher",
    "ip": 200.0,
    "er": 68.0,
    "bb": 50.0,
    "h_allowed": 170.0,
    "w": 15.0,
    "k": 230.0,
    "sv": 0.0,
    "era": 3.06,
    "whip": 1.10,
}


def test_get_blended_projections_empty(fake_redis):
    assert redis_store.get_blended_projections(fake_redis, "hitters") == []
    assert redis_store.get_blended_projections(fake_redis, "pitchers") == []


def test_set_and_get_blended_hitters(fake_redis):
    redis_store.set_blended_projections(fake_redis, "hitters", [HITTER_ROW])
    result = redis_store.get_blended_projections(fake_redis, "hitters")
    assert result == [HITTER_ROW]


def test_set_and_get_blended_pitchers(fake_redis):
    redis_store.set_blended_projections(fake_redis, "pitchers", [PITCHER_ROW])
    result = redis_store.get_blended_projections(fake_redis, "pitchers")
    assert result == [PITCHER_ROW]


def test_set_blended_rejects_bad_type(fake_redis):
    import pytest

    with pytest.raises(ValueError, match="player_type must be"):
        redis_store.set_blended_projections(fake_redis, "goalies", [])


def test_get_blended_projections_ignores_corrupt_json(fake_redis):
    fake_redis.set("blended_projections:hitters", "not valid json {{{")
    assert redis_store.get_blended_projections(fake_redis, "hitters") == []


def test_get_blended_projections_ignores_non_list_json(fake_redis):
    fake_redis.set("blended_projections:hitters", '{"unexpected": "object"}')
    assert redis_store.get_blended_projections(fake_redis, "hitters") == []


def test_get_blended_projections_returns_empty_when_client_none():
    assert redis_store.get_blended_projections(None, "hitters") == []
    assert redis_store.get_blended_projections(None, "pitchers") == []


# --- get_ros_projections ---------------------------------------------------


def test_get_ros_projections_returns_none_when_unset(fake_redis):
    assert redis_store.get_ros_projections(fake_redis) is None


def test_get_ros_projections_returns_none_when_client_none():
    assert redis_store.get_ros_projections(None) is None


def test_get_ros_projections_reads_payload(fake_redis):
    import json

    payload = {"hitters": [HITTER_ROW], "pitchers": [PITCHER_ROW]}
    fake_redis.set("cache:ros_projections", json.dumps(payload))
    assert redis_store.get_ros_projections(fake_redis) == payload


def test_get_ros_projections_returns_none_on_corrupt_json(fake_redis, caplog):
    import logging

    fake_redis.set("cache:ros_projections", "not valid json {{{")
    with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.redis_store"):
        assert redis_store.get_ros_projections(fake_redis) is None
    assert any("Corrupt JSON" in rec.message for rec in caplog.records)


def test_get_ros_projections_returns_none_on_non_dict_json(fake_redis):
    fake_redis.set("cache:ros_projections", '["unexpected", "list"]')
    assert redis_store.get_ros_projections(fake_redis) is None


# --- get_full_season_projections / set_full_season_projections ---------------


def test_full_season_projections_round_trip(tmp_path):
    from fantasy_baseball.data import redis_store as rs
    from fantasy_baseball.data.kv_store import SqliteKVStore

    kv = SqliteKVStore(tmp_path / "kv.db")
    rs.set_full_season_projections(
        kv,
        {
            "hitters": [{"name": "A", "r": 100}],
            "pitchers": [{"name": "B", "k": 200}],
        },
    )
    got = rs.get_full_season_projections(kv)
    assert got == {"hitters": [{"name": "A", "r": 100}], "pitchers": [{"name": "B", "k": 200}]}


def test_full_season_projections_missing_returns_none(tmp_path):
    from fantasy_baseball.data import redis_store as rs
    from fantasy_baseball.data.kv_store import SqliteKVStore

    kv = SqliteKVStore(tmp_path / "kv.db")
    assert rs.get_full_season_projections(kv) is None
