"""Tests for preseason_baseline:{year} helpers."""

from fantasy_baseball.data import redis_store

BASELINE = {
    "base": {"team_results": {"Team 01": {"median_pts": 72.5}}, "category_risk": {}},
    "with_management": {"team_results": {"Team 01": {"median_pts": 75.0}}, "category_risk": {}},
    "meta": {
        "frozen_at": "2026-04-18T12:00:00Z",
        "season_year": 2026,
        "roster_date": "2026-03-27",
        "projections_source": "blended",
    },
}


def test_get_preseason_baseline_empty(fake_redis):
    assert redis_store.get_preseason_baseline(fake_redis, 2026) is None


def test_set_and_get_round_trip(fake_redis):
    redis_store.set_preseason_baseline(fake_redis, 2026, BASELINE)
    result = redis_store.get_preseason_baseline(fake_redis, 2026)
    assert result == BASELINE


def test_different_seasons_isolated(fake_redis):
    redis_store.set_preseason_baseline(fake_redis, 2026, BASELINE)
    assert redis_store.get_preseason_baseline(fake_redis, 2025) is None


def test_get_returns_none_on_corrupt_json(fake_redis):
    fake_redis.set("preseason_baseline:2026", "not valid json {{{")
    assert redis_store.get_preseason_baseline(fake_redis, 2026) is None


def test_get_returns_none_on_non_dict_payload(fake_redis):
    import json

    fake_redis.set("preseason_baseline:2026", json.dumps(["not", "a", "dict"]))
    assert redis_store.get_preseason_baseline(fake_redis, 2026) is None


def test_get_returns_none_when_client_none():
    assert redis_store.get_preseason_baseline(None, 2026) is None


def test_set_noop_when_client_none():
    # Should not raise
    redis_store.set_preseason_baseline(None, 2026, BASELINE)
