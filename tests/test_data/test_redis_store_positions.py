"""Tests for positions helpers in redis_store."""

from fantasy_baseball.data import redis_store


def test_get_positions_returns_empty_dict_when_unset(fake_redis):
    assert redis_store.get_positions(fake_redis) == {}


def test_set_and_get_positions_roundtrip(fake_redis):
    positions = {
        "11100": ["SS", "2B"],
        "11101": ["SP"],
        "11102": ["OF", "1B", "Util"],
    }
    redis_store.set_positions(fake_redis, positions)
    assert redis_store.get_positions(fake_redis) == positions


def test_set_positions_overwrites_previous(fake_redis):
    redis_store.set_positions(fake_redis, {"1": ["C"]})
    redis_store.set_positions(fake_redis, {"2": ["SP"]})
    assert redis_store.get_positions(fake_redis) == {"2": ["SP"]}


def test_get_positions_ignores_corrupt_json(fake_redis):
    fake_redis.set("positions", "not valid json {{{")
    assert redis_store.get_positions(fake_redis) == {}


def test_get_positions_returns_empty_when_client_none():
    assert redis_store.get_positions(None) == {}


def test_set_positions_is_noop_when_client_none():
    # Should not raise.
    redis_store.set_positions(None, {"1": ["SP"]})
