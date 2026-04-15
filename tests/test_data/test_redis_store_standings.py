"""Tests for standings_history helpers."""
from fantasy_baseball.data import redis_store


STANDINGS_DAY_1 = {
    "teams": [
        {
            "team": "Alpha", "team_key": "431.l.1.t.1", "rank": 1,
            "r": 45, "hr": 12, "rbi": 40, "sb": 8, "avg": 0.268,
            "w": 3, "k": 85, "sv": 4, "era": 3.21, "whip": 1.14,
        },
        {
            "team": "Beta", "team_key": "431.l.1.t.2", "rank": 2,
            "r": 38, "hr": 9, "rbi": 32, "sb": 6, "avg": 0.255,
            "w": 2, "k": 72, "sv": 3, "era": 3.85, "whip": 1.22,
        },
    ],
}

STANDINGS_DAY_2 = {
    "teams": [
        {
            "team": "Alpha", "team_key": "431.l.1.t.1", "rank": 1,
            "r": 60, "hr": 16, "rbi": 55, "sb": 10, "avg": 0.272,
            "w": 5, "k": 110, "sv": 5, "era": 3.05, "whip": 1.10,
        },
    ],
}


def test_write_and_read_single_day(fake_redis):
    redis_store.write_standings_snapshot(
        fake_redis, "2026-04-15", STANDINGS_DAY_1
    )
    day = redis_store.get_standings_day(fake_redis, "2026-04-15")
    assert day == STANDINGS_DAY_1


def test_write_standings_snapshot_overwrites_same_date(fake_redis):
    redis_store.write_standings_snapshot(
        fake_redis, "2026-04-15", STANDINGS_DAY_1
    )
    redis_store.write_standings_snapshot(
        fake_redis, "2026-04-15", STANDINGS_DAY_2
    )
    day = redis_store.get_standings_day(fake_redis, "2026-04-15")
    assert day == STANDINGS_DAY_2


def test_get_latest_standings_picks_max_date(fake_redis):
    redis_store.write_standings_snapshot(
        fake_redis, "2026-04-08", STANDINGS_DAY_1
    )
    redis_store.write_standings_snapshot(
        fake_redis, "2026-04-15", STANDINGS_DAY_2
    )
    latest = redis_store.get_latest_standings(fake_redis)
    assert latest == STANDINGS_DAY_2


def test_get_standings_history_returns_all_dates(fake_redis):
    redis_store.write_standings_snapshot(
        fake_redis, "2026-04-08", STANDINGS_DAY_1
    )
    redis_store.write_standings_snapshot(
        fake_redis, "2026-04-15", STANDINGS_DAY_2
    )
    history = redis_store.get_standings_history(fake_redis)
    assert set(history.keys()) == {"2026-04-08", "2026-04-15"}
    assert history["2026-04-15"] == STANDINGS_DAY_2


def test_get_standings_history_empty(fake_redis):
    assert redis_store.get_standings_history(fake_redis) == {}


def test_write_standings_snapshot_none_client_noop():
    redis_store.write_standings_snapshot(None, "2026-04-15", STANDINGS_DAY_1)


def test_get_latest_standings_none_client_returns_empty():
    assert redis_store.get_latest_standings(None) == {}


def test_get_standings_day_none_client_returns_empty():
    assert redis_store.get_standings_day(None, "2026-04-15") == {}


def test_get_standings_history_none_client_returns_empty():
    assert redis_store.get_standings_history(None) == {}


def test_get_standings_day_ignores_corrupt_json(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", "not json {{{")
    assert redis_store.get_standings_day(fake_redis, "2026-04-15") == {}


def test_get_standings_history_skips_corrupt_entries(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-08", "not json {{{")
    redis_store.write_standings_snapshot(fake_redis, "2026-04-15", STANDINGS_DAY_1)
    hist = redis_store.get_standings_history(fake_redis)
    assert set(hist.keys()) == {"2026-04-15"}
