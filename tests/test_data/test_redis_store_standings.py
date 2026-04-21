"""Tests for standings_history helpers."""

from datetime import date

import pytest

from fantasy_baseball.data import redis_store
from fantasy_baseball.models.standings import (
    CategoryStats,
    Standings,
    StandingsEntry,
)


def _standings(eff: date, teams: list[tuple[str, str, int, dict, float | None]]) -> Standings:
    return Standings(
        effective_date=eff,
        entries=[
            StandingsEntry(
                team_name=name,
                team_key=team_key,
                rank=rank,
                stats=CategoryStats.from_dict(stats),
                yahoo_points_for=pf,
            )
            for name, team_key, rank, stats, pf in teams
        ],
    )


STANDINGS_DAY_1 = _standings(
    date(2026, 4, 15),
    [
        (
            "Alpha",
            "431.l.1.t.1",
            1,
            {
                "R": 45,
                "HR": 12,
                "RBI": 40,
                "SB": 8,
                "AVG": 0.268,
                "W": 3,
                "K": 85,
                "SV": 4,
                "ERA": 3.21,
                "WHIP": 1.14,
            },
            78.5,
        ),
        (
            "Beta",
            "431.l.1.t.2",
            2,
            {
                "R": 38,
                "HR": 9,
                "RBI": 32,
                "SB": 6,
                "AVG": 0.255,
                "W": 2,
                "K": 72,
                "SV": 3,
                "ERA": 3.85,
                "WHIP": 1.22,
            },
            60.0,
        ),
    ],
)

STANDINGS_DAY_2 = _standings(
    date(2026, 4, 22),
    [
        (
            "Alpha",
            "431.l.1.t.1",
            1,
            {
                "R": 60,
                "HR": 16,
                "RBI": 55,
                "SB": 10,
                "AVG": 0.272,
                "W": 5,
                "K": 110,
                "SV": 5,
                "ERA": 3.05,
                "WHIP": 1.10,
            },
            82.0,
        ),
    ],
)


def test_write_and_read_single_day(fake_redis):
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)
    loaded = redis_store.get_standings_day(fake_redis, "2026-04-15")
    assert loaded == STANDINGS_DAY_1


def test_write_standings_snapshot_overwrites_same_date(fake_redis):
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)
    same_date_new_content = _standings(
        date(2026, 4, 15),
        [("Alpha", "431.l.1.t.1", 1, {"R": 99}, 99.0)],
    )
    redis_store.write_standings_snapshot(fake_redis, same_date_new_content)
    loaded = redis_store.get_standings_day(fake_redis, "2026-04-15")
    assert loaded == same_date_new_content


def test_get_latest_standings_picks_max_date(fake_redis):
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_2)
    latest = redis_store.get_latest_standings(fake_redis)
    assert latest == STANDINGS_DAY_2


def test_get_standings_history_returns_all_dates(fake_redis):
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_2)
    history = redis_store.get_standings_history(fake_redis)
    assert set(history.keys()) == {"2026-04-15", "2026-04-22"}
    assert history["2026-04-22"] == STANDINGS_DAY_2


def test_get_standings_history_empty(fake_redis):
    assert redis_store.get_standings_history(fake_redis) == {}


def test_write_standings_snapshot_none_client_noop():
    redis_store.write_standings_snapshot(None, STANDINGS_DAY_1)


def test_get_latest_standings_none_client_returns_none():
    assert redis_store.get_latest_standings(None) is None


def test_get_standings_day_none_client_returns_none():
    assert redis_store.get_standings_day(None, "2026-04-15") is None


def test_get_standings_history_none_client_returns_empty():
    assert redis_store.get_standings_history(None) == {}


def test_get_standings_day_ignores_corrupt_json(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", "not json {{{")
    assert redis_store.get_standings_day(fake_redis, "2026-04-15") is None


def test_get_standings_history_raises_on_legacy_shape(fake_redis):
    import json

    legacy = {"teams": [{"team": "Alpha", "r": 10}]}
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", json.dumps(legacy))
    with pytest.raises(ValueError):
        redis_store.get_standings_day(fake_redis, "2026-04-15")
