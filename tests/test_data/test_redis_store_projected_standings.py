"""Tests for projected_standings_history helpers."""

from datetime import date

from fantasy_baseball.data import redis_store
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
)


def _projected(eff: date, teams: list[tuple[str, dict]]) -> ProjectedStandings:
    return ProjectedStandings(
        effective_date=eff,
        entries=[
            ProjectedStandingsEntry(team_name=name, stats=CategoryStats.from_dict(stats))
            for name, stats in teams
        ],
    )


PROJ_DAY_1 = _projected(
    date(2026, 4, 15),
    [
        (
            "Alpha",
            {
                "R": 880,
                "HR": 230,
                "RBI": 820,
                "SB": 110,
                "AVG": 0.265,
                "W": 75,
                "K": 1450,
                "SV": 60,
                "ERA": 3.55,
                "WHIP": 1.20,
            },
        ),
        (
            "Beta",
            {
                "R": 820,
                "HR": 200,
                "RBI": 780,
                "SB": 90,
                "AVG": 0.258,
                "W": 70,
                "K": 1380,
                "SV": 55,
                "ERA": 3.78,
                "WHIP": 1.25,
            },
        ),
    ],
)

PROJ_DAY_2 = _projected(
    date(2026, 4, 22),
    [
        (
            "Alpha",
            {
                "R": 890,
                "HR": 235,
                "RBI": 830,
                "SB": 112,
                "AVG": 0.266,
                "W": 76,
                "K": 1455,
                "SV": 62,
                "ERA": 3.50,
                "WHIP": 1.19,
            },
        ),
    ],
)


def test_write_and_read_single_day(fake_redis):
    redis_store.write_projected_standings_snapshot(fake_redis, PROJ_DAY_1)
    loaded = redis_store.get_projected_standings_day(fake_redis, "2026-04-15")
    assert loaded == PROJ_DAY_1


def test_overwrites_same_date(fake_redis):
    redis_store.write_projected_standings_snapshot(fake_redis, PROJ_DAY_1)
    same_date_new = _projected(
        date(2026, 4, 15),
        [("Alpha", {"R": 999})],
    )
    redis_store.write_projected_standings_snapshot(fake_redis, same_date_new)
    loaded = redis_store.get_projected_standings_day(fake_redis, "2026-04-15")
    assert loaded == same_date_new


def test_get_history_returns_all_dates(fake_redis):
    redis_store.write_projected_standings_snapshot(fake_redis, PROJ_DAY_1)
    redis_store.write_projected_standings_snapshot(fake_redis, PROJ_DAY_2)
    history = redis_store.get_projected_standings_history(fake_redis)
    assert set(history.keys()) == {"2026-04-15", "2026-04-22"}
    assert history["2026-04-22"] == PROJ_DAY_2


def test_get_history_empty(fake_redis):
    assert redis_store.get_projected_standings_history(fake_redis) == {}


def test_write_none_client_noop():
    redis_store.write_projected_standings_snapshot(None, PROJ_DAY_1)


def test_get_day_none_client_returns_none():
    assert redis_store.get_projected_standings_day(None, "2026-04-15") is None


def test_get_history_none_client_returns_empty():
    assert redis_store.get_projected_standings_history(None) == {}


def test_get_day_ignores_corrupt_json(fake_redis):
    fake_redis.hset(redis_store.PROJECTED_STANDINGS_HISTORY_KEY, "2026-04-15", "not json {{{")
    assert redis_store.get_projected_standings_day(fake_redis, "2026-04-15") is None
