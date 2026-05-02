"""End-to-end test for backfill_projected_standings_history.

Seeds fakeredis with weekly_rosters_history + ros_projections, runs the
script's main entry point, asserts projected_standings_history is
populated with one snapshot per roster date.
"""

import json
from unittest.mock import patch

import pytest

from fantasy_baseball.data import redis_store


@pytest.fixture
def seeded_redis(fake_redis, monkeypatch):
    """Populate fakeredis with 2 dates x 2 teams of roster data and
    minimal blended/ROS projections."""
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key

    rosters = {
        "2026-04-01": [
            {
                "team": "Alpha",
                "player_name": "Player One",
                "slot": "OF",
                "positions": "OF",
                "status": "",
                "yahoo_id": "p1",
            },
            {
                "team": "Beta",
                "player_name": "Player Two",
                "slot": "SP",
                "positions": "SP",
                "status": "",
                "yahoo_id": "p2",
            },
        ],
        "2026-04-15": [
            {
                "team": "Alpha",
                "player_name": "Player One",
                "slot": "OF",
                "positions": "OF",
                "status": "",
                "yahoo_id": "p1",
            },
            {
                "team": "Beta",
                "player_name": "Player Two",
                "slot": "SP",
                "positions": "SP",
                "status": "",
                "yahoo_id": "p2",
            },
        ],
    }
    for snap_date, entries in rosters.items():
        fake_redis.hset(
            redis_store.WEEKLY_ROSTERS_HISTORY_KEY,
            snap_date,
            json.dumps(entries),
        )

    # Two corresponding standings snapshots with non-empty team_keys
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-01",
        json.dumps(
            {
                "effective_date": "2026-04-01",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {
                            "R": 0,
                            "HR": 0,
                            "RBI": 0,
                            "SB": 0,
                            "AVG": 0,
                            "W": 0,
                            "K": 0,
                            "SV": 0,
                            "ERA": 99,
                            "WHIP": 99,
                        },
                        "yahoo_points_for": None,
                        "extras": {},
                    },
                    {
                        "name": "Beta",
                        "team_key": "T.2",
                        "rank": 2,
                        "stats": {
                            "R": 0,
                            "HR": 0,
                            "RBI": 0,
                            "SB": 0,
                            "AVG": 0,
                            "W": 0,
                            "K": 0,
                            "SV": 0,
                            "ERA": 99,
                            "WHIP": 99,
                        },
                        "yahoo_points_for": None,
                        "extras": {},
                    },
                ],
            }
        ),
    )

    hitter_row = {
        "name": "Player One",
        "fg_id": "fg1",
        "team": "TBD",
        "positions": "OF",
        "ab": 500,
        "pa": 580,
        "r": 80,
        "hr": 25,
        "rbi": 80,
        "sb": 8,
        "h": 145,
        "avg": 0.290,
        "player_type": "hitter",
    }
    pitcher_row = {
        "name": "Player Two",
        "fg_id": "fg2",
        "team": "TBD",
        "positions": "SP",
        "w": 12,
        "k": 180,
        "sv": 0,
        "ip": 180.0,
        "er": 70,
        "bb": 50,
        "h_allowed": 160,
        "era": 3.50,
        "whip": 1.17,
        "player_type": "pitcher",
    }

    fake_redis.set("blended_projections:hitters", json.dumps([hitter_row]))
    fake_redis.set("blended_projections:pitchers", json.dumps([pitcher_row]))
    fake_redis.set(
        redis_key(CacheKey.ROS_PROJECTIONS),
        json.dumps({"hitters": [hitter_row], "pitchers": [pitcher_row]}),
    )

    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")

    return fake_redis


def test_backfill_writes_snapshot_per_date(seeded_redis):
    from scripts import backfill_projected_standings_history as backfill

    with patch("fantasy_baseball.data.kv_store.get_kv", return_value=seeded_redis):
        backfill.main(season_year=2026)

    history = redis_store.get_projected_standings_history(seeded_redis)
    assert set(history.keys()) == {"2026-04-01", "2026-04-15"}
    for snap_date, projected in history.items():
        names = {e.team_name for e in projected.entries}
        assert names == {"Alpha", "Beta"}
        assert projected.effective_date.isoformat() == snap_date


def test_backfill_is_idempotent(seeded_redis):
    from scripts import backfill_projected_standings_history as backfill

    with patch("fantasy_baseball.data.kv_store.get_kv", return_value=seeded_redis):
        backfill.main(season_year=2026)
        first = redis_store.get_projected_standings_history(seeded_redis)
        backfill.main(season_year=2026)
        second = redis_store.get_projected_standings_history(seeded_redis)

    assert first.keys() == second.keys()
    for k in first:
        assert first[k] == second[k]
