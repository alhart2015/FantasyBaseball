import json

import pytest
from fantasy_baseball.analysis.transactions import score_transaction
from fantasy_baseball.data import redis_store
from fantasy_baseball.models.league import League


@pytest.fixture
def redis_league(fake_redis, monkeypatch):
    """Redirect ``redis_store.get_default_client()`` to the fake client
    so ``League.from_redis`` (and ``score_transaction``'s Redis
    projection loader) reads test data seeded via the ``redis_store``
    helpers.
    """
    monkeypatch.setattr(redis_store, "_default_client", fake_redis)
    monkeypatch.setattr(redis_store, "_default_client_initialized", True)
    yield fake_redis


def _league_from(_client, year=2026):
    return League.from_redis(year)


def _seed_standings(client, snapshot_date="2026-03-31"):
    """Write a standings snapshot in the lowercase-keys shape produced
    by the refresh pipeline."""
    payload = {
        "teams": [
            {
                "team": "Team A", "team_key": "", "rank": 1,
                "r": 30, "hr": 8, "rbi": 25, "sb": 3, "avg": 0.280,
                "w": 3, "k": 40, "sv": 2, "era": 3.20, "whip": 1.10,
            },
            {
                "team": "Team B", "team_key": "", "rank": 2,
                "r": 25, "hr": 6, "rbi": 20, "sb": 5, "avg": 0.260,
                "w": 2, "k": 35, "sv": 4, "era": 3.80, "whip": 1.25,
            },
        ],
    }
    redis_store.write_standings_snapshot(client, snapshot_date, payload)


def _seed_projections(client):
    """Seed the ``cache:ros_projections`` Redis key with two hitters.

    Mirrors the shape written by
    ``fantasy_baseball.data.ros_pipeline.blend_and_cache_ros``:
    ``{"hitters": [row, ...], "pitchers": [row, ...]}`` where each row
    is a per-player dict of lowercase stat columns (matching the old
    ``ros_blended_projections`` SQLite schema).
    """
    hitters = [
        {
            "year": 2026, "snapshot_date": "2026-03-30", "fg_id": "1",
            "name": "Otto Lopez", "team": "TOR", "player_type": "hitter",
            "pa": 500, "ab": 450, "h": 130, "r": 65, "hr": 12, "rbi": 55,
            "sb": 15, "avg": 0.289,
            "w": 0, "k": 0, "sv": 0, "ip": 0, "er": 0, "bb": 0,
            "h_allowed": 0, "era": 0, "whip": 0, "adp": 100,
        },
        {
            "year": 2026, "snapshot_date": "2026-03-30", "fg_id": "2",
            "name": "Marcus Semien", "team": "TEX", "player_type": "hitter",
            "pa": 600, "ab": 550, "h": 140, "r": 80, "hr": 20, "rbi": 70,
            "sb": 10, "avg": 0.255,
            "w": 0, "k": 0, "sv": 0, "ip": 0, "er": 0, "bb": 0,
            "h_allowed": 0, "era": 0, "whip": 0, "adp": 50,
        },
    ]
    client.set(
        "cache:ros_projections",
        json.dumps({"hitters": hitters, "pitchers": []}),
    )


class TestScoreTransaction:
    def test_returns_add_and_drop_wsgp(self, redis_league):
        _seed_standings(redis_league)
        _seed_projections(redis_league)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Otto Lopez",
            "add_positions": "2B, SS",
            "drop_name": "Marcus Semien",
            "drop_positions": "2B, SS",
        }
        result = score_transaction(
            _league_from(redis_league), redis_league, txn, 2026,
        )
        assert "add_wsgp" in result
        assert "drop_wsgp" in result
        assert "value" in result
        assert result["value"] == pytest.approx(
            result["add_wsgp"] - result["drop_wsgp"], abs=0.02
        )

    def test_add_only_has_zero_drop_wsgp(self, redis_league):
        _seed_standings(redis_league)
        _seed_projections(redis_league)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Otto Lopez",
            "add_positions": "2B, SS",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(
            _league_from(redis_league), redis_league, txn, 2026,
        )
        assert result["drop_wsgp"] == 0.0
        assert result["add_wsgp"] > 0

    def test_drop_only_has_zero_add_wsgp(self, redis_league):
        _seed_standings(redis_league)
        _seed_projections(redis_league)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": None,
            "add_positions": None,
            "drop_name": "Marcus Semien",
            "drop_positions": "2B, SS",
        }
        result = score_transaction(
            _league_from(redis_league), redis_league, txn, 2026,
        )
        assert result["add_wsgp"] == 0.0
        assert result["drop_wsgp"] > 0
        assert result["value"] < 0

    def test_unmatched_player_gets_zero_wsgp(self, redis_league):
        _seed_standings(redis_league)
        _seed_projections(redis_league)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Unknown Player",
            "add_positions": "OF",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(
            _league_from(redis_league), redis_league, txn, 2026,
        )
        assert result["add_wsgp"] == 0.0
