import pytest
from fantasy_baseball.analysis.transactions import score_transaction
from fantasy_baseball.data import redis_store
from fantasy_baseball.data.db import create_tables, get_connection
from fantasy_baseball.models.league import League


@pytest.fixture
def redis_league(fake_redis, monkeypatch):
    """Redirect ``redis_store.get_default_client()`` to the fake client
    so ``League.from_redis`` reads test data seeded via
    ``write_standings_snapshot``.
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


def _seed_projections(conn, snapshot_date="2026-03-30"):
    conn.executemany(
        "INSERT INTO ros_blended_projections "
        "(year, snapshot_date, fg_id, name, team, player_type, "
        "pa, ab, h, r, hr, rbi, sb, avg, w, k, sv, ip, er, bb, h_allowed, era, whip, adp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, snapshot_date, "1", "Otto Lopez", "TOR", "hitter",
             500, 450, 130, 65, 12, 55, 15, .289, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100),
            (2026, snapshot_date, "2", "Marcus Semien", "TEX", "hitter",
             600, 550, 140, 80, 20, 70, 10, .255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50),
        ],
    )
    conn.commit()


class TestScoreTransaction:
    def test_returns_add_and_drop_wsgp(self, redis_league):
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_standings(redis_league)
        _seed_projections(conn)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Otto Lopez",
            "add_positions": "2B, SS",
            "drop_name": "Marcus Semien",
            "drop_positions": "2B, SS",
        }
        result = score_transaction(_league_from(redis_league), conn, txn, 2026)
        assert "add_wsgp" in result
        assert "drop_wsgp" in result
        assert "value" in result
        assert result["value"] == pytest.approx(
            result["add_wsgp"] - result["drop_wsgp"], abs=0.02
        )
        conn.close()

    def test_add_only_has_zero_drop_wsgp(self, redis_league):
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_standings(redis_league)
        _seed_projections(conn)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Otto Lopez",
            "add_positions": "2B, SS",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(_league_from(redis_league), conn, txn, 2026)
        assert result["drop_wsgp"] == 0.0
        assert result["add_wsgp"] > 0
        conn.close()

    def test_drop_only_has_zero_add_wsgp(self, redis_league):
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_standings(redis_league)
        _seed_projections(conn)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": None,
            "add_positions": None,
            "drop_name": "Marcus Semien",
            "drop_positions": "2B, SS",
        }
        result = score_transaction(_league_from(redis_league), conn, txn, 2026)
        assert result["add_wsgp"] == 0.0
        assert result["drop_wsgp"] > 0
        assert result["value"] < 0
        conn.close()

    def test_unmatched_player_gets_zero_wsgp(self, redis_league):
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_standings(redis_league)
        _seed_projections(conn)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Unknown Player",
            "add_positions": "OF",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(_league_from(redis_league), conn, txn, 2026)
        assert result["add_wsgp"] == 0.0
        conn.close()
