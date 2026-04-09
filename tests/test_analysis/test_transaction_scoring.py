import pytest
from fantasy_baseball.analysis.transactions import score_transaction
from fantasy_baseball.data.db import create_tables, get_connection


def _seed_standings(conn, snapshot_date="2026-03-31"):
    conn.executemany(
        "INSERT INTO standings "
        "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, snapshot_date, "Team A", 1, 30, 8, 25, 3, .280, 3, 40, 2, 3.20, 1.10),
            (2026, snapshot_date, "Team B", 2, 25, 6, 20, 5, .260, 2, 35, 4, 3.80, 1.25),
        ],
    )
    conn.commit()


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
    def test_returns_add_and_drop_wsgp(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_standings(conn)
        _seed_projections(conn)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Otto Lopez",
            "add_positions": "2B, SS",
            "drop_name": "Marcus Semien",
            "drop_positions": "2B, SS",
        }
        result = score_transaction(conn, txn, 2026)
        assert "add_wsgp" in result
        assert "drop_wsgp" in result
        assert "value" in result
        assert result["value"] == pytest.approx(
            result["add_wsgp"] - result["drop_wsgp"], abs=0.02
        )
        conn.close()

    def test_add_only_has_zero_drop_wsgp(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_standings(conn)
        _seed_projections(conn)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Otto Lopez",
            "add_positions": "2B, SS",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(conn, txn, 2026)
        assert result["drop_wsgp"] == 0.0
        assert result["add_wsgp"] > 0
        conn.close()

    def test_drop_only_has_zero_add_wsgp(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_standings(conn)
        _seed_projections(conn)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": None,
            "add_positions": None,
            "drop_name": "Marcus Semien",
            "drop_positions": "2B, SS",
        }
        result = score_transaction(conn, txn, 2026)
        assert result["add_wsgp"] == 0.0
        assert result["drop_wsgp"] > 0
        assert result["value"] < 0
        conn.close()

    def test_unmatched_player_gets_zero_wsgp(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_standings(conn)
        _seed_projections(conn)

        txn = {
            "team": "Team A",
            "timestamp": "1775059200",
            "add_name": "Unknown Player",
            "add_positions": "OF",
            "drop_name": None,
            "drop_positions": None,
        }
        result = score_transaction(conn, txn, 2026)
        assert result["add_wsgp"] == 0.0
        conn.close()
