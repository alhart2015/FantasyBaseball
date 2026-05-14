import sqlite3

import pytest


@pytest.fixture
def conn():
    """Fresh in-memory SQLite connection."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def _columns(conn, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


class TestWeeklyRostersMigration:
    def test_fresh_db_has_status_and_yahoo_id(self, conn):
        from fantasy_baseball.data.db import create_tables

        create_tables(conn)
        cols = _columns(conn, "weekly_rosters")
        assert "status" in cols
        assert "yahoo_id" in cols

    def test_migration_is_idempotent_on_existing_db(self, conn):
        from fantasy_baseball.data.db import create_tables

        # Simulate a pre-migration DB: create weekly_rosters with the old
        # column set only.
        conn.executescript("""
            CREATE TABLE weekly_rosters (
                snapshot_date TEXT NOT NULL,
                week_num     INTEGER,
                team         TEXT NOT NULL,
                slot         TEXT NOT NULL,
                player_name  TEXT NOT NULL,
                positions    TEXT,
                PRIMARY KEY (snapshot_date, team, slot, player_name)
            );
        """)
        conn.commit()

        # Pre-migration state
        assert "status" not in _columns(conn, "weekly_rosters")

        # Run migration twice — should succeed both times
        create_tables(conn)
        create_tables(conn)

        cols = _columns(conn, "weekly_rosters")
        assert "status" in cols
        assert "yahoo_id" in cols

    def test_migration_preserves_existing_rows(self, conn):
        from fantasy_baseball.data.db import create_tables

        conn.executescript("""
            CREATE TABLE weekly_rosters (
                snapshot_date TEXT NOT NULL,
                week_num     INTEGER,
                team         TEXT NOT NULL,
                slot         TEXT NOT NULL,
                player_name  TEXT NOT NULL,
                positions    TEXT,
                PRIMARY KEY (snapshot_date, team, slot, player_name)
            );
        """)
        conn.execute(
            "INSERT INTO weekly_rosters VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-04-07", 2, "Hart of the Order", "C", "Ivan Herrera", "C, Util"),
        )
        conn.commit()

        create_tables(conn)

        row = conn.execute("SELECT player_name, status, yahoo_id FROM weekly_rosters").fetchone()
        assert row["player_name"] == "Ivan Herrera"
        assert row["status"] is None
        assert row["yahoo_id"] is None
