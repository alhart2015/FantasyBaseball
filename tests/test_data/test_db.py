import sqlite3
from fantasy_baseball.data.db import create_tables, get_connection, DB_PATH


def test_create_tables_creates_all_five(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor]
    assert "raw_projections" in tables
    assert "blended_projections" in tables
    assert "draft_results" in tables
    assert "weekly_rosters" in tables
    assert "standings" in tables
    conn.close()


def test_create_tables_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    create_tables(conn)  # should not raise
    conn.close()


def test_get_connection_returns_connection():
    conn = get_connection(":memory:")
    assert isinstance(conn, sqlite3.Connection)
    conn.close()
