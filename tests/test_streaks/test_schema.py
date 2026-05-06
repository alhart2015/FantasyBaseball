"""Tests for streaks DuckDB schema initialization."""

import duckdb

from fantasy_baseball.streaks.data.schema import get_connection, init_schema


def test_init_schema_creates_all_tables():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert tables == {
        "hitter_games",
        "hitter_statcast_pa",
        "hitter_windows",
        "thresholds",
        "hitter_streak_labels",
    }


def test_init_schema_is_idempotent():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    init_schema(conn)  # should not raise
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert len(tables) == 5


def test_get_connection_in_memory_string():
    conn = get_connection(":memory:")
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert "hitter_games" in tables
    conn.close()


def test_get_connection_creates_parent_dir(tmp_path):
    db = tmp_path / "nested" / "deeper" / "streaks.duckdb"
    conn = get_connection(db)
    assert db.parent.is_dir()
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert "hitter_games" in tables
    conn.close()
