"""Tests for streaks DuckDB schema initialization."""

import duckdb

from fantasy_baseball.streaks.data.schema import init_schema


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
