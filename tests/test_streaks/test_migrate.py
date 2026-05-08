"""Tests for the Phase 2 schema migration on an existing Phase 1 DB."""

from __future__ import annotations

import duckdb

from fantasy_baseball.streaks.data.migrate import migrate_to_phase_2


def _phase_1_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Recreate the Phase 1 (pre-migration) DDL for testing."""
    conn.execute(
        """
        CREATE TABLE hitter_games (
            player_id INTEGER NOT NULL,
            game_pk INTEGER NOT NULL,
            name VARCHAR NOT NULL,
            team VARCHAR,
            season INTEGER NOT NULL,
            date DATE NOT NULL,
            pa INTEGER NOT NULL,
            ab INTEGER NOT NULL,
            h INTEGER NOT NULL,
            hr INTEGER NOT NULL,
            r INTEGER NOT NULL,
            rbi INTEGER NOT NULL,
            sb INTEGER NOT NULL,
            bb INTEGER NOT NULL,
            k INTEGER NOT NULL,
            PRIMARY KEY (player_id, game_pk)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE hitter_statcast_pa (
            player_id INTEGER NOT NULL,
            date DATE NOT NULL,
            pa_index INTEGER NOT NULL,
            event VARCHAR,
            launch_speed DOUBLE,
            launch_angle DOUBLE,
            estimated_woba_using_speedangle DOUBLE,
            barrel BOOLEAN,
            PRIMARY KEY (player_id, date, pa_index)
        )
        """
    )
    conn.execute(
        "INSERT INTO hitter_games VALUES (1, 1, 'X', 'ABC', 2025, '2025-04-01', "
        "4, 3, 1, 0, 0, 0, 0, 1, 1)"
    )


def test_migrate_adds_new_columns_to_hitter_games() -> None:
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info('hitter_games')").fetchall()}
    for col in ("b2", "b3", "sf", "hbp", "ibb", "cs", "gidp", "sh", "ci", "is_home"):
        assert col in cols, f"missing column {col}"


def test_migrate_adds_new_columns_to_hitter_statcast_pa() -> None:
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info('hitter_statcast_pa')").fetchall()}
    for col in ("at_bat_number", "bb_type", "estimated_ba_using_speedangle", "hit_distance_sc"):
        assert col in cols, f"missing column {col}"


def test_migrate_deletes_existing_rows_to_force_refetch() -> None:
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    assert conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0] == 0


def test_migrate_is_idempotent() -> None:
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    # Second call should not raise (DROP IF EXISTS + CREATE IF NOT EXISTS).
    migrate_to_phase_2(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info('hitter_games')").fetchall()}
    assert "b2" in cols  # still present, didn't blow up


def test_migrate_preserves_not_null_constraints() -> None:
    """Migrated DB must match a fresh DB's nullability — not silently more
    permissive. PRAGMA table_info column 3 is the ``notnull`` flag."""
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    games_notnull = {
        r[1]: bool(r[3]) for r in conn.execute("PRAGMA table_info('hitter_games')").fetchall()
    }
    for col in ("b2", "b3", "sf", "hbp", "ibb", "cs", "gidp", "sh", "ci", "is_home"):
        assert games_notnull[col], f"hitter_games.{col} should be NOT NULL"
    statcast_notnull = {
        r[1]: bool(r[3]) for r in conn.execute("PRAGMA table_info('hitter_statcast_pa')").fetchall()
    }
    for col in ("at_bat_number", "bb_type", "estimated_ba_using_speedangle", "hit_distance_sc"):
        assert not statcast_notnull[col], f"hitter_statcast_pa.{col} should be nullable"


def test_migrate_creates_downstream_tables() -> None:
    """After migration, hitter_windows / thresholds / hitter_streak_labels exist."""
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    for table in ("hitter_windows", "thresholds", "hitter_streak_labels"):
        # Should not raise CatalogException.
        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
