"""Tests for the Phase 2 schema migration on an existing Phase 1 DB."""

from __future__ import annotations

import duckdb

from fantasy_baseball.streaks.data.migrate import migrate_to_phase_2
from fantasy_baseball.streaks.data.schema import get_connection


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


def test_migrate_to_phase_3_resets_labels_and_keeps_other_tables() -> None:
    """`migrate_to_phase_3` drops hitter_streak_labels and recreates it with
    the new PK, but does NOT touch hitter_games / hitter_windows / thresholds.
    """
    from fantasy_baseball.streaks.data.migrate import migrate_to_phase_3

    conn = get_connection(":memory:")
    # Seed something in hitter_games so we can assert it survives.
    conn.execute(
        "INSERT INTO hitter_games VALUES (1, 100, 'X', 'TEAM', 2025, '2025-04-01', "
        "4, 4, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, true)"
    )
    # Seed an old-shape label row to confirm it gets cleared.
    conn.execute(
        "INSERT INTO hitter_streak_labels (player_id, window_end, window_days, category, "
        "cold_method, label) VALUES (1, '2025-04-08', 7, 'hr', 'empirical', 'cold')"
    )

    migrate_to_phase_3(conn)

    # Labels are wiped...
    n_labels = conn.execute("SELECT COUNT(*) FROM hitter_streak_labels").fetchone()[0]
    assert n_labels == 0
    # ...but games are not.
    n_games = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    assert n_games == 1


def test_migrate_to_phase_3_is_idempotent() -> None:
    from fantasy_baseball.streaks.data.migrate import migrate_to_phase_3

    conn = get_connection(":memory:")
    migrate_to_phase_3(conn)
    migrate_to_phase_3(conn)  # second call must not raise
    info = conn.execute("PRAGMA table_info('hitter_streak_labels')").fetchall()
    pk_cols = [r[1] for r in info if r[5]]
    assert "cold_method" in pk_cols
