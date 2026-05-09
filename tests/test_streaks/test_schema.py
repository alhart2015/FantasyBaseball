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
        "hitter_projection_rates",
        "continuation_rates",
    }


def test_init_schema_is_idempotent():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    init_schema(conn)  # should not raise
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert len(tables) == 7


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


def test_hitter_games_has_new_phase_2_columns() -> None:
    conn = get_connection(":memory:")
    rows = conn.execute("PRAGMA table_info('hitter_games')").fetchall()
    cols = {r[1]: r[2] for r in rows}  # name -> type
    assert cols["b2"] == "INTEGER"
    assert cols["b3"] == "INTEGER"
    assert cols["sf"] == "INTEGER"
    assert cols["hbp"] == "INTEGER"
    assert cols["ibb"] == "INTEGER"
    assert cols["cs"] == "INTEGER"
    assert cols["gidp"] == "INTEGER"
    assert cols["sh"] == "INTEGER"
    assert cols["ci"] == "INTEGER"
    assert cols["is_home"] == "BOOLEAN"


def test_hitter_statcast_pa_has_new_phase_2_columns() -> None:
    conn = get_connection(":memory:")
    rows = conn.execute("PRAGMA table_info('hitter_statcast_pa')").fetchall()
    cols = {r[1]: r[2] for r in rows}
    assert cols["at_bat_number"] == "INTEGER"
    assert cols["bb_type"] == "VARCHAR"
    assert cols["estimated_ba_using_speedangle"] == "DOUBLE"
    assert cols["hit_distance_sc"] == "DOUBLE"


def test_hitter_streak_labels_has_cold_method_pk() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('hitter_streak_labels')").fetchall()
    cols = {r[1] for r in info}
    assert "cold_method" in cols, f"expected cold_method in {cols}"
    pk_cols = [r[1] for r in info if r[5]]  # column 5 is the pk position
    assert pk_cols == [
        "player_id",
        "window_end",
        "window_days",
        "category",
        "cold_method",
    ]


def test_hitter_projection_rates_table_exists() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('hitter_projection_rates')").fetchall()
    cols = [r[1] for r in info]
    assert cols == ["player_id", "season", "hr_per_pa", "sb_per_pa", "n_systems"]
    pk_cols = [r[1] for r in info if r[5]]
    assert pk_cols == ["player_id", "season"]


def test_continuation_rates_table_exists() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('continuation_rates')").fetchall()
    cols = {r[1] for r in info}
    expected_cols = {
        "season_set",
        "category",
        "window_days",
        "pt_bucket",
        "strength_bucket",
        "direction",
        "cold_method",
        "n_labeled",
        "n_continued",
        "p_continued",
        "p_baserate",
        "lift",
    }
    assert expected_cols.issubset(cols)
