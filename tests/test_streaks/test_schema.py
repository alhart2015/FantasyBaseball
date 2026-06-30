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
        "model_fits",
    }


def test_init_schema_is_idempotent():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    init_schema(conn)  # should not raise
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert len(tables) == 8


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
    assert cols == [
        "player_id",
        "season",
        "hr_per_pa",
        "sb_per_pa",
        "r_per_pa",
        "rbi_per_pa",
        "avg",
        "n_systems",
    ]
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


def test_hitter_projection_rates_has_dense_cat_columns() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('hitter_projection_rates')").fetchall()
    cols = {r[1] for r in info}
    assert {"r_per_pa", "rbi_per_pa", "avg"}.issubset(cols)


def test_init_schema_heals_missing_additive_column() -> None:
    """An existing table that predates an additive column must get the column
    back-filled by init_schema -- not silently left without it.

    Regression for the schema-drift bug: the gitignored streaks.duckdb on a dev
    box was created before ``launch_speed_angle`` was added, and
    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so
    ``upsert_statcast_pa`` threw ``BinderException`` on every refresh.
    """
    conn = duckdb.connect(":memory:")
    # Pre-Phase-5 shape: hitter_statcast_pa without launch_speed_angle.
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
            at_bat_number INTEGER,
            bb_type VARCHAR,
            estimated_ba_using_speedangle DOUBLE,
            hit_distance_sc DOUBLE,
            PRIMARY KEY (player_id, date, pa_index)
        )
        """
    )
    conn.execute(
        "INSERT INTO hitter_statcast_pa VALUES "
        "(1, '2025-04-01', 1, 'single', 95.0, 10.0, 0.4, 1, 'line_drive', 0.4, 200.0)"
    )

    init_schema(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info('hitter_statcast_pa')").fetchall()}
    assert "launch_speed_angle" in cols
    # Existing row survives with NULL in the healed column.
    n = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert n == 1
    lsa = conn.execute("SELECT launch_speed_angle FROM hitter_statcast_pa").fetchone()[0]
    assert lsa is None


def test_init_schema_heals_all_additive_columns() -> None:
    """Every column added after a table's original CREATE is reconciled onto a
    pre-existing DB -- not just launch_speed_angle. Guards the whole registry.
    """
    conn = duckdb.connect(":memory:")
    # Original (pre-Phase-4) hitter_projection_rates: hr/sb only.
    conn.execute(
        """
        CREATE TABLE hitter_projection_rates (
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            hr_per_pa DOUBLE NOT NULL,
            sb_per_pa DOUBLE NOT NULL,
            n_systems INTEGER NOT NULL,
            PRIMARY KEY (player_id, season)
        )
        """
    )
    # Pre-Phase-B model_fits: audit columns only.
    conn.execute(
        """
        CREATE TABLE model_fits (
            model_id VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            direction VARCHAR NOT NULL,
            season_set VARCHAR NOT NULL,
            window_days INTEGER NOT NULL,
            cold_method VARCHAR NOT NULL,
            chosen_C DOUBLE NOT NULL,
            cv_auc_mean DOUBLE NOT NULL,
            cv_auc_std DOUBLE NOT NULL,
            val_auc DOUBLE NOT NULL,
            n_train_rows INTEGER NOT NULL,
            n_val_rows INTEGER NOT NULL,
            fit_timestamp TIMESTAMP NOT NULL,
            PRIMARY KEY (model_id)
        )
        """
    )

    init_schema(conn)

    rates_cols = {
        r[1] for r in conn.execute("PRAGMA table_info('hitter_projection_rates')").fetchall()
    }
    assert {"r_per_pa", "rbi_per_pa", "avg"}.issubset(rates_cols)
    fits_cols = {r[1] for r in conn.execute("PRAGMA table_info('model_fits')").fetchall()}
    assert {
        "feature_columns",
        "coef",
        "intercept",
        "scaler_mean",
        "scaler_scale",
        "dense_quintile_cutoffs",
    }.issubset(fits_cols)


def test_init_schema_leaves_unknown_columns_untouched() -> None:
    """The healer is strictly additive: a legacy column absent from _SCHEMA_DDL
    (e.g. the pre-Phase-5 ``barrel`` column) must survive init_schema untouched,
    not be dropped. Pins the 'never drops or retypes' guarantee so a future
    symmetric-diff refactor can't silently delete data and stay green.
    """
    conn = duckdb.connect(":memory:")
    conn.execute(
        "CREATE TABLE hitter_statcast_pa ("
        "player_id INTEGER, date DATE, pa_index INTEGER, barrel BOOLEAN, "
        "PRIMARY KEY (player_id, date, pa_index))"
    )

    init_schema(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info('hitter_statcast_pa')").fetchall()}
    assert "barrel" in cols  # legacy column NOT in _SCHEMA_DDL -- must be preserved
    assert "launch_speed_angle" in cols  # the genuinely-missing column is still healed


def test_heal_skips_not_null_additive_column(monkeypatch, caplog) -> None:
    """A NOT NULL additive column can't be ALTER-added to a populated table, so
    the healer must skip it with a warning -- not silently add it as nullable
    (which would diverge healed DBs from fresh ones).
    """
    from fantasy_baseball.streaks.data import schema

    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, PRIMARY KEY (id))")
    conn.execute("INSERT INTO t VALUES (1)")  # populated -> NOT NULL add is impossible
    monkeypatch.setattr(
        schema,
        "_intended_schema",
        lambda: {"t": [("id", "INTEGER", True), ("req", "INTEGER", True)]},
    )

    with caplog.at_level("WARNING"):
        schema._heal_additive_drift(conn)

    cols = {r[1] for r in conn.execute("PRAGMA table_info('t')").fetchall()}
    assert "req" not in cols  # skipped, not silently added as nullable
    assert any("req" in m for m in caplog.messages)


def test_model_fits_table_exists() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('model_fits')").fetchall()
    cols = {r[1] for r in info}
    expected_cols = {
        "model_id",
        "category",
        "direction",
        "season_set",
        "window_days",
        "cold_method",
        "chosen_C",
        "cv_auc_mean",
        "cv_auc_std",
        "val_auc",
        "n_train_rows",
        "n_val_rows",
        "fit_timestamp",
    }
    assert expected_cols.issubset(cols)
    pk_cols = [r[1] for r in info if r[5]]
    assert pk_cols == ["model_id"]
