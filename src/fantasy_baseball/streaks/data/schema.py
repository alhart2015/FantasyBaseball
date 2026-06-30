"""DuckDB schema for the streaks analysis project.

All DDL is `CREATE TABLE IF NOT EXISTS` so init_schema is idempotent and
safe to call on every connection open.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/streaks/streaks.duckdb")

_SCHEMA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS hitter_games (
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
        b2 INTEGER NOT NULL,
        b3 INTEGER NOT NULL,
        sf INTEGER NOT NULL,
        hbp INTEGER NOT NULL,
        ibb INTEGER NOT NULL,
        cs INTEGER NOT NULL,
        gidp INTEGER NOT NULL,
        sh INTEGER NOT NULL,
        ci INTEGER NOT NULL,
        is_home BOOLEAN NOT NULL,
        PRIMARY KEY (player_id, game_pk)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_statcast_pa (
        player_id INTEGER NOT NULL,
        date DATE NOT NULL,
        pa_index INTEGER NOT NULL,
        event VARCHAR,
        launch_speed DOUBLE,
        launch_angle DOUBLE,
        estimated_woba_using_speedangle DOUBLE,
        launch_speed_angle INTEGER,
        at_bat_number INTEGER,
        bb_type VARCHAR,
        estimated_ba_using_speedangle DOUBLE,
        hit_distance_sc DOUBLE,
        PRIMARY KEY (player_id, date, pa_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_windows (
        player_id INTEGER NOT NULL,
        window_end DATE NOT NULL,
        window_days INTEGER NOT NULL,
        pa INTEGER NOT NULL,
        hr INTEGER NOT NULL,
        r INTEGER NOT NULL,
        rbi INTEGER NOT NULL,
        sb INTEGER NOT NULL,
        avg DOUBLE,
        babip DOUBLE,
        k_pct DOUBLE,
        bb_pct DOUBLE,
        iso DOUBLE,
        ev_avg DOUBLE,
        barrel_pct DOUBLE,
        xwoba_avg DOUBLE,
        pt_bucket VARCHAR NOT NULL,
        PRIMARY KEY (player_id, window_end, window_days)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS thresholds (
        season_set VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        window_days INTEGER NOT NULL,
        pt_bucket VARCHAR NOT NULL,
        p10 DOUBLE NOT NULL,
        p90 DOUBLE NOT NULL,
        PRIMARY KEY (season_set, category, window_days, pt_bucket)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_streak_labels (
        player_id INTEGER NOT NULL,
        window_end DATE NOT NULL,
        window_days INTEGER NOT NULL,
        category VARCHAR NOT NULL,
        cold_method VARCHAR NOT NULL,
        label VARCHAR NOT NULL,
        PRIMARY KEY (player_id, window_end, window_days, category, cold_method)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_projection_rates (
        player_id INTEGER NOT NULL,
        season INTEGER NOT NULL,
        hr_per_pa DOUBLE NOT NULL,
        sb_per_pa DOUBLE NOT NULL,
        r_per_pa DOUBLE,
        rbi_per_pa DOUBLE,
        avg DOUBLE,
        n_systems INTEGER NOT NULL,
        PRIMARY KEY (player_id, season)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS continuation_rates (
        season_set VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        window_days INTEGER NOT NULL,
        pt_bucket VARCHAR NOT NULL,
        strength_bucket VARCHAR NOT NULL,
        direction VARCHAR NOT NULL,
        cold_method VARCHAR NOT NULL,
        n_labeled INTEGER NOT NULL,
        n_continued INTEGER NOT NULL,
        p_continued DOUBLE NOT NULL,
        p_baserate DOUBLE NOT NULL,
        lift DOUBLE NOT NULL,
        PRIMARY KEY (season_set, category, window_days, pt_bucket, strength_bucket, direction, cold_method)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_fits (
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
        -- Phase B: persisted Pipeline parameters so the dashboard refresh can
        -- reconstruct fitted models without retraining. ``feature_columns``,
        -- ``coef``, ``scaler_mean``, ``scaler_scale`` are aligned 1:1 in the
        -- same column order. ``dense_quintile_cutoffs`` is the 4-tuple of
        -- quintile breakpoints used to recompute ``streak_strength_numeric``
        -- at inference time for dense cats; NULL for sparse cats (which use
        -- a Poisson z-score formula instead).
        feature_columns VARCHAR[],
        coef DOUBLE[],
        intercept DOUBLE,
        scaler_mean DOUBLE[],
        scaler_scale DOUBLE[],
        dense_quintile_cutoffs DOUBLE[],
        PRIMARY KEY (model_id)
    )
    """,
]


# Per-table intended columns as ``(name, duckdb_type, not_null)``. Immutable
# (read-only mapping of tuples) because ``_intended_schema`` caches and shares a
# single instance process-wide -- a caller must not be able to corrupt it.
IntendedSchema = Mapping[str, tuple[tuple[str, str, bool], ...]]

# Per-table raw catalog as ``(name, duckdb_type, not_null, is_pk)`` -- the single
# pass the intended-schema and primary-key views both project from.
_TableCatalog = Mapping[str, tuple[tuple[str, str, bool, bool], ...]]


@functools.cache
def _catalog() -> _TableCatalog:
    """Each table's columns as ``(name, duckdb_type, not_null, is_pk)``.

    Built once (cached) by running ``_SCHEMA_DDL`` into a throwaway in-memory DB
    and reading DuckDB's own catalog back -- ``PRAGMA table_info`` yields
    ``(cid, name, type, notnull, dflt, pk)``, so name/type/not_null/pk are
    ``r[1]/r[2]/r[3]/r[5]``. Keeps ``_SCHEMA_DDL`` the single source of truth
    (normalized type strings, no Python->SQL type map) and introspects the DDL
    *once* per process for both the column-catalog and primary-key views below.

    Lazy (not built at import) so importing this module never runs DDL: a
    malformed-DDL or DuckDB-engine error then surfaces inside the first
    ``init_schema`` call -- where the caller can handle it -- rather than as an
    ImportError that bricks every module that imports schema.
    """
    scratch = duckdb.connect(":memory:")
    try:
        for ddl in _SCHEMA_DDL:
            scratch.execute(ddl)
        return MappingProxyType(
            {
                table: tuple(
                    (r[1], r[2], bool(r[3]), bool(r[5]))
                    for r in scratch.execute(f'PRAGMA table_info("{table}")').fetchall()
                )
                for (table,) in scratch.execute("SHOW TABLES").fetchall()
            }
        )
    finally:
        scratch.close()


@functools.cache
def _intended_schema() -> IntendedSchema:
    """Intended column catalog ``(name, type, not_null)``, from the cached ``_catalog``."""
    return MappingProxyType(
        {table: tuple((c[0], c[1], c[2]) for c in cols) for table, cols in _catalog().items()}
    )


def _table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Column names currently on *table*. The table must exist (DuckDB's
    PRAGMA table_info raises CatalogException for a missing table)."""
    return {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()}


def table_primary_key(table: str) -> tuple[str, ...]:
    """PRIMARY KEY column names for *table*, from the cached ``_SCHEMA_DDL`` catalog.

    Keeps the DDL the single source of truth -- callers (e.g. ``load._bulk_upsert``'s
    dedupe) never hand-restate a PK that can drift -- without paying a catalog
    query per call on the ingestion hot path. PRIMARY KEY columns never drift
    additively (the self-heal only ADDs nullable columns), so the intended key
    always equals the live table's key."""
    return tuple(c[0] for c in _catalog()[table] if c[3])


def get_connection(path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (or create) the streaks DuckDB at *path* and return the connection.

    Parent directory is created if missing. Schema is initialized on every open.
    """
    path_str = str(path)
    if path_str != ":memory:":
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(path_str)
    init_schema(conn)
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all streaks tables if they don't already exist, then heal any
    additive column drift on pre-existing tables (see ``_heal_additive_drift``).
    """
    for ddl in _SCHEMA_DDL:
        conn.execute(ddl)
    _heal_additive_drift(conn)


def _heal_additive_drift(conn: duckdb.DuckDBPyConnection) -> None:
    """ALTER-ADD any column that ``_SCHEMA_DDL`` declares but an existing table
    is missing.

    ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a
    gitignored dev DB created before a column was added (e.g.
    ``launch_speed_angle``) silently lacks it -- and the next insert throws a
    BinderException. Every added column is logged so an auto-heal leaves a trail
    instead of mutating the schema silently.

    Strictly additive and nullable-only: the diff only ever finds columns
    *missing* from the real table (never drops or retypes), and ``ADD COLUMN``
    on a populated table can only add a nullable column. A NOT NULL additive
    column therefore cannot be healed in place -- it is skipped with a warning,
    leaving the loud failure for a real migrate.py migration. Other destructive
    changes (drop ``barrel``, PK reshape) also go through migrate.py.
    """
    for table, columns in _intended_schema().items():
        existing = _table_columns(conn, table)
        for name, sql_type, not_null in columns:
            if name in existing:
                continue
            if not_null:
                logger.warning(
                    "Cannot auto-heal NOT NULL column %s.%s onto an existing table; "
                    "run the appropriate migrate.py migration",
                    table,
                    name,
                )
                continue
            conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}')
            logger.info("Healed schema drift: added %s.%s (%s)", table, name, sql_type)
