"""Streaks DuckDB schema migrations.

Phase 2 (:func:`migrate_to_phase_2`) drops all streaks tables and lets
:func:`init_schema` recreate them — a full schema change paired with a
committed re-fetch, since ALTER ADD COLUMN cannot reinstate the
``NOT NULL`` constraints on the box-score additions. Phase 3
(:func:`migrate_to_phase_3`) drops only ``hitter_streak_labels`` to pick
up the PK shape change for ``cold_method``; other tables are untouched.
Both functions are idempotent (``DROP TABLE IF EXISTS`` plus
``CREATE TABLE IF NOT EXISTS``) and safe to re-run.
"""

from __future__ import annotations

import logging

import duckdb

from fantasy_baseball.streaks.data.schema import init_schema

logger = logging.getLogger(__name__)

_TABLES: tuple[str, ...] = (
    "hitter_games",
    "hitter_statcast_pa",
    "hitter_windows",
    "thresholds",
    "hitter_streak_labels",
)


def migrate_to_phase_2(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop all streaks tables and recreate them with the Phase 2 schema."""
    for table in _TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        logger.info("Dropped %s", table)
    init_schema(conn)
    logger.info("Recreated all streaks tables via init_schema")


def migrate_to_phase_3(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop only `hitter_streak_labels` (which has a PK shape change) and let
    `init_schema` recreate it with the Phase 3 PK — plus any new Phase 3 tables
    that are missing (`hitter_projection_rates`, `continuation_rates`).

    `hitter_games` / `hitter_statcast_pa` / `hitter_windows` / `thresholds` are
    untouched: their schema didn't change. Labels are pure derived data and are
    rebuilt by `apply_labels` after this migration.
    """
    conn.execute("DROP TABLE IF EXISTS hitter_streak_labels")
    logger.info("Dropped hitter_streak_labels (PK shape change for cold_method)")
    init_schema(conn)
    logger.info("Recreated hitter_streak_labels + Phase 3 tables via init_schema")


def migrate_to_phase_4(conn: duckdb.DuckDBPyConnection) -> None:
    """Add Phase 4 columns/tables. Idempotent and non-destructive.

    Adds three nullable rate columns to ``hitter_projection_rates``
    (``r_per_pa``, ``rbi_per_pa``, ``avg``) so dense-cat continuation
    models can take ``season_rate_in_category`` as a feature. Then calls
    ``init_schema`` to ensure the ``model_fits`` table exists.

    Existing Phase 3 rows (hr_per_pa + sb_per_pa only) survive with NULL
    in the new columns. Re-run ``scripts/streaks/load_projections.py``
    after this migration to backfill them.

    ``hitter_games`` / ``hitter_statcast_pa`` / ``hitter_windows`` /
    ``thresholds`` / ``hitter_streak_labels`` / ``continuation_rates``
    are untouched.
    """
    for col in ("r_per_pa", "rbi_per_pa", "avg"):
        conn.execute(f"ALTER TABLE hitter_projection_rates ADD COLUMN IF NOT EXISTS {col} DOUBLE")
        logger.info("ALTER hitter_projection_rates ADD COLUMN IF NOT EXISTS %s", col)
    init_schema(conn)
    logger.info("Recreated/ensured Phase 4 tables via init_schema (model_fits)")


def migrate_to_phase_5(conn: duckdb.DuckDBPyConnection) -> None:
    """Replace ``hitter_statcast_pa.barrel`` (BOOLEAN) with ``launch_speed_angle``
    (INTEGER, Statcast's 1-6 batted-ball classifier).

    ``barrel`` was always NULL in pre-Phase-5 fetches because pybaseball never
    returned a ``barrel`` column. The canonical Statcast classifier is
    ``launch_speed_angle`` where value 6 == barrel; lower values are weaker
    contact tiers (1=weak, 2=topped, 3=under, 4=flare/burner, 5=solid).
    ``windows.py`` derives ``barrel_pct`` from ``launch_speed_angle = 6``.

    Idempotent and non-destructive: existing PA rows survive with
    ``launch_speed_angle = NULL`` after the migration. Re-run
    ``scripts/streaks/fetch_history.py --force-statcast --season {year}``
    for each year of data to backfill the new column via INSERT OR REPLACE.
    """
    init_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info('hitter_statcast_pa')").fetchall()}
    if "launch_speed_angle" not in cols:
        conn.execute("ALTER TABLE hitter_statcast_pa ADD COLUMN launch_speed_angle INTEGER")
        logger.info("ALTER hitter_statcast_pa ADD COLUMN launch_speed_angle INTEGER")
    if "barrel" in cols:
        conn.execute("ALTER TABLE hitter_statcast_pa DROP COLUMN barrel")
        logger.info("ALTER hitter_statcast_pa DROP COLUMN barrel")


def migrate_to_phase_b(conn: duckdb.DuckDBPyConnection) -> None:
    """Add Phase B pipeline-state columns to ``model_fits``. Idempotent.

    Adds six nullable columns so ``load_models_from_fits`` can reconstruct
    fitted Pipelines without retraining:

    - ``feature_columns`` (VARCHAR[]) — the in-order feature names.
    - ``coef`` (DOUBLE[]) — LogisticRegression coefficient vector.
    - ``intercept`` (DOUBLE) — LogisticRegression intercept scalar.
    - ``scaler_mean`` / ``scaler_scale`` (DOUBLE[]) — StandardScaler params.
    - ``dense_quintile_cutoffs`` (DOUBLE[]) — quintile breakpoints used to
      reproduce ``streak_strength_numeric`` at inference time (NULL for
      sparse cats).

    Existing Phase 4 rows survive with NULL in the new columns; the loader
    skips rows it cannot reconstruct, so the next ``refit_models_for_report``
    call repopulates them automatically.
    """
    init_schema(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info('model_fits')").fetchall()}
    additions = (
        ("feature_columns", "VARCHAR[]"),
        ("coef", "DOUBLE[]"),
        ("intercept", "DOUBLE"),
        ("scaler_mean", "DOUBLE[]"),
        ("scaler_scale", "DOUBLE[]"),
        ("dense_quintile_cutoffs", "DOUBLE[]"),
    )
    for col, sql_type in additions:
        if col not in cols:
            conn.execute(f"ALTER TABLE model_fits ADD COLUMN {col} {sql_type}")
            logger.info("ALTER model_fits ADD COLUMN %s %s", col, sql_type)
