"""Streaks DuckDB schema migrations.

Phase 2 (:func:`migrate_to_phase_2`) drops all streaks tables and lets
:func:`init_schema` recreate them — a full schema change paired with a
committed re-fetch, since ALTER ADD COLUMN cannot reinstate the
``NOT NULL`` constraints on the box-score additions. Phase 3
(:func:`migrate_to_phase_3`) drops only ``hitter_streak_labels`` to pick
up the PK shape change for ``cold_method``; other tables are untouched.
Both functions are idempotent (``DROP TABLE IF EXISTS`` plus
``CREATE TABLE IF NOT EXISTS``) and safe to re-run.

Phases 4, 5, and B were additive column adds; that work now lives in
``init_schema``'s additive-drift healer, so those functions are thin wrappers
over ``init_schema`` (Phase 5 additionally drops the legacy ``barrel`` column,
the one step the additive healer cannot do).
"""

from __future__ import annotations

import logging

import duckdb

from fantasy_baseball.streaks.data.schema import _table_columns, init_schema

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

    Phase 4 adds three nullable rate columns to ``hitter_projection_rates``
    (``r_per_pa``, ``rbi_per_pa``, ``avg``) so dense-cat continuation models can
    take ``season_rate_in_category`` as a feature, plus the ``model_fits``
    table. ``init_schema``'s additive-drift healer now adds these columns, so
    this is a thin wrapper kept for the ``--phase 4`` CLI and back reference
    (see the module docstring for why 4/5/b are wrappers).

    Existing Phase 3 rows (hr_per_pa + sb_per_pa only) survive with NULL in the
    new columns. Re-run ``scripts/streaks/load_projections.py`` after this
    migration to backfill them.
    """
    init_schema(conn)
    logger.info("Ensured Phase 4 columns/tables via init_schema")


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

    Adding ``launch_speed_angle`` is now handled by ``init_schema``'s
    additive-drift healer; the only step unique to this migration is dropping
    the legacy ``barrel`` column, which the additive healer cannot do.
    """
    init_schema(conn)
    if "barrel" in _table_columns(conn, "hitter_statcast_pa"):
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

    ``init_schema``'s additive-drift healer now adds the six columns, so this
    is a thin wrapper kept for the ``--phase b`` CLI and back reference (see the
    module docstring for why 4/5/b are wrappers).
    """
    init_schema(conn)
    logger.info("Ensured Phase B model_fits columns via init_schema")
