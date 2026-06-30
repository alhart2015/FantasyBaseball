"""Streaks DuckDB schema migrations.

Phase 2 (:func:`migrate_to_phase_2`) drops all streaks tables and lets
:func:`init_schema` recreate them — a full schema change paired with a
committed re-fetch, since ALTER ADD COLUMN cannot reinstate the
``NOT NULL`` constraints on the box-score additions. Phase 3
(:func:`migrate_to_phase_3`) drops only ``hitter_streak_labels`` to pick
up the PK shape change for ``cold_method``; other tables are untouched.
Both functions are idempotent (``DROP TABLE IF EXISTS`` plus
``CREATE TABLE IF NOT EXISTS``) and safe to re-run.

Additive column adds are no longer migrations: ``init_schema``'s additive-drift
healer adds any missing nullable column on every connection open (see
``schema.py``). The old additive phase_4 / phase_b functions were therefore
removed. Phase 5 remains only to drop the legacy ``barrel`` column, the one
step the additive healer cannot do.
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
