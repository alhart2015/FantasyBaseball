"""One-shot Phase 2 schema migration.

DROPs all five streaks tables and lets :func:`init_schema` recreate them
with the canonical Phase 2 shape. We commit to a full re-fetch anyway
(see Task 7 of the Phase 2 plan), so wiping the data is intentional —
and DROP+CREATE is cleaner than ALTER+DELETE because ALTER ADD COLUMN
on an existing table cannot reinstate the ``NOT NULL`` constraints
``schema.py`` declares on the box-score additions, leaving a migrated
DB silently more permissive than a freshly-created one.

Idempotent: ``DROP TABLE IF EXISTS`` plus ``CREATE TABLE IF NOT EXISTS``.
Safe to re-run.
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
