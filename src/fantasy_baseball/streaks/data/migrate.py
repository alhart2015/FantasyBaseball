"""One-shot Phase 2 schema migration.

Adds the box-score expansion columns to ``hitter_games`` and the Statcast
peripheral columns to ``hitter_statcast_pa`` on a pre-existing Phase 1 DB.
DELETEs all rows from both tables so re-running :mod:`fetch_history`
repopulates with the new columns; downstream tables (``hitter_windows``,
``thresholds``, ``hitter_streak_labels``) are also cleared as a safety
measure (they should be empty in Phase 1 but the DELETE is cheap).

Idempotent: ALTER TABLE ADD COLUMN failures (column already exists) are
caught per column. Safe to re-run.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

_GAMES_NEW_COLUMNS: tuple[tuple[str, str], ...] = (
    ("b2", "INTEGER"),
    ("b3", "INTEGER"),
    ("sf", "INTEGER"),
    ("hbp", "INTEGER"),
    ("ibb", "INTEGER"),
    ("cs", "INTEGER"),
    ("gidp", "INTEGER"),
    ("sh", "INTEGER"),
    ("ci", "INTEGER"),
    ("is_home", "BOOLEAN"),
)
_STATCAST_NEW_COLUMNS: tuple[tuple[str, str], ...] = (
    ("at_bat_number", "INTEGER"),
    ("bb_type", "VARCHAR"),
    ("estimated_ba_using_speedangle", "DOUBLE"),
    ("hit_distance_sc", "DOUBLE"),
)
_TABLES_TO_CLEAR: tuple[str, ...] = (
    "hitter_games",
    "hitter_statcast_pa",
    "hitter_windows",
    "thresholds",
    "hitter_streak_labels",
)


def _column_names(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {r[1] for r in rows}


def _add_column_if_missing(
    conn: duckdb.DuckDBPyConnection, table: str, column: str, sql_type: str
) -> None:
    if column in _column_names(conn, table):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
    logger.info("Added %s.%s (%s)", table, column, sql_type)


def migrate_to_phase_2(conn: duckdb.DuckDBPyConnection) -> None:
    """Add Phase 2 columns and clear stale rows from a Phase 1 DB.

    No-op for already-migrated DBs.
    """
    for col, sql_type in _GAMES_NEW_COLUMNS:
        _add_column_if_missing(conn, "hitter_games", col, sql_type)
    for col, sql_type in _STATCAST_NEW_COLUMNS:
        _add_column_if_missing(conn, "hitter_statcast_pa", col, sql_type)
    for table in _TABLES_TO_CLEAR:
        # Some downstream tables may not exist on a fresh Phase 1 DB; tolerate that.
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        except duckdb.CatalogException:
            continue
        n = row[0] if row else 0
        if n:
            logger.info("Clearing %d rows from %s", n, table)
            conn.execute(f"DELETE FROM {table}")
