"""Apply calibrated thresholds to ``hitter_windows`` -> ``hitter_streak_labels``.

For each (window row x category), label is:
  - 'hot'     if value >= p90
  - 'cold'    if value <= p10
  - 'neutral' otherwise

Windows whose (window_days, pt_bucket) combo has no threshold row in the
named ``season_set`` are skipped entirely (no labels written).

Idempotent: rebuilds all rows on each call. The labels table has no
``season_set`` column, so we full-wipe before re-inserting -- labels are
tied to the latest threshold calibration.
"""

from __future__ import annotations

import logging

import duckdb

from fantasy_baseball.streaks.thresholds import CATEGORIES

logger = logging.getLogger(__name__)


def apply_labels(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Rebuild ``hitter_streak_labels`` from ``hitter_windows`` joined to thresholds.

    Returns total rows written across all categories. Note: only one season_set's
    labels are usable at a time — calling with a different ``season_set`` wipes
    prior labels (the labels table has no ``season_set`` column).
    """
    # Full wipe: the labels table has no season_set column, and labels are
    # tied to the most recent calibration. A scoped delete would leave stale
    # rows from prior season_sets in place.
    conn.execute("DELETE FROM hitter_streak_labels")

    for category in CATEGORIES:
        # Both counting cats (hr/r/rbi/sb) and the avg rate live under
        # same-named columns in hitter_windows.
        col = category
        # NULL check first: ``NULL >= x`` evaluates to NULL (not boolean),
        # so without the explicit guard the CASE would return NULL for
        # missing rate values, violating the NOT NULL label column.
        sql = f"""
            INSERT OR REPLACE INTO hitter_streak_labels
                (player_id, window_end, window_days, category, label)
            SELECT
                w.player_id,
                w.window_end,
                w.window_days,
                ? AS category,
                CASE
                    WHEN w.{col} IS NULL THEN 'neutral'
                    WHEN w.{col} >= t.p90 THEN 'hot'
                    WHEN w.{col} <= t.p10 THEN 'cold'
                    ELSE 'neutral'
                END AS label
            FROM hitter_windows w
            JOIN thresholds t
              ON t.season_set = ?
             AND t.category = ?
             AND t.window_days = w.window_days
             AND t.pt_bucket = w.pt_bucket
        """
        conn.execute(sql, [category, season_set, category])

    row = conn.execute("SELECT COUNT(*) FROM hitter_streak_labels").fetchone()
    written = int(row[0]) if row is not None else 0
    logger.info("Wrote %d label rows for season_set=%s", written, season_set)
    return written
