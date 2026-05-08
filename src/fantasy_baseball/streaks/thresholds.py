"""Empirical threshold calibration for streak labels.

For a given calibration set (e.g. ``"2023-2025"`` or a single season),
computes p10 and p90 per (category x window_days x pt_bucket) using only
windows from player-seasons with >= ``qualifying_pa`` PA. Writes results
to the ``thresholds`` table (idempotent -- DELETE-then-INSERT keyed on
(season_set, category, window_days, pt_bucket)).

The five categories match the project's hitter roto stats: HR, R, RBI,
SB (counting), and AVG (rate). For the counting stats we percentile the
raw window count; for AVG we percentile the rate column from
``hitter_windows.avg``.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = ("hr", "r", "rbi", "sb", "avg")


def compute_thresholds(
    conn: duckdb.DuckDBPyConnection,
    *,
    season_set: str,
    qualifying_pa: int = 150,
) -> int:
    """(Re)build ``thresholds`` rows for the given ``season_set``.

    ``season_set`` is a free-form label (e.g. ``"2025"``, ``"2023-2025"``);
    the SQL filter on which seasons to include is derived from it.
    Currently supports either a single season ``"YYYY"`` or a hyphenated
    range ``"YYYY-YYYY"``.

    Returns the number of rows written.
    """
    seasons = _parse_season_set(season_set)
    season_list_sql = ", ".join(str(s) for s in seasons)

    # Qualifying players: aggregate PA from hitter_games over the seasons
    # in scope, keep player_ids with sum(pa) >= qualifying_pa.
    # Both ``season_list_sql`` (built from ints we parsed ourselves) and
    # ``qualifying_pa`` (a Python int) are not user input, so f-string
    # interpolation is safe; DuckDB's ``percentile_cont`` doesn't accept
    # parameter markers for the GROUP BY columns we need anyway.
    qualifying_sql = f"""
        SELECT player_id, season FROM hitter_games
        WHERE season IN ({season_list_sql})
        GROUP BY player_id, season
        HAVING SUM(pa) >= {qualifying_pa}
    """

    # Windows from those qualifying player-seasons. We join hitter_windows
    # to a derived qualifying-player-seasons table on (player_id, season),
    # using the calendar year of window_end as the season key.
    windows_sql = f"""
        WITH qualified AS ({qualifying_sql})
        SELECT
            w.player_id,
            EXTRACT(YEAR FROM w.window_end)::INTEGER AS season,
            w.window_end, w.window_days, w.pt_bucket,
            w.hr, w.r, w.rbi, w.sb, w.avg
        FROM hitter_windows w
        JOIN qualified q
          ON q.player_id = w.player_id
         AND q.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
    """

    # Drop any pre-existing rows for this season_set before re-writing so
    # stale strata that no longer appear in the data are also cleared (an
    # INSERT OR REPLACE alone would leave them behind).
    conn.execute("DELETE FROM thresholds WHERE season_set = ?", [season_set])

    written = 0
    for category in CATEGORIES:
        # AVG is a rate column; counting cats are int columns -- both live
        # under a same-named column in hitter_windows, so the same template
        # works for either.
        col = category
        rows = conn.execute(
            f"""
            WITH src AS ({windows_sql})
            SELECT
                window_days,
                pt_bucket,
                percentile_cont(0.1) WITHIN GROUP (ORDER BY {col}) AS p10,
                percentile_cont(0.9) WITHIN GROUP (ORDER BY {col}) AS p90
            FROM src
            WHERE {col} IS NOT NULL
            GROUP BY window_days, pt_bucket
            HAVING COUNT(*) >= 1
            """
        ).fetchall()
        for window_days, pt_bucket, p10, p90 in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO thresholds
                (season_set, category, window_days, pt_bucket, p10, p90)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [season_set, category, int(window_days), pt_bucket, float(p10), float(p90)],
            )
            written += 1
    logger.info("Wrote %d threshold rows for season_set=%s", written, season_set)
    return written


def _parse_season_set(season_set: str) -> list[int]:
    """Parse ``"YYYY"`` or ``"YYYY-YYYY"`` into an inclusive list of seasons."""
    if "-" in season_set:
        start_str, end_str = season_set.split("-", 1)
        return list(range(int(start_str), int(end_str) + 1))
    return [int(season_set)]
