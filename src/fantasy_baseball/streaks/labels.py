"""Apply calibrated thresholds to hitter_windows -> hitter_streak_labels.

Two label paths, both written into the same table with `cold_method`
distinguishing them:

- **Dense categories (R, RBI, AVG):** uses calibrated empirical p10/p90
  from `thresholds`. One row per (player, window, category) with
  cold_method='empirical'.
- **Sparse categories (HR, SB):** uses skill-relative Poisson lower-tail
  thresholds against per-player projected rates. Two rows per (player,
  window, category) -- cold_method='poisson_p10' and cold_method='poisson_p20'.
  Hot uses the same empirical p90 in both rows. Players without a row in
  `hitter_projection_rates` get NO sparse-cat labels written (callers can
  still compute hot via the dense path; we omit them from sparse rows
  rather than fabricating a baseline).

Idempotent: full-wipe of `hitter_streak_labels` on each call (labels are
tied to the latest threshold + projection-rate calibration; no scoped
delete is meaningful).
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import poisson

from fantasy_baseball.streaks.models import StreakCategory

logger = logging.getLogger(__name__)

DENSE_CATEGORIES: tuple[StreakCategory, ...] = ("r", "rbi", "avg")
SPARSE_CATEGORIES: tuple[StreakCategory, ...] = ("hr", "sb")
POISSON_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("poisson_p10", 0.10),
    ("poisson_p20", 0.20),
)


def apply_labels(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Rebuild `hitter_streak_labels` from windows + thresholds + projection rates.

    Returns total rows written across all (category, cold_method) pairs.
    """
    conn.execute("DELETE FROM hitter_streak_labels")
    n_dense = _apply_dense_labels(conn, season_set=season_set)
    n_sparse = _apply_sparse_labels(conn, season_set=season_set)
    total = n_dense + n_sparse
    logger.info(
        "Wrote %d label rows for season_set=%s (dense=%d, sparse=%d)",
        total,
        season_set,
        n_dense,
        n_sparse,
    )
    return total


def _apply_dense_labels(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Empirical p10/p90 for R, RBI, AVG. Pure SQL, mirrors Phase 2 logic."""
    n_written = 0
    for category in DENSE_CATEGORIES:
        sql = f"""
            INSERT INTO hitter_streak_labels
                (player_id, window_end, window_days, category, cold_method, label)
            SELECT
                w.player_id,
                w.window_end,
                w.window_days,
                ? AS category,
                'empirical' AS cold_method,
                CASE
                    WHEN w.{category} IS NULL THEN 'neutral'
                    WHEN w.{category} >= t.p90 THEN 'hot'
                    WHEN w.{category} <= t.p10 THEN 'cold'
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
        row = conn.execute(
            "SELECT COUNT(*) FROM hitter_streak_labels WHERE category = ? AND cold_method = 'empirical'",
            [category],
        ).fetchone()
        n_written += int(row[0]) if row is not None else 0
    return n_written


def _apply_sparse_labels(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Skill-relative Poisson cold + empirical p90 hot for HR and SB.

    The math runs in pandas -- `scipy.stats.poisson.ppf` is vectorized and the
    join cardinality (~3-5M rows) is comfortably in-memory. SQL would have to
    UDF or LATERAL the Poisson call per row, which is messier.
    """
    df = conn.execute(
        """
        SELECT
            w.player_id,
            w.window_end,
            w.window_days,
            w.pa AS window_pa,
            w.hr,
            w.sb,
            w.pt_bucket,
            EXTRACT(YEAR FROM w.window_end)::INTEGER AS season,
            p.hr_per_pa,
            p.sb_per_pa
        FROM hitter_windows w
        INNER JOIN hitter_projection_rates p
          ON p.player_id = w.player_id
         AND p.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
        """
    ).df()
    if df.empty:
        logger.warning(
            "No (window, projection_rate) joined rows -- sparse cats get zero labels. "
            "Did you forget to load projection rates first?"
        )
        return 0

    # Empirical p90 for hot, fetched as a small per-(category, window_days,
    # pt_bucket) frame and merged onto df below.
    thresholds_df = conn.execute(
        "SELECT category, window_days, pt_bucket, p90 FROM thresholds WHERE season_set = ?",
        [season_set],
    ).df()

    # Per-row identity arrays — same across all (category, cold_method)
    # blocks. NumPy arrays so we can stack them with concatenate without
    # round-tripping through Python lists.
    pid_arr = df["player_id"].astype(int).to_numpy()
    end_arr = df["window_end"].to_numpy()
    wd_arr = df["window_days"].astype(int).to_numpy()
    n = len(df)

    pid_blocks: list[np.ndarray] = []
    end_blocks: list[np.ndarray] = []
    wd_blocks: list[np.ndarray] = []
    cat_blocks: list[np.ndarray] = []
    cm_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []

    for category in SPARSE_CATEGORIES:
        rate_col = f"{category}_per_pa"
        count_col = category
        expected = (df[rate_col] * df["window_pa"]).to_numpy(dtype=float)
        counts = df[count_col].to_numpy(dtype=int)

        # Hot: empirical p90 (same in both poisson methods). Vectorized
        # via a left-merge on a tiny per-category threshold frame; rows
        # without a matching threshold get NaN p90 and so can't be hot.
        cat_thresholds = thresholds_df.loc[
            thresholds_df["category"] == category, ["window_days", "pt_bucket", "p90"]
        ].rename(columns={"p90": "_hot_p90"})
        merged = df.merge(cat_thresholds, on=["window_days", "pt_bucket"], how="left")
        hot_p90 = merged["_hot_p90"].to_numpy(dtype=float)
        is_hot = (~np.isnan(hot_p90)) & (counts >= hot_p90)

        for cold_method, percentile in POISSON_PERCENTILES:
            # Poisson.ppf returns the smallest k such that P(X <= k) >= percentile.
            # Cold => window_count < k. For very low expected (<= ~0.5) ppf returns
            # 0 and cold can never fire -- the desired floor effect.
            k = poisson.ppf(percentile, expected)
            is_cold = counts < k
            # Build label: hot wins ties (a window in both buckets is hot).
            labels = np.where(is_hot, "hot", np.where(is_cold, "cold", "neutral"))
            pid_blocks.append(pid_arr)
            end_blocks.append(end_arr)
            wd_blocks.append(wd_arr)
            cat_blocks.append(np.full(n, category, dtype=object))
            cm_blocks.append(np.full(n, cold_method, dtype=object))
            label_blocks.append(labels)

    rows_df = pd.DataFrame(  # referenced by name in the DuckDB SQL below
        {
            "player_id": np.concatenate(pid_blocks),
            "window_end": np.concatenate(end_blocks),
            "window_days": np.concatenate(wd_blocks),
            "category": np.concatenate(cat_blocks),
            "cold_method": np.concatenate(cm_blocks),
            "label": np.concatenate(label_blocks),
        }
    )
    n_rows = len(rows_df)
    # Bulk INSERT via DuckDB's pandas scan — orders of magnitude faster
    # than Python-level executemany at multi-million-row sizes.
    conn.execute(
        "INSERT INTO hitter_streak_labels "
        "(player_id, window_end, window_days, category, cold_method, label) "
        "SELECT player_id, window_end, window_days, category, cold_method, label FROM rows_df"
    )
    return n_rows
