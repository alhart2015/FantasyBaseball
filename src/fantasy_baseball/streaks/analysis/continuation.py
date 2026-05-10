"""Continuation-rate computation for the Phase 3 go/no-go gate.

For each labeled window, look up the disjoint next window's outcome and
classify it relative to the player's expectation:

- Sparse cats (HR, SB): expectation = projected_rate * next_window_PA. The
  outcome direction is "above" if next_window_count > expected, "below"
  otherwise (ties break to "above" as the natural fantasy interpretation).
- Dense cats (R, RBI, AVG): expectation = empirical median of windows in the
  same (category, window_days, pt_bucket) cell. Direction same as above.

The output table `continuation_rates` has one row per
(season_set, category, window_days, pt_bucket, strength_bucket, direction,
cold_method). Each row reports:

- n_labeled: # of windows in this stratum with the matching label
- n_continued: # of those whose next-window outcome was on the matching side
- p_continued: n_continued / n_labeled
- p_baserate: unconditional rate of "next window on this direction's side" in
  the same (cat, window, bucket, direction) cell -- same value across all
  strength_bucket rows in that cell
- lift: p_continued - p_baserate

Only rows with N >= 1 in the labeled population are written. Phase 3
acceptance applies an N >= 1000 threshold against this table at read time.

Idempotent: full-wipe of `continuation_rates WHERE season_set = ?` on each call.
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd

from fantasy_baseball.streaks.labels import DENSE_CATEGORIES, SPARSE_CATEGORIES
from fantasy_baseball.streaks.models import StreakCategory

logger = logging.getLogger(__name__)

# One row per stratum we INSERT into `continuation_rates`. Column order matches
# the INSERT statement in compute_continuation_rates.
ContinuationRow = tuple[
    str,  # season_set
    str,  # category
    int,  # window_days
    str,  # pt_bucket
    str,  # strength_bucket
    str,  # direction
    str,  # cold_method
    int,  # n_labeled
    int,  # n_continued
    float,  # p_continued
    float,  # p_baserate
    float,  # lift
]


def compute_continuation_rates(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Rebuild `continuation_rates` for the given season_set. Returns rows written."""
    conn.execute("DELETE FROM continuation_rates WHERE season_set = ?", [season_set])

    rows: list[ContinuationRow] = []
    for category in DENSE_CATEGORIES:
        rows.extend(_continuation_rows_dense(conn, season_set=season_set, category=category))
    for category in SPARSE_CATEGORIES:
        rows.extend(_continuation_rows_sparse(conn, season_set=season_set, category=category))

    if rows:
        conn.executemany(
            """
            INSERT INTO continuation_rates
                (season_set, category, window_days, pt_bucket, strength_bucket,
                 direction, cold_method, n_labeled, n_continued, p_continued,
                 p_baserate, lift)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    logger.info("Wrote %d continuation_rates rows for season_set=%s", len(rows), season_set)
    return len(rows)


def _join_with_next_window(conn: duckdb.DuckDBPyConnection, category: str) -> pd.DataFrame:
    """Return a frame keyed by (player_id, window_end, window_days) joined with
    the player's next disjoint window's count and PA for the same window_days.

    Columns: player_id, window_end, window_days, pt_bucket, value (this window's
    count for the category), current_pa, next_value, next_pa, next_window_end.
    Rows whose next window doesn't exist (end-of-season trim) are dropped.
    """
    return conn.execute(
        f"""
        SELECT
            w.player_id,
            w.window_end,
            w.window_days,
            w.pt_bucket,
            w.{category} AS value,
            w.pa AS current_pa,
            n.{category} AS next_value,
            n.pa AS next_pa,
            n.window_end AS next_window_end
        FROM hitter_windows w
        INNER JOIN hitter_windows n
          ON n.player_id = w.player_id
         AND n.window_days = w.window_days
         AND n.window_end = w.window_end + INTERVAL (w.window_days) DAY
        """
    ).df()


def _continuation_rows_dense(
    conn: duckdb.DuckDBPyConnection, *, season_set: str, category: StreakCategory
) -> list[ContinuationRow]:
    df = _join_with_next_window(conn, category)
    if df.empty:
        return []

    # Median of `next_value` per (window_days, pt_bucket) -> the "expectation"
    # comparator for the next window.
    medians = df.groupby(["window_days", "pt_bucket"])["next_value"].median().rename("next_median")
    df = df.merge(medians, left_on=["window_days", "pt_bucket"], right_index=True)

    # Direction: above expectation if next_value > next_median; below otherwise.
    df["direction"] = np.where(df["next_value"] > df["next_median"], "above", "below")

    # Labels for this category x season_set, joined back.
    labels = conn.execute(
        """
        SELECT player_id, window_end, window_days, label
        FROM hitter_streak_labels
        WHERE category = ? AND cold_method = 'empirical'
        """,
        [category],
    ).df()
    df = df.merge(labels, on=["player_id", "window_end", "window_days"], how="inner")

    # Strength bucket: "p10_q1".."p10_q5" for cold; "p90_q1".."p90_q5" for hot;
    # "neutral" otherwise. Quintile within each label's population.
    df["strength_bucket"] = _dense_strength_buckets(df)
    return _aggregate_rows(df, season_set=season_set, category=category, cold_method="empirical")


def _dense_strength_buckets(df: pd.DataFrame) -> pd.Series:
    """Vectorized strength-bucket assignment for the dense-cat path.

    Replaces a per-row ``df.apply`` that recomputed the same-label quantiles
    inside every call (O(N x N_label) at hundreds-of-thousands of rows).
    Computes quintiles once per label and uses ``np.searchsorted`` for the
    bucket bin lookup.
    """
    out = pd.Series(np.full(len(df), "neutral", dtype=object), index=df.index)
    for label in ("hot", "cold"):
        mask = df["label"].to_numpy() == label
        if not mask.any():
            continue
        values = df.loc[mask, "value"].to_numpy()
        quintiles = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
        # ``np.searchsorted(quintiles, val, side='left')`` returns the
        # number of quintiles strictly less than val — equivalent to the
        # prior ``sum(val > q for q in quintiles)`` row-by-row logic.
        # Ties on a quintile boundary land in the lower bucket.
        qbins = np.searchsorted(quintiles, values, side="left")
        # Cap at 4 in case of floating-point edge cases.
        qbins = np.clip(qbins, 0, 4)
        out.loc[mask] = [f"{label}_q{int(b) + 1}" for b in qbins]
    return out


def _continuation_rows_sparse(
    conn: duckdb.DuckDBPyConnection, *, season_set: str, category: StreakCategory
) -> list[ContinuationRow]:
    rows: list[ContinuationRow] = []
    df_all = _join_with_next_window(conn, category)
    if df_all.empty:
        return rows

    rates = conn.execute(
        "SELECT player_id, season, hr_per_pa, sb_per_pa FROM hitter_projection_rates"
    ).df()
    rate_col = f"{category}_per_pa"
    df_all["season"] = pd.to_datetime(df_all["window_end"]).dt.year
    df_all = df_all.merge(rates[["player_id", "season", rate_col]], on=["player_id", "season"])
    df_all["expected_next"] = df_all[rate_col] * df_all["next_pa"]
    df_all["direction"] = np.where(df_all["next_value"] > df_all["expected_next"], "above", "below")

    for cold_method in ("poisson_p10", "poisson_p20"):
        labels = conn.execute(
            """
            SELECT player_id, window_end, window_days, label
            FROM hitter_streak_labels
            WHERE category = ? AND cold_method = ?
            """,
            [category, cold_method],
        ).df()
        df = df_all.merge(labels, on=["player_id", "window_end", "window_days"], how="inner")
        if df.empty:
            continue
        # current_pa is included in df_all by _join_with_next_window. Compute
        # expected_current = projected_rate * current_window_pa, then a
        # Poisson z-score (window_count - expected) / sqrt(expected).
        df["expected_current"] = df[rate_col] * df["current_pa"]
        denom = df["expected_current"].pow(0.5).replace(0, np.nan)
        df["z"] = (df["value"] - df["expected_current"]) / denom
        df["strength_bucket"] = _sparse_strength_buckets(df)
        rows.extend(
            _aggregate_rows(df, season_set=season_set, category=category, cold_method=cold_method)
        )
    return rows


def _sparse_strength_buckets(df: pd.DataFrame) -> pd.Series:
    """Vectorized strength-bucket assignment for the sparse-cat path.

    Replaces a per-row ``df.apply`` that ran the same arithmetic on each
    of ~1M rows. Bucket is ``{label}_{half:+.1f}sigma`` where ``half`` is
    the z-score rounded to the nearest half-integer and clamped to
    ``[-3, +3]``; rows with NaN z fall into ``{label}_zna``; neutral rows
    stay neutral.

    Implementation: ``half`` ranges over only 13 distinct values
    (-3.0, -2.5, ..., +3.0). We pre-format those 13 suffixes and look
    them up by integer index instead of formatting each row.
    """
    label = df["label"].to_numpy()
    z = df["z"].to_numpy(dtype=float)
    is_labeled = label != "neutral"
    is_zna = is_labeled & np.isnan(z)
    is_z = is_labeled & ~np.isnan(z)

    # 13 fixed-width suffixes indexed by ``half_idx`` (= 0..12 for
    # half = -3.0..+3.0 in 0.5 steps).
    half_step_table = np.array([f"_{(-3.0 + 0.5 * i):+.1f}sigma" for i in range(13)], dtype=object)

    out = np.full(len(df), "neutral", dtype=object)
    if is_z.any():
        # half = round(z*2)/2 clamped to [-3, +3]. half_idx = (half + 3) * 2.
        # Compute half_idx only over non-NaN labeled rows to avoid
        # ``invalid value encountered in cast`` from NaN -> int64.
        z_subset = z[is_z]
        half_idx_subset = np.clip(np.rint(z_subset * 2).astype(np.int64) + 6, 0, 12)
        out[is_z] = np.char.add(
            label[is_z].astype(str),
            half_step_table[half_idx_subset].astype(str),
        )
    if is_zna.any():
        out[is_zna] = np.char.add(label[is_zna].astype(str), "_zna")
    return pd.Series(out, index=df.index)


def _aggregate_rows(
    df: pd.DataFrame, *, season_set: str, category: str, cold_method: str
) -> list[ContinuationRow]:
    """Group the joined dataframe by (window_days, pt_bucket, strength_bucket)
    and compute n_labeled / n_continued / lift per row, with one row written
    per group at the label's natural matching direction (hot -> above,
    cold -> below).

    Base rate is the unconditional fraction of next-window outcomes on each
    direction's side within (window_days, pt_bucket).
    """
    # Base rate: unconditional fraction of next-window outcomes per
    # (window_days, pt_bucket, direction). Computed by counting all rows in
    # the joined frame (regardless of label) and dividing by the per-(wd,
    # bucket) total.
    counts = df.groupby(["window_days", "pt_bucket", "direction"]).size()
    totals = counts.groupby(level=[0, 1]).sum()
    base_rate = (counts / totals).rename("p_baserate")

    out: list[ContinuationRow] = []
    # Group by strength_bucket only (not direction): n_labeled is the size of
    # the labeled stratum regardless of next-window outcome, and we emit one
    # row per stratum at the label's natural matching direction.
    for (wd, bucket, strength), grp in df.groupby(["window_days", "pt_bucket", "strength_bucket"]):
        # n_labeled: count of (player, window) pairs *with this label* in this stratum.
        # The strength_bucket already encodes the label (hot_q1, cold_q1, neutral, ...).
        labeled_n = len(grp)
        if strength == "neutral":
            # Neutral rows aren't useful for the go/no-go gate (no signal claim).
            continue
        # n_continued: of those, how many had the next window on the direction-of-deviation matching the label.
        # For hot: direction == 'above'. For cold: direction == 'below'.
        label = (
            "hot"
            if strength.startswith("hot")
            else ("cold" if strength.startswith("cold") else None)
        )
        if label is None:
            continue
        match_dir = "above" if label == "hot" else "below"
        n_continued = int((grp["direction"] == match_dir).sum())
        try:
            p_base = float(base_rate.loc[(wd, bucket, match_dir)])
        except KeyError:
            p_base = 0.0
        if labeled_n == 0:
            continue
        p_cont = n_continued / labeled_n
        out.append(
            (
                season_set,
                category,
                int(wd),
                str(bucket),
                strength,
                match_dir,
                cold_method,
                int(labeled_n),
                int(n_continued),
                float(p_cont),
                float(p_base),
                float(p_cont - p_base),
            )
        )
    return out
