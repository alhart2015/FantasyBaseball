"""Phase 4 predictor pipeline: per-direction logistic regressions on
streak continuation, with player-grouped CV, bootstrap CIs, and
permutation feature importance.

See ``docs/superpowers/plans/2026-05-10-hot-streaks-phase-4-predictive-model.md``
for the design decisions captured at the top of this module.
"""

from __future__ import annotations

import logging
import re

import duckdb
import numpy as np
import pandas as pd

from fantasy_baseball.streaks.models import StreakCategory, StreakDirection

logger = logging.getLogger(__name__)

# All 8 Phase 4 models. (category, direction): hot ⇔ 'above'; cold ⇔ 'below'.
PHASE_4_MODELS: tuple[tuple[StreakCategory, StreakDirection], ...] = (
    ("r", "above"),
    ("r", "below"),
    ("rbi", "above"),
    ("rbi", "below"),
    ("avg", "above"),
    ("avg", "below"),
    ("hr", "above"),  # sparse hot only
    ("sb", "above"),  # sparse hot only
)

DENSE_CATS: frozenset[StreakCategory] = frozenset({"r", "rbi", "avg"})
SPARSE_CATS: frozenset[StreakCategory] = frozenset({"hr", "sb"})

# Sparse hot rows are duplicated across poisson_p10/poisson_p20 in
# hitter_streak_labels. Phase 4 dedupes to a single partition for training.
SPARSE_HOT_COLD_METHOD = "poisson_p20"

# Final ordered list of feature column names in the training frame.
EXPECTED_FEATURE_COLUMNS: tuple[str, ...] = (
    "streak_strength_numeric",
    "babip",
    "k_pct",
    "bb_pct",
    "iso",
    "ev_avg",
    "barrel_pct",
    "xwoba_avg",
    "season_rate_in_category",
    "pt_bucket_low",
    "pt_bucket_mid",
    "pt_bucket_high",
)

# Parsing rules for strength_bucket → numeric.
_DENSE_BUCKET_RE = re.compile(r"^(hot|cold)_q([1-5])$")
_SPARSE_BUCKET_RE = re.compile(r"^(hot|cold)_([+-]?\d+\.\d)sigma$")


def _parse_strength_numeric(bucket: str) -> float | None:
    """Encode Phase 3's strength_bucket string as a numeric feature.

    - Dense quintiles "hot_qN" / "cold_qN" → integer 1..5.
    - Sparse half-sigma buckets "hot_+1.5sigma" → 1.5 (signed float).
    - "{label}_zna" or any other shape → None (caller drops the row).
    """
    if m := _DENSE_BUCKET_RE.match(bucket):
        return float(m.group(2))
    if m := _SPARSE_BUCKET_RE.match(bucket):
        return float(m.group(2))
    return None


def _parse_season_set(season_set: str) -> list[int]:
    """Parse ``"YYYY"`` or ``"YYYY-YYYY"`` into an inclusive list of seasons.

    Mirrors the same-named helper in :mod:`thresholds`; duplicated here rather
    than re-exported to keep the streaks subsystem's modules independently
    importable.
    """
    if "-" in season_set:
        start_str, end_str = season_set.split("-", 1)
        return list(range(int(start_str), int(end_str) + 1))
    return [int(season_set)]


def _build_training_frame_dense(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    direction: StreakDirection,
    seasons: list[int],
    window_days: int,
) -> pd.DataFrame:
    """Dense-cat training frame. Target uses the (window_days, pt_bucket)
    median of next_value computed across the entire labeled population."""
    label = "hot" if direction == "above" else "cold"
    rate_col = "avg" if category == "avg" else f"{category}_per_pa"
    # ``seasons`` is built from ints parsed by ``_parse_season_set`` and never
    # user-supplied, so f-string interpolation is safe.
    season_list_sql = ", ".join(str(s) for s in seasons)

    df = conn.execute(
        f"""
        SELECT
            w.player_id,
            w.window_end,
            EXTRACT(YEAR FROM w.window_end)::INTEGER AS season,
            w.pt_bucket,
            w.{category} AS value,
            n.{category} AS next_value,
            w.babip,
            w.k_pct,
            w.bb_pct,
            w.iso,
            w.ev_avg,
            w.barrel_pct,
            w.xwoba_avg,
            p.{rate_col} AS season_rate_in_category,
            l.label,
            CASE
                WHEN ? = 'above' THEN
                    CASE WHEN n.{category} > t.next_median THEN 1 ELSE 0 END
                ELSE
                    CASE WHEN n.{category} < t.next_median THEN 1 ELSE 0 END
            END AS target
        FROM hitter_windows w
        INNER JOIN hitter_windows n
          ON n.player_id = w.player_id
         AND n.window_days = w.window_days
         AND n.window_end = w.window_end + INTERVAL (w.window_days) DAY
        INNER JOIN hitter_streak_labels l
          ON l.player_id = w.player_id
         AND l.window_end = w.window_end
         AND l.window_days = w.window_days
         AND l.category = ?
         AND l.cold_method = 'empirical'
         AND l.label = ?
        INNER JOIN hitter_projection_rates p
          ON p.player_id = w.player_id
         AND p.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
        INNER JOIN (
            SELECT window_days, pt_bucket, MEDIAN({category}) AS next_median
            FROM hitter_windows
            GROUP BY window_days, pt_bucket
        ) t
          ON t.window_days = w.window_days
         AND t.pt_bucket = w.pt_bucket
        WHERE w.window_days = ?
          AND p.{rate_col} IS NOT NULL
          AND EXTRACT(YEAR FROM w.window_end)::INTEGER IN ({season_list_sql})
        """,
        [direction, category, label, window_days],
    ).df()

    if df.empty:
        return df

    # Strength bucket = "{label}_q[1-5]". For dense, parse from a per-row
    # quintile within the labeled population (matches Phase 3's
    # _dense_strength_buckets convention).
    values = df["value"].to_numpy()
    quintiles = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    qbins = np.clip(np.searchsorted(quintiles, values, side="left"), 0, 4)
    df["streak_strength_numeric"] = (qbins + 1).astype(float)
    return df


def _build_training_frame_sparse(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    seasons: list[int],
    window_days: int,
) -> pd.DataFrame:
    """Sparse-cat hot-only training frame.

    Sparse hot rows are duplicated across poisson_p10 and poisson_p20 in
    hitter_streak_labels (the hot determination is identical in both). We
    filter to ``SPARSE_HOT_COLD_METHOD`` so the model is not trained on the
    same row twice.

    Target: next_value > expected_next = projected_rate * next_window_pa
    (mirrors Phase 3 continuation logic).

    Streak strength: parsed from the sparse strength_bucket
    ("hot_+1.5sigma" → 1.5). Rows with bucket "hot_zna" are dropped.
    """
    rate_col = f"{category}_per_pa"
    season_list_sql = ", ".join(str(s) for s in seasons)

    df = conn.execute(
        f"""
        SELECT
            w.player_id,
            w.window_end,
            EXTRACT(YEAR FROM w.window_end)::INTEGER AS season,
            w.pt_bucket,
            w.{category} AS value,
            w.pa AS current_pa,
            n.{category} AS next_value,
            n.pa AS next_pa,
            w.babip,
            w.k_pct,
            w.bb_pct,
            w.iso,
            w.ev_avg,
            w.barrel_pct,
            w.xwoba_avg,
            p.{rate_col} AS season_rate_in_category,
            l.label,
            CASE
                WHEN n.{category} > p.{rate_col} * n.pa THEN 1 ELSE 0
            END AS target
        FROM hitter_windows w
        INNER JOIN hitter_windows n
          ON n.player_id = w.player_id
         AND n.window_days = w.window_days
         AND n.window_end = w.window_end + INTERVAL (w.window_days) DAY
        INNER JOIN hitter_streak_labels l
          ON l.player_id = w.player_id
         AND l.window_end = w.window_end
         AND l.window_days = w.window_days
         AND l.category = ?
         AND l.cold_method = ?
         AND l.label = 'hot'
        INNER JOIN hitter_projection_rates p
          ON p.player_id = w.player_id
         AND p.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
        WHERE w.window_days = ?
          AND p.{rate_col} IS NOT NULL
          AND EXTRACT(YEAR FROM w.window_end)::INTEGER IN ({season_list_sql})
        """,
        [category, SPARSE_HOT_COLD_METHOD, window_days],
    ).df()

    if df.empty:
        return df

    # Sparse strength_bucket isn't stored on labels — recompute here from the
    # window's expected_current vs value (matches Phase 3 _sparse_strength_buckets).
    df["expected_current"] = df["season_rate_in_category"] * df["current_pa"]
    denom = df["expected_current"].pow(0.5).replace(0, np.nan)
    df["z"] = (df["value"] - df["expected_current"]) / denom
    # Drop rows with NaN z (would be _zna strength_bucket).
    df = df.dropna(subset=["z"])
    if df.empty:
        return df
    # Half-sigma value, clamped to [+0.5, +3.0] for hot models (z is positive
    # by definition of hot under empirical p90). Cold edge cases (z below 0)
    # shouldn't occur for hot rows; clip for safety.
    half = np.clip(np.round(df["z"].to_numpy() * 2) / 2.0, 0.5, 3.0)
    df["streak_strength_numeric"] = half
    return df


def build_training_frame(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    direction: StreakDirection,
    season_set: str,
    window_days: int,
) -> pd.DataFrame:
    """Return a feature + target DataFrame for one (category, direction) model.

    Columns: EXPECTED_FEATURE_COLUMNS + 'target' + 'player_id' + 'season' +
    'window_end' (audit only).

    Filters:
    - window_days = ``window_days`` (= 14 for Phase 4)
    - current label matches the model's direction (hot for above, cold for below)
    - season in ``season_set`` (e.g. ``"2023-2024"`` keeps two seasons of rows)
    - season_rate_in_category IS NOT NULL (drops Phase-3-only rate rows)
    - sparse cats: filtered to ``SPARSE_HOT_COLD_METHOD`` to dedupe
    - strength_bucket parses to a numeric (drops {label}_zna rows)

    Empty DataFrame is returned when no labeled rows survive — caller handles.
    """
    seasons = _parse_season_set(season_set)
    if category in SPARSE_CATS:
        if direction != "above":
            # Sparse cats are hot-only in Phase 4. Explicit assertion-style log
            # rather than silent empty: the orchestrator should never call this.
            raise ValueError(
                f"sparse category {category!r} only has a hot model in Phase 4; "
                f"got direction={direction!r}"
            )
        df = _build_training_frame_sparse(
            conn, category=category, seasons=seasons, window_days=window_days
        )
    else:
        df = _build_training_frame_dense(
            conn,
            category=category,
            direction=direction,
            seasons=seasons,
            window_days=window_days,
        )

    if df.empty:
        return df

    # Drop rows whose strength_bucket didn't parse (already filtered for sparse;
    # belt-and-suspenders for any future buckets).
    df = df[df["streak_strength_numeric"].notna()].copy()
    if df.empty:
        return df

    # One-hot encode pt_bucket. Phase 2 buckets are {'low', 'mid', 'high'}.
    for bucket in ("low", "mid", "high"):
        df[f"pt_bucket_{bucket}"] = (df["pt_bucket"] == bucket).astype(int)

    # Drop rows with any NULL peripheral feature (~3% of windows; tolerable loss).
    feature_cols_with_nulls = [
        "babip",
        "k_pct",
        "bb_pct",
        "iso",
        "ev_avg",
        "barrel_pct",
        "xwoba_avg",
    ]
    df = df.dropna(subset=feature_cols_with_nulls)
    if df.empty:
        return df

    keep_cols = [*EXPECTED_FEATURE_COLUMNS, "target", "player_id", "season", "window_end"]
    return df[keep_cols].reset_index(drop=True)
