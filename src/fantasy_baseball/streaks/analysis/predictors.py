"""Phase 4 predictor pipeline: per-direction logistic regressions on
streak continuation, with player-grouped CV, bootstrap CIs, and
permutation feature importance.

See ``docs/superpowers/plans/2026-05-10-hot-streaks-phase-4-predictive-model.md``
for the design decisions captured at the top of this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

# Defense-in-depth guard for SQL interpolation in the helpers below: even
# though ``category`` is typed as ``StreakCategory`` (a Literal), Python does
# not enforce Literal membership at runtime.
_VALID_CATEGORIES: frozenset[str] = frozenset({"hr", "r", "rbi", "sb", "avg"})

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
    # Half-sigma value, clamped to [0.0, 3.0] to match Phase 3's encoding;
    # hot rows by construction have z >= 0.
    half = np.clip(np.round(df["z"].to_numpy() * 2) / 2.0, 0.0, 3.0)
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
    if category not in _VALID_CATEGORIES:
        raise ValueError(f"unknown streak category: {category!r}")
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


DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

# Name of the LogisticRegression step inside the Pipeline. Extracted as a
# constant so `_build_pipeline` and `bootstrap_coef_ci` agree on the key.
_LR_STEP_NAME = "lr"


@dataclass(frozen=True)
class FitResult:
    """One fitted model + the CV metrics that selected it."""

    pipeline: Pipeline
    chosen_C: float
    cv_auc_mean: float
    cv_auc_std: float
    cv_auc_per_fold: tuple[float, ...]


def _build_pipeline(C: float, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                _LR_STEP_NAME,
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_one_model(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    *,
    C_grid: Iterable[float] = DEFAULT_C_GRID,
    n_splits: int = 5,
    random_state: int = 42,
) -> FitResult:
    """Player-grouped 5-fold CV over an L2 strength grid, then refit on full
    train. Returns the refit pipeline + per-fold AUC stats for the chosen C.

    All inputs are positional from the caller's perspective; ``X`` columns must
    match ``EXPECTED_FEATURE_COLUMNS`` (the scaler is column-order-agnostic but
    consumers of FitResult.pipeline.coef_ assume this order).
    """
    y_arr = np.asarray(y, dtype=int)
    groups_arr = np.asarray(groups, dtype=int)
    grid = tuple(C_grid)

    best_C: float | None = None
    best_mean: float = -np.inf
    best_std: float = 0.0
    best_per_fold: tuple[float, ...] = ()

    for C in grid:
        per_fold: list[float] = []
        cv = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in cv.split(X, y_arr, groups=groups_arr):
            pipe = _build_pipeline(C=C, random_state=random_state)
            pipe.fit(X.iloc[train_idx], y_arr[train_idx])
            val_proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            if len(np.unique(y_arr[val_idx])) < 2:
                # Degenerate fold — single class in val. Skip rather than
                # crash; sklearn.roc_auc_score requires both classes.
                continue
            per_fold.append(float(roc_auc_score(y_arr[val_idx], val_proba)))
        if not per_fold:
            logger.warning("No usable folds for C=%g (every fold had a single-class val set)", C)
            continue
        mean = float(np.mean(per_fold))
        std = float(np.std(per_fold))
        if mean > best_mean:
            best_C = C
            best_mean = mean
            best_std = std
            best_per_fold = tuple(per_fold)

    if best_C is None:
        raise RuntimeError("fit_one_model: no C value produced any usable CV fold")

    # Refit on full train with the chosen C.
    final = _build_pipeline(C=best_C, random_state=random_state)
    final.fit(X, y_arr)

    return FitResult(
        pipeline=final,
        chosen_C=best_C,
        cv_auc_mean=best_mean,
        cv_auc_std=best_std,
        cv_auc_per_fold=best_per_fold,
    )


def bootstrap_coef_ci(
    *,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    chosen_C: float,
    n_resamples: int = 200,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Player-grouped bootstrap CIs on L2-regularized coefficients.

    For each of ``n_resamples`` iterations:
      1. Sample N players with replacement from the unique groups.
      2. Assemble the bootstrap training set from all rows of those players.
      3. Refit a fresh pipeline with the same chosen_C on that resample.
      4. Append the coefficient vector to the running list.

    Returns ``{feature_name: (p5, p95)}`` — 5th / 95th percentiles over the
    bootstrap distribution.

    Note: this function refits internally given ``chosen_C`` — it does not
    depend on any externally fitted pipeline state.
    """
    y_arr = np.asarray(y, dtype=int)
    groups_arr = np.asarray(groups, dtype=int)
    feature_names = list(X.columns)
    n_features = len(feature_names)
    rng = np.random.default_rng(random_state)

    unique_players = np.unique(groups_arr)
    coef_samples = np.empty((n_resamples, n_features), dtype=float)

    for i in range(n_resamples):
        sampled_players = rng.choice(unique_players, size=len(unique_players), replace=True)
        # Assemble rows belonging to any sampled player. A player sampled twice
        # contributes their rows twice (the correct bootstrap behavior — it's
        # the players, not the rows, we resample).
        row_chunks: list[np.ndarray] = []
        for p in sampled_players:
            row_chunks.append(np.where(groups_arr == p)[0])
        rows = np.concatenate(row_chunks)
        X_boot = X.iloc[rows]
        y_boot = y_arr[rows]
        if len(np.unique(y_boot)) < 2:
            # Degenerate resample — single class. Skip; fill with NaN
            # placeholder so np.percentile later ignores it.
            coef_samples[i, :] = np.nan
            continue
        pipe = _build_pipeline(C=chosen_C, random_state=random_state)
        pipe.fit(X_boot, y_boot)
        coef_samples[i, :] = pipe.named_steps[_LR_STEP_NAME].coef_.ravel()

    out: dict[str, tuple[float, float]] = {}
    for j, name in enumerate(feature_names):
        col = coef_samples[:, j]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            out[name] = (float("nan"), float("nan"))
        else:
            out[name] = (float(np.percentile(col, 5)), float(np.percentile(col, 95)))
    return out
