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
from datetime import UTC, datetime
from typing import Literal

import duckdb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fantasy_baseball.streaks.data.load_model_fits import upsert_model_fits
from fantasy_baseball.streaks.models import ModelFit, StreakCategory, StreakDirection

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
SPARSE_HOT_COLD_METHOD: Literal["poisson_p20"] = "poisson_p20"

# Final ordered list of feature column names in the training frame.
#
# Phase 4 acceptance found that ``hitter_statcast_pa.barrel`` is NULL for the
# full 598K-row local Statcast corpus (Phase 1 fetch issue, surfaced 2026-05-10
# because Phase 3 did not use ``barrel_pct`` as a load-bearing feature). With
# all windows showing ``barrel_pct = NULL`` the dropna below would empty every
# model's training frame. We omit ``barrel_pct`` from the Phase 4 feature set
# pending a separate fix to the Statcast fetch; ``xwoba_avg`` is the dominant
# Statcast peripheral and captures most of barrel's signal anyway.
EXPECTED_FEATURE_COLUMNS: tuple[str, ...] = (
    "streak_strength_numeric",
    "babip",
    "k_pct",
    "bb_pct",
    "iso",
    "ev_avg",
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

    # Drop rows with any NULL peripheral feature (~3% of windows; tolerable
    # loss). ``barrel_pct`` is intentionally not in this list — see the
    # comment on ``EXPECTED_FEATURE_COLUMNS`` above.
    feature_cols_with_nulls = [
        "babip",
        "k_pct",
        "bb_pct",
        "iso",
        "ev_avg",
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


@dataclass(frozen=True)
class EvaluationResult:
    """ROC-AUC + reliability diagram for one model on one held-out set.

    Bin arrays have the same length and are aligned 1:1 — index k refers to
    the same (non-empty) bin in all three.
    """

    auc: float
    reliability_bin_centers: tuple[float, ...]
    reliability_observed: tuple[float, ...]
    reliability_bin_counts: tuple[int, ...]


def evaluate_model(
    *,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> EvaluationResult:
    """Held-out AUC + reliability diagram (n_bins equal-width bins).

    Empty bins are dropped from the returned arrays; the three reliability_*
    tuples stay aligned. ``bin_centers`` are the bin midpoints, not the mean
    predicted probability in the bin — simpler to interpret on a reliability
    plot, and accurate to within bin width / 2 of the mean.
    """
    y_arr = np.asarray(y, dtype=int)
    proba = pipeline.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y_arr, proba))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, proba, side="right") - 1, 0, n_bins - 1)

    bin_centers: list[float] = []
    bin_observed: list[float] = []
    bin_counts: list[int] = []
    for k in range(n_bins):
        mask = bin_idx == k
        if not mask.any():
            continue
        bin_centers.append(0.5 * (bin_edges[k] + bin_edges[k + 1]))
        bin_observed.append(float(y_arr[mask].mean()))
        bin_counts.append(int(mask.sum()))

    return EvaluationResult(
        auc=auc,
        reliability_bin_centers=tuple(bin_centers),
        reliability_observed=tuple(bin_observed),
        reliability_bin_counts=tuple(bin_counts),
    )


def permutation_feature_importance(
    *,
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Sklearn permutation importance on the validation set.

    For each feature: shuffle it, measure AUC drop, repeat ``n_repeats`` times,
    report (mean_drop, std_drop).
    """
    y_arr = np.asarray(y_val, dtype=int)

    def _scorer(estimator, X_, y_):
        return roc_auc_score(y_, estimator.predict_proba(X_)[:, 1])

    result = permutation_importance(
        pipeline,
        X_val,
        y_arr,
        scoring=_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return {
        name: (float(result.importances_mean[i]), float(result.importances_std[i]))
        for i, name in enumerate(X_val.columns)
    }


@dataclass(frozen=True)
class PerModelResult:
    """Everything the notebook needs for one (category, direction)."""

    fit: FitResult
    evaluation: EvaluationResult
    coef_ci: dict[str, tuple[float, float]]
    permutation_importance: dict[str, tuple[float, float]]
    n_train_rows: int
    n_val_rows: int
    cold_method: Literal["empirical", "poisson_p20"]


@dataclass(frozen=True)
class AllModelsResult:
    """Output of fit_all_models — one entry per (category, direction) pair."""

    fits: dict[tuple[StreakCategory, StreakDirection], PerModelResult | None]
    season_set_train: str
    season_set_val: str
    window_days: int


def _split_frame_by_season(
    df: pd.DataFrame, *, season_set_train: str, season_set_val: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition the unified training frame by season."""
    train_seasons = set(_parse_season_set(season_set_train))
    val_seasons = set(_parse_season_set(season_set_val))
    df_train = df[df["season"].isin(train_seasons)].copy()
    df_val = df[df["season"].isin(val_seasons)].copy()
    return df_train, df_val


def _fit_one_phase_4_model(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    direction: StreakDirection,
    season_set_train: str,
    season_set_val: str,
    window_days: int,
    C_grid: Iterable[float],
    n_bootstrap: int,
    random_state: int,
) -> PerModelResult | None:
    """Build train + val frames for one model; fit, evaluate, bootstrap, importance."""
    train_set = set(_parse_season_set(season_set_train))
    val_set = set(_parse_season_set(season_set_val))
    if train_set & val_set:
        raise ValueError(f"train and val season sets overlap: {sorted(train_set & val_set)}")
    all_seasons = sorted(train_set | val_set)
    season_set_combined = f"{min(all_seasons)}-{max(all_seasons)}"
    full = build_training_frame(
        conn,
        category=category,
        direction=direction,
        season_set=season_set_combined,
        window_days=window_days,
    )
    if full.empty:
        logger.info("No training frame rows for (%s, %s) — skipping", category, direction)
        return None

    df_train, df_val = _split_frame_by_season(
        full, season_set_train=season_set_train, season_set_val=season_set_val
    )
    if df_train.empty or df_val.empty:
        logger.info(
            "Train/val split empty for (%s, %s): train=%d val=%d — skipping",
            category,
            direction,
            len(df_train),
            len(df_val),
        )
        return None

    feature_cols = list(EXPECTED_FEATURE_COLUMNS)
    X_train = df_train[feature_cols]
    y_train = df_train["target"].to_numpy()
    groups = df_train["player_id"].to_numpy()
    X_val = df_val[feature_cols]
    y_val = df_val["target"].to_numpy()

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        logger.info("Single-class target for (%s, %s) — skipping", category, direction)
        return None

    fit_result = fit_one_model(
        X_train,
        y_train,
        groups,
        C_grid=C_grid,
        n_splits=5,
        random_state=random_state,
    )
    eval_result = evaluate_model(pipeline=fit_result.pipeline, X=X_val, y=y_val, n_bins=10)
    coef_ci = bootstrap_coef_ci(
        X=X_train,
        y=y_train,
        groups=groups,
        chosen_C=fit_result.chosen_C,
        n_resamples=n_bootstrap,
        random_state=random_state,
    )
    importance = permutation_feature_importance(
        pipeline=fit_result.pipeline,
        X_val=X_val,
        y_val=y_val,
        n_repeats=10,
        random_state=random_state,
    )
    cold_method: Literal["empirical", "poisson_p20"] = (
        SPARSE_HOT_COLD_METHOD if category in SPARSE_CATS else "empirical"
    )
    return PerModelResult(
        fit=fit_result,
        evaluation=eval_result,
        coef_ci=coef_ci,
        permutation_importance=importance,
        n_train_rows=len(df_train),
        n_val_rows=len(df_val),
        cold_method=cold_method,
    )


def fit_all_models(
    conn: duckdb.DuckDBPyConnection,
    *,
    season_set_train: str = "2023-2024",
    season_set_val: str = "2025",
    window_days: int = 14,
    C_grid: Iterable[float] = DEFAULT_C_GRID,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> AllModelsResult:
    """Fit all 8 Phase 4 models, persist metadata to model_fits, return results.

    Skips any model whose training or validation frame is empty (logs the
    skip and records ``None`` in the result dict). All non-skipped models
    write one row to ``model_fits``.
    """
    fits: dict[tuple[StreakCategory, StreakDirection], PerModelResult | None] = {}
    fit_rows: list[ModelFit] = []
    timestamp = datetime.now(UTC)

    for cat, direction in PHASE_4_MODELS:
        per_model = _fit_one_phase_4_model(
            conn,
            category=cat,
            direction=direction,
            season_set_train=season_set_train,
            season_set_val=season_set_val,
            window_days=window_days,
            C_grid=C_grid,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        fits[(cat, direction)] = per_model
        if per_model is None:
            continue
        model_id = f"{cat}_{'hot' if direction == 'above' else 'cold'}_{season_set_train}"
        fit_rows.append(
            ModelFit(
                model_id=model_id,
                category=cat,
                direction=direction,
                season_set=season_set_train,
                window_days=window_days,
                cold_method=per_model.cold_method,
                chosen_C=per_model.fit.chosen_C,
                cv_auc_mean=per_model.fit.cv_auc_mean,
                cv_auc_std=per_model.fit.cv_auc_std,
                val_auc=per_model.evaluation.auc,
                n_train_rows=per_model.n_train_rows,
                n_val_rows=per_model.n_val_rows,
                fit_timestamp=timestamp,
            )
        )

    upsert_model_fits(conn, fit_rows)
    logger.info("Wrote %d rows to model_fits", len(fit_rows))

    return AllModelsResult(
        fits=fits,
        season_set_train=season_set_train,
        season_set_val=season_set_val,
        window_days=window_days,
    )
