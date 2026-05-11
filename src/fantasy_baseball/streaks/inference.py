"""Phase 5 inference layer: apply fitted models to current windows.

Pure inference — no Yahoo, no I/O beyond DuckDB. The orchestrator in
:mod:`streaks.reports.sunday` wraps this with Yahoo fetching and report
rendering.

The two public entry points are:

- :func:`refit_models_for_report` — refit all 8 Phase 4 models on the
  full historical corpus (default ``2023-2025``). Returns the in-memory
  fitted pipelines so the report code does not have to re-load them
  from DuckDB. Also writes ``model_fits`` rows so the latest fit is
  inspectable post-run.
- :func:`score_player_windows` — for each ``(player_id, category)``,
  pull the player's most recent 14d window + current label, build the
  feature vector matching ``EXPECTED_FEATURE_COLUMNS``, run
  ``predict_proba``, and attribute the top peripheral drivers.

Phase 5 has no held-out validation set: 2026 is the live out-of-sample
data. The audit rows still need a ``val_auc`` column (Phase 4 schema), so
we populate it with the player-grouped CV mean. The report layer ignores
``val_auc``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fantasy_baseball.streaks.analysis.predictors import (
    DENSE_CATS,
    EXPECTED_FEATURE_COLUMNS,
    PHASE_4_MODELS,
    SPARSE_CATS,
    SPARSE_HOT_COLD_METHOD,
    build_training_frame,
    fit_one_model,
)
from fantasy_baseball.streaks.data.load_model_fits import upsert_model_fits
from fantasy_baseball.streaks.models import (
    ColdMethod,
    ModelFit,
    StreakCategory,
    StreakDirection,
    StreakLabel,
)

logger = logging.getLogger(__name__)


# Five categories scored in the Sunday report. Order is the canonical
# column order used by :mod:`streaks.reports.sunday`.
REPORT_CATEGORIES: tuple[StreakCategory, ...] = ("hr", "r", "rbi", "sb", "avg")

# Pipeline step name for the LogisticRegression layer; mirrors the
# constant in :mod:`streaks.analysis.predictors`. Re-declared here rather
# than imported because the predictors module marks it private.
_LR_STEP_NAME = "lr"
_SCALER_STEP_NAME = "scaler"


@dataclass(frozen=True)
class Driver:
    """One peripheral feature attributed as driving a prediction.

    ``feature`` is the column name from :data:`EXPECTED_FEATURE_COLUMNS`.
    ``z_score`` is the StandardScaler-transformed feature value — signed,
    so positive means the player's feature is above the training mean.
    """

    feature: str
    z_score: float


@dataclass(frozen=True)
class FittedModel:
    """One fitted Phase 4 pipeline + the metadata needed to score it.

    ``dense_quintile_cutoffs`` is the 4-tuple of quintile breakpoints from
    the training-population's category-value distribution; required to
    reproduce the ``streak_strength_numeric`` feature at inference time for
    dense cats. ``None`` for sparse cats, which use a Poisson z-score
    formula instead.
    """

    pipeline: Pipeline
    category: StreakCategory
    direction: StreakDirection
    cold_method: ColdMethod
    dense_quintile_cutoffs: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class PlayerCategoryScore:
    """Scored prediction for one (player, category).

    - ``label`` is the empirical/Poisson label for the player's most recent
      14d window. For sparse cats, we use the ``poisson_p20`` cold partition
      (matches training).
    - ``probability`` is ``P(continuation)`` from the model — ``None`` when
      the label is neutral, OR when the label is non-neutral but no model
      was trained for that ``(cat, direction)`` (sparse cats are hot-only
      in Phase 4, so ``sb cold`` ends up here).
    - ``drivers`` is empty when ``probability`` is ``None``.
    - ``window_end`` is the date the scored 14d window ends on (most
      recent in DB). Reports surface this in the header so the reader
      knows how stale the signal is.
    """

    player_id: int
    category: StreakCategory
    label: StreakLabel
    probability: float | None
    drivers: tuple[Driver, ...]
    window_end: date | None


@dataclass(frozen=True)
class ScoreSkip:
    """Why one (player_id, category) was dropped from scoring.

    Surfaced by the report layer as a one-line "Skipped N players: <reason>"
    footnote so the reader can see which players were not evaluated and
    why.
    """

    player_id: int
    reason: Literal["no_window", "no_projection_rate"]


def _parse_season_set(season_set: str) -> list[int]:
    """Mirror of the same helper in :mod:`streaks.analysis.predictors`.

    Duplicated rather than re-exported to keep this module independently
    importable; the streaks subsystem deliberately keeps inter-module
    dependencies shallow.
    """
    if "-" in season_set:
        start_str, end_str = season_set.split("-", 1)
        return list(range(int(start_str), int(end_str) + 1))
    return [int(season_set)]


def _dense_quintile_cutoffs(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    direction: StreakDirection,
    season_set: str,
    window_days: int,
) -> tuple[float, float, float, float]:
    """Re-derive the dense-cat quintile breakpoints used at training time.

    Mirrors the inline computation in
    :func:`streaks.analysis.predictors._build_training_frame_dense` — same
    SQL filter, same ``np.quantile([0.2, 0.4, 0.6, 0.8])`` over raw
    category values. We do not modify the predictors module to return
    these cutoffs because Phase 4 audit code paths are stable; instead
    we re-derive in inference, paying a ~5ms query cost for cleanliness.
    """
    label = "hot" if direction == "above" else "cold"
    rate_col = "avg" if category == "avg" else f"{category}_per_pa"
    seasons = _parse_season_set(season_set)
    season_list_sql = ", ".join(str(s) for s in seasons)
    # The SQL below mirrors _build_training_frame_dense exactly so the
    # quintile sample matches the population the training frame's
    # ``streak_strength_numeric`` was computed over.
    df = conn.execute(
        f"""
        SELECT w.{category} AS value
        FROM hitter_windows w
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
        WHERE w.window_days = ?
          AND p.{rate_col} IS NOT NULL
          AND EXTRACT(YEAR FROM w.window_end)::INTEGER IN ({season_list_sql})
        """,
        [category, label, window_days],
    ).df()
    if df.empty:
        raise RuntimeError(
            f"No labeled rows for dense quintile cutoffs ({category}, {direction}, "
            f"{season_set}); refit_models_for_report must be called only after the "
            f"label pipeline has been re-applied."
        )
    cutoffs = np.quantile(df["value"].to_numpy(dtype=float), [0.2, 0.4, 0.6, 0.8])
    return (float(cutoffs[0]), float(cutoffs[1]), float(cutoffs[2]), float(cutoffs[3]))


def refit_models_for_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    season_set_train: str = "2023-2025",
    window_days: int = 14,
) -> dict[tuple[StreakCategory, StreakDirection], FittedModel]:
    """Refit all Phase 4 models on the full historical corpus.

    Calls :func:`build_training_frame` + :func:`fit_one_model` directly
    rather than :func:`fit_all_models` because Phase 5 has no held-out
    val partition (2026 is the live out-of-sample). One ``model_fits``
    row is written per fitted model with ``n_val_rows=0`` and
    ``val_auc=cv_auc_mean``; the report layer ignores ``val_auc``.

    Returns ``{(cat, direction): FittedModel}``. Skipped models (empty
    frame or single-class target) are omitted from the dict — callers
    must handle missing keys.
    """
    fits: dict[tuple[StreakCategory, StreakDirection], FittedModel] = {}
    fit_rows: list[ModelFit] = []
    timestamp = datetime.now(UTC)

    for cat, direction in PHASE_4_MODELS:
        df = build_training_frame(
            conn,
            category=cat,
            direction=direction,
            season_set=season_set_train,
            window_days=window_days,
        )
        if df.empty:
            logger.info("refit_models_for_report: no rows for (%s, %s)", cat, direction)
            continue
        X = df[list(EXPECTED_FEATURE_COLUMNS)]
        y = df["target"].to_numpy()
        groups = df["player_id"].to_numpy()
        if len(np.unique(y)) < 2:
            logger.info("refit_models_for_report: single-class target for (%s, %s)", cat, direction)
            continue

        fit_result = fit_one_model(X, y, groups)
        cold_method: ColdMethod = SPARSE_HOT_COLD_METHOD if cat in SPARSE_CATS else "empirical"
        cutoffs: tuple[float, float, float, float] | None = None
        if cat in DENSE_CATS:
            cutoffs = _dense_quintile_cutoffs(
                conn,
                category=cat,
                direction=direction,
                season_set=season_set_train,
                window_days=window_days,
            )

        fits[(cat, direction)] = FittedModel(
            pipeline=fit_result.pipeline,
            category=cat,
            direction=direction,
            cold_method=cold_method,
            dense_quintile_cutoffs=cutoffs,
        )
        model_id = f"{cat}_{'hot' if direction == 'above' else 'cold'}_{season_set_train}"
        # Persist pipeline params so ``load_models_from_fits`` can reconstruct
        # the fitted Pipeline without retraining. Coef / mean / scale are stored
        # column-aligned with ``feature_columns`` — that order is the contract.
        scaler = fit_result.pipeline.named_steps[_SCALER_STEP_NAME]
        lr = fit_result.pipeline.named_steps[_LR_STEP_NAME]
        fit_rows.append(
            ModelFit(
                model_id=model_id,
                category=cat,
                direction=direction,
                season_set=season_set_train,
                window_days=window_days,
                cold_method=cold_method,
                chosen_C=fit_result.chosen_C,
                cv_auc_mean=fit_result.cv_auc_mean,
                cv_auc_std=fit_result.cv_auc_std,
                val_auc=fit_result.cv_auc_mean,
                n_train_rows=len(df),
                n_val_rows=0,
                fit_timestamp=timestamp,
                feature_columns=tuple(EXPECTED_FEATURE_COLUMNS),
                coef=tuple(float(c) for c in lr.coef_.ravel()),
                intercept=float(lr.intercept_[0]),
                scaler_mean=tuple(float(m) for m in scaler.mean_),
                scaler_scale=tuple(float(s) for s in scaler.scale_),
                dense_quintile_cutoffs=cutoffs,
            )
        )

    upsert_model_fits(conn, fit_rows)
    logger.info("refit_models_for_report: wrote %d model_fits rows", len(fit_rows))
    return fits


# Defense-in-depth: even though category/direction/cold_method are typed as
# Literals downstream, DuckDB returns plain strings and the report layer is
# downstream of any future schema corruption — narrow + validate here.
_VALID_REPORT_CATEGORIES: frozenset[str] = frozenset(REPORT_CATEGORIES)
_VALID_DIRECTIONS: frozenset[str] = frozenset({"above", "below"})
_VALID_COLD_METHODS: frozenset[str] = frozenset({"empirical", "poisson_p10", "poisson_p20"})


def _narrow_category(value: str) -> StreakCategory:
    if value not in _VALID_REPORT_CATEGORIES:
        raise RuntimeError(f"unexpected category {value!r} in model_fits")
    return value  # type: ignore[return-value]


def _narrow_direction(value: str) -> StreakDirection:
    if value not in _VALID_DIRECTIONS:
        raise RuntimeError(f"unexpected direction {value!r} in model_fits")
    return value  # type: ignore[return-value]


def _narrow_cold_method(value: str) -> ColdMethod:
    if value not in _VALID_COLD_METHODS:
        raise RuntimeError(f"unexpected cold_method {value!r} in model_fits")
    return value  # type: ignore[return-value]


def load_models_from_fits(
    conn: duckdb.DuckDBPyConnection,
) -> dict[tuple[StreakCategory, StreakDirection], FittedModel]:
    """Reconstruct fitted Pipelines from the most recent ``model_fits`` rows.

    Selects the most recent ``fit_timestamp`` group; for each ``(category,
    direction)`` row in that group, rebuilds a ``Pipeline`` of
    :class:`StandardScaler` + :class:`LogisticRegression` whose parameters are
    set directly from the persisted coefficients / intercept / scaler mean
    / scaler scale. The result produces byte-identical predictions to the
    original fit (no retraining).

    Skips rows where the persisted pipeline params are NULL — those are
    pre-Phase-B audit rows that don't carry the state needed to round-trip.
    Raises :class:`RuntimeError` if ``model_fits`` is empty.
    """
    rows = conn.execute(
        """
        SELECT category, direction, cold_method, dense_quintile_cutoffs,
               feature_columns, coef, intercept,
               scaler_mean, scaler_scale, fit_timestamp
        FROM model_fits
        WHERE fit_timestamp = (SELECT MAX(fit_timestamp) FROM model_fits)
        """
    ).fetchall()
    if not rows:
        raise RuntimeError("model_fits is empty; refit before loading")

    out: dict[tuple[StreakCategory, StreakDirection], FittedModel] = {}
    for (
        category,
        direction,
        cold_method,
        quintile_cutoffs,
        feature_columns,
        coef,
        intercept,
        scaler_mean,
        scaler_scale,
        _fit_timestamp,
    ) in rows:
        if (
            feature_columns is None
            or coef is None
            or intercept is None
            or scaler_mean is None
            or scaler_scale is None
        ):
            # Pre-Phase-B row — can't reconstruct, skip rather than crash.
            logger.info(
                "load_models_from_fits: skipping (%s, %s) — missing pipeline params",
                category,
                direction,
            )
            continue

        cat_narrow = _narrow_category(str(category))
        dir_narrow = _narrow_direction(str(direction))
        cm_narrow = _narrow_cold_method(str(cold_method))

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(scaler_mean, dtype=np.float64)
        scaler.scale_ = np.asarray(scaler_scale, dtype=np.float64)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(feature_columns)
        scaler.feature_names_in_ = np.asarray(feature_columns, dtype=object)

        clf = LogisticRegression()
        clf.coef_ = np.asarray([coef], dtype=np.float64)
        clf.intercept_ = np.asarray([intercept], dtype=np.float64)
        clf.classes_ = np.asarray([0, 1])
        clf.n_features_in_ = len(feature_columns)
        # Deliberately *do not* set ``clf.feature_names_in_`` — the LR step
        # inside a Pipeline receives the scaler's numpy output, never a
        # DataFrame, so populating feature_names_in_ here would trigger
        # sklearn's "X does not have valid feature names" UserWarning at every
        # predict_proba call. Setting it on the scaler is sufficient: the
        # scaler validates the inbound DataFrame, then forwards a feature-name
        # -less ndarray to the LR — matching the behavior of a freshly fit
        # Pipeline.

        # Step names must match the predictors module so ``top_drivers`` (and
        # any other consumer that pulls ``named_steps["lr"]``) keeps working
        # on loaded pipelines exactly as it does on freshly-fit ones.
        pipeline = Pipeline([(_SCALER_STEP_NAME, scaler), (_LR_STEP_NAME, clf)])
        cutoffs: tuple[float, float, float, float] | None
        if quintile_cutoffs is None:
            cutoffs = None
        else:
            # DuckDB returns DOUBLE[] as a Python list; the dataclass field
            # is a 4-tuple by contract.
            if len(quintile_cutoffs) != 4:
                raise RuntimeError(
                    f"dense_quintile_cutoffs must have length 4 for ({cat_narrow}, "
                    f"{dir_narrow}); got {len(quintile_cutoffs)}"
                )
            cutoffs = (
                float(quintile_cutoffs[0]),
                float(quintile_cutoffs[1]),
                float(quintile_cutoffs[2]),
                float(quintile_cutoffs[3]),
            )
        out[(cat_narrow, dir_narrow)] = FittedModel(
            pipeline=pipeline,
            category=cat_narrow,
            direction=dir_narrow,
            cold_method=cm_narrow,
            dense_quintile_cutoffs=cutoffs,
        )
    return out


def _dense_streak_strength(value: float, cutoffs: tuple[float, float, float, float]) -> float:
    """Bin a dense-cat value to its 1-5 quintile using stored cutoffs.

    Mirrors ``np.clip(np.searchsorted(quintiles, values, side='left'), 0, 4) + 1``
    from :func:`_build_training_frame_dense`.
    """
    cuts = np.asarray(cutoffs, dtype=float)
    bin_idx = int(np.clip(np.searchsorted(cuts, value, side="left"), 0, 4))
    return float(bin_idx + 1)


def _sparse_streak_strength(*, value: int, window_pa: int, season_rate: float) -> float | None:
    """Compute the sparse-cat half-sigma encoding for one live window.

    Returns ``None`` when expected is zero (a degenerate window — the
    Phase 4 training frame would have dropped the row to a ``_zna``
    strength bucket; we can't predict it either).
    """
    expected = season_rate * window_pa
    if expected <= 0:
        return None
    denom = expected**0.5
    z = (value - expected) / denom
    return float(np.clip(np.round(z * 2) / 2.0, 0.0, 3.0))


def _build_feature_row(
    *,
    window: pd.Series,
    season_rate: float,
    streak_strength_numeric: float,
) -> pd.DataFrame:
    """Assemble a single-row DataFrame in ``EXPECTED_FEATURE_COLUMNS`` order.

    The scaler is column-order-agnostic but the coefficients in
    ``pipeline.named_steps["lr"]`` are aligned with that order — we keep
    the same order so driver attribution does not pull the wrong
    feature-name labels.
    """
    pt_bucket = str(window["pt_bucket"])
    row = {
        "streak_strength_numeric": streak_strength_numeric,
        "babip": float(window["babip"]),
        "k_pct": float(window["k_pct"]),
        "bb_pct": float(window["bb_pct"]),
        "iso": float(window["iso"]),
        "ev_avg": float(window["ev_avg"]),
        "barrel_pct": float(window["barrel_pct"]),
        "xwoba_avg": float(window["xwoba_avg"]),
        "season_rate_in_category": float(season_rate),
        "pt_bucket_low": 1 if pt_bucket == "low" else 0,
        "pt_bucket_mid": 1 if pt_bucket == "mid" else 0,
        "pt_bucket_high": 1 if pt_bucket == "high" else 0,
    }
    return pd.DataFrame([row], columns=list(EXPECTED_FEATURE_COLUMNS))


def top_drivers(
    *,
    pipeline: Pipeline,
    feature_row: pd.DataFrame,
    k: int = 2,
    exclude: Iterable[str] = (
        "streak_strength_numeric",
        "season_rate_in_category",
        "pt_bucket_low",
        "pt_bucket_mid",
        "pt_bucket_high",
    ),
) -> tuple[Driver, ...]:
    """Rank features by ``|coef * x_scaled|`` and return the top-k as Drivers.

    ``exclude`` defaults to the non-peripheral inputs: streak strength,
    the projection rate, and the pt_bucket one-hots. Those columns are
    in the model but the report's "drivers" section is about *peripheral*
    signal (barrel%, xwOBA, etc.) — the columns Hart can read as
    independent evidence the streak is real. Pass ``exclude=()`` to
    include them.
    """
    scaler = pipeline.named_steps[_SCALER_STEP_NAME]
    lr = pipeline.named_steps[_LR_STEP_NAME]
    # Pass the DataFrame so sklearn's feature-name check (fit-time vs
    # predict-time) stays satisfied — converting to numpy here triggered
    # a UserWarning at every call site.
    x_scaled = scaler.transform(feature_row)[0]
    coefs = lr.coef_.ravel()
    feature_names = list(feature_row.columns)
    excluded = set(exclude)
    contributions: list[tuple[str, float, float]] = []
    for j, name in enumerate(feature_names):
        if name in excluded:
            continue
        magnitude = abs(float(coefs[j]) * float(x_scaled[j]))
        contributions.append((name, magnitude, float(x_scaled[j])))
    contributions.sort(key=lambda t: t[1], reverse=True)
    return tuple(Driver(feature=name, z_score=z) for name, _mag, z in contributions[:k])


def _label_cold_method(category: StreakCategory) -> ColdMethod:
    """Per-cat partition of ``hitter_streak_labels`` to read from.

    Mirrors the Phase 4 training partition: dense cats are labeled with
    ``empirical``; sparse cats are labeled twice (poisson_p10 and
    poisson_p20) and we read the p20 row to match the training cohort.
    """
    return SPARSE_HOT_COLD_METHOD if category in SPARSE_CATS else "empirical"


# Defense-in-depth guard for the label loader: even though ``category``
# is typed as :data:`StreakCategory` (a Literal), Python does not enforce
# Literal membership at runtime and the table is shared with Phase 4
# pipelines, so a corrupted row would otherwise leak through.
_VALID_CATEGORIES: frozenset[StreakCategory] = frozenset(REPORT_CATEGORIES)


def _coerce_label(label: str) -> StreakLabel:
    """Narrow a raw label string from DuckDB to the StreakLabel Literal."""
    if label == "hot":
        return "hot"
    if label == "cold":
        return "cold"
    if label == "neutral":
        return "neutral"
    raise RuntimeError(f"unexpected label {label!r} in hitter_streak_labels")


def _load_most_recent_windows(
    conn: duckdb.DuckDBPyConnection,
    *,
    player_ids: list[int],
    window_days: int,
    window_end_on_or_before: date,
) -> dict[int, pd.Series]:
    """Pull the most-recent ``hitter_windows`` row for every player in one query.

    Returns ``{player_id: window_series}`` for players that have at least
    one window on or before the cutoff date. Players without a window
    are absent from the returned dict — callers translate that to a
    ``no_window`` skip.

    Uses a registered DataFrame to feed the player_ids in, then a
    QUALIFY ROW_NUMBER pattern to keep only the latest row per player.
    """
    if not player_ids:
        return {}
    players_df = pd.DataFrame({"player_id": player_ids})  # noqa: F841 — registered via pandas scan
    df = conn.execute(
        """
        SELECT w.player_id, w.window_end, w.window_days, w.pa, w.hr, w.r, w.rbi,
               w.sb, w.avg, w.babip, w.k_pct, w.bb_pct, w.iso, w.ev_avg,
               w.barrel_pct, w.xwoba_avg, w.pt_bucket
        FROM hitter_windows w
        INNER JOIN players_df p ON p.player_id = w.player_id
        WHERE w.window_days = ?
          AND w.window_end <= ?
        QUALIFY ROW_NUMBER() OVER (PARTITION BY w.player_id ORDER BY w.window_end DESC) = 1
        """,
        [int(window_days), window_end_on_or_before],
    ).df()
    return {int(row["player_id"]): row for _, row in df.iterrows()}


def _load_labels(
    conn: duckdb.DuckDBPyConnection,
    *,
    windows: dict[int, pd.Series],
    window_days: int,
) -> dict[tuple[int, StreakCategory], StreakLabel]:
    """Pull labels for every scored (player, window_end, category) in one query.

    Filters in SQL to the per-cat ``cold_method`` partition used at
    training time so the in-memory dict has at most one entry per
    (player_id, category). Missing entries — e.g. sparse-cat rows
    skipped because the player has no projection rate — translate to a
    ``neutral`` label at the call site.
    """
    if not windows:
        return {}
    targets = pd.DataFrame(  # noqa: F841 — referenced via pandas scan below
        [
            {"player_id": pid, "window_end": pd.Timestamp(row["window_end"]).date()}
            for pid, row in windows.items()
        ]
    )
    df = conn.execute(
        """
        SELECT l.player_id, l.category, l.cold_method, l.label
        FROM hitter_streak_labels l
        INNER JOIN targets t
          ON t.player_id = l.player_id AND t.window_end = l.window_end
        WHERE l.window_days = ?
          AND (
            (l.category IN ('r', 'rbi', 'avg') AND l.cold_method = 'empirical')
            OR (l.category IN ('hr', 'sb') AND l.cold_method = 'poisson_p20')
          )
        """,
        [int(window_days)],
    ).df()
    out: dict[tuple[int, StreakCategory], StreakLabel] = {}
    for row in df.itertuples(index=False):
        cat = row.category
        if cat not in _VALID_CATEGORIES:
            continue
        out[(int(row.player_id), cat)] = _coerce_label(str(row.label))
    return out


def _load_projection_rates(
    conn: duckdb.DuckDBPyConnection,
    *,
    player_ids: list[int],
    season: int,
) -> dict[tuple[int, StreakCategory], float]:
    """Pull all per-category projection rates for the listed players in one query.

    Returns ``{(player_id, category): rate}`` with absent entries for any
    (player, cat) where the rate is NULL.
    """
    if not player_ids:
        return {}
    players_df = pd.DataFrame({"player_id": player_ids})  # noqa: F841 — pandas scan
    df = conn.execute(
        """
        SELECT p.player_id, p.hr_per_pa, p.sb_per_pa, p.r_per_pa, p.rbi_per_pa, p.avg
        FROM hitter_projection_rates p
        INNER JOIN players_df pl ON pl.player_id = p.player_id
        WHERE p.season = ?
        """,
        [int(season)],
    ).df()
    out: dict[tuple[int, StreakCategory], float] = {}
    cat_columns: tuple[tuple[StreakCategory, str], ...] = (
        ("hr", "hr_per_pa"),
        ("sb", "sb_per_pa"),
        ("r", "r_per_pa"),
        ("rbi", "rbi_per_pa"),
        ("avg", "avg"),
    )
    for row in df.itertuples(index=False):
        pid = int(row.player_id)
        for cat, col in cat_columns:
            value = getattr(row, col)
            if value is not None and not pd.isna(value):
                out[(pid, cat)] = float(value)
    return out


_PERIPHERAL_COLS: tuple[str, ...] = (
    "babip",
    "k_pct",
    "bb_pct",
    "iso",
    "ev_avg",
    "barrel_pct",
    "xwoba_avg",
)


def score_player_windows(
    conn: duckdb.DuckDBPyConnection,
    *,
    models: dict[tuple[StreakCategory, StreakDirection], FittedModel],
    player_ids: Iterable[int],
    window_end_on_or_before: date,
    window_days: int = 14,
    scoring_season: int,
) -> tuple[list[PlayerCategoryScore], list[ScoreSkip]]:
    """Score every (player, REPORT_CATEGORIES) for the listed player_ids.

    ``window_end_on_or_before`` is normally ``date.today()`` — Statcast
    has a 1-2 day publication lag so the most recent window typically
    ends 1-2 days before today. ``scoring_season`` is the season whose
    ``hitter_projection_rates`` rows are looked up for
    ``season_rate_in_category`` (the live 2026 rate).

    Returns ``(scores, skips)``. ``scores`` contains one entry per
    (player, category) for every player who has a window — neutral
    categories included so the report's roster grid is uniform.

    Performance: pulls windows / labels / projection rates in three
    bulk queries up front, then iterates in pure Python. The previous
    implementation issued one query per (player, category, lookup),
    which was ~500 round-trips per report run.
    """
    unique_ids = list(dict.fromkeys(int(p) for p in player_ids))
    if not unique_ids:
        return [], []

    windows = _load_most_recent_windows(
        conn,
        player_ids=unique_ids,
        window_days=window_days,
        window_end_on_or_before=window_end_on_or_before,
    )
    labels = _load_labels(conn, windows=windows, window_days=window_days)
    rates = _load_projection_rates(conn, player_ids=unique_ids, season=scoring_season)

    scores: list[PlayerCategoryScore] = []
    skips: list[ScoreSkip] = []

    for player_id in unique_ids:
        window = windows.get(player_id)
        if window is None:
            skips.append(ScoreSkip(player_id=player_id, reason="no_window"))
            continue
        window_end = pd.Timestamp(window["window_end"]).date()
        peripherals_null = any(pd.isna(window[c]) for c in _PERIPHERAL_COLS)

        for category in REPORT_CATEGORIES:
            label = labels.get((player_id, category), "neutral")
            score = _score_one(
                player_id=player_id,
                category=category,
                label=label,
                window=window,
                window_end=window_end,
                peripherals_null=peripherals_null,
                models=models,
                season_rate=rates.get((player_id, category)),
            )
            scores.append(score)

    return scores, skips


def _score_one(
    *,
    player_id: int,
    category: StreakCategory,
    label: StreakLabel,
    window: pd.Series,
    window_end: date,
    peripherals_null: bool,
    models: dict[tuple[StreakCategory, StreakDirection], FittedModel],
    season_rate: float | None,
) -> PlayerCategoryScore:
    """Build the score for one (player, category) given pre-loaded inputs.

    Returns a ``PlayerCategoryScore`` with ``probability=None`` whenever
    we can't score: neutral label, no trained model (sparse cold), no
    projection rate, or NULL peripherals in the current window.
    """
    base = PlayerCategoryScore(
        player_id=player_id,
        category=category,
        label=label,
        probability=None,
        drivers=(),
        window_end=window_end,
    )
    if label == "neutral":
        return base

    direction: StreakDirection = "above" if label == "hot" else "below"
    model = models.get((category, direction))
    # Sparse cats are hot-only in Phase 4 (sb cold / hr cold have no model);
    # missing rate or NULL peripherals mean we can't build the feature vector.
    if model is None or season_rate is None or peripherals_null:
        return base

    if category in DENSE_CATS:
        assert model.dense_quintile_cutoffs is not None
        strength = _dense_streak_strength(
            value=float(window[category]),
            cutoffs=model.dense_quintile_cutoffs,
        )
    else:
        strength = (
            _sparse_streak_strength(
                value=int(window[category]),
                window_pa=int(window["pa"]),
                season_rate=season_rate,
            )
            or 0.0
        )

    feature_row = _build_feature_row(
        window=window,
        season_rate=season_rate,
        streak_strength_numeric=strength,
    )
    proba = float(model.pipeline.predict_proba(feature_row)[0, 1])
    drivers = top_drivers(pipeline=model.pipeline, feature_row=feature_row, k=2)
    return PlayerCategoryScore(
        player_id=player_id,
        category=category,
        label=label,
        probability=proba,
        drivers=drivers,
        window_end=window_end,
    )
