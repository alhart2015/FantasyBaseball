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
from sklearn.pipeline import Pipeline

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
            )
        )

    upsert_model_fits(conn, fit_rows)
    logger.info("refit_models_for_report: wrote %d model_fits rows", len(fit_rows))
    return fits


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


def _most_recent_window(
    conn: duckdb.DuckDBPyConnection,
    *,
    player_id: int,
    window_days: int,
    window_end_on_or_before: date,
) -> pd.Series | None:
    """Pull the latest ``hitter_windows`` row for the player, with its label.

    The label row is joined per-category and pivoted in the caller — this
    helper only returns the window.
    """
    df = conn.execute(
        """
        SELECT player_id, window_end, window_days, pa, hr, r, rbi, sb, avg,
               babip, k_pct, bb_pct, iso, ev_avg, barrel_pct, xwoba_avg,
               pt_bucket
        FROM hitter_windows
        WHERE player_id = ?
          AND window_days = ?
          AND window_end <= ?
        ORDER BY window_end DESC
        LIMIT 1
        """,
        [int(player_id), int(window_days), window_end_on_or_before],
    ).df()
    if df.empty:
        return None
    return df.iloc[0]


def _category_label(
    conn: duckdb.DuckDBPyConnection,
    *,
    player_id: int,
    window_end: date,
    window_days: int,
    category: StreakCategory,
) -> StreakLabel | None:
    """Read the stored label for one (player, window, category).

    Returns ``None`` if no row exists (e.g. sparse-cat row missing
    because the player has no projection rate — the sparse label writer
    skips those).
    """
    cold_method = _label_cold_method(category)
    row = conn.execute(
        """
        SELECT label FROM hitter_streak_labels
        WHERE player_id = ? AND window_end = ? AND window_days = ?
              AND category = ? AND cold_method = ?
        """,
        [int(player_id), window_end, int(window_days), category, cold_method],
    ).fetchone()
    if row is None:
        return None
    label = row[0]
    if label == "hot":
        return "hot"
    if label == "cold":
        return "cold"
    if label == "neutral":
        return "neutral"
    raise RuntimeError(f"unexpected label {label!r} in hitter_streak_labels")


def _projection_rate(
    conn: duckdb.DuckDBPyConnection,
    *,
    player_id: int,
    season: int,
    category: StreakCategory,
) -> float | None:
    """Pull the per-category 2026 rate for one player; ``None`` if missing."""
    rate_col = "avg" if category == "avg" else f"{category}_per_pa"
    row = conn.execute(
        f"SELECT {rate_col} FROM hitter_projection_rates WHERE player_id = ? AND season = ?",
        [int(player_id), int(season)],
    ).fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


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
    """
    scores: list[PlayerCategoryScore] = []
    skips: list[ScoreSkip] = []

    for player_id in player_ids:
        window = _most_recent_window(
            conn,
            player_id=player_id,
            window_days=window_days,
            window_end_on_or_before=window_end_on_or_before,
        )
        if window is None:
            skips.append(ScoreSkip(player_id=player_id, reason="no_window"))
            continue
        window_end = pd.Timestamp(window["window_end"]).date()

        for category in REPORT_CATEGORIES:
            label = _category_label(
                conn,
                player_id=player_id,
                window_end=window_end,
                window_days=window_days,
                category=category,
            )
            # Missing sparse-cat label rows mean we have no projection
            # rate for the player — there's nothing the model can score.
            if label is None:
                scores.append(
                    PlayerCategoryScore(
                        player_id=player_id,
                        category=category,
                        label="neutral",
                        probability=None,
                        drivers=(),
                        window_end=window_end,
                    )
                )
                continue

            if label == "neutral":
                scores.append(
                    PlayerCategoryScore(
                        player_id=player_id,
                        category=category,
                        label="neutral",
                        probability=None,
                        drivers=(),
                        window_end=window_end,
                    )
                )
                continue

            direction: StreakDirection = "above" if label == "hot" else "below"
            model = models.get((category, direction))
            if model is None:
                # Sparse cats are hot-only in Phase 4: sb cold / hr cold have
                # no trained model. We still surface the label.
                scores.append(
                    PlayerCategoryScore(
                        player_id=player_id,
                        category=category,
                        label=label,
                        probability=None,
                        drivers=(),
                        window_end=window_end,
                    )
                )
                continue

            season_rate = _projection_rate(
                conn,
                player_id=player_id,
                season=scoring_season,
                category=category,
            )
            if season_rate is None:
                # Dense cat label says hot/cold but we have no projection rate
                # for the feature vector. Skip — the model can't predict.
                scores.append(
                    PlayerCategoryScore(
                        player_id=player_id,
                        category=category,
                        label=label,
                        probability=None,
                        drivers=(),
                        window_end=window_end,
                    )
                )
                continue

            # Compute streak_strength_numeric matching training-time encoding.
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

            # Drop rows with NULL peripherals — we can't fill them and the
            # model was trained on rows without nulls.
            peripheral_cols = (
                "babip",
                "k_pct",
                "bb_pct",
                "iso",
                "ev_avg",
                "barrel_pct",
                "xwoba_avg",
            )
            if any(pd.isna(window[c]) for c in peripheral_cols):
                scores.append(
                    PlayerCategoryScore(
                        player_id=player_id,
                        category=category,
                        label=label,
                        probability=None,
                        drivers=(),
                        window_end=window_end,
                    )
                )
                continue

            feature_row = _build_feature_row(
                window=window,
                season_rate=season_rate,
                streak_strength_numeric=strength,
            )
            proba = float(model.pipeline.predict_proba(feature_row)[0, 1])
            drivers = top_drivers(pipeline=model.pipeline, feature_row=feature_row, k=2)
            scores.append(
                PlayerCategoryScore(
                    player_id=player_id,
                    category=category,
                    label=label,
                    probability=proba,
                    drivers=drivers,
                    window_end=window_end,
                )
            )

    return scores, skips
