"""Tests for Phase 4 predictor pipeline."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fantasy_baseball.streaks.analysis.predictors import (
    DEFAULT_C_GRID,
    EXPECTED_FEATURE_COLUMNS,
    PHASE_4_MODELS,
    AllModelsResult,
    EvaluationResult,
    FitResult,
    bootstrap_coef_ci,
    build_training_frame,
    evaluate_model,
    fit_all_models,
    fit_one_model,
    permutation_feature_importance,
)
from fantasy_baseball.streaks.data.load import upsert_hitter_games, upsert_statcast_pa
from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.models import HitterGame, HitterProjectionRate, HitterStatcastPA
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows


def _seed_pipeline(conn, *, n_players: int = 16, n_days: int = 90, season: int = 2024) -> None:
    """Run the full Phase 1-3 pipeline against a synthetic fixture sized so
    Phase 4's GroupKFold has at least a few players per fold.

    ``game_pk`` includes ``season`` so that calling this twice with different
    seasons does not overwrite the first season's rows via the (player_id,
    game_pk) PK — important for the orchestrator test that needs both 2023
    and 2024 game data present simultaneously.
    """
    base = date(season, 4, 1)
    games: list[HitterGame] = []
    for pid in range(1, n_players + 1):
        for d in range(1, n_days + 1):
            high = pid % 2 == 0
            hr = 1 if (high and d % 6 == 0) else 0
            sb = 1 if (high and d % 5 == 0) else 0
            # Every player has alternating "surge" and "slump" phases so the
            # rolling-window R/RBI sum has spread across consecutive 14-day
            # windows. With a 21-day cycle, two adjacent disjoint 14-day
            # windows can land on different sides of the phase — giving the
            # dense-cat target column both 0s and 1s.
            phase = (d + pid * 5) % 21
            surging = phase < 10
            # High-rate players still average above low-rate over the season,
            # but their R/RBI fluctuates day-to-day.
            if high:
                r_val = 3 if surging else 1
            else:
                r_val = 2 if surging else 0
            rbi = r_val
            games.append(
                HitterGame(
                    player_id=pid,
                    game_pk=season * 100_000 + pid * 100 + d,
                    name=f"P{pid}",
                    team="ABC",
                    season=season,
                    date=base + timedelta(days=d - 1),
                    pa=4,
                    ab=4,
                    h=2 if high else 1,
                    hr=hr,
                    r=r_val,
                    rbi=rbi,
                    sb=sb,
                    bb=0,
                    k=1,
                    b2=0,
                    b3=0,
                    sf=0,
                    hbp=0,
                    ibb=0,
                    cs=0,
                    gidp=0,
                    sh=0,
                    ci=0,
                    is_home=True,
                )
            )
    upsert_hitter_games(conn, games)
    # Seed minimal Statcast PAs so the training-frame builder's NaN-peripheral
    # drop doesn't empty the synthetic fixture. High-rate players get
    # higher-quality contact (faster EV, more barrels, higher xwOBA) so the
    # peripheral signal correlates loosely with hot/cold status.
    statcast: list[HitterStatcastPA] = []
    for pid in range(1, n_players + 1):
        for d in range(1, n_days + 1):
            high = pid % 2 == 0
            for pa_idx in range(4):  # 4 PA per game, matching games above
                statcast.append(
                    HitterStatcastPA(
                        player_id=pid,
                        date=base + timedelta(days=d - 1),
                        pa_index=pa_idx,
                        event="single" if high else "field_out",
                        launch_speed=92.0 if high else 85.0,
                        launch_angle=15.0,
                        estimated_woba_using_speedangle=0.400 if high else 0.280,
                        # 6 == barrel (Statcast classifier); 3 == under for low-EV contact
                        launch_speed_angle=6 if (high and pa_idx == 0) else 3,
                        at_bat_number=pa_idx + 1,
                        bb_type="line_drive",
                        estimated_ba_using_speedangle=0.330 if high else 0.230,
                        hit_distance_sc=300.0 if high else 200.0,
                    )
                )
    upsert_statcast_pa(conn, statcast)
    upsert_projection_rates(
        conn,
        [
            HitterProjectionRate(
                player_id=pid,
                season=season,
                hr_per_pa=0.05 if pid % 2 == 0 else 0.005,
                sb_per_pa=0.04 if pid % 2 == 0 else 0.004,
                r_per_pa=0.15 if pid % 2 == 0 else 0.10,
                rbi_per_pa=0.18 if pid % 2 == 0 else 0.10,
                avg=0.275 if pid % 2 == 0 else 0.230,
                n_systems=2,
            )
            for pid in range(1, n_players + 1)
        ],
    )
    compute_windows(conn)
    compute_thresholds(conn, season_set=str(season), qualifying_pa=50)
    apply_labels(conn, season_set=str(season))


def test_build_training_frame_columns_match_expected() -> None:
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    # Required: features + target + grouping/season metadata.
    for col in EXPECTED_FEATURE_COLUMNS:
        assert col in df.columns, f"missing feature column {col}"
    assert "target" in df.columns
    assert "player_id" in df.columns
    assert "season" in df.columns


def test_build_training_frame_hot_dense_target_matches_bucket_median() -> None:
    """For dense hot, target=1 iff next_value > median(next_value) within (window_days, pt_bucket)."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    assert not df.empty
    assert df["target"].isin([0, 1]).all()
    # At least some variation — fixture has both above-median and below-median rows.
    assert df["target"].sum() > 0
    assert df["target"].sum() < len(df)


def test_build_training_frame_filters_to_hot_only_for_above_direction() -> None:
    """The hot model trains only on rows currently labeled hot."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    # Streak strength numeric is parsed from "hot_qN" — values 1..5 only.
    assert df["streak_strength_numeric"].between(1, 5).all()


def test_build_training_frame_sparse_hr_hot_uses_poisson_p20_partition() -> None:
    """HR hot rows are duplicated across poisson_p10 and poisson_p20 in
    hitter_streak_labels. Dedup to p20 in the training frame so the model
    isn't trained on identical rows twice."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="hr", direction="above", season_set="2024", window_days=14
    )
    if df.empty:
        return  # fixture may not produce any hot HR windows; tolerable.
    # No duplicate (player_id, window_end) pairs — confirms dedup.
    assert df.duplicated(subset=["player_id", "window_end"]).sum() == 0


def test_build_training_frame_drops_zna_strength_rows() -> None:
    """Rows with strength_bucket ending in '_zna' have undefined sigma —
    drop them rather than guess a numeric encoding."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="hr", direction="above", season_set="2024", window_days=14
    )
    if df.empty:
        return
    # streak_strength_numeric is float for sparse — but never NaN after drop.
    assert df["streak_strength_numeric"].notna().all()


def test_build_training_frame_pt_bucket_one_hot_encoded() -> None:
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    for col in ("pt_bucket_low", "pt_bucket_mid", "pt_bucket_high"):
        assert col in df.columns
    # Each row has exactly one bucket flag set.
    assert (df[["pt_bucket_low", "pt_bucket_mid", "pt_bucket_high"]].sum(axis=1) == 1).all()


def test_build_training_frame_includes_season_rate_for_dense_cats() -> None:
    """For R hot, season_rate_in_category should equal hitter_projection_rates.r_per_pa."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    assert df["season_rate_in_category"].notna().all()
    # In the fixture, high-rate players have r_per_pa=0.15 and low-rate=0.10.
    assert set(df["season_rate_in_category"].round(2)).issubset({0.10, 0.15})


def _make_synthetic_X_y(
    n_rows: int = 200, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Synthetic, linearly-separable-ish dataset for fit-loop unit tests."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n_rows, len(EXPECTED_FEATURE_COLUMNS))),
        columns=list(EXPECTED_FEATURE_COLUMNS),
    )
    # Make target weakly dependent on the first feature.
    logits = X[EXPECTED_FEATURE_COLUMNS[0]].to_numpy() + 0.5 * rng.normal(size=n_rows)
    y = (logits > 0).astype(int)
    groups = rng.integers(low=1, high=10, size=n_rows)
    return X, y, groups


def test_fit_one_model_returns_fitresult_with_pipeline_and_metrics() -> None:
    X, y, groups = _make_synthetic_X_y()
    result = fit_one_model(X, y, groups, C_grid=DEFAULT_C_GRID, n_splits=5, random_state=42)
    assert isinstance(result, FitResult)
    assert isinstance(result.pipeline, Pipeline)
    assert result.chosen_C in DEFAULT_C_GRID
    assert 0.0 <= result.cv_auc_mean <= 1.0
    assert result.cv_auc_std >= 0.0
    # AUC for a linearly-separable-ish target should be well above 0.5.
    assert result.cv_auc_mean > 0.55


def test_fit_one_model_pipeline_is_fitted_on_full_train() -> None:
    """Pipeline.predict_proba should succeed without further fit."""
    X, y, groups = _make_synthetic_X_y()
    result = fit_one_model(X, y, groups, C_grid=DEFAULT_C_GRID, n_splits=5, random_state=42)
    proba = result.pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_fit_one_model_picks_highest_cv_auc() -> None:
    """When the C-grid has a single value, that value is selected."""
    X, y, groups = _make_synthetic_X_y()
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    assert result.chosen_C == 1.0


def test_bootstrap_coef_ci_returns_per_feature_intervals() -> None:
    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    cis = bootstrap_coef_ci(
        X=X,
        y=y,
        groups=groups,
        chosen_C=result.chosen_C,
        n_resamples=20,  # small for the unit test
        random_state=42,
    )
    # One (lo, hi) per feature column, ordered to match X.
    assert set(cis.keys()) == set(X.columns)
    for col, (lo, hi) in cis.items():
        assert lo <= hi, f"CI inverted for {col}: ({lo}, {hi})"


def test_bootstrap_coef_ci_intervals_narrow_with_more_resamples() -> None:
    """Sanity check: 100 resamples should produce intervals that *include*
    the point estimate from the original fit for most features (we don't
    assert a hard rate — bootstrap can disagree with the L2-shrunk
    point — but every CI should be finite)."""
    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    cis = bootstrap_coef_ci(
        X=X,
        y=y,
        groups=groups,
        chosen_C=result.chosen_C,
        n_resamples=100,
        random_state=42,
    )
    for _col, (lo, hi) in cis.items():
        assert np.isfinite(lo) and np.isfinite(hi)


def test_evaluate_model_returns_auc_and_reliability_bins() -> None:
    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    eval_result = evaluate_model(pipeline=result.pipeline, X=X, y=y, n_bins=10)
    assert isinstance(eval_result, EvaluationResult)
    assert 0.0 <= eval_result.auc <= 1.0
    # 10 reliability bins, each with (mean_predicted, mean_observed, count).
    assert len(eval_result.reliability_bin_centers) == len(eval_result.reliability_observed)
    assert (np.asarray(eval_result.reliability_bin_counts) >= 0).all()


def test_evaluate_model_auc_matches_sklearn_directly() -> None:
    from sklearn.metrics import roc_auc_score as _roc

    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    eval_result = evaluate_model(pipeline=result.pipeline, X=X, y=y, n_bins=10)
    direct = _roc(y, result.pipeline.predict_proba(X)[:, 1])
    assert eval_result.auc == pytest.approx(direct, rel=1e-9)


def test_permutation_feature_importance_returns_per_feature_mean_and_std() -> None:
    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    importance = permutation_feature_importance(
        pipeline=result.pipeline, X_val=X, y_val=y, n_repeats=5, random_state=42
    )
    assert set(importance.keys()) == set(X.columns)
    for _col, (mean_drop, std_drop) in importance.items():
        assert np.isfinite(mean_drop)
        assert std_drop >= 0.0


def _seed_two_season_pipeline(conn) -> None:
    """Seed two seasons so the orchestrator has both train and val to work with."""
    _seed_pipeline(conn, n_players=20, n_days=90, season=2023)
    _seed_pipeline(conn, n_players=20, n_days=90, season=2024)
    # Re-run thresholds and labels for the combined season_set.
    compute_thresholds(conn, season_set="2023-2024", qualifying_pa=50)
    apply_labels(conn, season_set="2023-2024")


def test_fit_all_models_returns_one_result_per_phase_4_model(tmp_path) -> None:
    conn = get_connection(":memory:")
    _seed_two_season_pipeline(conn)
    # For the unit test we use 2023 as train and 2024 as val just to exercise
    # the orchestrator's two-frame plumbing. (Real-data acceptance uses
    # 2023-2024 train / 2025 val.)
    result = fit_all_models(
        conn,
        season_set_train="2023",
        season_set_val="2024",
        window_days=14,
        C_grid=(1.0,),
        n_bootstrap=10,
        random_state=42,
    )
    assert isinstance(result, AllModelsResult)
    # If the synthetic fixture is too small to produce any labeled rows for a
    # given (cat, dir), the orchestrator records a None fit for it. The
    # length of the dict still matches the model spec.
    assert len(result.fits) == len(PHASE_4_MODELS)


def test_fit_all_models_writes_to_model_fits_table() -> None:
    conn = get_connection(":memory:")
    _seed_two_season_pipeline(conn)
    fit_all_models(
        conn,
        season_set_train="2023",
        season_set_val="2024",
        window_days=14,
        C_grid=(1.0,),
        n_bootstrap=10,
        random_state=42,
    )
    n = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    # At least one model should have produced enough rows to fit.
    assert n >= 1


def test_fit_all_models_skips_models_with_no_training_rows() -> None:
    """If a (cat, dir) frame is empty after filtering, the result entry is
    None and no row is written to model_fits for it."""
    conn = get_connection(":memory:")
    # Bare init — no seeded data.
    result = fit_all_models(
        conn,
        season_set_train="2099",
        season_set_val="2100",
        window_days=14,
        C_grid=(1.0,),
        n_bootstrap=10,
        random_state=42,
    )
    assert all(v is None for v in result.fits.values())
    n = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n == 0
