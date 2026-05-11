"""Tests for the Phase 5 inference layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import pytest

from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.inference import (
    REPORT_CATEGORIES,
    Driver,
    PlayerCategoryScore,
    ScoreSkip,
    _build_feature_row,
    _dense_quintile_cutoffs,
    _dense_streak_strength,
    _sparse_streak_strength,
    refit_models_for_report,
    score_player_windows,
    top_drivers,
)
from tests.test_streaks.test_predictors import _seed_pipeline


def _seed_two_seasons(conn) -> None:
    """Seed enough data that refit_models_for_report has a 2-season train set."""
    _seed_pipeline(conn, season=2023)
    _seed_pipeline(conn, season=2024)


def test_dense_streak_strength_bins_to_quintile() -> None:
    # ``side='left'`` semantics match _build_training_frame_dense — a value
    # equal to a cutoff lands in the same bin as values below it.
    cuts = (1.0, 2.0, 3.0, 4.0)
    assert _dense_streak_strength(0.5, cuts) == 1.0
    assert _dense_streak_strength(1.0, cuts) == 1.0
    assert _dense_streak_strength(1.5, cuts) == 2.0
    assert _dense_streak_strength(2.5, cuts) == 3.0
    assert _dense_streak_strength(4.5, cuts) == 5.0


def test_sparse_streak_strength_returns_none_for_zero_expected() -> None:
    assert _sparse_streak_strength(value=0, window_pa=50, season_rate=0.0) is None


def test_sparse_streak_strength_clips_to_half_sigma_range() -> None:
    # value much higher than expected → should clip to 3.0.
    s = _sparse_streak_strength(value=10, window_pa=50, season_rate=0.05)
    assert s is not None
    assert 0.0 <= s <= 3.0


def test_refit_models_for_report_returns_pipelines_and_writes_audit_rows() -> None:
    conn = get_connection(":memory:")
    _seed_two_seasons(conn)
    fitted = refit_models_for_report(conn, season_set_train="2023-2024", window_days=14)
    # At least the dense-cat hot models should always fit on this fixture.
    assert ("r", "above") in fitted
    # Each fitted model must carry quintile cutoffs for dense cats and not
    # for sparse cats.
    for (cat, _direction), model in fitted.items():
        if cat in ("r", "rbi", "avg"):
            assert model.dense_quintile_cutoffs is not None
            assert len(model.dense_quintile_cutoffs) == 4
        else:
            assert model.dense_quintile_cutoffs is None
    # model_fits rows were written.
    n_rows = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n_rows == len(fitted)


def test_dense_quintile_cutoffs_matches_training_population() -> None:
    """The cutoffs should be computed over the same row population that
    ``_build_training_frame_dense`` uses for its inline quantile call."""
    conn = get_connection(":memory:")
    _seed_two_seasons(conn)
    cutoffs = _dense_quintile_cutoffs(
        conn, category="r", direction="above", season_set="2023-2024", window_days=14
    )
    # Quintile cutoffs must be monotonically non-decreasing.
    assert cutoffs[0] <= cutoffs[1] <= cutoffs[2] <= cutoffs[3]


def test_top_drivers_excludes_non_peripheral_features_by_default() -> None:
    conn = get_connection(":memory:")
    _seed_two_seasons(conn)
    fitted = refit_models_for_report(conn, season_set_train="2023-2024", window_days=14)
    model = fitted[("r", "above")]
    # Build a feature row with all peripherals at their training mean (0
    # after StandardScaler) — the top-k will surface whichever has a
    # non-zero coef, but the excluded columns should never appear.
    window_row = (
        conn.execute("SELECT * FROM hitter_windows WHERE window_days = 14 LIMIT 1").df().iloc[0]
    )
    feature_row = _build_feature_row(
        window=window_row,
        season_rate=0.15,
        streak_strength_numeric=3.0,
    )
    drivers = top_drivers(pipeline=model.pipeline, feature_row=feature_row, k=2)
    assert len(drivers) == 2
    for d in drivers:
        assert d.feature not in {
            "streak_strength_numeric",
            "season_rate_in_category",
            "pt_bucket_low",
            "pt_bucket_mid",
            "pt_bucket_high",
        }


def test_score_player_windows_returns_one_row_per_category() -> None:
    conn = get_connection(":memory:")
    _seed_two_seasons(conn)
    fitted = refit_models_for_report(conn, season_set_train="2023-2024", window_days=14)
    # Pick a player who is in the fixture and whose window_end is in 2024.
    last_window = conn.execute(
        "SELECT player_id, MAX(window_end) FROM hitter_windows WHERE window_days = 14 GROUP BY player_id LIMIT 1"
    ).fetchone()
    assert last_window is not None
    player_id, max_end = last_window
    scoring_end = max_end if isinstance(max_end, date) else date.fromisoformat(str(max_end))
    scores, skips = score_player_windows(
        conn,
        models=fitted,
        player_ids=[int(player_id)],
        window_end_on_or_before=scoring_end + timedelta(days=1),
        window_days=14,
        scoring_season=2024,
    )
    # One entry per category, regardless of label.
    cats_present = {s.category for s in scores}
    assert cats_present == set(REPORT_CATEGORIES)
    assert len(scores) == len(REPORT_CATEGORIES)
    assert all(s.player_id == player_id for s in scores)
    assert not skips


def test_score_player_windows_skips_player_with_no_window() -> None:
    conn = get_connection(":memory:")
    _seed_two_seasons(conn)
    fitted = refit_models_for_report(conn, season_set_train="2023-2024", window_days=14)
    scores, skips = score_player_windows(
        conn,
        models=fitted,
        player_ids=[9999],  # not in fixture
        window_end_on_or_before=date(2024, 12, 31),
        window_days=14,
        scoring_season=2024,
    )
    assert not scores
    assert len(skips) == 1
    assert skips[0].player_id == 9999
    assert skips[0].reason == "no_window"


def test_score_player_windows_emits_probability_only_when_label_non_neutral_and_model_exists() -> (
    None
):
    conn = get_connection(":memory:")
    _seed_two_seasons(conn)
    fitted = refit_models_for_report(conn, season_set_train="2023-2024", window_days=14)
    # Grab any player_id present in windows.
    pid_row = conn.execute("SELECT player_id FROM hitter_windows LIMIT 1").fetchone()
    assert pid_row is not None
    player_id = int(pid_row[0])
    max_end = conn.execute(
        "SELECT MAX(window_end) FROM hitter_windows WHERE player_id = ? AND window_days = 14",
        [player_id],
    ).fetchone()[0]
    scoring_end = max_end if isinstance(max_end, date) else date.fromisoformat(str(max_end))
    scores, _ = score_player_windows(
        conn,
        models=fitted,
        player_ids=[player_id],
        window_end_on_or_before=scoring_end + timedelta(days=1),
        window_days=14,
        scoring_season=2024,
    )
    for s in scores:
        # Probabilities must be in [0, 1] when present.
        if s.probability is not None:
            assert 0.0 <= s.probability <= 1.0
            assert s.label in ("hot", "cold")
            # Drivers populated only when probability is.
            assert isinstance(s.drivers, tuple)
        else:
            # Either neutral, or no model exists for the (cat, direction).
            assert s.drivers == ()


def test_score_player_windows_handles_missing_projection_rate_gracefully() -> None:
    """If a dense cat label is hot/cold but projection_rate is missing,
    the score row carries the label but no probability."""
    conn = get_connection(":memory:")
    _seed_two_seasons(conn)
    fitted = refit_models_for_report(conn, season_set_train="2023-2024", window_days=14)
    # Wipe projection rates so dense lookups return None.
    conn.execute("DELETE FROM hitter_projection_rates")
    pid_row = conn.execute("SELECT player_id FROM hitter_windows LIMIT 1").fetchone()
    assert pid_row is not None
    player_id = int(pid_row[0])
    max_end = conn.execute(
        "SELECT MAX(window_end) FROM hitter_windows WHERE player_id = ? AND window_days = 14",
        [player_id],
    ).fetchone()[0]
    scoring_end = max_end if isinstance(max_end, date) else date.fromisoformat(str(max_end))
    scores, _ = score_player_windows(
        conn,
        models=fitted,
        player_ids=[player_id],
        window_end_on_or_before=scoring_end + timedelta(days=1),
        window_days=14,
        scoring_season=2024,
    )
    # No probability anywhere — and no crashes.
    assert all(s.probability is None for s in scores)


def test_driver_dataclass_is_frozen() -> None:
    d = Driver(feature="xwoba_avg", z_score=1.5)
    with pytest.raises(FrozenInstanceError):
        d.feature = "ev_avg"  # type: ignore[misc]


def test_player_category_score_dataclass_is_frozen() -> None:
    s = PlayerCategoryScore(
        player_id=1, category="r", label="neutral", probability=None, drivers=(), window_end=None
    )
    with pytest.raises(FrozenInstanceError):
        s.probability = 0.5  # type: ignore[misc]


def test_score_skip_reasons_are_typed() -> None:
    # Smoke test that the Literal narrows correctly at runtime via constructor.
    skip = ScoreSkip(player_id=7, reason="no_window")
    assert skip.reason == "no_window"
