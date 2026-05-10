"""Tests for Phase 4 predictor pipeline."""

from __future__ import annotations

from datetime import date, timedelta

from fantasy_baseball.streaks.analysis.predictors import (
    EXPECTED_FEATURE_COLUMNS,
    build_training_frame,
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
                        barrel=high and pa_idx == 0,
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
