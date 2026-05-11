"""Cheap regression guard: dataclass field surface stays in lockstep with the DDL.

If this fails after a column add, update both `models.py` and `schema.py`
together — they are co-load-bearing for `load.py`'s attrgetter-based upsert.
"""

from __future__ import annotations

from dataclasses import fields

from fantasy_baseball.streaks.models import (
    ContinuationRate,
    HitterGame,
    HitterProjectionRate,
    HitterStatcastPA,
    HitterStreakLabel,
    HitterWindow,
)


def test_hitter_game_fields_in_expected_order() -> None:
    expected = (
        "player_id",
        "game_pk",
        "name",
        "team",
        "season",
        "date",
        "pa",
        "ab",
        "h",
        "hr",
        "r",
        "rbi",
        "sb",
        "bb",
        "k",
        "b2",
        "b3",
        "sf",
        "hbp",
        "ibb",
        "cs",
        "gidp",
        "sh",
        "ci",
        "is_home",
    )
    assert tuple(f.name for f in fields(HitterGame)) == expected


def test_hitter_statcast_pa_fields_in_expected_order() -> None:
    expected = (
        "player_id",
        "date",
        "pa_index",
        "event",
        "launch_speed",
        "launch_angle",
        "estimated_woba_using_speedangle",
        "barrel",
        "at_bat_number",
        "bb_type",
        "estimated_ba_using_speedangle",
        "hit_distance_sc",
    )
    assert tuple(f.name for f in fields(HitterStatcastPA)) == expected


def test_hitter_window_fields_in_expected_order() -> None:
    expected = (
        "player_id",
        "window_end",
        "window_days",
        "pa",
        "hr",
        "r",
        "rbi",
        "sb",
        "avg",
        "babip",
        "k_pct",
        "bb_pct",
        "iso",
        "ev_avg",
        "barrel_pct",
        "xwoba_avg",
        "pt_bucket",
    )
    assert tuple(f.name for f in fields(HitterWindow)) == expected


def test_hitter_streak_label_includes_cold_method() -> None:
    expected = ("player_id", "window_end", "window_days", "category", "cold_method", "label")
    assert tuple(f.name for f in fields(HitterStreakLabel)) == expected


def test_hitter_projection_rate_includes_dense_cat_rates() -> None:
    expected = (
        "player_id",
        "season",
        "hr_per_pa",
        "sb_per_pa",
        "r_per_pa",
        "rbi_per_pa",
        "avg",
        "n_systems",
    )
    assert tuple(f.name for f in fields(HitterProjectionRate)) == expected


def test_model_fit_fields_in_expected_order() -> None:
    from fantasy_baseball.streaks.models import ModelFit

    expected = (
        "model_id",
        "category",
        "direction",
        "season_set",
        "window_days",
        "cold_method",
        "chosen_C",
        "cv_auc_mean",
        "cv_auc_std",
        "val_auc",
        "n_train_rows",
        "n_val_rows",
        "fit_timestamp",
    )
    assert tuple(f.name for f in fields(ModelFit)) == expected


def test_continuation_rate_fields_in_expected_order() -> None:
    expected = (
        "season_set",
        "category",
        "window_days",
        "pt_bucket",
        "strength_bucket",
        "direction",
        "cold_method",
        "n_labeled",
        "n_continued",
        "p_continued",
        "p_baserate",
        "lift",
    )
    assert tuple(f.name for f in fields(ContinuationRate)) == expected
