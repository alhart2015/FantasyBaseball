import pytest

from fantasy_baseball.analysis.recency import (
    predict_exponential_decay,
    predict_fixed_blend,
    predict_preseason,
    predict_reliability_blend,
    predict_season_to_date,
)

PROJ_HITTER = {
    "hr_per_pa": 0.040,
    "r_per_pa": 0.120,
    "rbi_per_pa": 0.110,
    "sb_per_pa": 0.015,
    "avg": 0.270,
}

# 30 hot games: 4 PA each, 2 H, 1 HR, 1 R, 1 RBI, 0 SB per game
HOT_GAMES = [
    {"date": f"2025-06-{d:02d}", "pa": 4, "ab": 4, "h": 2, "hr": 1, "r": 1, "rbi": 1, "sb": 0}
    for d in range(1, 31)
]
# Totals: 120 PA, 120 AB, 60 H, 30 HR, 30 R, 30 RBI, 0 SB


def test_preseason_ignores_actuals():
    result = predict_preseason(PROJ_HITTER, HOT_GAMES, "2025-07-01")
    assert result["hr_per_pa"] == pytest.approx(0.040)
    assert result["avg"] == pytest.approx(0.270)


def test_season_to_date_ignores_projection():
    result = predict_season_to_date(PROJ_HITTER, HOT_GAMES, "2025-07-01")
    assert result["hr_per_pa"] == pytest.approx(30 / 120, abs=0.001)
    assert result["avg"] == pytest.approx(60 / 120, abs=0.001)


def test_fixed_blend_between_proj_and_actual():
    result = predict_fixed_blend(PROJ_HITTER, HOT_GAMES, "2025-07-01")
    # 30% actual (0.25 HR/PA) + 70% proj (0.04) = 0.103
    assert result["hr_per_pa"] == pytest.approx(0.103, abs=0.005)


def test_reliability_blend_weights_by_sample_size():
    result = predict_reliability_blend(PROJ_HITTER, HOT_GAMES, "2025-07-01")
    # 120 PA, HR reliability constant = 200 PA
    # actual_weight = 120 / (120 + 200) = 0.375
    # blended = 0.375 * 0.25 + 0.625 * 0.04 = 0.119
    assert result["hr_per_pa"] == pytest.approx(0.119, abs=0.01)


def test_exponential_decay_weights_recent_more():
    # First 15 days cold (0 hits), last 15 days hot (3 hits per game)
    mixed = []
    for d in range(1, 16):
        mixed.append(
            {
                "date": f"2025-06-{d:02d}",
                "pa": 4,
                "ab": 4,
                "h": 0,
                "hr": 0,
                "r": 0,
                "rbi": 0,
                "sb": 0,
            }
        )
    for d in range(16, 31):
        mixed.append(
            {
                "date": f"2025-06-{d:02d}",
                "pa": 4,
                "ab": 4,
                "h": 3,
                "hr": 1,
                "r": 1,
                "rbi": 2,
                "sb": 1,
            }
        )
    result_decay = predict_exponential_decay(PROJ_HITTER, mixed, "2025-07-01")
    # Flat reliability blend uses the same blending method but weights all games equally.
    # Decay should weight recent hot games more heavily, producing a higher HR/PA estimate.
    result_flat_blend = predict_reliability_blend(PROJ_HITTER, mixed, "2025-07-01")
    assert result_decay["hr_per_pa"] > result_flat_blend["hr_per_pa"]


def test_fixed_blend_no_games_falls_back():
    """If no games in the 30-day window, fall back to projection."""
    old_games = [
        {"date": "2025-03-15", "pa": 4, "ab": 4, "h": 2, "hr": 1, "r": 1, "rbi": 1, "sb": 0}
    ]
    result = predict_fixed_blend(PROJ_HITTER, old_games, "2025-07-01")
    assert result["hr_per_pa"] == pytest.approx(0.040)


def test_season_to_date_no_games_returns_zeros():
    result = predict_season_to_date(PROJ_HITTER, [], "2025-07-01")
    assert result["hr_per_pa"] == 0
