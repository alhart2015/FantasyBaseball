from __future__ import annotations

import pandas as pd
import pytest

from fantasy_baseball.trajectory.era import (
    era_factors,
    era_normalize,
    league_rates,
    normalize_frame,
)
from fantasy_baseball.trajectory.panel import score


def _hitters(rows: list[dict]) -> pd.DataFrame:
    base = {
        "mlbam_id": 1,
        "season": 2024,
        "pa": 600.0,
        "ab_pa": 0.9,
        "h_ab": 0.280,
        "hr_pa": 0.05,
        "r_pa": 0.15,
        "rbi_pa": 0.14,
        "sb_pa": 0.02,
    }
    return score(pd.DataFrame([base | row for row in rows]), "hitter")


def _pitchers(rows: list[dict]) -> pd.DataFrame:
    base = {
        "mlbam_id": 1,
        "season": 2024,
        "ip": 180.0,
        "k_ip": 1.0,
        "w_ip": 0.07,
        "sv_ip": 0.0,
        "er_ip": 0.40,
        "bb_ip": 0.30,
        "h_ip": 0.85,
    }
    return score(pd.DataFrame([base | row for row in rows]), "pitcher")


REF = (2024,)


def test_league_rates_weight_by_volume_not_by_player() -> None:
    # 600 PA at .050 and 100 PA at .010 is 34 HR / 700 PA, not the .030 a flat mean
    # of the two rates would give.
    frame = _hitters([{"pa": 600.0, "hr_pa": 0.05}, {"mlbam_id": 2, "pa": 100.0, "hr_pa": 0.01}])
    assert league_rates(frame, "hitter").loc[2024, "hr_pa"] == pytest.approx(31.0 / 700)


def test_league_rates_use_ab_for_batting_average() -> None:
    frame = _hitters([{"h_ab": 0.300}, {"mlbam_id": 2, "pa": 200.0, "h_ab": 0.200}])
    # 540 AB at .300 and 180 AB at .200 -> 198 H / 720 AB
    assert league_rates(frame, "hitter").loc[2024, "h_ab"] == pytest.approx(198.0 / 720)


def test_league_rates_reject_an_unscored_panel() -> None:
    with pytest.raises(KeyError, match="panel must be scored"):
        league_rates(pd.DataFrame({"season": [2024], "h_ab": [0.28], "pa": [600.0]}), "hitter")


def test_a_high_offence_season_is_scaled_down_to_the_reference() -> None:
    # 2000 hits twice the reference HR rate, so its HRs should halve.
    frame = _hitters([{"season": 2000, "hr_pa": 0.10}, {"mlbam_id": 2, "season": 2024}])
    out = era_normalize(frame, "hitter", reference_seasons=REF).set_index("season")
    assert out.loc[2000, "hr_pa"] == pytest.approx(0.05)
    assert out.loc[2024, "hr_pa"] == pytest.approx(0.05)  # the reference is left alone


def test_normalizing_makes_equal_seasons_score_equally() -> None:
    frame = _hitters([{"season": 2000, "hr_pa": 0.10}, {"mlbam_id": 2, "season": 2024}])
    out = era_normalize(frame, "hitter", reference_seasons=REF).set_index("season")
    assert out.loc[2000, "sgp"] == pytest.approx(out.loc[2024, "sgp"])


def test_volume_is_never_era_normalized() -> None:
    # A run environment says nothing about how many times a player batted.
    frame = _hitters([{"season": 2000, "hr_pa": 0.10}, {"mlbam_id": 2, "season": 2024}])
    out = era_normalize(frame, "hitter", reference_seasons=REF).set_index("season")
    assert out.loc[2000, "pa"] == pytest.approx(600.0)


def test_ab_pa_is_left_alone_as_a_structural_ratio() -> None:
    frame = _hitters([{"season": 2000, "ab_pa": 0.95}, {"mlbam_id": 2, "season": 2024}])
    out = era_normalize(frame, "hitter", reference_seasons=REF).set_index("season")
    assert out.loc[2000, "ab_pa"] == pytest.approx(0.95)


def test_the_factor_applied_is_recorded_for_tracing() -> None:
    frame = _hitters([{"season": 2000, "hr_pa": 0.10}, {"mlbam_id": 2, "season": 2024}])
    out = era_normalize(frame, "hitter", reference_seasons=REF).set_index("season")
    assert out.loc[2000, "era_factor_hr_pa"] == pytest.approx(0.5)


def test_a_zero_league_rate_neutralizes_rather_than_exploding() -> None:
    # No saves at all in a season gives reference/0 = inf, which must not multiply
    # every other season's saves. This is the top-90-by-innings pool bug in miniature.
    frame = _pitchers(
        [{"season": 2000, "sv_ip": 0.0}, {"mlbam_id": 2, "season": 2024, "sv_ip": 0.3}]
    )
    out = era_normalize(frame, "pitcher", reference_seasons=REF).set_index("season")
    assert out.loc[2000, "era_factor_sv_ip"] == 1.0
    assert out.loc[2000, "sv_ip"] == 0.0
    assert out.loc[2024, "sv_ip"] == pytest.approx(0.3)


def test_a_missing_reference_window_is_an_error_not_a_silent_no_op() -> None:
    frame = _hitters([{"season": 2000}])
    with pytest.raises(ValueError, match="reference seasons"):
        era_normalize(frame, "hitter", reference_seasons=(2023, 2024, 2025))


def test_a_PARTIAL_reference_window_is_also_refused() -> None:
    # Accepting 2023 alone silently restates every season onto a third of the window
    # league.yaml's denominators were calibrated on -- wrong units, no warning.
    frame = _hitters([{"season": 2000}, {"mlbam_id": 2, "season": 2023}])
    with pytest.raises(ValueError, match=r"\[2024, 2025\]"):
        era_normalize(frame, "hitter", reference_seasons=(2023, 2024, 2025))


def test_the_reference_averages_across_the_whole_window() -> None:
    frame = _hitters(
        [
            {"season": 2023, "hr_pa": 0.04},
            {"mlbam_id": 2, "season": 2025, "hr_pa": 0.06},
            {"mlbam_id": 3, "season": 2000, "hr_pa": 0.10},
        ]
    )
    out = era_normalize(frame, "hitter", reference_seasons=(2023, 2025)).set_index("season")
    assert out.loc[2000, "hr_pa"] == pytest.approx(0.05)  # reference is (.04 + .06) / 2


def test_era_factors_reproduces_the_scaling_era_normalize_applies() -> None:
    """era_factors is an EXTRACTION, not a reimplementation.

    era_normalize's own output is the contract. If these two ever disagree, the
    keeper-value side of the backtest is normalized onto a different reference than
    the shape side and every comparison in it is silently wrong.
    """
    panel = _hitters(
        [
            {"mlbam_id": 1, "season": 2022, "hr_pa": 0.040},
            {"mlbam_id": 2, "season": 2023, "hr_pa": 0.050},
            {"mlbam_id": 3, "season": 2024, "hr_pa": 0.060},
            {"mlbam_id": 4, "season": 2025, "hr_pa": 0.055},
        ]
    )
    factors = era_factors(panel, "hitter")
    normalized = era_normalize(panel, "hitter")

    for season in (2022, 2023, 2024, 2025):
        raw_row = panel.loc[panel["season"] == season].iloc[0]
        norm_row = normalized.loc[normalized["season"] == season].iloc[0]
        assert norm_row["hr_pa"] == pytest.approx(raw_row["hr_pa"] * factors.loc[season, "hr_pa"])
        assert norm_row["era_factor_hr_pa"] == pytest.approx(factors.loc[season, "hr_pa"])


def test_era_factors_raises_on_a_missing_reference_season_like_era_normalize() -> None:
    panel = _hitters([{"mlbam_id": 1, "season": 2022}])
    with pytest.raises(ValueError, match="reference seasons"):
        era_factors(panel, "hitter")


def _reference_panel() -> pd.DataFrame:
    """A panel spanning the 2023-2025 reference window, so era_factors can be built."""
    return _hitters(
        [
            {"mlbam_id": 1, "season": 2022, "hr_pa": 0.040},
            {"mlbam_id": 2, "season": 2023, "hr_pa": 0.050},
            {"mlbam_id": 3, "season": 2024, "hr_pa": 0.060},
            {"mlbam_id": 4, "season": 2025, "hr_pa": 0.055},
        ]
    )


def test_normalize_frame_scales_each_rate_by_its_season_factor() -> None:
    factors = era_factors(_reference_panel(), "hitter")
    frame = pd.DataFrame({"pa": [600.0], "hr_pa": [0.040]}, index=pd.Index([99], name="mlbam_id"))

    out = normalize_frame(frame, 2022, "hitter", factors)

    assert out["hr_pa"].iloc[0] == pytest.approx(0.040 * factors.loc[2022, "hr_pa"])
    # Volume is never era-normalized -- a 600-PA season is 600 PA in any year.
    assert out["pa"].iloc[0] == 600.0
    # The input is not mutated; callers hold on to raw frames.
    assert frame["hr_pa"].iloc[0] == 0.040


def test_normalize_frame_leaves_an_unknown_season_alone() -> None:
    """1.0, not KeyError. A season with no factor means no usable adjustment, which is
    what era_normalize's own fillna(1.0) already decided."""
    factors = era_factors(_reference_panel(), "hitter")
    frame = pd.DataFrame({"pa": [600.0], "hr_pa": [0.040]}, index=pd.Index([99], name="mlbam_id"))

    out = normalize_frame(frame, 1998, "hitter", factors)

    assert out["hr_pa"].iloc[0] == 0.040


def test_normalize_frame_ignores_rate_columns_the_frame_does_not_carry() -> None:
    """A vintage export can be missing a category entirely (the 2027/2028 ZiPS files
    carry no SV). Skipping is right; raising would refuse a usable vintage."""
    factors = era_factors(_reference_panel(), "hitter")
    frame = pd.DataFrame({"pa": [600.0]}, index=pd.Index([99], name="mlbam_id"))

    out = normalize_frame(frame, 2022, "hitter", factors)

    assert list(out.columns) == ["pa"]
