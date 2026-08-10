from __future__ import annotations

import pandas as pd
import pytest

from fantasy_baseball.trajectory.era import (
    RATE_DENOMINATORS,
    era_factors,
    era_normalize,
    league_rates,
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

    Asserts the WHOLE normalized frame against one rebuilt from the factor table, with
    `assert_frame_equal` -- not one column. The earlier version checked `hr_pa` only and
    would have stayed green if the extraction had changed both halves the same way,
    which is exactly the failure mode a characterization test exists to catch.
    """
    panel = _hitters(
        [
            {"mlbam_id": 1, "season": 2022, "hr_pa": 0.040, "sb_pa": 0.03},
            {"mlbam_id": 2, "season": 2023, "hr_pa": 0.050, "sb_pa": 0.01},
            {"mlbam_id": 3, "season": 2024, "hr_pa": 0.060, "sb_pa": 0.04},
            {"mlbam_id": 4, "season": 2025, "hr_pa": 0.055, "sb_pa": 0.02},
        ]
    )
    factors = era_factors(panel, "hitter")
    normalized = era_normalize(panel, "hitter")

    # Rebuild the frame from the factor table alone and demand it match, column for
    # column, row for row -- every rate, the untouched volume, and the re-scored sgp.
    rebuilt = panel.copy()
    for rate in RATE_DENOMINATORS["hitter"]:
        factor = rebuilt["season"].map(factors[rate]).astype(float).fillna(1.0)
        rebuilt[f"era_factor_{rate}"] = factor
        rebuilt[rate] = rebuilt[rate] * factor
    rebuilt = score(rebuilt, "hitter")

    pd.testing.assert_frame_equal(normalized, rebuilt)


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


def test_precomputed_factors_override_the_frames_own_rates() -> None:
    """The base season's run environment is a fact about what HAPPENED in it, so the
    factor must be able to come from a frame other than the one being rescaled (#348).

    The trajectory board replaces its in-progress season with a YTD + rest-of-season
    line before normalizing. Projections are regressed toward the mean, so league rates
    computed off the injected rows sit closer to the reference than the season really
    did -- and the factor for the one season the whole board is anchored on would
    silently shrink.
    """
    actual = _hitters(
        [
            {"mlbam_id": 1, "season": 2023, "hr_pa": 0.050},
            {"mlbam_id": 2, "season": 2024, "hr_pa": 0.060},
            {"mlbam_id": 3, "season": 2025, "hr_pa": 0.055},
            # The real 2026: a depressed home-run environment, so its factor is > 1.
            {"mlbam_id": 4, "season": 2026, "hr_pa": 0.030},
        ]
    )
    factors = era_factors(actual, "hitter")
    assert factors.loc[2026, "hr_pa"] > 1.2

    # The same panel with 2026 replaced by a regressed projection sitting right on the
    # reference. Left to itself, era_normalize would call that season a neutral one.
    injected = actual.copy()
    injected.loc[injected["season"] == 2026, "hr_pa"] = 0.055
    assert era_factors(injected, "hitter").loc[2026, "hr_pa"] == pytest.approx(1.0, abs=0.02)

    normalized = era_normalize(injected, "hitter", factors=factors)
    applied = normalized.loc[normalized["season"] == 2026, "era_factor_hr_pa"]
    assert applied.iloc[0] == pytest.approx(factors.loc[2026, "hr_pa"])
    assert normalized.loc[normalized["season"] == 2026, "hr_pa"].iloc[0] == pytest.approx(
        0.055 * factors.loc[2026, "hr_pa"]
    )


def test_a_supplied_factor_table_must_cover_the_frame() -> None:
    """`factors` bypasses `era_factors`, and with it the reference-window check that is
    the only thing standing between a caller and a silently wrong restatement.

    A season the table does not cover does not fail -- `.map` yields NaN and the
    `fillna(1.0)` below treats it as a neutral era, so that season silently keeps its
    raw rates while every other season is restated around it. That is a season priced in
    different units from the ones it is ranked against, and nothing says so.
    """
    panel = _hitters(
        [
            {"mlbam_id": 1, "season": 2023, "hr_pa": 0.050},
            {"mlbam_id": 2, "season": 2024, "hr_pa": 0.060},
            {"mlbam_id": 3, "season": 2025, "hr_pa": 0.055},
            {"mlbam_id": 4, "season": 2026, "hr_pa": 0.030},
        ]
    )
    partial = era_factors(panel[panel["season"] != 2026], "hitter")
    with pytest.raises(ValueError, match="2026"):
        era_normalize(panel, "hitter", factors=partial)


def test_a_factor_table_missing_a_rate_column_is_refused() -> None:
    """Same failure with a different shape: a dropped column takes the whole category
    out of the restatement while the rest of the frame is scaled around it."""
    panel = _reference_panel()
    factors = era_factors(panel, "hitter").drop(columns=["hr_pa"])
    with pytest.raises(KeyError, match="hr_pa"):
        era_normalize(panel, "hitter", factors=factors)
