import math

import pytest

from fantasy_baseball.analysis.pace import compute_sgp_deviation
from fantasy_baseball.utils.constants import Category

# Explicit denominators so expected values are hand-computable.
DENOMS = {
    Category.R: 10.0,
    Category.HR: 5.0,
    Category.RBI: 10.0,
    Category.SB: 5.0,
    Category.AVG: 0.0015,
    Category.W: 2.0,
    Category.K: 12.0,
    Category.SV: 3.5,
    Category.ERA: 0.10,
    Category.WHIP: 0.03,
}


def test_hitter_full_deviation():
    # 100/500 PA -> proration factor 0.2. Overperforming across the board.
    actual = {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}
    projected = {"pa": 500, "ab": 450, "avg": 0.280, "r": 100, "hr": 30, "rbi": 100, "sb": 20}
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    # counting dev: R (25-20)/10=.5, HR (8-6)/5=.4, RBI (30-20)/10=1.0, SB (6-4)/5=.4 -> 2.3
    # AVG dev: (0.300-0.280)*90/(0.0015*5500) = 1.8/8.25 = 0.2182
    assert out["sgp_dev"] == pytest.approx(2.518, abs=0.005)
    assert out["actual_sgp"] == pytest.approx(8.845, abs=0.005)
    assert out["expected_sgp"] == pytest.approx(6.327, abs=0.005)
    # replacement cancels in the delta
    assert out["sgp_dev"] == pytest.approx(out["actual_sgp"] - out["expected_sgp"], abs=1e-6)


def test_hitter_rate_gated_out_below_30_pa():
    # 20 PA: counting colored (>=10), AVG NOT colored (<30) -> AVG excluded.
    actual = {"pa": 20, "ab": 18, "h": 8, "r": 10, "hr": 4, "rbi": 12, "sb": 2}
    projected = {"pa": 200, "ab": 180, "avg": 0.300, "r": 100, "hr": 40, "rbi": 100, "sb": 20}
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    # factor 0.1: R exp10 act10 ->0, HR exp4 act4 ->0, RBI exp10 act12 ->0.2, SB exp2 act2 ->0
    assert out["sgp_dev"] == pytest.approx(0.2, abs=1e-6)
    # actual_sgp counts only R/HR/RBI/SB (no AVG term): 1.0+0.8+1.2+0.4 = 3.4
    assert out["actual_sgp"] == pytest.approx(3.4, abs=1e-6)


def test_below_counting_gate_returns_none():
    actual = {"pa": 5, "ab": 4, "h": 1, "r": 1, "hr": 0, "rbi": 1, "sb": 0}
    projected = {"pa": 500, "ab": 450, "avg": 0.280, "r": 100, "hr": 30, "rbi": 100, "sb": 20}
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    assert out["sgp_dev"] is None


def test_no_projection_returns_none():
    actual = {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}
    out = compute_sgp_deviation(actual, {}, "hitter", DENOMS)
    assert out["sgp_dev"] is None


def test_pitcher_outperforming_positive_dev():
    # Lower ERA/WHIP than projected -> positive deviation (inverse stats).
    actual = {"ip": 50, "k": 60, "w": 5, "sv": 0, "er": 15, "bb": 12, "h_allowed": 38}
    projected = {"ip": 180, "k": 200, "w": 12, "sv": 0, "era": 3.50, "whip": 1.10}
    out = compute_sgp_deviation(actual, projected, "pitcher", DENOMS)
    # actual ERA 2.70 < 3.50, actual WHIP 1.00 < 1.10, K/W ahead of pace
    assert out["sgp_dev"] > 0


def test_pitcher_underperforming_negative_dev():
    actual = {"ip": 50, "k": 30, "w": 1, "sv": 0, "er": 35, "bb": 25, "h_allowed": 60}
    projected = {"ip": 180, "k": 200, "w": 12, "sv": 0, "era": 3.50, "whip": 1.10}
    out = compute_sgp_deviation(actual, projected, "pitcher", DENOMS)
    # actual ERA 6.30 > 3.50, actual WHIP 1.70 > 1.10, K/W behind pace
    assert out["sgp_dev"] < 0


def test_zero_projected_pa_returns_none():
    # Phantom projection: 0 projected PA has no basis for "expected" -> excluded
    # (else full actuals credit against ~0 expected inflates sgp_dev).
    actual = {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}
    projected = {"pa": 0, "ab": 0, "avg": 0.0, "r": 0, "hr": 0, "rbi": 0, "sb": 0}
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    assert out["sgp_dev"] is None


def test_nan_projected_pa_returns_none():
    actual = {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}
    projected = {
        "pa": float("nan"),
        "ab": 90,
        "avg": 0.280,
        "r": 100,
        "hr": 30,
        "rbi": 100,
        "sb": 20,
    }
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    assert out["sgp_dev"] is None


def test_zero_projected_avg_skips_rate_term():
    # Valid projected PA but 0 projected AVG -> AVG term skipped (not scored
    # against a .000 baseline); sgp_dev is the counting-only delta.
    actual = {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}
    projected = {"pa": 500, "ab": 450, "avg": 0.0, "r": 100, "hr": 30, "rbi": 100, "sb": 20}
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    assert out["sgp_dev"] == pytest.approx(2.3, abs=0.005)  # counting only, no AVG


def test_zero_projected_era_skips_rate_term():
    # 0 projected ERA -> ERA term skipped; result stays finite/defined.
    actual = {"ip": 50, "k": 60, "w": 5, "sv": 0, "er": 15, "bb": 12, "h_allowed": 38}
    projected = {"ip": 180, "k": 200, "w": 12, "sv": 0, "era": 0.0, "whip": 1.10}
    out = compute_sgp_deviation(actual, projected, "pitcher", DENOMS)
    assert out["sgp_dev"] is not None
    assert not math.isnan(out["sgp_dev"])
