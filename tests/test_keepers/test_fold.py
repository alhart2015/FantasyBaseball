import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.keepers.fold import (
    fold_rates,
    gate_mask,
    gate_ramp,
    reconstruct_hitter,
    reconstruct_pitcher,
    shrink,
)


def test_shrink_is_bounded_and_monotone() -> None:
    n = pd.Series([0.0, 50.0, 200.0, 600.0, 5000.0])
    w = shrink(n, n0=200.0)
    assert w.iloc[0] == 0.0
    assert w.iloc[2] == pytest.approx(0.5)
    assert (w < 1.0).all()  # never amplifies
    assert w.is_monotonic_increasing


def test_gate_excludes_low_playing_time() -> None:
    realized = pd.Series([0.0, 30.0, 200.0], index=[1, 2, 3])
    mask = gate_mask(realized, threshold=50.0)
    assert list(mask) == [False, False, True]


def test_gate_treats_missing_as_ungated() -> None:
    # Absence from the MLB leaderboard means AAA, not zero PA -- it must fall
    # through to passthrough, never to a large negative residual. Spec 5.4.
    realized = pd.Series([np.nan], index=[9])
    assert list(gate_mask(realized, threshold=50.0)) == [False]


def test_fold_rates_endpoints() -> None:
    base = pd.DataFrame({"hr_pa": [0.04]})
    resid = pd.DataFrame({"hr_pa": [0.02]})
    weight = pd.Series([1.0])
    assert fold_rates(base, resid, weight, k=0.0)["hr_pa"].iloc[0] == pytest.approx(0.04)
    assert fold_rates(base, resid, weight, k=1.0)["hr_pa"].iloc[0] == pytest.approx(0.06)
    assert fold_rates(base, resid, weight, k=0.5)["hr_pa"].iloc[0] == pytest.approx(0.05)


def test_fold_rates_floors_at_zero() -> None:
    base = pd.DataFrame({"sb_pa": [0.01]})
    resid = pd.DataFrame({"sb_pa": [-0.30]})
    out = fold_rates(base, resid, pd.Series([1.0]), k=1.0)
    assert out["sb_pa"].iloc[0] == 0.0  # never negative


def test_fold_rates_passes_through_a_nan_residual() -> None:
    # No observation must mean "do not move", not "move by zero-minus-base".
    base = pd.DataFrame({"hr_pa": [0.04]})
    resid = pd.DataFrame({"hr_pa": [float("nan")]})
    out = fold_rates(base, resid, pd.Series([1.0]), k=1.0)
    assert out["hr_pa"].iloc[0] == pytest.approx(0.04)


HITTER_RATE_ROW = pd.DataFrame(
    {
        "ab_pa": [0.9],
        "h_ab": [0.300],
        "hr_pa": [0.05],
        "r_pa": [0.15],
        "rbi_pa": [0.16],
        "sb_pa": [0.02],
    }
)
PITCHER_RATE_ROW = pd.DataFrame(
    {"k_ip": [1.0], "w_ip": [0.08], "er_ip": [0.35], "bb_ip": [0.25], "h_ip": [0.80]}
)


def test_reconstruct_hitter_uses_ab_for_hits_and_pa_for_counting() -> None:
    out = reconstruct_hitter(HITTER_RATE_ROW, pd.Series([600.0]))
    assert out["ab"].iloc[0] == pytest.approx(540.0)
    assert out["h"].iloc[0] == pytest.approx(540.0 * 0.300)  # AB, not PA
    assert out["avg"].iloc[0] == pytest.approx(0.300)  # not inflated by 1/0.9
    assert out["hr"].iloc[0] == pytest.approx(30.0)
    assert out["ab"].iloc[0] <= out["pa"].iloc[0]  # structural


def test_reconstruct_hitter_guards_zero_ab() -> None:
    out = reconstruct_hitter(HITTER_RATE_ROW, pd.Series([0.0]))
    assert out["avg"].iloc[0] == 0.0  # 0/0 guarded, not NaN


def test_reconstruct_pitcher_builds_era_and_whip_from_components() -> None:
    out = reconstruct_pitcher(PITCHER_RATE_ROW, pd.Series([180.0]))
    assert out["era"].iloc[0] == pytest.approx(9 * 0.35)
    assert out["whip"].iloc[0] == pytest.approx(0.25 + 0.80)
    assert out["k"].iloc[0] == pytest.approx(180.0)
    assert out["bb"].iloc[0] == pytest.approx(45.0)
    assert out["h_allowed"].iloc[0] == pytest.approx(144.0)


def test_reconstruct_pitcher_guards_zero_ip() -> None:
    out = reconstruct_pitcher(PITCHER_RATE_ROW, pd.Series([0.0]))
    assert out["era"].iloc[0] == 0.0
    assert out["whip"].iloc[0] == 0.0


def test_gate_ramp_removes_the_cliff_at_the_threshold() -> None:
    realized = pd.Series([0.0, 99.0, 100.0, 150.0, 200.0, 600.0], index=range(6))
    ramp = gate_ramp(realized, threshold=100.0, width=100.0)
    assert list(ramp) == pytest.approx([0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
    assert ramp.is_monotonic_increasing


def test_gate_ramp_agrees_with_the_hard_gate_off_the_ramp() -> None:
    realized = pd.Series([50.0, 400.0], index=[1, 2])
    ramp = gate_ramp(realized, threshold=100.0, width=100.0)
    mask = gate_mask(realized, threshold=100.0)
    assert list(ramp.astype(bool)) == list(mask)


def test_gate_ramp_treats_missing_as_unfolded() -> None:
    # Absence from the MLB leaderboard is AAA, not zero PA -- same rule as the
    # hard gate: it must resolve to no fold at all.
    assert gate_ramp(pd.Series([np.nan], index=[9]), threshold=100.0, width=100.0).iloc[0] == 0.0


def test_gate_ramp_rejects_a_nonpositive_width() -> None:
    with pytest.raises(ValueError, match="width must be positive"):
        gate_ramp(pd.Series([100.0]), threshold=100.0, width=0.0)


def test_fold_rates_accepts_a_per_column_coefficient_mapping() -> None:
    # The study fits one k PER column, so the shipped model cannot be expressed
    # by a scalar. A column absent from the mapping is not folded.
    base = pd.DataFrame({"hr_pa": [0.040], "sb_pa": [0.020], "r_pa": [0.150]})
    resid = pd.DataFrame({"hr_pa": [0.010], "sb_pa": [0.010], "r_pa": [0.010]})
    out = fold_rates(base, resid, pd.Series([1.0]), k={"hr_pa": 0.5, "sb_pa": 1.0})
    assert out["hr_pa"].iloc[0] == pytest.approx(0.045)
    assert out["sb_pa"].iloc[0] == pytest.approx(0.030)
    assert out["r_pa"].iloc[0] == pytest.approx(0.150)  # unmapped -> unfolded


def test_fold_rates_scalar_and_mapping_agree() -> None:
    base = pd.DataFrame({"hr_pa": [0.04, 0.05], "sb_pa": [0.02, 0.01]})
    resid = pd.DataFrame({"hr_pa": [0.01, -0.02], "sb_pa": [-0.03, 0.02]})
    weight = pd.Series([0.75, 0.5])
    scalar = fold_rates(base, resid, weight, k=0.6)
    mapping = fold_rates(base, resid, weight, k={"hr_pa": 0.6, "sb_pa": 0.6})
    pd.testing.assert_frame_equal(scalar, mapping)
