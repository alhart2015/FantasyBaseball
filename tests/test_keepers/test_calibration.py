from dataclasses import replace

import pandas as pd
import pytest

from fantasy_baseball.keepers.calibration import (
    PAIR_YEARS,
    FullTransfer,
    YearPair,
    ZeroTransfer,
    leave_one_out,
    survivorship,
    weighted_mse,
)


def test_pair_years_are_the_three_usable_ones() -> None:
    # 2025->2026 needs a complete 2026 season; 2021->2022 has no ZiPS 2021 on disk.
    assert PAIR_YEARS == (2022, 2023, 2024)


def test_survivorship_counts_players_who_kept_playing() -> None:
    pair = YearPair(
        year=2022,
        base=pd.DataFrame({"hr_pa": [0.04, 0.03]}, index=[1, 2]),
        residual=pd.DataFrame({"hr_pa": [0.01, -0.01]}, index=[1, 2]),
        target=pd.DataFrame({"hr_pa": [0.045, float("nan")]}, index=[1, 2]),
        realized_pt=pd.Series([500.0, 300.0], index=[1, 2]),
        target_pt=pd.Series([550.0, 0.0], index=[1, 2]),
    )
    out = survivorship([pair], threshold=100.0)
    row = out.iloc[0]
    assert row["n_in_year"] == 2
    assert row["n_survived"] == 1
    assert row["survival_rate"] == 0.5


def _pair(year: int) -> YearPair:
    idx = [1, 2, 3]
    return YearPair(
        year=year,
        base=pd.DataFrame({"hr_pa": [0.030, 0.040, 0.050]}, index=idx),
        residual=pd.DataFrame({"hr_pa": [0.010, 0.000, -0.010]}, index=idx),
        target=pd.DataFrame({"hr_pa": [0.040, 0.040, 0.040]}, index=idx),
        realized_pt=pd.Series([600.0, 600.0, 600.0], index=idx),
        target_pt=pd.Series([600.0, 600.0, 600.0], index=idx),
    )


def test_weighted_mse_weights_by_playing_time() -> None:
    pred = pd.Series([0.0, 0.0])
    actual = pd.Series([1.0, 0.0])
    assert weighted_mse(pred, actual, pd.Series([1.0, 1.0])) == pytest.approx(0.5)
    assert weighted_mse(pred, actual, pd.Series([3.0, 1.0])) == pytest.approx(0.75)


def test_endpoints_predict_as_documented() -> None:
    pair = _pair(2022)
    zero = ZeroTransfer().fit([pair], "hr_pa", n0=200.0)
    full = FullTransfer().fit([pair], "hr_pa", n0=200.0)
    w = pd.Series([1.0, 1.0, 1.0], index=pair.base.index)
    b, r = pair.base["hr_pa"], pair.residual["hr_pa"]
    assert list(zero.predict(b, r, w)) == pytest.approx(list(b))
    assert list(full.predict(b, r, w)) == pytest.approx(list(b + r))


def test_leave_one_out_holds_out_each_pair() -> None:
    pairs = [_pair(2022), _pair(2023), _pair(2024)]
    out = leave_one_out(ZeroTransfer(), pairs, "hr_pa", n0=200.0, gate=0.0)
    assert sorted(out["held_out_year"]) == [2022, 2023, 2024]
    assert out["error"].notna().all()


def test_leave_one_out_gate_drops_rows_below_the_floor() -> None:
    pair = _pair(2022)
    low = YearPair(
        year=2023,
        base=pair.base,
        residual=pair.residual,
        target=pair.target,
        realized_pt=pd.Series([600.0, 10.0, 10.0], index=pair.base.index),
        target_pt=pair.target_pt,
    )
    out = leave_one_out(ZeroTransfer(), [pair, low], "hr_pa", n0=200.0, gate=100.0)
    assert int(out.loc[out["held_out_year"] == 2023, "n"].iloc[0]) == 1
    assert int(out.loc[out["held_out_year"] == 2022, "n"].iloc[0]) == 3


def test_unweighted_evaluation_keeps_rows_with_zero_target_playing_time() -> None:
    # The PT coefficient is scored unweighted so non-survivors (target_pt == 0)
    # stay in the sample -- weighting by target_pt would delete exactly the
    # players whose lost playing time the coefficient exists to learn from.
    idx = [1, 2]
    pair = YearPair(
        year=2022,
        base=pd.DataFrame({"pa": [500.0, 500.0]}, index=idx),
        residual=pd.DataFrame({"pa": [50.0, -400.0]}, index=idx),
        target=pd.DataFrame({"pa": [520.0, 0.0]}, index=idx),
        realized_pt=pd.Series([550.0, 100.0], index=idx),
        target_pt=pd.Series([520.0, 0.0], index=idx),
    )
    pairs = [pair, replace(pair, year=2023)]
    weighted = leave_one_out(
        ZeroTransfer(), pairs, "pa", n0=200.0, gate=0.0, shrunk=False, weighted=True
    )
    unweighted = leave_one_out(
        ZeroTransfer(), pairs, "pa", n0=200.0, gate=0.0, shrunk=False, weighted=False
    )
    # Weighted drops the non-survivor entirely (weight 0); unweighted sees his
    # 500-PA miss, so the error is far larger.
    assert unweighted["error"].iloc[0] > weighted["error"].iloc[0]


def test_unshrunk_weight_is_one_so_playing_time_is_not_damped() -> None:
    pair = _pair(2022)
    out = leave_one_out(
        FullTransfer(), [pair, _pair(2023)], "hr_pa", n0=200.0, gate=0.0, shrunk=False
    )
    # k=1 unshrunk predicts base + residual exactly, which equals the target here.
    assert out["error"].iloc[0] == pytest.approx(0.0)
