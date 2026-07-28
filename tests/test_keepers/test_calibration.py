from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.keepers import calibration
from fantasy_baseball.keepers.calibration import (
    PAIR_YEARS,
    FullTransfer,
    ShrunkTransfer,
    YearPair,
    ZeroTransfer,
    gated,
    leave_one_out,
    survivorship,
    weighted_mse,
)
from fantasy_baseball.keepers.fold import shrink
from tests.test_keepers.conftest import mlb_hitting, write_zips_vintage


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
    out = leave_one_out(ZeroTransfer(), pairs, "hr_pa", n0=200.0)
    assert sorted(out["held_out_year"]) == [2022, 2023, 2024]
    assert out["error"].notna().all()


def test_gated_drops_rows_below_the_floor() -> None:
    # Gating is the caller's job and happens once, before leave_one_out sees the
    # pairs -- so the study has exactly one place that decides which rows it uses.
    pair = _pair(2022)
    low = YearPair(
        year=2023,
        base=pair.base,
        residual=pair.residual,
        target=pair.target,
        realized_pt=pd.Series([600.0, 10.0, 10.0], index=pair.base.index),
        target_pt=pair.target_pt,
    )
    kept = [gated(p, 100.0) for p in (pair, low)]
    out = leave_one_out(ZeroTransfer(), kept, "hr_pa", n0=200.0)
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
    weighted = leave_one_out(ZeroTransfer(), pairs, "pa", n0=200.0, shrunk=False, weighted=True)
    unweighted = leave_one_out(ZeroTransfer(), pairs, "pa", n0=200.0, shrunk=False, weighted=False)
    # Weighted drops the non-survivor entirely (weight 0); unweighted sees his
    # 500-PA miss, so the error is far larger.
    assert unweighted["error"].iloc[0] > weighted["error"].iloc[0]


def test_unshrunk_weight_is_one_so_playing_time_is_not_damped() -> None:
    pair = _pair(2022)
    out = leave_one_out(FullTransfer(), [pair, _pair(2023)], "hr_pa", n0=200.0, shrunk=False)
    # k=1 unshrunk predicts base + residual exactly, which equals the target here.
    assert out["error"].iloc[0] == pytest.approx(0.0)


def test_build_pairs_keeps_non_survivors_with_zero_target_playing_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Absence from the Y+1 leaderboard: no Y+1 rate (NaN), but a real Y+1 MLB
    # playing time of zero. Dropping these rows would delete the survivorship
    # signal the PT coefficient exists to measure.
    write_zips_vintage(tmp_path / "2024", MLBAMID=[1, 2])

    def _raw(ids: list[int], pas: list[int]) -> pd.DataFrame:
        return mlb_hitting(
            **{
                "player.id": ids,
                "stat.plateAppearances": pas,
                "stat.atBats": [int(p * 0.9) for p in pas],
                "stat.hits": [int(p * 0.25) for p in pas],
                "stat.runs": [int(p * 0.13) for p in pas],
                "stat.homeRuns": [int(p * 0.04) for p in pas],
                "stat.rbi": [int(p * 0.13) for p in pas],
                "stat.stolenBases": [int(p * 0.02) for p in pas],
            }
        )

    frames = {2024: _raw([1, 2], [620, 580]), 2025: _raw([1], [600])}
    monkeypatch.setattr(
        calibration, "fetch_mlb_season", lambda cache_dir, year, group: frames[year]
    )
    pair = calibration.build_pairs("hitter", tmp_path, tmp_path, years=(2024,))[0]

    assert list(pair.base.index) == [1, 2]  # non-survivor still present
    assert pd.isna(pair.target.loc[2, "hr_pa"])  # no Y+1 rate
    assert pair.target.loc[2, "pa"] == 0.0  # but a real Y+1 playing time
    assert pair.target_pt.loc[2] == 0.0


def _planted_pair(
    year: int,
    k_true: float,
    c_true: float,
    n: int,
    n0: float,
    seed: int,
    resid_mean: float = 0.0,
) -> YearPair:
    """One pair whose target is base + c_true + k_true * shrink * residual + noise.

    `resid_mean` reproduces the playing-time case: ZiPS hedges pool-wide, so the
    residual itself has a large systematic mean alongside the level offset.
    """
    rng = np.random.default_rng(seed)
    idx = list(range(n))
    base = pd.Series(rng.uniform(0.02, 0.06, n), index=idx)
    resid = pd.Series(rng.normal(resid_mean, 0.01, n), index=idx)
    realized = pd.Series(rng.uniform(150.0, 650.0, n), index=idx)
    weight = shrink(realized, n0)
    target = (
        base + c_true + k_true * weight * resid + pd.Series(rng.normal(0.0, 1e-4, n), index=idx)
    )
    return YearPair(
        year=year,
        base=base.to_frame("hr_pa"),
        residual=resid.to_frame("hr_pa"),
        target=target.to_frame("hr_pa"),
        realized_pt=realized,
        target_pt=pd.Series(np.full(n, 600.0), index=idx),
    )


def test_estimator_recovers_a_planted_coefficient() -> None:
    pair = _planted_pair(2022, k_true=0.6, c_true=0.0, n=500, n0=200.0, seed=0)
    fitted = ShrunkTransfer().fit([pair], "hr_pa", n0=200.0)
    assert fitted.params["k"] == pytest.approx(0.6, abs=0.05)


def test_estimator_separates_a_level_offset_from_the_slope() -> None:
    # Spec 6.2 requirement 12: the playing-time residual carries a large
    # systematic MEAN that is not surprise. A pure multiplicative fit cannot
    # separate the level from the signal; the nuisance intercept can.
    pair = _planted_pair(2022, k_true=0.6, c_true=0.004, n=2000, n0=200.0, seed=1, resid_mean=0.006)
    fitted = ShrunkTransfer().fit([pair], "hr_pa", n0=200.0)
    assert fitted.params["k_raw"] == pytest.approx(0.6, abs=0.05)
    assert fitted.params["c_fit"] == pytest.approx(0.004, abs=5e-4)

    no_intercept = ShrunkTransfer(use_intercept=False).fit([pair], "hr_pa", n0=200.0)
    # Without the intercept the level offset is forced onto the slope, which is
    # the failure mode requirement 12 describes.
    assert abs(no_intercept.params["k_raw"] - 0.6) > 0.2


def test_shipped_prediction_does_not_apply_the_calibration_intercept() -> None:
    # Requirement 1: the form applied in production is base + k * w * residual.
    # The intercept is a calibration-only nuisance term whose production value
    # is 0, because ZiPS_2027 is already aged forward and ZiPS_Y is not.
    pair = _planted_pair(2022, k_true=0.6, c_true=0.01, n=400, n0=200.0, seed=2)
    fitted = ShrunkTransfer().fit([pair], "hr_pa", n0=200.0)
    base = pd.Series([0.04], index=[0])
    resid = pd.Series([0.02], index=[0])
    weight = pd.Series([1.0], index=[0])
    expected = 0.04 + fitted.params["k"] * 0.02
    assert fitted.predict(base, resid, weight).iloc[0] == pytest.approx(expected)


def test_estimator_clamps_an_amplifying_coefficient_but_reports_the_raw_fit() -> None:
    # Requirement 7: a coefficient that would amplify residuals must not ship
    # silently. The shipped k is bounded; the raw fit stays visible.
    pair = _planted_pair(2022, k_true=1.8, c_true=0.0, n=800, n0=200.0, seed=3)
    fitted = ShrunkTransfer().fit([pair], "hr_pa", n0=200.0)
    assert fitted.params["k_raw"] > 1.0
    assert fitted.params["k"] == 1.0
    assert fitted.params["clamped"] == 1.0


def test_estimator_reports_a_confidence_interval() -> None:
    pair = _planted_pair(2022, k_true=0.6, c_true=0.0, n=1000, n0=200.0, seed=4)
    fitted = ShrunkTransfer().fit([pair], "hr_pa", n0=200.0)
    assert fitted.params["ci_lo"] < fitted.params["k_raw"] < fitted.params["ci_hi"]
    assert fitted.params["n_fit"] == 1000


def test_estimator_drops_rows_the_metric_would_drop() -> None:
    # Fit sample == eval sample: a NaN target (a non-survivor, who has no Y+1
    # rate) must not enter the weighted fit. The zero-weight half of `usable` is
    # covered by test_unweighted_evaluation_keeps_rows_with_zero_target_playing_time.
    pair = _planted_pair(2022, k_true=0.6, c_true=0.0, n=300, n0=200.0, seed=5)
    target = pair.target.copy()
    target.iloc[:100, 0] = np.nan
    holed = replace(pair, target=target)
    fitted = ShrunkTransfer().fit([holed], "hr_pa", n0=200.0)
    assert fitted.params["n_fit"] == 200


def test_build_pairs_refuses_an_incomplete_season(tmp_path: Path) -> None:
    """Spec 9 requires this test and it never existed.

    `keepers/cache.py` `fetch_or_cache` NEVER invalidates, so a mid-season pull
    freezes permanently -- the one failure mode here that cannot self-heal. The
    guard must fire BEFORE any fetch, not after.
    """
    fetched: list[int] = []
    monkey = pytest.MonkeyPatch()
    monkey.setattr(
        calibration,
        "fetch_mlb_season",
        lambda cache_dir, year, group: fetched.append(year) or pd.DataFrame(),
    )
    try:
        with pytest.raises(ValueError, match="needs a complete"):
            calibration.build_pairs(
                "hitter", tmp_path, tmp_path, years=(calibration.LAST_COMPLETE_SEASON,)
            )
    finally:
        monkey.undo()
    assert fetched == [], "the guard must fire before any season is fetched and cached"


def test_build_pairs_rejects_an_unknown_player_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="player_type must be"):
        calibration.build_pairs("goalie", tmp_path, tmp_path)
