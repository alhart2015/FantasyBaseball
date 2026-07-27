import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import keeper_calibration as script

from fantasy_baseball.keepers.calibration import YearPair


def _synthetic_pairs(k_true: float = 0.5) -> list[YearPair]:
    """Three pairs of deterministic data -- no network, no files."""
    rng = np.random.default_rng(0)
    pairs = []
    for year in (2022, 2023, 2024):
        n = 200
        idx = list(range(n))
        base = pd.DataFrame(
            {"hr_pa": rng.uniform(0.02, 0.06, n), "sb_pa": rng.uniform(0.0, 0.05, n)}, index=idx
        )
        resid = pd.DataFrame(
            {"hr_pa": rng.normal(0, 0.01, n), "sb_pa": rng.normal(0, 0.01, n)}, index=idx
        )
        realized = pd.Series(np.full(n, 600.0), index=idx)
        weight = realized / (realized + 200.0)
        pairs.append(
            YearPair(
                year=year,
                base=base,
                residual=resid,
                target=base + k_true * resid.mul(weight, axis=0),
                realized_pt=realized,
                target_pt=pd.Series(np.full(n, 600.0), index=idx),
            )
        )
    return pairs


def test_build_report_covers_every_coefficient_and_estimator() -> None:
    report = script.build_report(_synthetic_pairs(), columns=("hr_pa", "sb_pa"), n0=200.0)
    assert set(report["column"]) == {"hr_pa", "sb_pa"}
    assert {"k=0", "k=1"} <= set(report["estimator"])
    # Acceptance is per coefficient, not pooled -- spec 6.6.
    assert "verdict" in report.columns


def test_report_passes_a_coefficient_the_fit_recovers() -> None:
    # k_true = 0.5 sits strictly between the two endpoints, so a working fit
    # must beat both on every held-out pair.
    report = script.build_report(_synthetic_pairs(0.5), columns=("hr_pa",), n0=200.0)
    row = report.loc[report["estimator"] == "fitted-k"].iloc[0]
    assert row["verdict"] == "pass"
    assert row["k_full"] == pytest.approx(0.5, abs=0.05)


def test_report_falls_back_when_an_endpoint_wins() -> None:
    # k_true = 0 IS the k=0 endpoint, so the fitted estimator cannot beat it.
    report = script.build_report(_synthetic_pairs(0.0), columns=("hr_pa",), n0=200.0)
    assert set(report["verdict"]) == {"fallback:k=0"}


def test_report_flags_a_coefficient_outside_the_unit_interval() -> None:
    report = script.build_report(_synthetic_pairs(1.6), columns=("hr_pa",), n0=200.0)
    row = report.loc[report["estimator"] == "fitted-k"].iloc[0]
    assert bool(row["out_of_range"]) is True
    assert row["k_full"] == 1.0  # shipped value is clamped
    assert row["k_raw_full"] > 1.0  # the raw fit stays visible


def test_playing_time_column_is_fit_unshrunk_and_unweighted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spec 5.3 / finding A.1: the PT coefficient must not be damped by the
    # shrink, and must not be weighted by the very quantity it predicts.
    calls: list[dict[str, Any]] = []
    real = script.leave_one_out

    def spy(estimator: Any, pairs: Any, column: str, n0: float, **kwargs: Any) -> pd.DataFrame:
        calls.append({"column": column, **kwargs})
        return real(estimator, pairs, column, n0, **kwargs)

    monkeypatch.setattr(script, "leave_one_out", spy)
    script.build_report(_synthetic_pairs(), columns=("hr_pa", "sb_pa"), n0=200.0, pt_col="sb_pa")

    pt_calls = [c for c in calls if c["column"] == "sb_pa"]
    rate_calls = [c for c in calls if c["column"] == "hr_pa"]
    assert pt_calls and all(c["shrunk"] is False and c["weighted"] is False for c in pt_calls)
    assert rate_calls and all(c["shrunk"] is True and c["weighted"] is True for c in rate_calls)
