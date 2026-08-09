"""Guards on the historical mode added for #325.

These assert the things whose failure produces a PLAUSIBLE WRONG NUMBER rather than an
exception -- a 2026 lag inside a 2022 forecast, a panel that still contains the future,
a query player who can match himself. None of them would raise; all of them would
silently change the verdict this backtest exists to produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _pt_panel(seasons: list[int]) -> pd.DataFrame:
    """Minimal playing-time panel: one row per (player, season) for two players."""
    rows = []
    for pid in (1, 2):
        for season in seasons:
            rows.append(
                {
                    "mlbam_id": pid,
                    "season": season,
                    "pa": 600.0,
                    "ip": 0.0,
                    "games": 150,
                    "starts": 0,
                    "age": 25 + (season - min(seasons)),
                    "partial_season": False,
                }
            )
    return pd.DataFrame(rows)


class _StubCurve:
    """Stands in for the fitted playing-time curve; returns a flat projection."""

    def predict(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(600.0, index=features.index)


def test_volume_forecast_reads_no_season_after_its_base_year(monkeypatch) -> None:
    """BASE_YEAR was a module constant read in SIX places inside volume_forecast.

    Threading five of the six leaves a 2026 lag in a 2022 forecast, and nothing raises
    -- the forecast simply uses a season the estimator was never supposed to see.
    """
    import keeper_forecast

    seen: list[int] = []
    panel = _pt_panel(list(range(2018, 2027)))

    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: Path("fake.csv"))
    monkeypatch.setattr(keeper_forecast.pd, "read_csv", lambda *_a, **_k: panel)
    monkeypatch.setattr(
        keeper_forecast,
        "lag_panel",
        lambda *_a, **_k: pd.DataFrame(
            {feature: [1.0, 2.0] for feature in keeper_forecast.FEATURES["hitter"]}
            | {"target": [600.0, 610.0]}
        ),
    )
    monkeypatch.setattr(keeper_forecast, "fit_curve", lambda *_a, **_k: _StubCurve())

    real_series_for = keeper_forecast._series_for

    def spy(panel_arg, year, column, index):
        seen.append(year)
        return real_series_for(panel_arg, year, column, index)

    monkeypatch.setattr(keeper_forecast, "_series_for", spy)

    observed = pd.Series([600.0, 600.0], index=pd.Index([1, 2], name="mlbam_id"))
    keeper_forecast.volume_forecast("hitter", 2022, 2023, observed)

    assert seen, "volume_forecast never consulted the panel"
    assert max(seen) <= 2022, f"read seasons after the base year: {sorted(set(seen))}"


def test_volume_forecast_walks_age_to_the_target_year(monkeypatch) -> None:
    """A two-years-out target applies the curve twice, walking age forward each step.
    The base year is now a parameter, so the step count must come from it and not from
    a constant that happens to be 2026."""
    import keeper_forecast

    ages: list[float] = []
    panel = _pt_panel(list(range(2018, 2027)))

    class _RecordingCurve:
        def predict(self, features: pd.DataFrame) -> pd.Series:
            ages.append(float(features["age"].iloc[0]))
            return pd.Series(600.0, index=features.index)

    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: Path("fake.csv"))
    monkeypatch.setattr(keeper_forecast.pd, "read_csv", lambda *_a, **_k: panel)
    monkeypatch.setattr(
        keeper_forecast,
        "lag_panel",
        lambda *_a, **_k: pd.DataFrame(
            {feature: [1.0, 2.0] for feature in keeper_forecast.FEATURES["hitter"]}
            | {"target": [600.0, 610.0]}
        ),
    )
    monkeypatch.setattr(keeper_forecast, "fit_curve", lambda *_a, **_k: _RecordingCurve())

    observed = pd.Series([600.0, 600.0], index=pd.Index([1, 2], name="mlbam_id"))
    keeper_forecast.volume_forecast("hitter", 2022, 2024, observed)

    # Two steps for a +2 target, and the first projects 2023 (one year younger than
    # the 2024 target), not the target itself.
    assert len(ages) == 2
    assert ages[1] == pytest.approx(ages[0] + 1.0)
