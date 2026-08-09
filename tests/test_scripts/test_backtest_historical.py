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


def test_transitions_for_matches_the_counts_the_spec_discloses() -> None:
    """The leakage disclosure in the writeup is computed from these.

    Base 2024 is the year where loto and causal COINCIDE, which is why the sensitivity
    check runs on 2023. Running it on 2024 would return a difference of exactly zero
    and read as "the leakage is negligible" when nothing had been measured.
    """
    from backtest_trajectory import transitions_for

    assert transitions_for(2022, "loto") == ((2023, 2024), (2024, 2025))
    assert transitions_for(2023, "loto") == ((2022, 2023), (2024, 2025))
    assert transitions_for(2024, "loto") == ((2022, 2023), (2023, 2024))

    assert transitions_for(2022, "causal") == ()
    assert transitions_for(2023, "causal") == ((2022, 2023),)
    assert transitions_for(2024, "causal") == ((2022, 2023), (2023, 2024))

    # The sensitivity check is only meaningful where the two differ.
    assert transitions_for(2024, "loto") == transitions_for(2024, "causal")
    assert transitions_for(2023, "loto") != transitions_for(2023, "causal")


def test_loto_never_includes_the_transition_being_predicted() -> None:
    """That is the ONE thing leave-one-transition-out guarantees. It does not make the
    fit causal -- for base 2022 both survivors are LATER than the transition predicted,
    which is why the writeup lists it as a third advantage keeper-value keeps."""
    from backtest_trajectory import transitions_for

    for base in (2022, 2023, 2024):
        assert (base, base + 1) not in transitions_for(base, "loto")


def test_future_transition_counts_are_2_1_0() -> None:
    """Printed per base year in the report. If these ever change, the leakage
    disclosure in the PR body is wrong."""
    from backtest_trajectory import transitions_for

    counts = {
        base: sum(1 for _, end in transitions_for(base, "loto") if end > base + 1)
        for base in (2022, 2023, 2024)
    }
    assert counts == {2022: 2, 2023: 1, 2024: 0}


def test_transitions_for_rejects_an_unknown_mode() -> None:
    from backtest_trajectory import transitions_for

    with pytest.raises(ValueError, match="loto"):
        transitions_for(2023, "whatever")


def test_volume_forecast_censors_the_training_panel_to_the_base_year(monkeypatch) -> None:
    """A curve fit on seasons after Y has seen the future it is asked to predict.

    Regression guard: the censor itself landed with the base-year parameterization,
    since threading base_year without it would have been a half-change.
    """
    import keeper_forecast

    captured: dict[str, pd.DataFrame] = {}

    def fake_lag_panel(panel, kind, **kwargs):
        captured["panel"] = panel
        return pd.DataFrame(
            {feature: [1.0, 2.0] for feature in keeper_forecast.FEATURES["hitter"]}
            | {"target": [600.0, 610.0]}
        )

    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: Path("fake.csv"))
    monkeypatch.setattr(
        keeper_forecast.pd, "read_csv", lambda *_a, **_k: _pt_panel(list(range(2018, 2027)))
    )
    monkeypatch.setattr(keeper_forecast, "lag_panel", fake_lag_panel)
    monkeypatch.setattr(keeper_forecast, "fit_curve", lambda *_a, **_k: _StubCurve())

    observed = pd.Series([600.0, 600.0], index=pd.Index([1, 2], name="mlbam_id"))
    keeper_forecast.volume_forecast("hitter", 2022, 2023, observed)

    assert "panel" in captured, "volume_forecast never reached lag_panel"
    assert int(captured["panel"]["season"].max()) <= 2022


def test_fallback_report_counts_per_player_misses() -> None:
    import keeper_forecast

    report = keeper_forecast.FallbackReport(whole_pool=False, per_player=30, total=100)
    assert report.share == pytest.approx(0.30)
    assert report.exceeds_headline_threshold is True


def test_fallback_report_tolerates_a_share_at_the_threshold() -> None:
    import keeper_forecast

    report = keeper_forecast.FallbackReport(whole_pool=False, per_player=25, total=100)
    assert report.exceeds_headline_threshold is False


def test_fallback_report_flags_a_whole_pool_miss_regardless_of_share() -> None:
    """No panel at all fails the base year outright: there is no curve to measure, so
    the number would be about the gap model rather than about keeper-value."""
    import keeper_forecast

    report = keeper_forecast.FallbackReport(whole_pool=True, per_player=0, total=100)
    assert report.share == 0.0
    assert report.exceeds_headline_threshold is True


def test_fallback_report_handles_an_empty_pool_without_dividing_by_zero() -> None:
    import keeper_forecast

    report = keeper_forecast.FallbackReport(whole_pool=False, per_player=0, total=0)
    assert report.share == 0.0
    assert report.exceeds_headline_threshold is False


def test_volume_forecast_reports_a_whole_pool_fallback(monkeypatch) -> None:
    """The missing-panel path returns None; the caller has to be able to tell that
    apart from a curve that simply scored nobody."""
    import keeper_forecast

    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: None)

    observed = pd.Series([600.0], index=pd.Index([1], name="mlbam_id"))
    projected, whole_pool = keeper_forecast.volume_forecast("hitter", 2022, 2023, observed)

    assert projected is None
    assert whole_pool is True
