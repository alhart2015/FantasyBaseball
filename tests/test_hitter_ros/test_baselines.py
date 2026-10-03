import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.baselines import (
    REGRESS_PA,
    baseline_predictions,
    league_rates,
    marcel,
)
from fantasy_baseball.hitter_ros.evaluate import mae_table, paired_bootstrap, scored_players, spread
from fantasy_baseball.hitter_ros.features import TARGETS

COUNTS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")


def _row(player, season, week, **counts):
    row = {"player_id": player, "season": season, "week": week, "season_complete": True}
    row["as_of"] = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(weeks=week)
    for window in ("std", "p1", "p3", "ros"):
        for c in COUNTS:
            row[f"{window}_{c}"] = counts.get(f"{window}_{c}", 0)
    return row


def _league_table():
    # One complete season, 2024: 1000 PA, 900 AB, 270 H, 120 R, 30 HR, 110 RBI, 10 SB.
    return pd.DataFrame(
        [
            _row(
                1,
                2024,
                0,
                ros_pa=1000,
                ros_ab=900,
                ros_h=270,
                ros_r=120,
                ros_hr=30,
                ros_rbi=110,
                ros_sb=10,
            )
        ]
    )


def test_league_rates_come_from_earlier_complete_seasons():
    lg = league_rates(_league_table(), 2025)
    assert lg["hr"] == pytest.approx(0.03) and lg["avg"] == pytest.approx(0.3)
    assert lg["ab_per_pa"] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        league_rates(_league_table(), 2024)  # nothing before 2024


def test_marcel_with_no_history_is_league_average():
    lg = league_rates(_league_table(), 2025)
    rookie = pd.DataFrame([_row(9, 2025, 0)])
    out = marcel(rookie, lg).iloc[0]
    for s in TARGETS:
        assert out[s] == pytest.approx(lg[s])


def test_marcel_weights_this_season_last_season_and_older():
    lg = league_rates(_league_table(), 2025)
    # 100 PA / 10 HR this season, 600 / 30 last season, 1200 / 36 over the two before.
    rows = pd.DataFrame(
        [
            _row(
                9,
                2025,
                5,
                std_pa=100,
                std_hr=10,
                p1_pa=600,
                p1_hr=30,
                p3_pa=1800,
                p3_hr=66,
            )
        ]
    )
    expected_hr = (6 * 10 + 5 * 30 + 3.5 * 36 + REGRESS_PA * 0.03) / (
        6 * 100 + 5 * 600 + 3.5 * 1200 + REGRESS_PA
    )
    assert marcel(rows, lg).iloc[0]["hr"] == pytest.approx(expected_hr)


def test_baseline_predictions_have_the_prediction_shape():
    table = pd.concat([_league_table(), pd.DataFrame([_row(9, 2025, 0), _row(9, 2025, 1)])])
    preds = baseline_predictions(table.reset_index(drop=True), 2025)
    assert set(preds) == {"league_avg", "marcel"}
    for p in preds.values():
        assert list(p.columns) == ["player_id", "season", "week", "as_of", *TARGETS]
        assert len(p) == 2


def _scored(errs_a, errs_b):
    n = len(errs_a)
    actual = pd.DataFrame({s: [0.1] * n for s in TARGETS}, index=range(n))
    actual["pa"] = 500
    a = pd.DataFrame({s: 0.1 + np.array(errs_a) / 600 for s in TARGETS}, index=range(n))
    b = pd.DataFrame({s: 0.1 + np.array(errs_b) / 600 for s in TARGETS}, index=range(n))
    return scored_players({"a": a, "b": b}, actual, min_pa=1)


def test_paired_bootstrap_sign_and_interval():
    rng = np.random.default_rng(1)
    scored = _scored(rng.uniform(0, 2, 400), rng.uniform(3, 5, 400))
    r = paired_bootstrap(scored, "a", "b", n_boot=500).loc["hr"]
    assert r["diff"] == pytest.approx(-3.0, abs=0.2)
    assert r["lo"] < r["diff"] < r["hi"] < 0  # a clearly better: interval below 0


def test_paired_bootstrap_cannot_separate_equal_systems():
    rng = np.random.default_rng(2)
    scored = _scored(rng.uniform(0, 4, 400), rng.uniform(0, 4, 400))
    r = paired_bootstrap(scored, "a", "b", n_boot=500).loc["r"]
    assert r["lo"] < 0 < r["hi"]


def test_spread_reports_projection_and_outcome_sd():
    scored = _scored([0.0, 6.0], [3.0, 3.0])
    sd = spread(scored)
    assert sd.loc["a", "hr"] == pytest.approx(np.std([0.0, 6.0], ddof=1) * 600 / 600)
    assert sd.loc["b", "hr"] == pytest.approx(0.0)
    assert "(actual)" in sd.index


def _stacked_units():
    """Two players scored in two seasons, with different outcomes each season."""
    parts = []
    for season, (a_hr, b_hr) in ((2024, (0.0, 0.1)), (2025, (0.2, 0.3))):
        actual = pd.DataFrame({s: [0.1, 0.1] for s in TARGETS}, index=[1, 2])
        actual["hr"] = [a_hr, b_hr]
        actual["pa"] = 500
        proj = pd.DataFrame({s: [0.1, 0.1] for s in TARGETS}, index=[1, 2])
        parts.append(
            scored_players({"x": proj, "y": proj + 0.01}, actual, min_pa=1).assign(season=season)
        )
    return pd.concat(parts, ignore_index=True)


def test_spread_actual_row_uses_every_season():
    sd = spread(_stacked_units())
    expected = np.std(np.array([0.0, 0.1, 0.2, 0.3]) * 600, ddof=1)
    assert sd.loc["(actual)", "hr"] == pytest.approx(expected)


def test_scored_units_are_player_seasons():
    scored = _stacked_units()
    assert mae_table(scored).loc["x", "n"] == 4
    b = paired_bootstrap(scored, "x", "y", n_boot=50)
    assert b.loc["hr", "diff"] == pytest.approx(
        scored[(scored.stat == "hr") & (scored.system == "x")].abs_err.mean()
        - scored[(scored.stat == "hr") & (scored.system == "y")].abs_err.mean()
    )


def test_write_scores_removes_a_stale_file(tmp_path):
    from fantasy_baseball.hitter_ros.backtest import write_scores

    frame = _stacked_units()
    write_scores(tmp_path, frame, frame)
    assert (tmp_path / "scored_snapshots.parquet").exists()
    write_scores(tmp_path, frame, None)
    assert not (tmp_path / "scored_snapshots.parquet").exists()


def test_mean_over_seasons_weights_each_season_once():
    from fantasy_baseball.hitter_ros.backtest import mean_over_seasons

    scored = _stacked_units()
    # Drop one player from 2025 so the seasons have different sizes.
    scored = scored[~((scored.season == 2025) & (scored.player_id == 2))]
    per_season = (
        scored[(scored.system == "x") & (scored.stat == "hr")].groupby("season").abs_err.mean()
    )
    assert mean_over_seasons(scored).loc["x", "hr"] == pytest.approx(per_season.mean())
