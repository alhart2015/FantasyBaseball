"""#424: scores that ignore the league's level, the relative target and the rank loss."""

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.evaluate import (
    level_free_error,
    mae_table,
    order_table,
    pairwise_accuracy,
    scored_players,
    spearman,
)
from fantasy_baseball.hitter_ros.features import TARGETS


def test_a_pure_league_level_miss_is_free_only_in_the_level_free_score():
    actual = np.array([0.02, 0.04, 0.06])
    pa = np.array([600.0, 300.0, 100.0])
    np.testing.assert_allclose(level_free_error(actual * 0.9, actual, pa), 0.0, atol=1e-12)
    # A wrong order still costs.
    assert level_free_error(actual[::-1], actual, pa).sum() > 0


def test_level_free_rescaling_is_pa_weighted():
    actual = np.array([0.10, 0.20])
    projected = np.array([0.10, 0.10])
    # PA-weighted means: actual 0.18 with weights 1:4, projected 0.10 -> projected x 1.8.
    err = level_free_error(projected, actual, np.array([100.0, 400.0]))
    np.testing.assert_allclose(err, [0.08, 0.02])


def test_pairwise_accuracy():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert pairwise_accuracy(actual * 10, actual, weighted=False) == 1.0
    assert pairwise_accuracy(-actual, actual, weighted=False) == 0.0
    assert pairwise_accuracy(np.zeros(4), actual, weighted=False) == 0.5  # ties: half
    # Only the 1-vs-2 pair is wrong: 5 of 6 pairs right; weighted, it is the smallest gap.
    swapped = np.array([2.0, 1.0, 3.0, 4.0])
    assert pairwise_accuracy(swapped, actual, weighted=False) == pytest.approx(5 / 6)
    gaps = [1, 2, 3, 1, 2, 1]
    assert pairwise_accuracy(swapped, actual, weighted=True) == pytest.approx(
        (sum(gaps) - 1) / sum(gaps)
    )


def test_pairs_that_tied_in_reality_are_skipped():
    actual = np.array([0.0, 0.0, 1.0])
    # The 0-vs-0 pair has no right order, whatever the projection says.
    assert pairwise_accuracy(np.array([5.0, 0.0, 9.0]), actual, weighted=False) == 1.0
    assert np.isnan(pairwise_accuracy(np.array([1.0, 2.0]), np.zeros(2), weighted=False))


def test_spearman_is_nan_for_a_constant_projection():
    actual = np.array([1.0, 3.0, 2.0])
    assert spearman(np.array([10.0, 30.0, 20.0]), actual) == pytest.approx(1.0)
    assert np.isnan(spearman(np.ones(3), actual))


def _season_scored(season, actual_hr, projected):
    actual = pd.DataFrame({s: 0.1 for s in TARGETS}, index=range(len(actual_hr)))
    actual["hr"] = actual_hr
    actual["pa"] = 500
    proj = pd.DataFrame({s: 0.1 for s in TARGETS}, index=actual.index)
    proj["hr"] = projected
    return scored_players({"x": proj}, actual, min_pa=1).assign(season=season)


def test_scores_compare_players_only_within_a_season():
    # Each season's projection is the actuals times one league-wide factor (2x in 2024,
    # half in 2025): ordered perfectly within each season, while pairs across seasons
    # would be wrong, and raw MAE pays the whole league miss.
    scored = pd.concat(
        [
            _season_scored(2024, [0.01, 0.02, 0.03], [0.02, 0.04, 0.06]),
            _season_scored(2025, [0.05, 0.06, 0.07], [0.025, 0.03, 0.035]),
        ],
        ignore_index=True,
    )
    assert order_table(scored, "pairwise").loc["x", "hr"] == pytest.approx(100.0)
    assert order_table(scored, "spearman").loc["x", "hr"] == pytest.approx(1.0)
    assert mae_table(scored).loc["x", "hr"] > 10
    assert mae_table(scored, "lf_err").loc["x", "hr"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        order_table(scored, "kendall")


def test_league_forecast_lines_report_forecast_minus_actual():
    from fantasy_baseball.hitter_ros.backtest import league_forecast_lines

    rows = []
    for player, ros_hr in ((1, 30), (2, 10)):
        row = {"player_id": player, "season": 2025, "week": 0, "lg_p3_pa": 1000.0}
        for c, v in {"pa": 1000, "ab": 900, "h": 225, "r": 120, "hr": 30}.items():
            row[f"lg_p3_{c}"] = float(v)
            row[f"lg_std_{c}"] = 0.0
        for c in ("rbi", "sb"):
            row[f"lg_p3_{c}"], row[f"lg_std_{c}"] = 100.0, 0.0
        row.update(ros_pa=500, ros_ab=450, ros_h=110, ros_r=60, ros_hr=ros_hr, ros_rbi=50)
        row["ros_sb"] = 5
        rows.append(row)
    md = "\n".join(league_forecast_lines(pd.DataFrame(rows), [2025]))
    # Forecast HR/PA 30/1000; actual (30 + 10)/1000 -> -0.01 x 600 = -6.00.
    assert "| 2025 | " in md and "-6.00" in md


def test_net_config_checks_the_new_options():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig

    with pytest.raises(ValueError):
        NetConfig(loss="huber")
    with pytest.raises(ValueError):
        NetConfig(relative_target="yes")
    with pytest.raises(ValueError):
        NetConfig(loss="rank", micro_batch=64)


def test_rank_terms_count_only_ordered_pairs_in_the_same_group():
    torch = pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import rank_terms

    y = torch.tensor([[1.0], [2.0], [3.0], [3.0]])
    w = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
    groups = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([[0.0], [0.0], [5.0], [0.0]])
    num, den = rank_terms(pred, y, w, groups)
    # One pair counts: rows 0 and 1 (group 0). Rows 2-3 tie and row 3 has weight 0.
    assert den.item() == pytest.approx(0.5)  # 1 * 1 / (1 + 1)
    assert num.item() == pytest.approx(0.5 * np.log(2))  # equal predictions: -log(1/2)


def test_rank_loss_learns_the_order_and_calibration_restores_the_scale():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.net import NetConfig, linear_calibration, predict, train

    rng = np.random.default_rng(0)
    n = 1200
    x = rng.normal(size=(n, 3)).astype(np.float32)
    y = (2 * x[:, [0]] - x[:, [1]]).astype(np.float32)
    w = np.ones_like(y)
    groups = rng.integers(0, 6, size=n)
    val = rng.random(n) < 0.2
    config = NetConfig(
        hidden=[16], dropout=0.0, lr=1e-2, batch_size=256, max_epochs=40, patience=10, loss="rank"
    )
    with pytest.raises(ValueError):
        train(x, y, w, val, config)  # no groups
    result = train(x, y, w, val, config, groups=groups)
    assert result.val_loss[result.best_epoch] < 0.5 * result.val_loss[0]
    z = predict(result.model, x)
    assert spearman(z[val, 0], y[val, 0]) > 0.9
    slope, intercept = linear_calibration(z[~val], y[~val], w[~val])
    assert slope[0] > 0
    calibrated = z * slope + intercept
    assert np.average(calibrated[val]) == pytest.approx(y[val].mean(), abs=0.3)


def test_linear_calibration_recovers_a_line_and_ignores_weight_zero_rows():
    from fantasy_baseball.hitter_ros.net import linear_calibration

    z = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([[1.0], [3.0], [5.0], [np.nan]])
    w = np.array([[1.0], [1.0], [1.0], [0.0]])
    slope, intercept = linear_calibration(z, y, w)
    assert slope[0] == pytest.approx(2.0) and intercept[0] == pytest.approx(1.0)
