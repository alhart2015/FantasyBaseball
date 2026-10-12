import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.features import input_frame, target_frame, weight_frame
from fantasy_baseball.pitcher_ros.model import ID_COLS, fit_season, plain_config
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.pitcher_ros.table import build_table

from .test_table import RELIEVER, STARTER, _seasons, _write


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    _write(root, _seasons())
    return build_table(root)


def test_answers_and_their_weights(table):
    y, w = target_frame(table), weight_frame(table)
    row = table[(table.player_id == STARTER) & (table.season == 2025) & (table.week == 0)].index[0]
    # 21 starts: 5 K and 24 BF, 18 outs, 1 ER, 5 H, 2 BB each.
    assert y.loc[row, "k"] == pytest.approx(5 / 24)
    assert y.loc[row, "era"] == pytest.approx(27 * 1 / 18)
    assert y.loc[row, "whip"] == pytest.approx(3 * 7 / 18)
    assert w.loc[row, "k"] == 21 * 24 and w.loc[row, "era"] == pytest.approx(21 * 6)
    # A row with no games left has no answer, and no weight.
    empty = (table["ros_bf"] == 0).to_numpy()
    if empty.any():
        assert y[empty].isna().all().all() and (w[empty] == 0).all().all()


def test_inputs_never_read_the_answer(table):
    """Scramble every ros_ column: not one input may move."""
    scrambled = table.copy()
    rng = np.random.default_rng(0)
    for c in (c for c in table.columns if c.startswith("ros_")):
        scrambled[c] = rng.permutation(table[c].to_numpy())
    pd.testing.assert_frame_equal(input_frame(table), input_frame(scrambled))
    assert not (input_frame(table) == np.inf).any().any()


def test_fit_season_predicts_every_test_row_as_rates(table):
    x, y, w = input_frame(table), target_frame(table), weight_frame(table)
    config = plain_config(hidden=[8], max_epochs=3, patience=1, val_frac=0.34)
    preds, info = fit_season(table, x, y, w, 2025, config)
    assert list(preds.columns) == [*ID_COLS, *TARGETS]
    assert len(preds) == (table.season == 2025).sum()
    assert (preds[list(TARGETS)] >= 0).all().all() and preds[list(TARGETS)].notna().all().all()
    assert info["test_season"] == 2025 and info["train_rows"] > 0


def test_plain_config_switches_off_the_hitter_options():
    c = plain_config(seed=3)
    assert (c.seq, c.relative_target, c.count_loss, c.avg_pieces) == ("none",) * 4
    assert c.seed == 3 and not c.horizons


def test_our_rows_are_picked_as_known_on_the_date():
    preds = pd.DataFrame(
        {
            "player_id": [RELIEVER] * 3,
            "season": [2025] * 3,
            "week": [0, 1, 2],
            "as_of": pd.to_datetime(["2025-04-01", "2025-04-08", "2025-04-15"]),
            **{s: [0.1, 0.2, 0.3] for s in TARGETS},
        }
    )
    pre = backtest._ours_at({"ours": preds}, 2025, None)["ours"]
    assert pre.loc[RELIEVER, "k"] == 0.1
    mid = backtest._ours_at({"ours": preds}, 2025, pd.Timestamp("2025-04-14"))["ours"]
    assert mid.loc[RELIEVER, "k"] == 0.2  # week 1's row: week 2's is after the date
