import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.features import answers, input_frame
from fantasy_baseball.pitcher_ros.model import (
    APPLIED,
    ID_COLS,
    applied_settings,
    check_plain,
    check_week_zero,
    fit_season,
    plain_config,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.pitcher_ros.table import build_table

from .test_table import RELIEVER, STARTER, _seasons, _write


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    _write(root, _seasons())
    return build_table(root)


def _starter_week0(table):
    return table[(table.player_id == STARTER) & (table.season == 2025) & (table.week == 0)].index[0]


def test_answers_and_their_weights(table):
    y, w = answers(table)
    row = _starter_week0(table)
    # 21 starts: 5 K and 24 BF, 18 outs, 1 ER, 5 H, 2 BB each.
    assert y.loc[row, "k"] == pytest.approx(5 / 24)
    assert y.loc[row, "era"] == pytest.approx(27 * 1 / 18)
    assert y.loc[row, "whip"] == pytest.approx(3 * 7 / 18)
    assert w.loc[row, "k"] == 21 * 24 and w.loc[row, "era"] == pytest.approx(21 * 6)
    # The scorer charges (bf x err)^2 on K/W/SV: "bf2" weights each row by bf^2.
    _, w2 = answers(table, "bf2")
    assert w2.loc[row, "k"] == (21 * 24) ** 2 and w2.loc[row, "era"] == w.loc[row, "era"]
    with pytest.raises(ValueError, match="per_bf_weight"):
        answers(table, "pa")


def test_a_row_with_no_answer_has_no_weight(table):
    empty = table.copy()
    row = _starter_week0(table)
    for c in (c for c in table.columns if c.startswith("ros_")):
        empty.loc[row, c] = 0
    y, w = answers(empty, "bf2")
    assert y.loc[row].isna().all() and (w.loc[row] == 0).all()


def test_inputs_never_read_the_answer(table):
    """Scramble every ros_ column: not one input may move."""
    scrambled = table.copy()
    rng = np.random.default_rng(0)
    for c in (c for c in table.columns if c.startswith("ros_")):
        scrambled[c] = rng.permutation(table[c].to_numpy())
    pd.testing.assert_frame_equal(input_frame(table), input_frame(scrambled))
    assert not (input_frame(table) == np.inf).any().any()


def test_fit_season_predicts_every_test_row_as_rates(table):
    x, (y, w) = input_frame(table), answers(table)
    config = plain_config(hidden=[8], max_epochs=3, patience=1, val_frac=0.34)
    preds, info = fit_season(table, x, y, w, 2025, config)
    assert list(preds.columns) == [*ID_COLS, *TARGETS]
    assert len(preds) == (table.season == 2025).sum()
    assert (preds[list(TARGETS)] >= 0).all().all() and preds[list(TARGETS)].notna().all().all()
    assert info["test_season"] == 2025 and info["train_rows"] > 0


def test_only_settings_the_pitcher_net_uses_are_accepted_and_recorded():
    c = plain_config(seed=3, dropout=0.2)
    assert (c.seq, c.relative_target, c.count_loss, c.avg_pieces) == ("none",) * 4
    check_plain(c)
    assert set(applied_settings(c)) == set(APPLIED) and applied_settings(c)["seed"] == 3
    with pytest.raises(ValueError, match="split"):
        check_plain(plain_config(split=True))


def test_week_zero_must_be_the_whole_season_left(table):
    check_week_zero(table)
    bad = table.copy()
    bad.loc[_starter_week0(table), "frac_season_left"] = 0.9
    with pytest.raises(ValueError, match="week 0"):
        check_week_zero(bad)


def _preds(player, weeks, dates):
    return pd.DataFrame(
        {
            "player_id": [player] * len(weeks),
            "season": [2025] * len(weeks),
            "week": weeks,
            "as_of": pd.to_datetime(dates),
            **{s: [0.1 * (i + 1) for i in range(len(weeks))] for s in TARGETS},
        }
    )


def test_our_rows_are_picked_as_known_on_the_date():
    preds = _preds(RELIEVER, [0, 1, 2], ["2025-04-01", "2025-04-08", "2025-04-15"])
    pre = backtest._ours_at({"ours": preds}, 2025, None)["ours"]
    assert pre.loc[RELIEVER, "k"] == pytest.approx(0.1)
    mid = backtest._ours_at({"ours": preds}, 2025, pd.Timestamp("2025-04-14"))["ours"]
    assert mid.loc[RELIEVER, "k"] == pytest.approx(0.2)  # week 2's row is after the date
    twice = pd.concat([preds, preds])
    with pytest.raises(ValueError, match="more than one"):
        backtest._ours_at({"ours": twice}, 2025, None)


def test_ours_must_cover_every_scored_pitcher_and_not_borrow_a_name():
    actual = pd.DataFrame(
        {"bf": [200.0, 200.0], **{s: [0.2, 0.2] for s in TARGETS}},
        index=pd.Index([STARTER, RELIEVER], name="player_id"),
    )
    systems = {"steamer": actual[list(TARGETS)]}
    partial = {"ours": actual.loc[[STARTER], list(TARGETS)]}
    with pytest.raises(ValueError, match="no projection for 1 of the 2"):
        backtest._check_ours(partial, systems, actual, 100, "2025")
    with pytest.raises(ValueError, match="named like"):
        backtest._check_ours({"marcel": actual[list(TARGETS)]}, systems, actual, 100, "2025")
    backtest._check_ours({"ours": actual[list(TARGETS)]}, systems, actual, 100, "2025")
