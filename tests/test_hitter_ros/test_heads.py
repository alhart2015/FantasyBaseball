"""#422: a shared body with a preseason head and a mid-season head."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.features import balance_by_group  # noqa: E402
from fantasy_baseball.hitter_ros.net import (  # noqa: E402
    MultiHeadMLP,
    NetConfig,
    loss_weights,
    predict,
    train,
)


def test_each_row_uses_its_own_head_and_the_head_column_is_not_a_feature():
    torch.manual_seed(0)
    model = MultiHeadMLP(3, 2, [8], 0.0, n_heads=2).eval()
    x = torch.randn(4, 3)
    head = torch.tensor([0.0, 1.0, 0.0, 1.0])[:, None]
    out = model(torch.cat([x, head], dim=1))
    z = model.body(x)
    torch.testing.assert_close(out[0], model.heads[0](z)[0])
    torch.testing.assert_close(out[1], model.heads[1](z)[1])


def test_a_row_trains_only_its_own_head():
    torch.manual_seed(0)
    model = MultiHeadMLP(3, 2, [8], 0.0, n_heads=2)
    x = torch.cat([torch.randn(5, 3), torch.zeros(5, 1)], dim=1)  # all preseason rows
    model(x).pow(2).sum().backward()
    assert model.heads[0].weight.grad.abs().sum() > 0
    assert model.heads[1].weight.grad is None or model.heads[1].weight.grad.abs().sum() == 0


def test_two_heads_learn_opposite_rules_one_head_cannot():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=(n, 2)).astype(np.float32)
    head = (rng.random(n) < 0.5).astype(np.float32)
    y = np.where(head[:, None] == 1, -x[:, [0]], x[:, [0]]).astype(np.float32)
    w = np.ones_like(y)
    val = rng.random(n) < 0.2
    # Synthetic rows have no season time and aren't the 5 ROS targets: pin plain PA
    # weighting and squared error (not the pre_mid / Poisson defaults).
    base = dict(
        hidden=[16],
        dropout=0.0,
        lr=1e-2,
        batch_size=256,
        max_epochs=60,
        patience=10,
        weighting="pa",
        count_loss="none",
    )
    two = train(np.column_stack([x, head]), y, w, val, NetConfig(**base, heads=2))
    one = train(x, y, w, val, NetConfig(**base))  # no head flag: can't tell rows apart
    assert two.val_loss[two.best_epoch] < 0.05
    assert one.val_loss[one.best_epoch] > 0.5
    pred = predict(two.model, np.column_stack([x, head]))
    assert np.corrcoef(pred[val, 0], y[val, 0])[0, 1] > 0.95


def test_balance_by_group_gives_each_group_the_same_total():
    w = pd.DataFrame({"a": [1.0, 1.0, 1.0, 7.0], "b": [2.0, 2.0, 0.0, 4.0]})
    out = balance_by_group(w, np.array([0, 0, 0, 1]))
    assert out.loc[:2, "a"].sum() == pytest.approx(out.loc[3, "a"])
    assert out["a"].sum() == pytest.approx(w["a"].sum())
    assert out.loc[0, "a"] == pytest.approx(out.loc[1, "a"])  # relative weights kept


def test_head_options_are_checked():
    with pytest.raises(ValueError):
        NetConfig(heads=3)
    with pytest.raises(ValueError):
        NetConfig(heads=2, split=True)
    with pytest.raises(ValueError):
        NetConfig(heads=2, seq="gru")


def test_pre_mid_weighting_balances_week_0_against_the_rest():
    """pre_mid groups rows by season time exactly as the head index does (week 0 vs
    1+), so --heads 2 under pre_mid is the balanced-heads run of #422."""
    from fantasy_baseball.hitter_ros.features import balance_by_group

    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.random((6, 2)))
    season_time = np.array([1.0, 1.0, 0.9, 0.5, 0.2, 0.01])
    by_time = balance_by_group(w, (season_time < 1).astype(int))
    by_head = balance_by_group(w, np.array([0, 0, 1, 1, 1, 1]))
    pd.testing.assert_frame_equal(by_time, by_head)
    weights = loss_weights(w.to_numpy(), NetConfig(weighting="pre_mid"), season_time)
    np.testing.assert_allclose(weights, by_head.to_numpy(), rtol=1e-6)
    with pytest.raises(ValueError):
        train(
            np.zeros((4, 2), np.float32),
            np.zeros((4, 1), np.float32),
            np.ones((4, 1), np.float32),
            np.array([False, False, False, True]),
            NetConfig(weighting="pre_mid", max_epochs=1),
        )


def test_a_missing_head_column_is_refused():
    x = np.random.default_rng(0).normal(size=(8, 3)).astype(np.float32)  # no head column
    with pytest.raises(ValueError, match="head index"):
        train(
            x,
            np.zeros((8, 1), np.float32),
            np.ones((8, 1), np.float32),
            np.arange(8) < 2,
            NetConfig(heads=2, weighting="pa", max_epochs=1),
        )
