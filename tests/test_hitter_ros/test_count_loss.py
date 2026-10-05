"""#413: a Poisson loss on counts for the targets that are mostly zeros."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.features import TARGETS  # noqa: E402
from fantasy_baseball.hitter_ros.net import (  # noqa: E402
    NetConfig,
    device,
    poisson_mask,
    predict,
    row_losses,
    train,
)


def test_poisson_deviance_is_zero_when_right_and_handles_zero_counts():
    y = torch.tensor([[0.0, 0.02, 0.5]])
    w = torch.tensor([[100.0, 100.0, 10.0]])
    poisson = torch.tensor([True, True, False])
    right = torch.tensor([[np.log(1e-9), np.log(0.02), 0.5]], dtype=torch.float32)
    loss = row_losses(right, y, w, poisson)
    assert loss[0, 1].item() == pytest.approx(0.0, abs=1e-4)
    assert loss[0, 2].item() == pytest.approx(0.0)  # squared error column, exact
    wrong = torch.tensor([[np.log(0.01), np.log(0.04), 0.7]], dtype=torch.float32)
    bad = row_losses(wrong, y, w, poisson)
    assert (bad > 0).all()
    assert bad[0, 0].item() == pytest.approx(100 * 0.01, rel=1e-4)  # no steals: w * rate
    assert bad[0, 2].item() == pytest.approx(10 * 0.2**2, rel=1e-4)
    # Without a mask it's plain weighted squared error.
    assert torch.allclose(row_losses(wrong, y, w), w * (wrong - y) ** 2)


def test_count_loss_options():
    with pytest.raises(ValueError):
        NetConfig(count_loss="gamma")
    mask = poisson_mask(NetConfig(count_loss="sb"), len(TARGETS), device())
    assert mask.tolist() == [t == "sb" for t in TARGETS]
    assert poisson_mask(NetConfig(), len(TARGETS), device()) is None


def test_a_poisson_target_learns_a_skewed_rate():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=(n, 2)).astype(np.float32)
    rate = np.exp(-3.0 + 1.0 * x[:, 0])  # skewed: most near 0, a long tail
    exposure = rng.integers(50, 600, size=n).astype(np.float32)
    counts = rng.poisson(rate * exposure)
    sb = counts / exposure
    k = TARGETS.index("sb")
    y = np.zeros((n, len(TARGETS)), np.float32)
    y[:, k] = sb
    w = np.zeros_like(y)
    w[:, k] = exposure
    val = rng.random(n) < 0.2
    config = NetConfig(
        hidden=[16],
        dropout=0.0,
        lr=1e-2,
        batch_size=256,
        max_epochs=80,
        patience=10,
        weighting="pa",
        count_loss="sb",
    )
    result = train(x, y, w, val, config)
    pred = np.exp(predict(result.model, x)[:, k])
    assert np.corrcoef(np.log(pred[val]), np.log(rate[val]))[0, 1] > 0.95
    assert np.median(pred[val] / rate[val]) == pytest.approx(1.0, abs=0.15)
