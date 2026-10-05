"""#413: a Poisson loss on counts for the targets that are mostly zeros."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.features import TARGETS  # noqa: E402
from fantasy_baseball.hitter_ros.net import (  # noqa: E402
    CountLoss,
    NetConfig,
    count_loss,
    device,
    predict,
    row_losses,
    train,
)


def test_poisson_deviance_is_zero_when_right_and_handles_zero_counts():
    y = torch.tensor([[0.0, 0.02, 0.5]])
    w = torch.tensor([[100.0, 100.0, 10.0]])
    poisson = CountLoss(torch.tensor([True, True, False]), torch.ones(3))
    right = torch.tensor([[np.log(1e-9), np.log(0.02), 0.5]], dtype=torch.float32)
    loss = row_losses(right, y, w, poisson)
    assert loss[0, 0].item() == pytest.approx(0.0, abs=1e-4)  # no steals, rate ~0
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
    y = np.ones((4, len(TARGETS)))
    w = np.ones_like(y)
    spec = count_loss(NetConfig(count_loss="sb"), y, w, device())
    assert spec.mask.tolist() == [t == "sb" for t in TARGETS]
    assert count_loss(NetConfig(count_loss="none"), y, w, device()) is None
    # The default (since #413) is Poisson on the four counts, squared error on AVG.
    default = count_loss(NetConfig(), y, w, device())
    assert default.mask.tolist() == [t != "avg" for t in TARGETS]


def test_poisson_targets_are_scaled_to_a_standardized_squared_error():
    """Predicting the average costs 1 per unit weight for every target, Poisson or not,
    so the per-target average weighs them equally (the PR #429 review's finding)."""
    rng = np.random.default_rng(0)
    n = 5000
    exposure = rng.integers(50, 600, size=n).astype(float)
    y = np.zeros((n, len(TARGETS)))
    w = np.tile(exposure[:, None], (1, len(TARGETS)))
    y[:, TARGETS.index("r")] = rng.poisson(0.12 * exposure) / exposure  # R: not rare
    y[:, TARGETS.index("sb")] = rng.poisson(0.01 * exposure) / exposure  # SB: rare
    spec = count_loss(NetConfig(count_loss="counts"), y, w, device())
    for name in ("r", "sb"):
        k = TARGETS.index(name)
        mean = np.average(y[:, k], weights=w[:, k])
        dev = spec.mask.device
        f = torch.full((n, len(TARGETS)), float(np.log(mean)), device=dev)
        losses = row_losses(
            f,
            torch.as_tensor(y, dtype=torch.float32, device=dev),
            torch.as_tensor(w, dtype=torch.float32, device=dev),
            spec,
        )
        assert float(losses[:, k].sum().cpu()) / w[:, k].sum() == pytest.approx(1.0, rel=1e-3)
    # Unscaled, R's deviance per unit weight is far smaller than SB's, so R scales up more.
    assert spec.scale[TARGETS.index("r")] > spec.scale[TARGETS.index("sb")]


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


def test_target_weights_scale_each_column():
    from fantasy_baseball.hitter_ros.features import horizon_columns, target_stat

    cols = horizon_columns((25,))
    y = np.ones((10, len(cols)))
    w = np.ones_like(y)
    weights = np.array([1.0] * 5 + [0.25] * 5)
    spec = count_loss(
        NetConfig(count_loss="none"), y, w, device(), [target_stat(c) for c in cols], weights
    )
    assert not spec.mask.any()
    np.testing.assert_allclose(spec.scale.cpu().numpy(), weights)
    with pytest.raises(ValueError):
        count_loss(NetConfig(), y, w, device())  # 10 columns but ROS stats only
