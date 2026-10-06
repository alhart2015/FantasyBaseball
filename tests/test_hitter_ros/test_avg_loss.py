"""#433: AVG as hits out of at-bats -- a binomial loss on the log-odds of a hit."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.features import TARGETS  # noqa: E402
from fantasy_baseball.hitter_ros.net import (  # noqa: E402
    CountLoss,
    NetConfig,
    binomial_deviance,
    count_loss,
    device,
    predict,
    row_losses,
    train,
)
from scripts import train_hitter_ros  # noqa: E402

AVG = TARGETS.index("avg")


def _logit(p):
    return float(np.log(p / (1 - p)))


def test_binomial_deviance_is_zero_when_right_and_matches_the_formula():
    y = torch.tensor([0.25, 0.0, 1.0, 0.3])
    w = torch.tensor([400.0, 10.0, 3.0, 100.0])
    right = torch.tensor([_logit(0.25), -30.0, 30.0, _logit(0.3)])
    assert torch.allclose(binomial_deviance(right, y, w), torch.zeros(4), atol=1e-4)
    # -[H log p + (AB - H) log(1 - p)] minus the same at p = y, with H = 30 of AB = 100.
    p = 0.2
    loss = binomial_deviance(torch.tensor([_logit(p)]), torch.tensor([0.3]), torch.tensor([100.0]))
    nll = -(30 * np.log(p) + 70 * np.log(1 - p))
    best = -(30 * np.log(0.3) + 70 * np.log(0.7))
    assert loss.item() == pytest.approx(nll - best, rel=1e-4)
    assert (binomial_deviance(torch.tensor([0.0, 2.0]), y[:2], w[:2]) > 0).all()


def test_binomial_gradient_pushes_toward_the_actual_rate():
    y, w = torch.tensor([0.3]), torch.tensor([100.0])
    for start, sign in ((_logit(0.2), -1.0), (_logit(0.4), 1.0)):
        f = torch.tensor([start], requires_grad=True)
        binomial_deviance(f, y, w).sum().backward()
        # d/df = w * (sigmoid(f) - y): too low -> negative gradient (raise f), too high -> up.
        assert torch.sign(f.grad).item() == sign
        assert f.grad.item() == pytest.approx(100 * (torch.sigmoid(f).item() - 0.3), rel=1e-4)


def test_count_loss_marks_and_scales_avg_as_binomial():
    with pytest.raises(ValueError):
        NetConfig(avg_loss="beta")
    rng = np.random.default_rng(0)
    n = 4000
    ab = rng.integers(20, 500, size=n).astype(float)
    y = np.ones((n, len(TARGETS)))
    w = np.tile(ab[:, None], (1, len(TARGETS)))
    y[:, AVG] = rng.binomial(ab.astype(int), 0.25) / ab
    spec = count_loss(NetConfig(avg_loss="binomial"), y, w, device())
    assert spec.binomial.tolist() == [t == "avg" for t in TARGETS]
    assert not spec.mask[AVG]  # binomial, not Poisson
    # Predicting the average costs 1 per unit weight, like a standardized squared error.
    dev = spec.mask.device
    mean = np.average(y[:, AVG], weights=w[:, AVG])
    f = torch.full((n, len(TARGETS)), _logit(mean), device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    wt = torch.as_tensor(w, dtype=torch.float32, device=dev)
    losses = row_losses(f, yt, wt, spec)
    assert float(losses[:, AVG].sum().cpu()) / w[:, AVG].sum() == pytest.approx(1.0, rel=1e-3)
    # avg_loss alone (count_loss none) still gives a spec; the default has no binomial.
    alone = count_loss(NetConfig(count_loss="none", avg_loss="binomial"), y, w, device())
    assert alone.binomial[AVG] and not alone.mask.any()
    assert count_loss(NetConfig(), y, w, device()).binomial is None


def test_row_losses_mixes_all_three_losses():
    y = torch.tensor([[0.02, 0.5, 0.3]])
    w = torch.tensor([[100.0, 10.0, 100.0]])
    spec = CountLoss(
        mask=torch.tensor([True, False, False]),
        scale=torch.ones(3),
        binomial=torch.tensor([False, False, True]),
    )
    pred = torch.tensor([[np.log(0.04), 0.7, _logit(0.2)]], dtype=torch.float32)
    loss = row_losses(pred, y, w, spec)[0]
    assert loss[1].item() == pytest.approx(10 * 0.2**2, rel=1e-4)
    assert loss[2].item() == pytest.approx(
        binomial_deviance(pred[0, 2:], y[0, 2:], w[0, 2:]).item(), rel=1e-6
    )


def test_a_binomial_target_learns_the_rate_relative_to_an_offset():
    """The net learns the log-odds relative to a per-row league log-odds (the offset);
    adding the offset back and taking the sigmoid recovers each hitter's rate."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 2)).astype(np.float32)
    league = rng.choice([0.24, 0.26], size=n)  # two "seasons" with different leagues
    true = 1 / (1 + np.exp(-(np.log(league / (1 - league)) + 0.3 * x[:, 0])))
    ab = rng.integers(100, 600, size=n)
    y = np.zeros((n, len(TARGETS)), np.float32)
    y[:, AVG] = rng.binomial(ab, true) / ab
    w = np.zeros_like(y)
    w[:, AVG] = ab
    offset = np.zeros_like(y)
    offset[:, AVG] = np.log(league / (1 - league))
    val = rng.random(n) < 0.2
    config = NetConfig(
        hidden=[16],
        dropout=0.0,
        lr=1e-2,
        batch_size=256,
        max_epochs=80,
        patience=10,
        weighting="pa",
        count_loss="none",
        avg_loss="binomial",
    )
    result = train(x, y, w, val, config, offset=offset)
    z = predict(result.model, x)[:, AVG]
    pred = train_hitter_ros.expit(z + offset[:, AVG])
    assert np.abs(pred[val] - true[val]).mean() < 0.005
    # The net's own output is relative: about 0.3 * x, with no league level in it.
    assert np.corrcoef(z[val], 0.3 * x[val, 0])[0, 1] > 0.98
    with pytest.raises(ValueError):
        train(x, y, w, val, config, offset=offset[:, :2])


def test_logit_and_expit_round_trip():
    rate = pd.Series([0.001, 0.25, 0.5, 0.999, np.nan])
    back = train_hitter_ros.expit(train_hitter_ros.logit(rate))
    pd.testing.assert_series_equal(back, rate, rtol=1e-12)
    assert train_hitter_ros.expit(np.array([0.0]))[0] == 0.5
