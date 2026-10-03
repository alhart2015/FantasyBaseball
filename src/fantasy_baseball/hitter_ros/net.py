"""The neural net: a multilayer perceptron (MLP) from input rates to five ROS rates (#404).

Training in one picture:

    inputs (standardized rates) -> [Linear -> GELU -> Dropout] x N -> Linear -> 5 outputs

Each output is a standardized rest-of-season rate (R/PA, HR/PA, RBI/PA, SB/PA, AVG).
The loss is mean squared error weighted by rest-of-season PA (AB for AVG), so a
600-PA season counts 600 times as much as a 1-PA cameo -- the same as fitting counts.
Training stops when the loss on held-out validation players stops improving
("early stopping"), and the best epoch's weights are kept.

torch is an optional dependency (``pip install -e ".[nn]"`` plus the CUDA wheel, see
``scripts/train_hitter_ros.py``), so it is imported only in this module.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from fantasy_baseball.hitter_ros.sequence import SequenceBatcher

logger = logging.getLogger(__name__)


@dataclass
class NetConfig:
    """Everything that defines one training run. Saved next to every run's results."""

    # Defaults from #408: over 4 seeds this beat run 001's 256-256-128 / dropout 0.1 /
    # lr 1e-3 on every preseason stat (small margins) and halved seed-to-seed spread,
    # and it trains ~18 epochs instead of stopping at epoch 0-1.
    hidden: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.3
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 200
    patience: int = 10
    val_frac: float = 0.15
    seed: int = 0
    # Sequence encoder over each hitter's recent plate appearances (#414): "none" (the
    # plain MLP), "gru" or "transformer". Its summary joins the MLP's inputs.
    seq: str = "none"
    seq_len: int = 600
    seq_dim: int = 64
    seq_layers: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_out))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.body(x)
        return out


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_mse(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Per-target weighted MSE, averaged over targets so each counts the same."""
    per_target = (w * (pred - y) ** 2).sum(dim=0) / w.sum(dim=0).clamp(min=1e-9)
    return per_target.mean()


@dataclass
class TrainResult:
    model: nn.Module
    best_epoch: int
    train_loss: list[float]
    val_loss: list[float]


# Rows per forward pass when evaluating; keeps sequence batches within GPU memory.
EVAL_BATCH = 4096


def build_model(n_in: int, n_out: int, config: NetConfig) -> nn.Module:
    if config.seq == "none":
        return MLP(n_in, n_out, config.hidden, config.dropout)
    from fantasy_baseball.hitter_ros.sequence import HybridNet, make_encoder

    encoder = make_encoder(config.seq, config.seq_dim, config.seq_layers)
    return HybridNet(n_in, n_out, config.hidden, config.dropout, encoder)


def _forward(
    model: nn.Module,
    x: torch.Tensor,
    rows: torch.Tensor | None,
    batcher: SequenceBatcher | None,
    shuffle_order: bool = False,
) -> torch.Tensor:
    if batcher is None:
        out: torch.Tensor = model(x)
        return out
    assert rows is not None, "a sequence model needs table row positions"
    seq, lengths = batcher.batch(rows, shuffle_order=shuffle_order)
    out = model(x, seq, lengths)
    return out


def _eval_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    rows: torch.Tensor | None,
    batcher: SequenceBatcher | None,
) -> float:
    """weighted_mse over all rows, computed in chunks."""
    num = torch.zeros(y.shape[1], device=y.device)
    den = torch.zeros(y.shape[1], device=y.device)
    with torch.no_grad():
        for start in range(0, len(x), EVAL_BATCH):
            sl = slice(start, start + EVAL_BATCH)
            pred = _forward(model, x[sl], None if rows is None else rows[sl], batcher)
            num += (w[sl] * (pred - y[sl]) ** 2).sum(dim=0)
            den += w[sl].sum(dim=0)
    return float((num / den.clamp(min=1e-9)).mean().item())


def train(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    val_mask: np.ndarray,
    config: NetConfig,
    *,
    rows: np.ndarray | None = None,
    batcher: SequenceBatcher | None = None,
) -> TrainResult:
    """Fit the net. ``y`` is standardized targets (NaN allowed where ``w`` is 0).

    With a sequence model, ``rows`` gives each row's position in the table the
    ``batcher`` was built on, so it can fetch that row's plate appearances.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dev = device()
    y = np.nan_to_num(y, nan=0.0)

    def tensors(mask: np.ndarray) -> tuple[torch.Tensor, ...]:
        return tuple(torch.tensor(a[mask], dtype=torch.float32, device=dev) for a in (x, y, w))

    xt, yt, wt = tensors(~val_mask)
    xv, yv, wv = tensors(val_mask)
    rt = rv = None
    if rows is not None:
        rt = torch.as_tensor(rows[~val_mask], dtype=torch.long, device=dev)
        rv = torch.as_tensor(rows[val_mask], dtype=torch.long, device=dev)
    model = build_model(x.shape[1], y.shape[1], config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    gen = torch.Generator(device=dev).manual_seed(config.seed)

    best = (float("inf"), -1, copy.deepcopy(model.state_dict()))
    train_hist: list[float] = []
    val_hist: list[float] = []
    n = len(xt)
    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(n, device=dev, generator=gen)
        total = 0.0
        for start in range(0, n, config.batch_size):
            idx = order[start : start + config.batch_size]
            pred = _forward(model, xt[idx], None if rt is None else rt[idx], batcher)
            loss = weighted_mse(pred, yt[idx], wt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_hist.append(total / n)

        model.eval()
        val = _eval_loss(model, xv, yv, wv, rv, batcher)
        val_hist.append(val)
        if not np.isfinite(val):
            raise FloatingPointError(
                f"validation loss is {val} at epoch {epoch}: training diverged or an input "
                "is not finite; try a lower --lr"
            )
        if val < best[0]:
            best = (val, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= config.patience:
            break
        logger.info("epoch %d: train %.4f, val %.4f", epoch, train_hist[-1], val)

    model.load_state_dict(best[2])
    logger.info("best epoch %d of %d, val loss %.4f", best[1], len(val_hist), best[0])
    return TrainResult(model=model, best_epoch=best[1], train_loss=train_hist, val_loss=val_hist)


def predict(
    model: nn.Module,
    x: np.ndarray,
    *,
    rows: np.ndarray | None = None,
    batcher: SequenceBatcher | None = None,
    shuffle_order: bool = False,
) -> np.ndarray:
    """Model outputs for ``x`` (and, for a sequence model, table positions ``rows``)."""
    model.eval()
    dev = device()
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    rt = None if rows is None else torch.as_tensor(rows, dtype=torch.long, device=dev)
    parts = []
    with torch.no_grad():
        for start in range(0, len(xt), EVAL_BATCH):
            sl = slice(start, start + EVAL_BATCH)
            out = _forward(model, xt[sl], None if rt is None else rt[sl], batcher, shuffle_order)
            parts.append(out.cpu().numpy())
    return np.concatenate(parts)
