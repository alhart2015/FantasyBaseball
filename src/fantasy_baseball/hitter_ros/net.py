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
import pandas as pd
import torch
from torch import nn

if TYPE_CHECKING:
    from fantasy_baseball.hitter_ros.sequence import SequenceBatcher

logger = logging.getLogger(__name__)


# Attention heads in the transformer encoder; seq_dim must divide by it.
TRANSFORMER_HEADS = 4

RELATIVE_TARGETS = ("none", "known", "answer")
# count_loss (#413): which targets get a Poisson loss instead of squared error.
COUNT_LOSS_TARGETS = {"none": (), "sb": ("sb",), "counts": ("r", "hr", "rbi", "sb")}
# Poisson outputs are log rates; clamp before exp so a wild output can't overflow.
LOG_RATE_CLAMP = (-15.0, 8.0)


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
    # Rows per forward pass inside a batch (0 = the whole batch). Gradients from the
    # slices are added up, so the update is the same as one full batch; this only saves
    # GPU memory (a transformer over 600 PAs x 2048 rows does not fit in 12 GB). Also
    # used as the evaluation chunk size.
    micro_batch: int = 0
    # Run the sequence encoder in bfloat16 on the GPU (about 2x faster for the
    # transformer). Outputs and the loss stay float32.
    amp: bool = False
    # "pa": each row's loss counts by its rest-of-season PA. "balanced": the same, then
    # rescaled so each fifth of the season carries equal total weight (late-season rows
    # otherwise get ~4% of it). "pre_mid": rescaled so week-0 rows and week 1+ rows carry
    # equal totals (#422: this, not separate heads, is what helps preseason; with PA
    # weights week-0 rows carry ~5% of the loss).
    weighting: str = "pre_mid"
    # Era handling (#421): inputs from features.ERA_MODES.
    era: str = "none"
    # Predict each player's rates relative to a league rate, then multiply back by the
    # league forecast (features.league_reference) for rates. "none": absolute rates.
    # "known": divide by that same forecast (#421). "answer": divide by the league's
    # actual rate over the answer window (features.league_answer_rates, #424), so the
    # net learns only "how much better than the league", never the league's level.
    relative_target: str = "answer"
    # Probe features (#417) from this pretraining run (e.g. "p003"), read from
    # data/hitter_ros/probes_<run>.parquet and added to the inputs. "none": no probes.
    probes: str = "p004"
    # Multiple heads (#422): 1 = one output layer for every row (the plain MLP). 2 = a
    # shared body with a preseason head (week 0) and a mid-season head (week 1+); the
    # head index rides in the last input column (see MultiHeadMLP). With the default
    # pre_mid weighting both heads carry the same total loss; --weighting pa gives the
    # unbalanced #422 run. split: instead, two separate models, one trained only on
    # week-0 rows and one only on week 1+ rows.
    heads: int = 1
    split: bool = False
    # Steal inputs (#413): steal opportunities and attempts per opportunity, success
    # rate, starts by position, and the team's green light (features._steal_inputs).
    steal_inputs: bool = True
    # Loss shape (#413). SB per PA is mostly zeros with a long tail, which squared error
    # on a standardized rate fits poorly. "sb" / "counts": those targets (SB, or R, HR,
    # RBI and SB) get a Poisson loss on the count, with the row's PA as exposure. Their
    # outputs are then the log of the (league-relative) rate, so 0 = league average.
    count_loss: str = "none"

    def __post_init__(self) -> None:
        if self.heads not in (1, 2):
            raise ValueError(f"heads must be 1 or 2, got {self.heads}")
        if self.heads > 1 and self.seq != "none":
            raise ValueError("multiple heads are built on the plain MLP (seq none) only")
        if self.split and self.heads > 1:
            raise ValueError("split trains two separate models; it can't also have heads")
        if self.micro_batch < 0:
            raise ValueError(f"micro_batch must be >= 0 (0 = whole batch), got {self.micro_batch}")
        if self.count_loss not in COUNT_LOSS_TARGETS:
            raise ValueError(f"unknown count_loss {self.count_loss!r}")
        if self.relative_target not in RELATIVE_TARGETS:
            raise ValueError(f"unknown relative_target {self.relative_target!r}")
        from fantasy_baseball.hitter_ros.features import ERA_MODES

        if self.era not in ERA_MODES:
            raise ValueError(f"unknown era {self.era!r}")
        if self.weighting not in ("pa", "balanced", "pre_mid"):
            raise ValueError(f"unknown weighting {self.weighting!r}")
        if self.seq not in ("none", "gru", "transformer"):
            raise ValueError(f"unknown seq {self.seq!r}")
        if self.seq == "transformer" and self.seq_dim % TRANSFORMER_HEADS:
            raise ValueError(
                f"seq_dim {self.seq_dim} must be divisible by the "
                f"{TRANSFORMER_HEADS} attention heads"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hidden_layers(n_in: int, hidden: list[int], dropout: float) -> tuple[list[nn.Module], int]:
    """[Linear -> GELU -> Dropout] per hidden width, and the width they end on."""
    layers: list[nn.Module] = []
    width = n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(dropout)]
        width = h
    return layers, width


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers, width = _hidden_layers(n_in, hidden, dropout)
        self.body = nn.Sequential(*layers, nn.Linear(width, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.body(x)
        return out


class MultiHeadMLP(nn.Module):
    """A shared body (the MLP's hidden layers) and one output layer ("head") per job.

    The **last input column is the head index** (0, 1, ...), not a feature: the body
    reads the other columns, and each row's output comes from its own head. Every head
    is computed for every row and the row's own is picked, so a row's loss trains only
    its head (and the shared body).
    """

    def __init__(
        self, n_in: int, n_out: int, hidden: list[int], dropout: float, n_heads: int
    ) -> None:
        super().__init__()
        layers, width = _hidden_layers(n_in, hidden, dropout)
        self.body = nn.Sequential(*layers)
        self.heads = nn.ModuleList(nn.Linear(width, n_out) for _ in range(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = x[:, -1].long()
        z = self.body(x[:, :-1])
        every = torch.stack([h(z) for h in self.heads], dim=1)  # [rows, heads, outputs]
        out: torch.Tensor = every[torch.arange(len(x), device=x.device), head]
        return out


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def row_losses(
    pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor, poisson: torch.Tensor | None = None
) -> torch.Tensor:
    """Each row's weighted loss per target: ``w * (pred - y)^2``, or for the ``poisson``
    targets the Poisson deviance of the count with exposure ``w``. There ``pred`` is the
    log rate and ``y`` the rate, so ``w * exp(pred)`` is the expected count and ``w * y``
    the actual one; the deviance is 0 when they match and never negative."""
    loss = w * (pred - y) ** 2
    if poisson is None or not bool(poisson.any()):
        return loss
    f = pred.clamp(*LOG_RATE_CLAMP)
    dev = w * (torch.exp(f) - y * f - y + torch.xlogy(y, y))
    return torch.where(poisson, dev, loss)


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
    if config.heads > 1:  # the last input column is the head index, not a feature
        return MultiHeadMLP(n_in - 1, n_out, config.hidden, config.dropout, config.heads)
    if config.seq == "none":
        return MLP(n_in, n_out, config.hidden, config.dropout)
    from fantasy_baseball.hitter_ros.sequence import HybridNet, make_encoder

    encoder = make_encoder(
        config.seq, config.seq_dim, config.seq_layers, config.seq_len, config.dropout
    )
    return HybridNet(n_in, n_out, config.hidden, config.dropout, encoder)


def _forward(
    model: nn.Module,
    x: torch.Tensor,
    rows: torch.Tensor | None,
    batcher: SequenceBatcher | None,
    shuffle_order: bool = False,
    amp: bool = False,
) -> torch.Tensor:
    if batcher is None:
        out: torch.Tensor = model(x)
        return out
    assert rows is not None, "a sequence model needs table row positions"
    seq, lengths = batcher.batch(rows, shuffle_order=shuffle_order)
    out = model(x, seq, lengths, amp=amp)
    return out


def _eval_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    rows: torch.Tensor | None,
    batcher: SequenceBatcher | None,
    chunk: int = EVAL_BATCH,
    amp: bool = False,
    poisson: torch.Tensor | None = None,
) -> float:
    """The training loss (``row_losses``) over all rows, computed in chunks."""
    num = torch.zeros(y.shape[1], device=y.device)
    den = torch.zeros(y.shape[1], device=y.device)
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            sl = slice(start, start + chunk)
            pred = _forward(model, x[sl], None if rows is None else rows[sl], batcher, amp=amp)
            num += row_losses(pred, y[sl], w[sl], poisson).sum(dim=0)
            den += w[sl].sum(dim=0)
    return float((num / den.clamp(min=1e-9)).mean().item())


def accumulate_batch(
    model: nn.Module,
    idx: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    rows: torch.Tensor | None,
    batcher: SequenceBatcher | None,
    micro_batch: int,
    amp: bool = False,
    poisson: torch.Tensor | None = None,
) -> float:
    """Backpropagate the loss (``row_losses``; plain weighted MSE without ``poisson``)
    for batch ``idx``, in slices of ``micro_batch`` rows.

    Each slice's loss is normalized by the whole batch's weight per target, so the
    slices' gradients sum to exactly the full-batch gradient. Returns the batch loss.
    """
    w_total = w[idx].sum(dim=0).clamp(min=1e-9)
    step = micro_batch or len(idx)
    total = torch.zeros((), device=x.device)
    for start in range(0, len(idx), step):
        sub = idx[start : start + step]
        pred = _forward(model, x[sub], None if rows is None else rows[sub], batcher, amp=amp)
        part = (row_losses(pred, y[sub], w[sub], poisson).sum(dim=0) / w_total).mean()
        part.backward()
        total += part.detach()
    return float(total.item())  # one GPU->CPU sync per batch, not per slice


def loss_weights(w: np.ndarray, config: NetConfig, season_time: np.ndarray | None) -> np.ndarray:
    """Each row's loss weight after ``config.weighting``. ``season_time`` (each row's
    ``frac_season_left``) is required by the "balanced" and "pre_mid" weightings."""
    from fantasy_baseball.hitter_ros.features import balance_by_group, balance_by_season_time

    if config.weighting in ("balanced", "pre_mid") and season_time is None:
        raise ValueError(f"{config.weighting} weighting needs each row's frac_season_left")
    if config.weighting == "balanced":
        w = balance_by_season_time(pd.DataFrame(w), pd.Series(season_time)).to_numpy(np.float32)
    if config.weighting == "pre_mid":
        # Week 0 (the season's first date) is the only row with the whole season left.
        mid = (np.asarray(season_time) < 1).astype(int)
        w = balance_by_group(pd.DataFrame(w), mid).to_numpy(np.float32)
    return w


def poisson_mask(config: NetConfig, n_targets: int, dev: torch.device) -> torch.Tensor | None:
    """Which output columns get the Poisson loss (targets in ``features.TARGETS`` order),
    or None when none do."""
    from fantasy_baseball.hitter_ros.features import TARGETS

    names = COUNT_LOSS_TARGETS[config.count_loss]
    if not names:
        return None
    if n_targets != len(TARGETS):
        raise ValueError(f"count_loss needs the {len(TARGETS)} ROS targets, got {n_targets}")
    return torch.tensor([t in names for t in TARGETS], device=dev)


def train(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    val_mask: np.ndarray,
    config: NetConfig,
    *,
    rows: np.ndarray | None = None,
    batcher: SequenceBatcher | None = None,
    season_time: np.ndarray | None = None,
) -> TrainResult:
    """Fit the net. ``y`` is standardized targets (NaN allowed where ``w`` is 0).

    With a sequence model, ``rows`` gives each row's position in the table the
    ``batcher`` was built on, so it can fetch that row's plate appearances.
    ``w`` is rescaled here by ``loss_weights`` (``season_time`` is each row's
    ``frac_season_left``). With ``config.heads > 1`` the last column of ``x`` is each
    row's head index.
    """
    if config.heads > 1 and not np.isin(x[:, -1], np.arange(config.heads)).all():
        # A missing head column would otherwise be read silently: .long() truncates the
        # last standardized feature into a head index, and -1 picks the last head.
        raise ValueError(
            f"with heads {config.heads} the last column of x must be each row's head "
            f"index (0..{config.heads - 1})"
        )
    w = loss_weights(w, config, season_time)
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
    poisson = poisson_mask(config, y.shape[1], dev)
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
            opt.zero_grad()
            loss = accumulate_batch(
                model, idx, xt, yt, wt, rt, batcher, config.micro_batch, config.amp, poisson
            )
            opt.step()
            total += loss * len(idx)
        train_hist.append(total / n)

        model.eval()
        val = _eval_loss(
            model, xv, yv, wv, rv, batcher, config.micro_batch or EVAL_BATCH, config.amp, poisson
        )
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
    chunk: int = EVAL_BATCH,
    amp: bool = False,
) -> np.ndarray:
    """Model outputs for ``x`` (and, for a sequence model, table positions ``rows``)."""
    model.eval()
    dev = device()
    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    rt = None if rows is None else torch.as_tensor(rows, dtype=torch.long, device=dev)
    parts = []
    with torch.no_grad():
        for start in range(0, len(xt), chunk):
            sl = slice(start, start + chunk)
            out = _forward(
                model, xt[sl], None if rt is None else rt[sl], batcher, shuffle_order, amp
            )
            parts.append(out.cpu().numpy())
    if not parts:
        return np.zeros((0, _n_outputs(model)), dtype=np.float32)
    return np.concatenate(parts)


def _n_outputs(model: nn.Module) -> int:
    """Width of the model's last Linear layer."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return int(linears[-1].out_features)
