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
from scipy.special import logit
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
# avg_loss (#433): "mse" = squared error on the standardized AVG; "binomial" = each AB is
# a trial, the output is the log-odds of a hit.
AVG_LOSSES = ("mse", "binomial")
# avg_pieces (#433): "none"; "extra" = also predict AVG's pieces (features.PIECES), AVG
# still predicted directly; "derived" = predict the pieces and build AVG from them.
AVG_PIECES = ("none", "extra", "derived")
# Each piece's loss weight: the three together count like one stat.
PIECE_LOSS_WEIGHT = 1 / 3


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
    count_loss: str = "counts"
    # AVG's loss (#433). "binomial": hits out of at-bats, a binomial loss on the log-odds
    # (logit) of AVG, every horizon. With a relative target the output is the log-odds
    # relative to the league: the league's logit is added to it (an offset), not divided.
    avg_loss: str = "mse"
    # AVG as its pieces (#433): K/AB, HR/AB and BABIP (features.PIECES), each with a
    # binomial loss (K and HR out of AB, hits out of balls in play), every horizon, each
    # weighted PIECE_LOSS_WEIGHT. "extra": extra outputs that only shape the shared
    # body; AVG is still its own output. "derived": no AVG output; AVG is built from the
    # predicted pieces (features.avg_from_pieces). Default from #433: "extra". Over 6
    # seeds the 2026 mid-season AVG gap to FanGraphs went from -2.25 to -1.57 (vets
    # -2.14 -> -1.25) and preseason AVG held (-0.04 -> +0.11); mid-season RBI slipped
    # (+1.03 -> +0.78). "derived" was worse (preseason AVG -0.54). Neither makes the net
    # use this season's actual AVG (the #433 blind spot).
    avg_pieces: str = "extra"
    # Short horizons (#419): also predict the next 25 / 100 / 250 PA, each with its own
    # 5 outputs on the shared body (so one more "head" per horizon). horizon_weights:
    # how much each horizon's loss counts, for next 25 / 100 / 250 PA and rest of season.
    # recent_inputs: add the last 7 / 14 days as input windows. head_layers: a hidden
    # layer of this width inside each horizon's head (0 = a plain linear head).
    horizons: bool = True
    horizon_weights: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    recent_inputs: bool = False
    head_layers: int = 0
    # Minor-league inputs (#435): a graded minor-league line per window, from
    # data/hitter_ros/milb_<name>.parquet (hitter_ros.milb_features). "none": none.
    # Default from #435: only for players under 300 MLB PA (vets' rehab stints made
    # noise), small samples shrunk 100 PA. Over 3 seeds it closed most of the rookie gap
    # to FanGraphs (preseason HR -7.0 -> 0.0, SB -8.9 -> -0.4, AVG -12.5 -> -4.5) and left
    # vets within seed noise. Build: build_hitter_ros_milb.py --name rookies-s100
    # --vets-blank-from 300 --shrink-pa 100.
    milb: str = "rookies-s100"
    # Park inputs (#433): park factors of his team's home park and of the parks he hit in
    # (hitter_ros.parks), from data/hitter_ros/parks_<name>.parquet. "none": none.
    # Default from #433: over 3 seeds the preseason AVG gap to FanGraphs went from -0.55
    # to -0.01 and Coors hitters' AVG bias from -8.7 to -3.2 points, other stats within
    # seed noise. Build: build_hitter_ros_parks.py --name p3.
    parks: str = "p3"

    def __post_init__(self) -> None:
        if self.heads not in (1, 2):
            raise ValueError(f"heads must be 1 or 2, got {self.heads}")
        if self.heads > 1 and self.seq != "none":
            raise ValueError("multiple heads are built on the plain MLP (seq none) only")
        if self.split and self.heads > 1:
            raise ValueError("split trains two separate models; it can't also have heads")
        if self.micro_batch < 0:
            raise ValueError(f"micro_batch must be >= 0 (0 = whole batch), got {self.micro_batch}")
        if len(self.horizon_weights) != 4 or not all(
            np.isfinite(w) and w >= 0 for w in self.horizon_weights
        ):
            raise ValueError("horizon_weights: 4 non-negative weights (25, 100, 250 PA, ROS)")
        # Without horizons only the ROS weight is used; a zero there zeroes the whole loss.
        trained = self.horizon_weights if self.horizons else self.horizon_weights[3:]
        if not any(w > 0 for w in trained):
            raise ValueError("horizon_weights: every trained output would have weight 0")
        if self.head_layers < 0:
            raise ValueError(f"head_layers must be >= 0, got {self.head_layers}")
        if self.head_layers and (self.heads > 1 or self.seq != "none"):
            raise ValueError("head_layers is built on the plain MLP (heads 1, seq none)")
        if self.count_loss not in COUNT_LOSS_TARGETS:
            raise ValueError(f"unknown count_loss {self.count_loss!r}")
        if self.avg_loss not in AVG_LOSSES:
            raise ValueError(f"unknown avg_loss {self.avg_loss!r}")
        if self.avg_pieces not in AVG_PIECES:
            raise ValueError(f"unknown avg_pieces {self.avg_pieces!r}")
        if self.avg_pieces == "derived" and self.avg_loss != "mse":
            raise ValueError("avg_pieces derived has no AVG output for avg_loss to apply to")
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


@dataclass
class CountLoss:
    """Which targets get the Poisson loss (``mask``) or the binomial loss (``binomial``),
    and a per-target ``scale`` that puts each such target's loss on the same footing as
    a standardized squared error.

    Squared error on a standardized target is ~1 when predicting the average. A Poisson
    deviance is not: on the real table it is ~0.04 for R and ~0.7 for SB. Without the
    scale, the per-target average would weight AVG ~27x more than R (PR #429 review).
    """

    mask: torch.Tensor  # bool, one per target
    # float, one per target: the Poisson or binomial normalization (1 for squared-error
    # targets) times the target's weight (e.g. its horizon's weight, #419).
    scale: torch.Tensor
    # bool, one per target: binomial (#433). None = no binomial target.
    binomial: torch.Tensor | None = None


def poisson_deviance(f: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted Poisson deviance of log rate ``f`` against rate ``y`` with exposure ``w``:
    ``w * exp(f)`` is the expected count and ``w * y`` the actual one; 0 when they match
    and never negative."""
    f = f.clamp(*LOG_RATE_CLAMP)
    return w * (torch.exp(f) - y * f - y + torch.xlogy(y, y))


def binomial_deviance(f: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted binomial deviance of log-odds ``f`` against success rate ``y`` with ``w``
    trials (#433: hits out of at-bats): ``w * sigmoid(f)`` is the expected hits and
    ``w * y`` the actual ones; 0 when they match and never negative."""
    y = y.clamp(0.0, 1.0)
    log_p, log_q = nn.functional.logsigmoid(f), nn.functional.logsigmoid(-f)
    return w * (torch.xlogy(y, y) - y * log_p + torch.xlogy(1 - y, 1 - y) - (1 - y) * log_q)


def row_losses(
    pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor, count: CountLoss | None = None
) -> torch.Tensor:
    """Each row's weighted loss per target: ``w * (pred - y)^2``, or for ``count``'s
    Poisson targets the scaled Poisson deviance (``pred`` is then the log rate and ``y``
    the rate), and for its binomial targets the scaled binomial deviance (``pred`` is the
    log-odds and ``y`` the rate)."""
    loss = w * (pred - y) ** 2
    if count is None:
        return loss
    loss = torch.where(count.mask, poisson_deviance(pred, y, w), loss)
    if count.binomial is not None:
        loss = torch.where(count.binomial, binomial_deviance(pred, y, w), loss)
    return loss * count.scale


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


class HorizonHeadsMLP(nn.Module):
    """The MLP body, then one small head per group of 5 outputs (one per horizon,
    #419): [Linear -> GELU -> Dropout -> Linear] of width ``head_width`` each."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden: list[int],
        dropout: float,
        head_width: int,
        group: int = 5,
    ) -> None:
        super().__init__()
        if n_out % group:
            raise ValueError(f"{n_out} outputs don't split into heads of {group}")
        self.n_out = n_out
        layers, width = _hidden_layers(n_in, hidden, dropout)
        self.body = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, head_width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_width, group),
            )
            for _ in range(n_out // group)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        return torch.cat([h(z) for h in self.heads], dim=1)


def build_model(n_in: int, n_out: int, config: NetConfig) -> nn.Module:
    if config.head_layers:
        return HorizonHeadsMLP(n_in, n_out, config.hidden, config.dropout, config.head_layers)
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
    count: CountLoss | None = None,
    offset: torch.Tensor | None = None,
) -> float:
    """The training loss (``row_losses``) over all rows, computed in chunks. ``offset``
    (per row and target) is added to the net's output before the loss."""
    num = torch.zeros(y.shape[1], device=y.device)
    den = torch.zeros(y.shape[1], device=y.device)
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            sl = slice(start, start + chunk)
            pred = _forward(model, x[sl], None if rows is None else rows[sl], batcher, amp=amp)
            if offset is not None:
                pred = pred + offset[sl]
            num += row_losses(pred, y[sl], w[sl], count).sum(dim=0)
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
    count: CountLoss | None = None,
    offset: torch.Tensor | None = None,
) -> float:
    """Backpropagate the loss (``row_losses``; plain weighted MSE without ``count``)
    for batch ``idx``, in slices of ``micro_batch`` rows. ``offset`` (per row and
    target) is added to the net's output before the loss.

    Each slice's loss is normalized by the whole batch's weight per target, so the
    slices' gradients sum to exactly the full-batch gradient. Returns the batch loss.
    """
    w_total = w[idx].sum(dim=0).clamp(min=1e-9)
    step = micro_batch or len(idx)
    total = torch.zeros((), device=x.device)
    for start in range(0, len(idx), step):
        sub = idx[start : start + step]
        pred = _forward(model, x[sub], None if rows is None else rows[sub], batcher, amp=amp)
        if offset is not None:
            pred = pred + offset[sub]
        part = (row_losses(pred, y[sub], w[sub], count).sum(dim=0) / w_total).mean()
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


def count_loss(
    config: NetConfig,
    y: np.ndarray,
    w: np.ndarray,
    dev: torch.device,
    stats: list[str] | None = None,
    target_weights: np.ndarray | None = None,
) -> CountLoss | None:
    """The Poisson targets of ``config.count_loss``, the binomial ones (AVG with
    ``config.avg_loss`` binomial; AVG's pieces with ``config.avg_pieces``, when
    ``stats`` names them), and every target's scale; None when no target is any of
    those and every weight is 1. ``stats``: each column's stat (default ``features.TARGETS``, the
    rest-of-season columns). A Poisson or binomial target's scale is 1 / its deviance
    when predicting the weighted average rate on these rows (``y`` rates, ``w`` PA or
    AB), times its ``target_weights`` entry (default 1)."""
    from fantasy_baseball.hitter_ros.features import PIECES, TARGETS

    names = COUNT_LOSS_TARGETS[config.count_loss]
    binomial_names = (
        *(("avg",) if config.avg_loss == "binomial" else ()),
        *(PIECES if config.avg_pieces != "none" else ()),
    )
    unweighted = target_weights is None or np.all(np.asarray(target_weights) == 1)
    # Pieces are only ever named in ``stats`` (the default, TARGETS, has none).
    has_pieces = stats is not None and any(s in PIECES for s in stats)
    if not names and config.avg_loss != "binomial" and not has_pieces and unweighted:
        return None
    stats = list(TARGETS) if stats is None else stats
    if y.shape[1] != len(stats):
        raise ValueError(f"{len(stats)} target stats for {y.shape[1]} target columns")
    weights = np.ones(len(stats)) if target_weights is None else np.asarray(target_weights)
    mask = [t in names for t in stats]
    binomial = [t in binomial_names for t in stats]
    scale = np.ones(len(stats))
    for k in range(len(stats)):
        if not (mask[k] or binomial[k]):
            continue
        yk = torch.as_tensor(np.nan_to_num(y[:, k]), dtype=torch.float64)
        wk = torch.as_tensor(w[:, k], dtype=torch.float64)
        mean = float((wk * yk).sum() / wk.sum())
        if mask[k]:
            f, deviance = np.log(mean), poisson_deviance
        else:
            f, deviance = float(logit(mean)), binomial_deviance
        base = deviance(torch.full_like(yk, f), yk, wk).sum() / wk.sum()
        scale[k] = 1.0 / float(base) if float(base) > 0 else 1.0  # no spread: leave as is
    return CountLoss(
        mask=torch.tensor(mask, device=dev),
        scale=torch.tensor(scale * weights, dtype=torch.float32, device=dev),
        binomial=torch.tensor(binomial, device=dev) if any(binomial) else None,
    )


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
    target_stats: list[str] | None = None,
    target_weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
) -> TrainResult:
    """Fit the net. ``y`` is standardized targets, except the ``count_loss`` targets,
    which are rates (the net predicts their log), and the ``avg_loss`` binomial targets,
    which are rates (the net predicts their log-odds). NaN allowed where ``w`` is 0.
    ``offset`` (same shape as ``y``, NaN = 0) is added to the net's output before the
    loss, e.g. the league's log-odds of a hit (#433).

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
    ot = ov = None
    if offset is not None:
        if offset.shape != y.shape:
            raise ValueError(f"offset shape {offset.shape} != targets' {y.shape}")
        off = np.nan_to_num(offset, nan=0.0)
        ot = torch.tensor(off[~val_mask], dtype=torch.float32, device=dev)
        ov = torch.tensor(off[val_mask], dtype=torch.float32, device=dev)
    rt = rv = None
    if rows is not None:
        rt = torch.as_tensor(rows[~val_mask], dtype=torch.long, device=dev)
        rv = torch.as_tensor(rows[val_mask], dtype=torch.long, device=dev)
    model = build_model(x.shape[1], y.shape[1], config).to(dev)
    count = count_loss(config, y[~val_mask], w[~val_mask], dev, target_stats, target_weights)
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
                model, idx, xt, yt, wt, rt, batcher, config.micro_batch, config.amp, count, ot
            )
            opt.step()
            total += loss * len(idx)
        train_hist.append(total / n)

        model.eval()
        val = _eval_loss(
            model,
            xv,
            yv,
            wv,
            rv,
            batcher,
            config.micro_batch or EVAL_BATCH,
            config.amp,
            count,
            ov,
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
    """Width of the model's output: its last Linear layer, or every head's together."""
    if isinstance(model, HorizonHeadsMLP):  # the last Linear is only the last head's
        return model.n_out
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return int(linears[-1].out_features)
