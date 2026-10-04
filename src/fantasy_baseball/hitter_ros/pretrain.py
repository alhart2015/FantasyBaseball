"""Self-supervised pretraining on pitch outcomes (#415).

The ROS model has ~6,500 labeled player-seasons; every pitch has its own outcome, so
there are ~8.15M labels here. A causal transformer (:class:`PitchEncoder`) reads a
hitter's pitches in order and, at each step, predicts the **next** pitch's outcome
(``pitch_tokens.OUTCOMES``) from his history so far plus that next pitch's context
(type, speed, movement, location, count, ...). To do that well it has to learn what a
hitter's history says about him -- which is what the ROS model wants from it later.

Walk-forward, per season: one model is pretrained per season S, on seasons before S
only (``scripts/pretrain_hitter_ros.py``). Anything later read off it about a row from
season S (e.g. #417's probe features) is then out of sample for that row -- training
rows included, not just test rows.

Read the **prediction head's outputs** off these models, not the raw hidden state. Each
model learns its own coordinate system, so hidden-state numbers from the 2019 and 2023
models don't mean the same thing, while a predicted probability does. (The first version
of PR #420 fed one model's raw hidden states to every row; the training rows' features
had then seen those rows' own futures. It was removed.)

Reference point for the pretraining loss: ``count_baseline_ce`` is the cross-entropy
of predicting each pitch's outcome from the ball-strike count alone. A model that can't
beat it has learned nothing about hitters or pitches.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from fantasy_baseball.hitter_ros.history import HitterTimeline, to_days
from fantasy_baseball.hitter_ros.pitch_tokens import (
    CONTEXT_FEATURES,
    OUTCOMES,
    TOKEN_FEATURES,
)

logger = logging.getLogger(__name__)

N_CONTEXT = len(CONTEXT_FEATURES)
N_TOKEN = len(TOKEN_FEATURES)
N_OUTCOMES = len(OUTCOMES)


@dataclass
class PretrainConfig:
    dim: int = 128
    layers: int = 4
    heads: int = 4
    window: int = 1024  # pitches per training window, and history length when reading
    batch_size: int = 32  # windows per optimizer step
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_epochs: int = 30
    patience: int = 3
    val_frac: float = 0.1  # share of hitters held out to measure the pretraining loss
    dropout: float = 0.1
    seed: int = 0
    amp: bool = True  # bfloat16 on the GPU (stored features and math)

    def __post_init__(self) -> None:
        if self.dim % self.heads:
            raise ValueError(f"dim {self.dim} must be divisible by heads {self.heads}")
        if not 0 < self.val_frac < 1:
            raise ValueError(f"val_frac must be between 0 and 1, got {self.val_frac}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PitchStore:
    """Every pitch token on one device, sorted by hitter then time, with per-hitter
    offsets so a window or an "everything before date X" slice is index arithmetic."""

    def __init__(
        self, tokens: pd.DataFrame, device: torch.device, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """``dtype`` for the stored features: bfloat16 halves GPU memory; use float32
        for a full-precision run (``--no-amp``)."""
        order = ["player_id", "game_date", "game_pk", "at_bat_number", "pitch_number"]
        tokens = tokens.sort_values(order, ignore_index=True)
        self.device = device
        self.days = to_days(tokens["game_date"])
        self.season = tokens["season"].to_numpy()
        self.timeline = HitterTimeline(tokens["player_id"].to_numpy(), self.days)
        feats = tokens[list(TOKEN_FEATURES)].to_numpy(np.float32)
        self.feats = torch.as_tensor(feats, device=device).to(dtype)
        self.outcome = torch.as_tensor(tokens["outcome"].to_numpy(), device=device)

    @property
    def players(self) -> np.ndarray:
        return self.timeline.players

    def windows(
        self, players: np.ndarray, before_season: int, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(start, n_pitches) of every window (stride length/2) over these hitters'
        pitches from seasons before ``before_season``. A short career gives one shorter
        window; the newest pitches are always covered."""
        starts: list[np.ndarray] = []
        cuts: list[np.ndarray] = []
        stride = max(1, length // 2)
        rank, known = self.timeline.rank(np.asarray(players))
        for i in rank[known]:
            a = int(self.timeline.first[i])
            stop = int(self.timeline.last[i])
            b = a + int(np.searchsorted(self.season[a:stop], before_season, "left"))
            if b - a < 2:
                continue
            s = np.arange(a, max(a + 1, b - length + 1), stride)
            if (b - a) > length and (b - length - a) % stride:
                s = np.append(s, b - length)  # cover the newest pitches too
            starts.append(s)
            cuts.append(np.full(len(s), b))
        if not starts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        start = np.concatenate(starts)
        return start, np.minimum(start + length, np.concatenate(cuts)) - start

    def gather(
        self, starts: torch.Tensor, lengths: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(tokens [B, length, N_TOKEN], outcomes [B, length]) for windows starting at
        ``starts`` with ``lengths`` real pitches each, right-padded with zeros / -1."""
        steps = torch.arange(length, device=self.device)
        valid = steps[None, :] < lengths[:, None]
        idx = torch.where(valid, starts[:, None] + steps[None, :], torch.zeros_like(steps)[None, :])
        x = self.feats[idx] * valid[..., None]
        y = torch.where(valid, self.outcome[idx], torch.full_like(idx, -1))
        return x, y

    def history(
        self, player_ids: np.ndarray, as_of_days: np.ndarray, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(start, n) of each row's last ``length`` pitches strictly before its as-of day."""
        first, end = self.timeline.before(player_ids, as_of_days)
        n = np.minimum(end - first, length)
        return end - n, n


class PitchEncoder(nn.Module):
    """Causal transformer over pitch tokens: position t sees pitches 0..t only."""

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__()
        d = config.dim
        self.inp = nn.Linear(N_TOKEN, d)
        self.position = nn.Embedding(config.window, d)
        layer = nn.TransformerEncoderLayer(
            d, config.heads, 4 * d, config.dropout, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.shape[1]
        h = self.inp(x) + self.position(torch.arange(steps, device=x.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(steps, device=x.device)
        out: torch.Tensor = self.norm(self.body(h, mask=mask, is_causal=True))
        return out


class PretrainModel(nn.Module):
    """Encoder + a head that predicts pitch t+1's outcome from the history through t
    and pitch t+1's context."""

    def __init__(self, config: PretrainConfig) -> None:
        super().__init__()
        d = config.dim
        self.encoder = PitchEncoder(config)
        self.context = nn.Linear(N_CONTEXT, d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, N_OUTCOMES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits [B, L-1, N_OUTCOMES]: entry t predicts the outcome of pitch t+1."""
        h = self.encoder(x)[:, :-1]
        nxt = self.context(x[:, 1:, :N_CONTEXT])
        out: torch.Tensor = self.head(torch.cat([h, nxt], dim=-1))
        return out


def next_pitch_loss(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(sum of cross-entropy, number of predicted pitches); padding (-1) is skipped."""
    target = y[:, 1:]
    flat = logits.reshape(-1, N_OUTCOMES).float()
    ce = F.cross_entropy(flat, target.reshape(-1), ignore_index=-1, reduction="sum")
    return ce, (target >= 0).sum()


def count_baseline_ce(tokens: pd.DataFrame) -> float:
    """Cross-entropy of predicting each outcome from the ball-strike count alone."""
    balls = (tokens["balls"] * 3).round().astype(int)
    strikes = (tokens["strikes"] * 2).round().astype(int)
    count = balls * 3 + strikes
    probs = pd.crosstab(count, tokens["outcome"], normalize="index")
    probs = probs.reindex(columns=range(N_OUTCOMES), fill_value=0.0)  # outcomes never seen
    p = probs.to_numpy()[np.searchsorted(probs.index, count), tokens["outcome"].to_numpy()]
    return float(-np.log(np.clip(p, 1e-9, None)).mean())


@dataclass
class PretrainResult:
    model: PretrainModel
    best_epoch: int
    train_ce: list[float]
    val_ce: list[float]
    baseline_ce: float


def _lr_at(step: int, total: int, config: PretrainConfig) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < config.warmup_steps:
        return config.lr * (step + 1) / config.warmup_steps
    frac = (step - config.warmup_steps) / max(1, total - config.warmup_steps)
    return config.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))


def pretrain(store: PitchStore, before_season: int, config: PretrainConfig) -> PretrainResult:
    """Pretrain on every pitch from seasons before ``before_season``."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    dev = store.device
    players = store.players
    val_players = rng.choice(players, size=int(len(players) * config.val_frac), replace=False)
    train_players = np.setdiff1d(players, val_players)
    L = config.window

    def window_set(ps: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        starts, lengths = store.windows(ps, before_season, L)
        return torch.as_tensor(starts, device=dev), torch.as_tensor(lengths, device=dev)

    tr_s, tr_n = window_set(train_players)
    va_s, va_n = window_set(val_players)
    if len(tr_s) == 0 or len(va_s) == 0:
        raise ValueError(f"no pitches before {before_season} to pretrain on")
    logger.info(
        "pretrain < %s: %d train windows, %d val windows", before_season, len(tr_s), len(va_s)
    )

    model = PretrainModel(config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(tr_s) / config.batch_size)
    total_steps = steps_per_epoch * config.max_epochs
    gen = torch.Generator(device=dev).manual_seed(config.seed)
    amp = config.amp and dev.type == "cuda"

    step = 0

    def run(starts: torch.Tensor, lengths: torch.Tensor, train: bool) -> float:
        nonlocal step
        ce_sum, n_sum = torch.zeros((), device=dev), torch.zeros((), device=dev)
        order = (
            torch.randperm(len(starts), device=dev, generator=gen)
            if train
            else torch.arange(len(starts), device=dev)
        )
        for b in range(0, len(starts), config.batch_size):
            idx = order[b : b + config.batch_size]
            x, y = store.gather(starts[idx], lengths[idx], L)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x.float())
            ce, n = next_pitch_loss(logits, y)
            if train:
                for g in opt.param_groups:
                    g["lr"] = _lr_at(step, total_steps, config)
                opt.zero_grad()
                (ce / n.clamp(min=1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
            ce_sum += ce.detach()
            n_sum += n
        return float((ce_sum / n_sum.clamp(min=1)).item())

    before = store.season < before_season
    baseline = count_baseline_ce(
        pd.DataFrame(
            {
                "balls": store.feats[:, TOKEN_FEATURES.index("balls")].float().cpu().numpy(),
                "strikes": store.feats[:, TOKEN_FEATURES.index("strikes")].float().cpu().numpy(),
                "outcome": store.outcome.cpu().numpy(),
            }
        )[before]
    )
    best = (float("inf"), -1, copy.deepcopy(model.state_dict()))
    train_hist: list[float] = []
    val_hist: list[float] = []
    for epoch in range(config.max_epochs):
        model.train()
        train_hist.append(run(tr_s, tr_n, train=True))
        model.eval()
        with torch.no_grad():
            val = run(va_s, va_n, train=False)
        val_hist.append(val)
        logger.info(
            "pretrain < %s epoch %d: train CE %.4f, val CE %.4f (count-only baseline %.4f)",
            before_season,
            epoch,
            train_hist[-1],
            val,
            baseline,
        )
        if not math.isfinite(val):
            raise FloatingPointError(f"pretraining val CE is {val} at epoch {epoch}")
        if val < best[0]:
            best = (val, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= config.patience:
            break
    model.load_state_dict(best[2])
    return PretrainResult(model, best[1], train_hist, val_hist, baseline)
