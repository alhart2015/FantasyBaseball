"""Fine-tune the pretrained pitch encoder on the projection loss (#492).

The frozen readouts of the pitcher-side pretrained model (probe features,
``pitcher_ros/probes.py``) made the projections worse: 23 fixed numbers carry little of
what the model learned. Here the encoder itself joins the pitcher net and trains on its
loss: :class:`FinetuneNet` is the pitcher net (an MLP over the table inputs plus the
appearance encoder's summary, ``hitter_ros.sequence.HybridNet``) with a third input,
the pretrained :class:`~fantasy_baseball.hitter_ros.pretrain.PitchEncoder`'s state after
the pitcher's last pitch before the row's date, through a small projection. The encoder
starts from its pretrained weights and learns at ``lr_scale`` times the rest's learning
rate, so the projection loss reshapes it without wiping out what 13M pitches taught it.

Walk-forward: the net for test season T fine-tunes model T of the pretraining run
(pretrained on seasons before T), on seasons before T. Test rows are then out of
sample. Training rows are not, for the encoder: model T was pretrained on their
seasons' pitches, their futures included (PR #420's concern). That can make the net
lean on the encoder more than it should, never flatter the test score.

History: each row's last ``length`` pitches strictly before its as-of date
(:meth:`PitchStore.history`), left-aligned from position 0 as in pretraining
(``length`` < the pretraining window, whose last position never predicted anything).

Cross-fitting removes that leak while every pitcher still trains: two runs pretrained on
opposite halves of the pitchers (``pretrain_hitter_ros.py --half``, ``player_half``),
and each row read by the encoder of the half it is not in (:func:`cross_encoders`), so
no row's encoder ever read that pitcher. The two encoders were trained apart and their
states mean different things, so each has its own projection into the MLP.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from fantasy_baseball.hitter_ros.history import to_days
from fantasy_baseball.hitter_ros.net import MLP, NetConfig
from fantasy_baseball.hitter_ros.pretrain import (
    PitchEncoder,
    PitchStore,
    PretrainConfig,
    PretrainModel,
    player_half,
)
from fantasy_baseball.hitter_ros.sequence import (
    GRUEncoder,
    SequenceBatcher,
    TransformerEncoder,
    make_encoder,
)


@dataclass(frozen=True)
class FinetuneConfig:
    run: str  # pitcher pretraining run under data/pitcher_ros/pretrain/
    length: int = 1023  # pitches of history each row reads
    lr_scale: float = 0.1  # the encoder's learning rate, as a share of the net's
    out_dim: int = 16  # the encoder summary's width as the MLP sees it
    cross: str | None = None  # a run pretrained on the other half of the pitchers

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError(f"length must be >= 1, got {self.length}")
        if not 0 <= self.lr_scale <= 1:
            raise ValueError(f"lr_scale must be in [0, 1], got {self.lr_scale}")
        if self.out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {self.out_dim}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PretrainRun:
    """A pitcher pretraining run's settings and its season models."""

    def __init__(self, root: Path, name: str) -> None:
        self.dir = root / name
        meta_path = self.dir / "run.json"
        if not meta_path.exists():
            raise ValueError(f"{self.dir} has no run.json; is the pretraining run name right?")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        self.config = PretrainConfig(**self.meta["config"])

    def encoder(self, season: int, device: torch.device) -> PitchEncoder:
        """Model ``season``'s encoder (pretrained on seasons before it), on ``device``."""
        path = self.dir / str(season) / "model.pt"
        if not path.exists():
            raise ValueError(f"{self.dir} has no model for {season}; pretrain it with --add")
        model = PretrainModel(self.config)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.encoder.to(device)


class FinetuneBatcher:
    """The appearance batcher (``SequenceBatcher``) plus each table row's pitch history,
    for :class:`FinetuneNet`. ``batch`` returns ((appearances, pitches, n pitches,
    encoder), appearance lengths); the rest is the appearance batcher's, so the pitcher
    net's checks on it still hold. ``encoder``: which of the net's encoders reads each
    table row (:func:`cross_encoders`); all 0 when omitted."""

    def __init__(
        self,
        appearances: SequenceBatcher,
        store: PitchStore,
        table: pd.DataFrame,
        length: int,
        encoder: np.ndarray | None = None,
    ) -> None:
        self.appearances = appearances
        self.store = store
        self.length = length
        start, n = store.history(table["player_id"].to_numpy(), to_days(table["as_of"]), length)
        dev = store.device
        self.pitch_start = torch.as_tensor(start, dtype=torch.long, device=dev)
        self.pitch_n = torch.as_tensor(n, dtype=torch.long, device=dev)
        if encoder is not None and len(encoder) != len(table):
            raise ValueError("encoder must have one entry per table row")
        which = np.zeros(len(table)) if encoder is None else np.asarray(encoder)
        self.encoder = torch.as_tensor(which, dtype=torch.long, device=dev)

    @property
    def max_len(self) -> int:
        return self.appearances.max_len

    @property
    def n_features(self) -> int:
        return self.appearances.n_features

    @property
    def row_start(self) -> torch.Tensor:
        return self.appearances.row_start

    def batch(
        self, rows: torch.Tensor, *, shuffle_order: bool = False
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        app, lengths = self.appearances.batch(rows, shuffle_order=shuffle_order)
        rows = rows.to(self.store.device)
        n = self.pitch_n[rows]
        pitches, _ = self.store.gather(self.pitch_start[rows], n, self.length)
        return (app, pitches, n, self.encoder[rows]), lengths


class FinetuneNet(nn.Module):
    """MLP over [table inputs, appearance summary, projected pitch-encoder state].

    ``pitches``: one pretrained encoder, or two (cross-fitting); each row is read by the
    one its batch says, through that encoder's own projection. With ``amp`` the encoders
    run in bfloat16; their summaries are cast back to float32 before the MLP, so
    predictions and the loss stay float32.
    """

    def __init__(
        self,
        n_static: int,
        n_out: int,
        hidden: list[int],
        dropout: float,
        appearances: GRUEncoder | TransformerEncoder,
        pitches: list[PitchEncoder],
        out_dim: int,
        lr_scale: float,
    ) -> None:
        super().__init__()
        if not 1 <= len(pitches) <= 2:
            raise ValueError(f"one pitch encoder, or two to cross-fit; got {len(pitches)}")
        self.appearances = appearances
        self.pitches = nn.ModuleList(pitches)
        self.project = nn.ModuleList(nn.Linear(p.dim, out_dim) for p in pitches)
        self.out_dim = out_dim
        self.head = MLP(n_static + appearances.out_dim + out_dim, n_out, hidden, dropout)
        self.lr_scale = lr_scale

    def forward(
        self,
        x: torch.Tensor,
        seq: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        lengths: torch.Tensor,
        amp: bool = False,
    ) -> torch.Tensor:
        app, pitches, n, which = seq
        if len(which) and (int(which.min()) < 0 or int(which.max()) >= len(self.pitches)):
            raise ValueError(f"a row asks for an encoder outside 0..{len(self.pitches) - 1}")
        use_amp = amp and x.is_cuda
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            app_summary = self.appearances(app, lengths)
        summary = torch.zeros(len(n), self.out_dim, device=x.device)
        for k, (encoder, project) in enumerate(zip(self.pitches, self.project, strict=True)):
            rows = torch.nonzero(which == k).flatten()
            if len(rows) == 0:
                continue
            nk = n[rows]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                h = encoder(pitches[rows].float())
                last = h[torch.arange(len(rows), device=h.device), (nk - 1).clamp(min=0)]
            last = last.float() * (nk > 0).float()[:, None]  # no history: a zero state
            summary = summary.index_put((rows,), project(last))
        out: torch.Tensor = self.head(torch.cat([x, app_summary.float(), summary], 1))
        return out

    def optimizer_groups(self, lr: float) -> list[dict[str, Any]]:
        """The pretrained encoder at ``lr_scale`` x ``lr``; everything else at ``lr``."""
        encoder = list(self.pitches.parameters())
        ids = {id(p) for p in encoder}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": rest, "lr": lr}, {"params": encoder, "lr": lr * self.lr_scale}]


def model_maker(
    run: PretrainRun,
    season: int,
    finetune: FinetuneConfig,
    device: torch.device,
    cross: PretrainRun | None = None,
) -> Any:
    """A ``hitter_ros.net.train`` ``make_model`` building :class:`FinetuneNet` around
    ``run``'s model for ``season`` (and with ``cross``, the cross run's as encoder 1)."""

    def make(n_in: int, n_out: int, config: NetConfig, n_token_features: int | None) -> nn.Module:
        if config.seq == "none" or n_token_features is None:
            raise ValueError("fine-tuning joins the appearance encoder: seq can't be 'none'")
        appearances = make_encoder(
            config.seq,
            config.seq_dim,
            config.seq_layers,
            config.seq_len,
            config.dropout,
            n_token_features,
        )
        return FinetuneNet(
            n_in,
            n_out,
            config.hidden,
            config.dropout,
            appearances,
            [r.encoder(season, device) for r in (run, cross) if r is not None],
            finetune.out_dim,
            finetune.lr_scale,
        )

    return make


def check_length(finetune: FinetuneConfig, run: PretrainRun) -> None:
    """The history must fit the encoder's positions, less the last (never trained)."""
    if finetune.length > run.config.window - 1:
        raise ValueError(
            f"length {finetune.length} > the pretraining window less one ({run.config.window - 1})"
        )


def cross_encoders(player_ids: np.ndarray | pd.Series, run: PretrainRun) -> np.ndarray:
    """Per row, the encoder that never read that pitcher: 0 (``run``'s) for a pitcher
    outside ``run``'s half, 1 (the cross run's, pretrained on the other half) for one
    inside it."""
    half = run.meta.get("half")
    if half is None:
        raise ValueError(f"{run.dir} was pretrained on every pitcher: no half to cross-fit")
    which: np.ndarray = (player_half(player_ids) == half).astype(np.int64)
    return which


def check_cross(run: PretrainRun, cross: PretrainRun) -> None:
    """Two runs that can cross-fit: opposite halves, one token file, one window."""
    halves = (run.meta.get("half"), cross.meta.get("half"))
    if None in halves or halves[0] == halves[1]:
        raise ValueError(f"the runs must be pretrained on opposite halves; got {halves}")
    if run.meta["tokens"] != cross.meta["tokens"]:
        raise ValueError("the runs were pretrained on different token files")
    if run.config.window != cross.config.window:
        raise ValueError("the runs have different pretraining windows")
