"""Plate-appearance sequences for the hitter ROS net (#414, stage 1 of the sequence track).

Three pieces:

1. :func:`build_pa_tokens` turns the pitch store into one row ("token") per plate
   appearance: what happened (result, contact quality, pitches seen, swings, whiffs) and
   the situation (pitcher hand, runners on, outs). About 2.1M PAs for 2015-2026.

2. :class:`SequenceBatcher` holds every token on the GPU and, for a batch of training-
   table rows, hands back each hitter's last ``max_len`` PAs **strictly before** the
   row's as-of date. The PAs are oldest first, right-padded with zeros, plus a length per
   row. Two features are added per token at batch time, because they depend on the row:
   how many days before the as-of date the PA happened (log-scaled), and whether it was
   in the row's own season. So recency is something the net can learn.

3. Two encoders that turn a sequence into one summary vector -- :class:`GRUEncoder`
   (reads the PAs in order) and :class:`TransformerEncoder` (attention over all of them)
   -- and :class:`HybridNet`, which joins that vector to the existing per-window inputs
   before the output layers.

A hitter with no PA before the date (a debut) gets an all-zero summary; the existing
inputs' missing flags already say "no history".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from fantasy_baseball.hitter_ros.net import MLP, TRANSFORMER_HEADS
from fantasy_baseball.hitter_ros.statcast_sql import SPRAY_SQL, SWING_SQL, WHIFF_SQL, sql_in
from fantasy_baseball.pitch_data.store import connect

# Statcast PA-ending events -> result class. truncated_pa (the inning ended on a runner
# while this batter was up) and game_advisory are not plate appearances and are dropped.
RESULT_CLASSES = ("1b", "2b", "3b", "hr", "bb", "hbp", "k", "out", "sac", "other")
EVENT_TO_RESULT = {
    "single": "1b",
    "double": "2b",
    "triple": "3b",
    "home_run": "hr",
    "walk": "bb",
    "intent_walk": "bb",
    "hit_by_pitch": "hbp",
    "strikeout": "k",
    "strikeout_double_play": "k",
    "field_out": "out",
    "force_out": "out",
    "grounded_into_double_play": "out",
    "double_play": "out",
    "triple_play": "out",
    "fielders_choice": "out",
    "fielders_choice_out": "out",
    "sac_fly": "sac",
    "sac_bunt": "sac",
    "sac_fly_double_play": "sac",
    "sac_bunt_double_play": "sac",
    "field_error": "other",
    "catcher_interf": "other",
}

# Per-PA features stored in the token table, each roughly on a 0-1 or -1..1 scale.
STORED_FEATURES = (
    *(f"res_{c}" for c in RESULT_CLASSES),
    "contact",  # ball in play
    "has_ev",  # exit velo measured (2015-16 radar gaps)
    "ev",  # exit velo / 100
    "la",  # launch angle / 50
    "spray",  # spray angle / 45; negative = left field
    "xwoba_con",  # Statcast xwOBA on contact
    "woba_value",  # wOBA credit for the actual result
    "n_pitches",  # / 6
    "swings",  # / 4
    "whiffs",  # / 3
    "vs_lhp",
    "bats_left",
    "on_1b",
    "on_2b",
    "on_3b",
    "outs",  # / 2
)
# Added per batch row: log1p(days before the as-of date) / 5, and same-season flag.
BATCH_FEATURES = ("days_ago", "same_season")
N_TOKEN_FEATURES = len(STORED_FEATURES) + len(BATCH_FEATURES)


def build_pa_tokens(store: Path) -> pd.DataFrame:
    """One row per regular-season PA, sorted by hitter then time.

    Within a day, PAs are ordered by game_pk then at_bat_number. The store has no game
    start time or doubleheader game number, and game_pk order is not guaranteed to be
    play order in a doubleheader (~3% of games), so a doubleheader's two games can come
    out swapped. Only within-day order is affected; days-ago is the same for both.
    """
    conn = connect(store)
    try:
        events = sql_in(EVENT_TO_RESULT)
        df = conn.execute(
            f"""
            WITH pa_pitches AS (
                SELECT game_pk, at_bat_number, batter,
                       count(*) AS n_pitches,
                       count(*) FILTER (WHERE {SWING_SQL}) AS swings,
                       count(*) FILTER (WHERE {WHIFF_SQL}) AS whiffs
                FROM pitches WHERE game_type = 'R'
                GROUP BY 1, 2, 3
            )
            SELECT p.batter AS player_id, CAST(p.game_date AS DATE) AS game_date,
                   CAST(p.season AS INTEGER) AS season, p.game_pk, p.at_bat_number, p.events,
                   p.type = 'X' AS contact, p.launch_speed, p.launch_angle,
                   CASE WHEN p.hc_x IS NOT NULL AND p.hc_y IS NOT NULL THEN {SPRAY_SQL} END AS spray,
                   p.estimated_woba_using_speedangle AS xwoba_con, p.woba_value,
                   pp.n_pitches, pp.swings, pp.whiffs,
                   p.p_throws = 'L' AS vs_lhp, p.stand = 'L' AS bats_left,
                   p.on_1b IS NOT NULL AS on_1b, p.on_2b IS NOT NULL AS on_2b,
                   p.on_3b IS NOT NULL AS on_3b, p.outs_when_up AS outs
            FROM pitches p
            JOIN pa_pitches pp USING (game_pk, at_bat_number, batter)
            WHERE p.game_type = 'R' AND p.events IN {events}
            ORDER BY player_id, game_date, game_pk, at_bat_number
            """
        ).df()
    finally:
        conn.close()
    result = df["events"].map(EVENT_TO_RESULT)
    out = df[["player_id", "game_date", "season", "game_pk", "at_bat_number"]].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    for c in RESULT_CLASSES:
        out[f"res_{c}"] = (result == c).astype(np.float32)
    out["contact"] = df["contact"].astype(np.float32)
    out["has_ev"] = df["launch_speed"].notna().astype(np.float32)
    out["ev"] = (df["launch_speed"] / 100).fillna(0).astype(np.float32)
    out["la"] = (df["launch_angle"] / 50).fillna(0).astype(np.float32)
    out["spray"] = (df["spray"] / 45).fillna(0).astype(np.float32)
    out["xwoba_con"] = df["xwoba_con"].fillna(0).astype(np.float32)
    out["woba_value"] = df["woba_value"].fillna(0).astype(np.float32)
    out["n_pitches"] = (df["n_pitches"] / 6).astype(np.float32)
    out["swings"] = (df["swings"] / 4).astype(np.float32)
    out["whiffs"] = (df["whiffs"] / 3).astype(np.float32)
    for c in ("vs_lhp", "bats_left", "on_1b", "on_2b", "on_3b"):
        out[c] = df[c].astype(np.float32)
    out["outs"] = (df["outs"] / 2).astype(np.float32)
    return out


def _days(dates: pd.Series) -> np.ndarray:
    """Dates as whole days since 1970."""
    days: np.ndarray = pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype(np.int64)
    return days


# Sort key that orders tokens by hitter, then day: player rank * _DAY_SPAN + day.
_DAY_SPAN = 1_000_000


class SequenceBatcher:
    """Every PA token on one device, plus per-table-row slice bounds.

    ``rows`` is the training table (needs ``player_id``, ``as_of``, ``season``). Row ``i``
    owns the tokens ``[start[i], end[i])``: that hitter's PAs strictly before ``as_of``.
    """

    def __init__(
        self,
        tokens: pd.DataFrame,
        rows: pd.DataFrame,
        max_len: int,
        device: torch.device,
    ) -> None:
        tokens = tokens.sort_values(["player_id", "game_date", "game_pk", "at_bat_number"])
        self.max_len = max_len
        self.device = device
        tok_player = tokens["player_id"].to_numpy()
        tok_days = _days(tokens["game_date"])
        players, first = np.unique(tok_player, return_index=True)
        tok_key = np.searchsorted(players, tok_player) * _DAY_SPAN + tok_days

        row_player = rows["player_id"].to_numpy()
        row_days = _days(rows["as_of"])
        rank = np.searchsorted(players, row_player)
        known = (rank < len(players)) & (players[np.minimum(rank, len(players) - 1)] == row_player)
        start = np.where(known, first[np.minimum(rank, len(players) - 1)], 0)
        # First token of this hitter on or after the as-of date = end of the "before" slice.
        end = np.where(known, np.searchsorted(tok_key, rank * _DAY_SPAN + row_days, "left"), 0)

        def as_t(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(a, dtype=torch.long, device=device)

        self.feats = torch.as_tensor(
            tokens[list(STORED_FEATURES)].to_numpy(np.float32), device=device
        )
        self.tok_days = as_t(tok_days)
        self.tok_season = as_t(tokens["season"].to_numpy())
        self.row_start = as_t(start)
        self.row_end = as_t(end)
        self.row_days = as_t(row_days)
        self.row_season = as_t(rows["season"].to_numpy())

    def lengths(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.row_end[rows] - self.row_start[rows], max=self.max_len)

    def batch(
        self, rows: torch.Tensor, *, shuffle_order: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(tokens [B, max_len, N_TOKEN_FEATURES], lengths [B]) for table positions ``rows``.

        ``shuffle_order`` scrambles each row's PAs (a check: if scores don't drop, the
        net isn't using order).
        """
        n = self.lengths(rows)
        steps = torch.arange(self.max_len, device=self.device)
        valid = steps[None, :] < n[:, None]
        if shuffle_order:
            noise = torch.rand(valid.shape, device=self.device).masked_fill(~valid, 2.0)
            steps_b = noise.argsort(dim=1)
        else:
            steps_b = steps[None, :].expand(len(rows), -1)
        idx = self.row_end[rows][:, None] - n[:, None] + steps_b
        idx = torch.where(valid, idx, torch.zeros_like(idx))
        feats = self.feats[idx]
        days_ago = (self.row_days[rows][:, None] - self.tok_days[idx]).clamp(min=0)
        same = (self.tok_season[idx] == self.row_season[rows][:, None]).float()
        extra = torch.stack([torch.log1p(days_ago.float()) / 5, same], dim=-1)
        x = torch.cat([feats, extra], dim=-1) * valid[..., None]
        return x, n


class GRUEncoder(nn.Module):
    """Reads the PAs oldest to newest; the summary is the state after the newest one."""

    def __init__(self, n_in: int, dim: int, layers: int = 1) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, dim), nn.GELU())
        self.gru = nn.GRU(dim, dim, num_layers=layers, batch_first=True)
        self.out_dim = dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        has = (lengths > 0).float()[:, None]
        packed = pack_padded_sequence(
            self.inp(x), lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        out: torch.Tensor = h[-1] * has
        return out


class TransformerEncoder(nn.Module):
    """Self-attention over the PAs; a learned summary token reads them all.

    Attention by itself ignores order, so each PA also gets a learned embedding of its
    position counted back from the newest (0 = most recent PA). That lets the encoder
    tell PAs apart even on the same day, where days-ago is identical.
    """

    def __init__(
        self,
        n_in: int,
        dim: int,
        layers: int,
        max_len: int,
        dropout: float,
        heads: int = TRANSFORMER_HEADS,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"seq_dim {dim} must be divisible by the {heads} attention heads")
        self.inp = nn.Linear(n_in, dim)
        self.position = nn.Embedding(max_len, dim)
        # Small random start: an all-zero summary would attend to every PA equally (a
        # plain average, blind to order) until training moved it.
        self.summary = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=2 * dim, dropout=dropout, batch_first=True
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_dim = dim

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, steps, _ = x.shape
        step = torch.arange(steps, device=x.device)[None, :]
        back = (lengths[:, None] - 1 - step).clamp(min=0)  # 0 = newest PA
        tokens = self.inp(x) + self.position(back)
        tokens = torch.cat([self.summary.expand(b, -1, -1), tokens], dim=1)
        pad = torch.arange(steps, device=x.device)[None, :] >= lengths[:, None]
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), pad], dim=1)
        out: torch.Tensor = self.body(tokens, src_key_padding_mask=pad)[:, 0]
        return out * (lengths > 0).float()[:, None]


class HybridNet(nn.Module):
    """The plain MLP, with a sequence summary joined to its inputs.

    With ``amp``, only the encoder runs in bfloat16; its summary is cast back to float32
    before the MLP head, so predictions and the loss stay float32.
    """

    def __init__(
        self,
        n_static: int,
        n_out: int,
        hidden: list[int],
        dropout: float,
        encoder: GRUEncoder | TransformerEncoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = MLP(n_static + encoder.out_dim, n_out, hidden, dropout)

    def forward(
        self, x: torch.Tensor, seq: torch.Tensor, lengths: torch.Tensor, amp: bool = False
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and x.is_cuda):
            summary = self.encoder(seq, lengths)
        out: torch.Tensor = self.head(torch.cat([x, summary.float()], dim=1))
        return out


def make_encoder(
    kind: str, dim: int, layers: int, max_len: int, dropout: float
) -> GRUEncoder | TransformerEncoder:
    if kind == "gru":
        return GRUEncoder(N_TOKEN_FEATURES, dim, layers)
    if kind == "transformer":
        return TransformerEncoder(N_TOKEN_FEATURES, dim, layers, max_len, dropout)
    raise ValueError(f"unknown sequence encoder {kind!r}")
