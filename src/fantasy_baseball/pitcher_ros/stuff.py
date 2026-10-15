"""Per-pitch pitch-quality ("stuff") model for the pitcher net (#489, stage 1).

FanGraphs beats us on the top starters' ERA, all of it in earned runs per batter and
mostly home runs; our misses line up with batted-ball shape and damage (#486, #489).
The net sees season-level rates only. This model looks at every pitch: from what the
pitch *is* -- type, speed, movement, spin, release, extension, arm angle, and how it
differs from the same pitcher's fastball that game -- it predicts what batters do with
it:

* ``OUTCOMES`` (ball / called strike / whiff / foul / HBP / ball in play by contact
  quality, up to a barrel; ``hitter_ros.pitch_tokens``),
* on a ball in play, its type (``BB_TYPES``: ground ball, line drive, fly ball, popup),
* on a ball in play with Statcast's estimate (2015+), its xwOBA.

Two input sets: ``STUFF_FEATURES`` (the pitch only) and ``LOCATED_FEATURES`` (also where
it crossed the plate and the count). The gap between them is command.

Walk-forward like ``hitter_ros.pretrain``: the model for season S is trained on pitches
before S only, so anything it says about a season-S pitch is out of sample. Read its
predicted probabilities, never hidden states: each season's model has its own
coordinates (PR #420).

Movement and release side are mirrored for left-handed pitchers, so "arm side" means
the same thing for everyone. A missing tracking number is 0 with a flag.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from fantasy_baseball.hitter_ros.features import FIXED_ZONE
from fantasy_baseball.hitter_ros.net import device
from fantasy_baseball.hitter_ros.pitch_tokens import OUTCOMES, PITCH_GROUPS, outcome_index
from fantasy_baseball.pitch_data.store import connect
from fantasy_baseball.pitcher_ros.stats import GAME_DATE_SQL

logger = logging.getLogger(__name__)

GROUPS = (*PITCH_GROUPS, "other")
GRADED_CONTACT_FROM = 2015
FB_GROUPS = ("ff", "si")  # PITCH_GROUPS names, not Statcast codes
BB_TYPES = ("ground_ball", "line_drive", "fly_ball", "popup")

STUFF_FEATURES = (
    *(f"pt_{g}" for g in GROUPS),
    "has_track",  # speed and movement measured
    "speed",  # (release_speed - 90) / 5
    "move_arm",  # horizontal break toward the pitcher's arm side, feet
    "move_up",  # induced vertical break, feet
    "has_spin",
    "spin",  # release_spin_rate / 2500
    "has_axis",
    "axis_sin",  # spin axis, mirrored for lefties
    "axis_cos",
    "has_release",
    "rel_side",  # release point toward the arm side, feet
    "rel_height",  # release height - 6, feet
    "has_ext",
    "ext",  # release extension - 6, feet
    "has_arm",
    "arm_angle",  # / 45 degrees
    "has_fb",  # he threw a fastball (ff / si) in this game
    "fb_speed_gap",  # (speed - his fastball's average speed this game) / 5
    "fb_arm_gap",  # move_arm - his fastball's, feet
    "fb_up_gap",  # move_up - his fastball's, feet
    "same_side",  # pitcher and batter throw / bat from the same side
    # Statcast grades contact (weak / flare / solid / barrel) from 2015 on; before, every
    # ball in play is "bip_other". A model trained on both eras is read with this at 1
    # (:func:`predict_stuff`) to get the modern answer for every pitch.
    "graded_contact",
)
LOCATION_FEATURES = (
    "loc_in",  # plate_x toward the batter (inside), feet
    "loc_up",  # height in features.FIXED_ZONE: 0 = bottom, 1 = top
    "balls",  # / 3
    "strikes",  # / 2
)
LOCATED_FEATURES = (*STUFF_FEATURES, *LOCATION_FEATURES)
FEATURE_SETS = {"stuff": STUFF_FEATURES, "located": LOCATED_FEATURES}
KEY_COLUMNS = ("pitcher", "season", "game_date", "game_pk", "at_bat_number", "pitch_number")
LABEL_COLUMNS = ("outcome", "bb", "xwoba")


def stuff_path(root: Path) -> Path:
    """Where ``scripts/build_pitcher_stuff.py`` writes the per-pitch frame under ``root``."""
    return root / "stuff_pitches.parquet"


def build_stuff_pitches(store: Path) -> pd.DataFrame:
    """:func:`stuff_frame` for every regular-season pitch in the store, a season at a
    time to keep memory down."""
    conn = connect(store)
    try:
        seasons = [
            int(r[0])
            for r in conn.execute("SELECT DISTINCT season FROM pitches ORDER BY 1").fetchall()
        ]
        parts = []
        for s in seasons:
            parts.append(stuff_frame(_season_pitches(conn, s)))
            logger.info("season %d: %d pitches", s, len(parts[-1]))
    finally:
        conn.close()
    return pd.concat(parts, ignore_index=True)


def _season_pitches(conn: duckdb.DuckDBPyConnection, season: int) -> pd.DataFrame:
    # Dated as the pitcher table dates the game (table.py: the pitching lines' date, a
    # suspended game's resume date), so a row's as-of cutoff sees the same games here.
    return conn.execute(
        f"""
        WITH game_dates AS (
            SELECT game_pk, max({GAME_DATE_SQL}) AS line_date FROM pitching GROUP BY 1
        )
        SELECT pitcher, CAST(season AS INTEGER) AS season,
               COALESCE(line_date, CAST(game_date AS DATE)) AS game_date,
               game_pk, at_bat_number, pitch_number, pitch_type, p_throws, stand,
               release_speed, pfx_x, pfx_z, release_spin_rate, spin_axis,
               release_pos_x, release_pos_z, release_extension, arm_angle,
               plate_x, plate_z, balls, strikes,
               description, type, launch_speed_angle, bb_type,
               estimated_woba_using_speedangle AS xwoba
        FROM pitches LEFT JOIN game_dates USING (game_pk)
        WHERE game_type = 'R' AND season = ?
        ORDER BY pitcher, game_date, game_pk, at_bat_number, pitch_number
        """,
        [season],
    ).df()


def stuff_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keys, ``LOCATED_FEATURES`` and labels (``outcome`` index into ``OUTCOMES``; ``bb``
    index into ``BB_TYPES`` on a ball in play, else -1; ``xwoba`` on a ball in play,
    else NaN) from raw pitch rows; split out so tests can feed rows."""
    f32 = np.float32
    out = df[list(KEY_COLUMNS)].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    group = pd.Series("other", index=df.index)
    for g, codes in PITCH_GROUPS.items():
        group[df["pitch_type"].isin(codes)] = g
    for g in GROUPS:
        out[f"pt_{g}"] = (group == g).astype(f32)

    lefty = df["p_throws"] == "L"
    # Catcher's view: + x = toward first base. A righty's arm side is third base (-),
    # so flip righties; + is then arm side for everyone.
    arm_sign = np.where(lefty, 1.0, -1.0)
    out["has_track"] = (df["release_speed"].notna() & df["pfx_x"].notna()).astype(f32)
    out["speed"] = ((df["release_speed"] - 90) / 5).fillna(0).astype(f32)
    move_arm = df["pfx_x"] * arm_sign
    out["move_arm"] = move_arm.fillna(0).astype(f32)
    out["move_up"] = df["pfx_z"].fillna(0).astype(f32)
    out["has_spin"] = df["release_spin_rate"].notna().astype(f32)
    out["spin"] = (df["release_spin_rate"] / 2500).fillna(0).astype(f32)
    # Spin axis in degrees, 180 = pure backspin from a righty; mirror lefties around it.
    axis = np.deg2rad(np.where(lefty, 360 - df["spin_axis"], df["spin_axis"]).astype(float))
    out["has_axis"] = df["spin_axis"].notna().astype(f32)
    out["axis_sin"] = np.nan_to_num(np.sin(axis)).astype(f32)
    out["axis_cos"] = np.nan_to_num(np.cos(axis)).astype(f32)
    out["has_release"] = (df["release_pos_x"].notna() & df["release_pos_z"].notna()).astype(f32)
    out["rel_side"] = (df["release_pos_x"] * arm_sign).fillna(0).astype(f32)
    out["rel_height"] = (df["release_pos_z"] - 6).fillna(0).astype(f32)
    out["has_ext"] = df["release_extension"].notna().astype(f32)
    out["ext"] = (df["release_extension"] - 6).fillna(0).astype(f32)
    out["has_arm"] = df["arm_angle"].notna().astype(f32)
    out["arm_angle"] = (df["arm_angle"] / 45).fillna(0).astype(f32)

    # His fastball this game: the average of his tracked ff / si pitches.
    is_fb = group.isin(FB_GROUPS) & df["release_speed"].notna() & df["pfx_x"].notna()
    fb = (
        pd.DataFrame(
            {
                "pitcher": df["pitcher"],
                "game_pk": df["game_pk"],
                "speed": df["release_speed"],
                "arm": move_arm,
                "up": df["pfx_z"],
            }
        )[is_fb.to_numpy()]
        .groupby(["pitcher", "game_pk"])[["speed", "arm", "up"]]
        .mean()
    )
    ref = fb.reindex(pd.MultiIndex.from_frame(df[["pitcher", "game_pk"]]))
    ref.index = df.index
    has_fb = ref["speed"].notna() & (out["has_track"] > 0)
    out["has_fb"] = has_fb.astype(f32)
    out["fb_speed_gap"] = ((df["release_speed"] - ref["speed"]) / 5).where(has_fb, 0).astype(f32)
    out["fb_arm_gap"] = (move_arm - ref["arm"]).where(has_fb, 0).astype(f32)
    out["fb_up_gap"] = (df["pfx_z"] - ref["up"]).where(has_fb, 0).astype(f32)
    out["same_side"] = (df["p_throws"] == df["stand"]).astype(f32)
    out["graded_contact"] = (df["season"] >= GRADED_CONTACT_FROM).astype(f32)

    lefty_b = df["stand"] == "L"
    # Clipped: a few tracking glitches sit tens of feet off the plate.
    loc_in = df["plate_x"].where(lefty_b, -df["plate_x"]).clip(-3, 3)
    out["loc_in"] = loc_in.fillna(0).astype(f32)
    height = (df["plate_z"] - FIXED_ZONE[1]) / (FIXED_ZONE[2] - FIXED_ZONE[1])
    out["loc_up"] = height.clip(-2, 3).fillna(0.5).astype(f32)
    out["balls"] = (df["balls"] / 3).astype(f32)
    out["strikes"] = (df["strikes"] / 2).astype(f32)

    in_play = df["type"] == "X"
    out["outcome"] = outcome_index(df["description"], df["launch_speed_angle"], df["type"])
    bb = df["bb_type"].map({b: i for i, b in enumerate(BB_TYPES)})
    out["bb"] = bb.where(in_play).fillna(-1).astype(np.int64)
    out["xwoba"] = df["xwoba"].where(in_play).astype(f32)
    return out


@dataclass
class StuffConfig:
    hidden: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8192
    max_epochs: int = 20
    patience: int = 2
    val_frac: float = 0.05  # of the training pitchers' games, for early stopping
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StuffNet(nn.Module):
    """A shared MLP with three heads: outcome logits, batted-ball-type logits, xwOBA."""

    def __init__(self, n_in: int, config: StuffConfig):
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for h in config.hidden:
            layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(config.dropout)]
            width = h
        self.trunk = nn.Sequential(*layers)
        self.outcome = nn.Linear(width, len(OUTCOMES))
        self.bb = nn.Linear(width, len(BB_TYPES))
        self.xwoba = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.outcome(h), self.bb(h), self.xwoba(h).squeeze(-1)


def _losses(
    model: StuffNet, x: torch.Tensor, y_out: torch.Tensor, y_bb: torch.Tensor, y_xw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-head mean losses: outcome cross-entropy over every pitch, batted-ball-type
    cross-entropy over balls in play, xwOBA squared error over balls in play with one."""
    lo, lb, lx = model(x)
    out_loss = F.cross_entropy(lo, y_out)
    has_bb = y_bb >= 0
    bb_loss = F.cross_entropy(lb[has_bb], y_bb[has_bb]) if has_bb.any() else lo.sum() * 0
    has_xw = ~torch.isnan(y_xw)
    xw_loss = ((lx[has_xw] - y_xw[has_xw]) ** 2).mean() if has_xw.any() else lo.sum() * 0
    return out_loss, bb_loss, xw_loss


# xwOBA's squared error is ~0.15 per ball in play against cross-entropies near 1.5:
# scale it up so the damage head pulls on the trunk about as hard as the others.
XWOBA_LOSS_WEIGHT = 5.0


@dataclass
class StuffFit:
    model: StuffNet
    features: tuple[str, ...]
    best_epoch: int
    val_loss: list[float]
    # Whether training saw graded-contact pitches (``graded_contact`` 1): only then has
    # the model learned what that input means (see :func:`predict_stuff`).
    graded_contact: bool


def pitch_tensors(
    frame: pd.DataFrame, features: tuple[str, ...], dev: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs, outcome, batted-ball type and xwOBA of ``frame`` as tensors on ``dev``."""
    return (
        torch.as_tensor(frame[list(features)].to_numpy(np.float32), device=dev),
        torch.as_tensor(frame["outcome"].to_numpy(np.int64, copy=True), device=dev),
        torch.as_tensor(frame["bb"].to_numpy(np.int64, copy=True), device=dev),
        torch.as_tensor(frame["xwoba"].to_numpy(np.float32), device=dev),
    )


def fit_stuff(
    pitches: pd.DataFrame, features: tuple[str, ...], config: StuffConfig | None = None
) -> StuffFit:
    """Train on ``pitches`` (a :func:`stuff_frame`, already cut to the training seasons).
    A random ``val_frac`` of games is held out for early stopping on the summed loss."""
    config = config or StuffConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    games = pitches["game_pk"].unique()
    val_games = rng.choice(games, max(1, int(len(games) * config.val_frac)), replace=False)
    val = pitches["game_pk"].isin(val_games).to_numpy()
    dev = device()
    xt, ot, bt, wt = pitch_tensors(pitches[~val], features, dev)
    xv, ov, bv, wv = pitch_tensors(pitches[val], features, dev)
    model = StuffNet(len(features), config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    gen = torch.Generator(device=dev).manual_seed(config.seed)
    best: tuple[float, int, dict[str, torch.Tensor] | None] = (math.inf, -1, None)
    history: list[float] = []
    n = len(xt)
    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(n, device=dev, generator=gen)
        for start in range(0, n, config.batch_size):
            i = order[start : start + config.batch_size]
            lo, lb, lx = _losses(model, xt[i], ot[i], bt[i], wt[i])
            opt.zero_grad()
            (lo + lb + XWOBA_LOSS_WEIGHT * lx).backward()
            opt.step()
        # A head with nothing to score (no xwOBA before 2015) adds nothing.
        val_loss = float(
            np.nansum(
                np.array((1, 1, XWOBA_LOSS_WEIGHT)) * np.array(evaluate(model, xv, ov, bv, wv))
            )
        )
        history.append(val_loss)
        logger.info("epoch %d: val %.4f", epoch, val_loss)
        if val_loss < best[0]:
            best = (val_loss, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        elif epoch - best[1] >= config.patience:
            break
    assert best[2] is not None
    model.load_state_dict(best[2])
    graded = "graded_contact" in features and bool((pitches["graded_contact"] > 0).any())
    return StuffFit(
        model=model,
        features=features,
        best_epoch=best[1],
        val_loss=history,
        graded_contact=graded,
    )


@torch.no_grad()
def evaluate(
    model: StuffNet,
    x: torch.Tensor,
    y_out: torch.Tensor,
    y_bb: torch.Tensor,
    y_xw: torch.Tensor,
    chunk: int = 262144,
) -> tuple[float, float, float]:
    """Mean outcome cross-entropy, batted-ball-type cross-entropy (balls in play) and
    xwOBA squared error (balls in play with one), over chunks; NaN for a head with
    nothing to score."""
    model.eval()
    sums = np.zeros(3)
    counts = np.zeros(3)
    for s in range(0, len(x), chunk):
        sl = slice(s, s + chunk)
        lo, lb, lx = model(x[sl])
        sums[0] += F.cross_entropy(lo, y_out[sl], reduction="sum").item()
        counts[0] += len(lo)
        has_bb = y_bb[sl] >= 0
        sums[1] += F.cross_entropy(lb[has_bb], y_bb[sl][has_bb], reduction="sum").item()
        counts[1] += int(has_bb.sum())
        has_xw = ~torch.isnan(y_xw[sl])
        sums[2] += ((lx[has_xw] - y_xw[sl][has_xw]) ** 2).sum().item()
        counts[2] += int(has_xw.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        out = sums / counts
    return float(out[0]), float(out[1]), float(out[2])


@torch.no_grad()
def predict_stuff(fit: StuffFit, pitches: pd.DataFrame, chunk: int = 262144) -> pd.DataFrame:
    """Predicted outcome probabilities (``p_<outcome>``), batted-ball-type probabilities
    (``p_<bb type>``, given a ball in play) and xwOBA given a ball in play, per pitch, as
    a pitch thrown in the graded-contact era (``graded_contact`` set to 1) when ``fit``
    was trained on any; a model that only saw earlier pitches never learned that input,
    so it reads every pitch as it was trained (0)."""
    model = fit.model.eval()
    dev = next(model.parameters()).device
    frame = pitches[list(fit.features)].assign(graded_contact=np.float32(fit.graded_contact))
    x = torch.as_tensor(frame.to_numpy(np.float32), device=dev)
    parts = []
    for s in range(0, len(x), chunk):
        lo, lb, lx = model(x[s : s + chunk])
        parts.append(
            torch.cat([F.softmax(lo, -1), F.softmax(lb, -1), lx.unsqueeze(-1)], dim=-1)
            .cpu()
            .numpy()
        )
    cols = [*(f"p_{o}" for o in OUTCOMES), *(f"p_{b}" for b in BB_TYPES), "xwoba_bip"]
    return pd.DataFrame(np.concatenate(parts), columns=cols, index=pitches.index)


def type_baseline(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float, float]:
    """The same three losses for a pitch-type-only guess: each pitch group's outcome and
    batted-ball-type frequencies, and its mean xwOBA on balls in play, from ``train``.
    A stuff model that can't beat this has learned nothing beyond the pitch type."""
    group = lambda f: f[[f"pt_{g}" for g in GROUPS]].to_numpy().argmax(axis=1)  # noqa: E731
    g_tr, g_te = group(train), group(test)
    k = len(GROUPS)
    out_freq = np.full((k, len(OUTCOMES)), 1.0)  # add-one smoothing
    np.add.at(out_freq, (g_tr, train["outcome"].to_numpy()), 1)
    out_freq /= out_freq.sum(axis=1, keepdims=True)
    bb_tr = train["bb"].to_numpy()
    bb_freq = np.full((k, len(BB_TYPES)), 1.0)
    np.add.at(bb_freq, (g_tr[bb_tr >= 0], bb_tr[bb_tr >= 0]), 1)
    bb_freq /= bb_freq.sum(axis=1, keepdims=True)
    xw_tr = train["xwoba"].to_numpy()
    xw_ok = ~np.isnan(xw_tr)
    overall = xw_tr[xw_ok].mean()
    xw_mean = np.array(
        [
            xw_tr[xw_ok & (g_tr == j)].mean() if (xw_ok & (g_tr == j)).any() else overall
            for j in range(k)
        ]
    )
    ce = -np.log(out_freq[g_te, test["outcome"].to_numpy()]).mean()
    bb_te = test["bb"].to_numpy()
    on = bb_te >= 0
    bb_ce = -np.log(bb_freq[g_te[on], bb_te[on]]).mean()
    xw_te = test["xwoba"].to_numpy()
    ok = ~np.isnan(xw_te)
    xw_se = ((xw_mean[g_te[ok]] - xw_te[ok]) ** 2).mean()
    return float(ce), float(bb_ce), float(xw_se)


def load_stuff(path: Path) -> StuffFit:
    """A model saved by ``scripts/build_pitcher_stuff.py`` (``model-<season>.pt``)."""
    saved = torch.load(path, map_location=device(), weights_only=False)
    config = StuffConfig(**{**saved["config"], "hidden": tuple(saved["config"]["hidden"])})
    features = tuple(saved["features"])
    model = StuffNet(len(features), config).to(device())
    model.load_state_dict(saved["state_dict"])
    # Models saved before this was recorded: model-<season> trained on earlier seasons
    # only, so it saw graded contact exactly when an earlier season had it.
    graded = saved.get("graded_contact", saved["season"] > GRADED_CONTACT_FROM)
    return StuffFit(
        model=model.eval(),
        features=features,
        best_epoch=-1,
        val_loss=[],
        graded_contact=bool(graded),
    )
