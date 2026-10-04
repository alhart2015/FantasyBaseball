"""Probe features (#417): use the pretrained pitch model as a scout.

The pretrained model (:mod:`hitter_ros.pretrain`) reads a hitter's pitch history and
predicts the outcome of a next pitch it is shown. So after it reads a hitter's history
before a row's as-of date, show it a **fixed set of standard pitches** and read its
predicted outcome probabilities. Every hitter faces the same probes, so the answers are
comparable, pitch-difficulty-adjusted skill measures: "how often does he whiff on a
good slider low and away", "how hard does he hit a fastball down the middle".

The probes (``PROBE_PITCHES`` x ``PROBE_COUNTS`` x both pitcher hands) are fixed in
code: typical pitch shapes (median speed, movement and spin per pitch type and pitcher
hand, from tracked 2015+ pitches) at standard locations. The ~100 probabilities per row
are then compressed to ``PROBE_FEATURES`` (about 25 numbers) by averaging over pitch
families, counts and zone / chase locations.

Walk-forward: a row from season S is read by the model pretrained on seasons before S
only (one model per season, ``scripts/pretrain_hitter_ros.py``), so every row's
features -- training rows included -- are out of sample. The history is the hitter's
last ``window - 1`` pitches strictly before the as-of date (the model's last position
embedding never predicted anything in pretraining, so it is not used). Only the
prediction head's probabilities are read: hidden states from different seasons' models
are in different coordinates (PR #420). A row with no pitch history, or from a season
with no model, gets NaN (the Standardizer's missing flags handle it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from fantasy_baseball.hitter_ros.history import to_days
from fantasy_baseball.hitter_ros.pitch_tokens import CONTEXT_FEATURES, OUTCOMES, TOKEN_FEATURES
from fantasy_baseball.hitter_ros.pretrain import N_CONTEXT, PitchStore, PretrainModel

# Median (speed, move_arm, move_up, spin) in token units, per pitch group and pitcher
# hand (0 = RHP, 1 = LHP), from tracked pitches 2015-2026 with spin. Token units:
# speed = (mph - 90) / 5, movement in feet, spin = rpm / 2500 (see pitch_tokens.py).
PITCH_SHAPES: dict[tuple[str, int], tuple[float, float, float, float]] = {
    ("ff", 0): (0.84, 0.62, 1.35, 0.918),
    ("ff", 1): (0.54, 0.67, 1.35, 0.902),
    ("si", 0): (0.66, 1.24, 0.76, 0.864),
    ("si", 1): (0.42, 1.26, 0.80, 0.850),
    ("fc", 0): (-0.10, -0.23, 0.68, 0.960),
    ("fc", 1): (-0.52, -0.14, 0.65, 0.909),
    ("sl", 0): (-0.82, -0.36, 0.18, 0.964),
    ("sl", 1): (-1.14, -0.34, 0.16, 0.939),
    ("sweep", 0): (-1.46, -1.17, 0.05, 1.040),
    ("sweep", 1): (-1.82, -1.17, 0.02, 0.994),
    ("cu", 0): (-2.02, -0.78, -0.87, 1.020),
    ("cu", 1): (-2.36, -0.65, -0.77, 0.970),
    ("ch", 0): (-0.84, 1.17, 0.51, 0.700),
    ("ch", 1): (-1.20, 1.20, 0.61, 0.699),
    ("fs", 0): (-0.80, 0.94, 0.29, 0.570),
    ("fs", 1): (-1.28, 0.92, 0.42, 0.494),
}

# (loc_in, loc_up, in the zone?): loc_in is feet toward the hitter (inside +, away -),
# loc_up is height in his zone (0 = bottom, 1 = top); the zone's side edge is ~0.83 ft.
LOCATIONS: dict[str, tuple[float, float, bool]] = {
    "heart": (0.0, 0.5, True),
    "up": (0.0, 0.85, True),
    "in": (0.5, 0.5, True),
    "away": (-0.5, 0.4, True),
    "low": (0.0, 0.15, True),
    "low_away": (-0.5, 0.15, True),
    "chase_up": (0.0, 1.3, False),
    "chase_away": (-1.2, 0.3, False),
    "chase_low": (0.0, -0.35, False),
    "chase_low_away": (-1.1, -0.25, False),
}

# Where each pitch type is typically thrown: in the zone, and as a chase pitch.
PROBE_PITCHES: tuple[tuple[str, str], ...] = (
    ("ff", "heart"),
    ("ff", "up"),
    ("ff", "in"),
    ("ff", "chase_up"),
    ("si", "in"),
    ("si", "low"),
    ("fc", "in"),
    ("fc", "away"),
    ("sl", "low_away"),
    ("sl", "chase_low_away"),
    ("sweep", "away"),
    ("sweep", "chase_away"),
    ("cu", "low"),
    ("cu", "chase_low"),
    ("ch", "low_away"),
    ("ch", "chase_low_away"),
    ("fs", "low"),
    ("fs", "chase_low"),
)
FAMILIES = {"fb": ("ff", "si", "fc"), "brk": ("sl", "sweep", "cu"), "off": ("ch", "fs")}
PROBE_COUNTS = {"even": (1, 1), "two_strike": (1, 2), "ahead": (2, 0)}
# Probes read as if the next day, nobody on, nobody out.
PROBE_GAP = float(np.log1p(1) / 5)

_SWING = ("whiff", "foul", "bip_weak", "bip_flare", "bip_solid", "bip_barrel", "bip_other")
_BIP = ("bip_weak", "bip_flare", "bip_solid", "bip_barrel", "bip_other")


@dataclass(frozen=True)
class Probe:
    group: str
    location: str
    count: str
    lhp: int

    @property
    def in_zone(self) -> bool:
        return LOCATIONS[self.location][2]

    @property
    def family(self) -> str:
        return next(f for f, groups in FAMILIES.items() if self.group in groups)


PROBES: tuple[Probe, ...] = tuple(
    Probe(g, loc, count, lhp)
    for lhp in (0, 1)
    for count in PROBE_COUNTS
    for g, loc in PROBE_PITCHES
)


def probe_contexts(bats_left: float) -> np.ndarray:
    """[len(PROBES), N_CONTEXT] context rows for a hitter batting from ``bats_left``
    (1 = left) -- the same pitches for everyone, mirrored by the token conventions."""
    out = np.zeros((len(PROBES), N_CONTEXT), dtype=np.float32)
    col = {c: i for i, c in enumerate(CONTEXT_FEATURES)}
    for k, p in enumerate(PROBES):
        speed, move_arm, move_up, spin = PITCH_SHAPES[(p.group, p.lhp)]
        loc_in, loc_up, _ = LOCATIONS[p.location]
        balls, strikes = PROBE_COUNTS[p.count]
        row = out[k]
        row[col[f"pt_{p.group}"]] = 1.0
        row[col["has_track"]] = 1.0
        row[col["speed"]] = speed
        row[col["move_arm"]] = move_arm
        row[col["move_up"]] = move_up
        row[col["has_spin"]] = 1.0
        row[col["spin"]] = spin
        row[col["loc_in"]] = loc_in
        row[col["loc_up"]] = loc_up
        row[col["balls"]] = balls / 3
        row[col["strikes"]] = strikes / 2
        row[col["vs_lhp"]] = float(p.lhp)
        row[col["bats_left"]] = bats_left
        row[col["gap"]] = PROBE_GAP
    return out


def _probs_to_features(probs: np.ndarray) -> pd.DataFrame:
    """[rows, len(PROBES), N_OUTCOMES] probabilities -> ``PROBE_FEATURES`` per row."""
    o = {name: probs[..., i] for i, name in enumerate(OUTCOMES)}
    swing = sum(o[k] for k in _SWING)
    bip = sum(o[k] for k in _BIP)
    per = {
        "swing": swing,
        "whiff": o["whiff"] / np.clip(swing, 1e-6, None),  # whiffs per swing
        "hard": (o["bip_solid"] + o["bip_barrel"]) / np.clip(bip, 1e-6, None),  # per BIP
        "barrel": o["bip_barrel"] / np.clip(bip, 1e-6, None),
    }

    def mean(kind: str, keep: list[bool]) -> np.ndarray:
        cols = np.flatnonzero(keep)
        return np.asarray(per[kind][:, cols].mean(axis=1))

    def pick(**want: object) -> list[bool]:
        return [all(getattr(p, k) == v for k, v in want.items()) for p in PROBES]

    out: dict[str, np.ndarray] = {}
    for fam in FAMILIES:
        zone = pick(family=fam, in_zone=True, count="even", lhp=0)
        for kind in ("swing", "whiff", "hard", "barrel"):
            out[f"probe_zone_{kind}_{fam}"] = mean(kind, zone)
        out[f"probe_chase_swing_{fam}"] = mean(
            "swing", pick(family=fam, in_zone=False, count="even", lhp=0)
        )
    out["probe_chase_whiff"] = mean("whiff", pick(in_zone=False, count="even", lhp=0))
    out["probe_two_strike_zone_whiff"] = mean(
        "whiff", pick(in_zone=True, count="two_strike", lhp=0)
    )
    out["probe_two_strike_chase_swing"] = mean(
        "swing", pick(in_zone=False, count="two_strike", lhp=0)
    )
    ahead_fb = pick(family="fb", in_zone=True, count="ahead", lhp=0)
    out["probe_ahead_zone_swing_fb"] = mean("swing", ahead_fb)
    out["probe_ahead_hard_fb"] = mean("hard", ahead_fb)
    for kind, in_zone in (("whiff", True), ("hard", True), ("swing", False)):
        name = f"probe_platoon_{'zone' if in_zone else 'chase'}_{kind}"
        lhp = mean(kind, pick(in_zone=in_zone, count="even", lhp=1))
        rhp = mean(kind, pick(in_zone=in_zone, count="even", lhp=0))
        out[name] = lhp - rhp  # how much worse/better vs lefties
    return pd.DataFrame(out)


PROBE_FEATURES: tuple[str, ...] = tuple(
    _probs_to_features(np.full((1, len(PROBES), len(OUTCOMES)), 0.1)).columns
)


def _bats_left(store: PitchStore, start: np.ndarray, n: np.ndarray) -> dict[int, np.ndarray]:
    """Per pitcher hand (0 = RHP, 1 = LHP), per row: which side he bats from (1 = left)
    -- the majority side over his history pitches vs that hand, else over all of them
    (a switch hitter's side depends on the hand)."""
    bl = store.feats[:, TOKEN_FEATURES.index("bats_left")].float().cpu().numpy()
    vs = store.feats[:, TOKEN_FEATURES.index("vs_lhp")].float().cpu().numpy()

    def window_sum(values: np.ndarray) -> np.ndarray:
        prefix = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
        return np.asarray(prefix[start + n] - prefix[start])

    left_all, total = window_sum(bl), n.astype(np.float64)
    out = {}
    for lhp in (0, 1):
        hand = (vs == lhp).astype(np.float64)
        left, seen = window_sum(bl * hand), window_sum(hand)
        share = np.where(seen > 0, left / np.maximum(seen, 1), left_all / np.maximum(total, 1))
        out[lhp] = (share >= 0.5).astype(np.float32)
    return out


def probe_features(
    model: PretrainModel,
    store: PitchStore,
    player_ids: np.ndarray,
    as_of: pd.Series | np.ndarray,
    *,
    window: int,
    batch_size: int = 64,
    amp: bool = True,
) -> pd.DataFrame:
    """``PROBE_FEATURES`` for each (player, as-of date), read off ``model``.

    History = his last ``window - 1`` pitches strictly before the as-of date. NaN for a
    row with no history. ``model`` must not have seen the rows' season (walk-forward).
    """
    dev = store.device
    model.eval()
    use_amp = amp and dev.type == "cuda"
    start, n = store.history(np.asarray(player_ids), to_days(pd.Series(as_of)), window - 1)
    has = n > 0
    sides = _bats_left(store, start, n)
    # Per probe, which side he bats from depends on the probe's pitcher hand.
    hand = np.array([p.lhp for p in PROBES])
    ctx_r = torch.as_tensor(probe_contexts(0.0), device=dev)
    ctx_l = torch.as_tensor(probe_contexts(1.0), device=dev)
    probs = np.full((len(start), len(PROBES), len(OUTCOMES)), np.nan, dtype=np.float32)
    rows = np.flatnonzero(has)
    with torch.no_grad():
        for b in range(0, len(rows), batch_size):
            idx = rows[b : b + batch_size]
            s_t = torch.as_tensor(start[idx], device=dev)
            n_t = torch.as_tensor(n[idx], device=dev)
            x, _ = store.gather(s_t, n_t, int(n[idx].max()))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                h = model.encoder(x.float())
                h = h[torch.arange(len(idx), device=dev), n_t - 1]  # after his last pitch
                # Context per row and probe: the side he bats from vs that probe's hand.
                left = torch.as_tensor(
                    np.stack([sides[lhp][idx] for lhp in hand], axis=1), device=dev
                )[..., None]
                ctx = left * ctx_l[None] + (1 - left) * ctx_r[None]
                nxt = model.context(ctx)
                logits = model.head(torch.cat([h[:, None].expand_as(nxt), nxt], dim=-1))
            probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    feats = _probs_to_features(np.nan_to_num(probs, nan=0.1))
    feats.loc[~has, :] = np.nan
    return feats


def probe_inputs(table: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """``PROBE_FEATURES`` aligned to ``table``'s rows by (player, season, week); NaN for
    a row the probe file doesn't cover."""
    keys = ["player_id", "season", "week"]
    merged = table[keys].merge(probes[[*keys, *PROBE_FEATURES]], on=keys, how="left")
    merged.index = table.index
    return merged[list(PROBE_FEATURES)]
