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
are then compressed to ``PROBE_FEATURES`` (23 numbers) by averaging over pitch
families, counts and zone / chase locations.

Each season is read by a different model with its own scale, so features are
standardized per season against a **reference group known before the season**: every
hitter-season of the season before (no batting pitchers), read on Opening Day
(``reference_cohort``). Not
against the table's own rows: the table holds only hitters who go on to play after the
as-of date, so that cohort would carry a little hindsight.

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
from pathlib import Path

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
# Probes read as a pitch in the middle of a game (gap 0), nobody on, nobody out. In the
# tokens a gap > 0 only ever comes on a hitter's first pitch of a day, which is always
# at 0-0, so a 1-2 or 2-0 probe with a gap would be a combination never seen.
PROBE_GAP = 0.0

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
# Features that read contact quality (solid / barrel classes). A model pretrained only
# on pitches before Statcast's contact classes (2015) has never seen one, so these come
# out exactly 0 for every hitter: the build blanks them for those seasons
# (``first_contact_season``).
CONTACT_FEATURES: tuple[str, ...] = tuple(
    f for f in PROBE_FEATURES if "_hard" in f or "_barrel" in f
)


def first_contact_season(tokens: pd.DataFrame) -> int:
    """The first season whose balls in play carry Statcast contact classes."""
    quality = (tokens["out_bip_solid"] > 0) | (tokens["out_bip_barrel"] > 0)
    if not quality.any():
        raise ValueError("no pitch has a contact class")
    return int(tokens.loc[quality, "season"].min())


@dataclass(frozen=True)
class SidePrefix:
    """Running sums over the store's pitches, so any history window's batting-side
    counts are two lookups. Build once per store with :func:`side_prefix`."""

    left: np.ndarray  # bats_left
    left_vs: dict[int, np.ndarray]  # bats_left on pitches vs that hand
    vs: dict[int, np.ndarray]  # pitches vs that hand


def side_prefix(store: PitchStore) -> SidePrefix:
    bl = store.feats[:, TOKEN_FEATURES.index("bats_left")].float().cpu().numpy()
    vs = store.feats[:, TOKEN_FEATURES.index("vs_lhp")].float().cpu().numpy()

    def running(values: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])

    hands = {lhp: (vs == lhp).astype(np.float64) for lhp in (0, 1)}
    return SidePrefix(
        left=running(bl),
        left_vs={lhp: running(bl * h) for lhp, h in hands.items()},
        vs={lhp: running(h) for lhp, h in hands.items()},
    )


def _bats_left(prefix: SidePrefix, start: np.ndarray, n: np.ndarray) -> dict[int, np.ndarray]:
    """Per pitcher hand (0 = RHP, 1 = LHP), per row: which side he bats from (1 = left)
    -- the majority side over his history pitches vs that hand, else over all of them
    (a switch hitter's side depends on the hand)."""

    def window(run: np.ndarray) -> np.ndarray:
        return np.asarray(run[start + n] - run[start])

    left_all, total = window(prefix.left), n.astype(np.float64)
    out = {}
    for lhp in (0, 1):
        left, seen = window(prefix.left_vs[lhp]), window(prefix.vs[lhp])
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
    prefix: SidePrefix | None = None,
) -> pd.DataFrame:
    """``PROBE_FEATURES`` (raw probabilities, not standardized) for each (player, as-of
    date), read off ``model``.

    History = his last ``window - 1`` pitches strictly before the as-of date. NaN for a
    row with no history. ``model`` must not have seen the rows' season (walk-forward).
    ``prefix``: :func:`side_prefix` of ``store``, if already built.
    """
    dev = store.device
    model.eval()
    use_amp = amp and dev.type == "cuda"
    start, n = store.history(np.asarray(player_ids), to_days(pd.Series(as_of)), window - 1)
    has = n > 0
    sides = _bats_left(prefix or side_prefix(store), start, n)
    # Per row and probe: 1 if he bats left against that probe's pitcher hand.
    hand = np.array([p.lhp for p in PROBES])
    left_all = np.stack([sides[lhp] for lhp in hand], axis=1)
    probs = np.full((len(start), len(PROBES), len(OUTCOMES)), np.nan, dtype=np.float32)
    rows = np.flatnonzero(has)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # The context layer is affine, so a row's context embedding is the blend of
            # the two sides' embeddings: compute each side's once.
            nxt_l = model.context(torch.as_tensor(probe_contexts(1.0), device=dev))
            nxt_r = model.context(torch.as_tensor(probe_contexts(0.0), device=dev))
        for b in range(0, len(rows), batch_size):
            idx = rows[b : b + batch_size]
            s_t = torch.as_tensor(start[idx], device=dev)
            n_t = torch.as_tensor(n[idx], device=dev)
            x, _ = store.gather(s_t, n_t, int(n[idx].max()))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                h = model.encoder(x.float())
                h = h[torch.arange(len(idx), device=dev), n_t - 1]  # after his last pitch
                left = torch.as_tensor(left_all[idx], device=dev)[..., None]
                nxt = left * nxt_l[None] + (1 - left) * nxt_r[None]
                logits = model.head(torch.cat([h[:, None].expand_as(nxt), nxt], dim=-1))
            probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    feats = _probs_to_features(np.nan_to_num(probs, nan=0.1))
    feats.loc[~has, :] = np.nan
    return feats


def reference_cohort(table: pd.DataFrame, season: int) -> np.ndarray:
    """Last season's hitters (the training table's hitter-seasons, so pitchers who batted
    before the universal DH are left out): a group known before ``season`` starts, used
    to standardize that season's probe features."""
    return np.asarray(table.loc[table["season"] == season - 1, "player_id"].unique())


def standardize(feats: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Each feature as standard deviations from the reference group's mean (blank where
    the reference has no spread, e.g. a blanked feature)."""
    sd = reference.std()
    return (feats - reference.mean()) / sd.where(sd > 0)


PROBE_KEYS = ["player_id", "season", "week"]


def probe_path(root: Path, run: str) -> Path:
    """Where ``scripts/build_hitter_ros_probes.py`` writes a run's probe features."""
    return root / f"probes_{run}.parquet"


def check_probes(table: pd.DataFrame, probes: pd.DataFrame) -> str | None:
    """Why ``probes`` can't be used with ``table`` (None if it can): duplicate keys, a
    table row it lacks, or an as-of date that differs (a probe file built for another
    table would read history cut at the wrong date)."""
    if probes.duplicated(PROBE_KEYS).any():
        return "it has duplicate (player, season, week) rows"
    merged = table[[*PROBE_KEYS, "as_of"]].merge(
        probes[[*PROBE_KEYS, "as_of"]], on=PROBE_KEYS, how="left", suffixes=("", "_probe")
    )
    missing = int(merged["as_of_probe"].isna().sum())
    if missing:
        return f"it lacks {missing} table rows"
    moved = int((pd.to_datetime(merged["as_of"]) != pd.to_datetime(merged["as_of_probe"])).sum())
    if moved:
        return f"{moved} rows have a different as-of date than the table"
    return None


def probe_inputs(table: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    """``PROBE_FEATURES`` aligned to ``table``'s rows by (player, season, week); NaN for
    a row the probe file doesn't cover. The file's features are already standardized
    (see ``scripts/build_hitter_ros_probes.py``)."""
    merged = table[PROBE_KEYS].merge(
        probes[[*PROBE_KEYS, *PROBE_FEATURES]], on=PROBE_KEYS, how="left", validate="one_to_one"
    )
    merged.index = table.index
    return merged[list(PROBE_FEATURES)]
