"""Pitcher probe features (#492): the pretrained pitch model as a scout of the pitcher.

The pitcher-side pretrained model (``scripts/pretrain_hitter_ros.py --side pitcher``)
reads a pitcher's pitch history and predicts the outcome of a next pitch it is shown.
After it reads his history before a row's as-of date, show it a **fixed set of standard
pitches** -- the hitter probes' league-typical shapes and locations
(``hitter_ros.probes``), for his throwing hand, to batters of both sides -- and read its
predicted outcome probabilities. Every pitcher throws the same probes, so the answers
measure what his history says **beyond the pitch itself**: hitters swinging through or
squaring up a typical fastball from him more or less than from others (deception,
sequencing, how hitters approach him). The pitch's own quality is the stuff inputs'
job (#489).

The ~100 probabilities per row are compressed to ``PROBE_FEATURES`` (23 numbers) by
averaging over pitch families, counts, zone / chase locations and batter sides, as for
hitters. Standardized per season against a reference group known before the season:
last season's pitchers, read on Opening Day (``scripts/build_pitcher_ros_probes.py``).

Walk-forward as the hitter probes: a row of season S is read by the model pretrained on
seasons before S only; the history is his last ``window - 1`` pitches strictly before
the as-of date; only the head's probabilities are read (PR #420). A row with no pitch
history gets NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fantasy_baseball.hitter_ros.features import ROW_KEYS, load_feature_file
from fantasy_baseball.hitter_ros.history import to_days
from fantasy_baseball.hitter_ros.pitch_tokens import CONTEXT_FEATURES, OUTCOMES, TOKEN_FEATURES
from fantasy_baseball.hitter_ros.pretrain import N_CONTEXT, PitchStore, PretrainModel
from fantasy_baseball.hitter_ros.probes import (
    FAMILIES,
    LOCATIONS,
    PITCH_SHAPES,
    PROBE_COUNTS,
    PROBE_GAP,
    PROBE_PITCHES,
)

_SWING = ("whiff", "foul", "bip_weak", "bip_flare", "bip_solid", "bip_barrel", "bip_other")
_BIP = ("bip_weak", "bip_flare", "bip_solid", "bip_barrel", "bip_other")


@dataclass(frozen=True)
class Probe:
    group: str
    location: str
    count: str
    bats_left: int

    @property
    def in_zone(self) -> bool:
        return LOCATIONS[self.location][2]

    @property
    def family(self) -> str:
        return next(f for f, groups in FAMILIES.items() if self.group in groups)


PROBES: tuple[Probe, ...] = tuple(
    Probe(g, loc, count, left)
    for left in (0, 1)
    for count in PROBE_COUNTS
    for g, loc in PROBE_PITCHES
)


def probe_contexts(lhp: int) -> np.ndarray:
    """[len(PROBES), N_CONTEXT] context rows for a pitcher throwing with hand ``lhp``
    (1 = left): every probe pitch in his hand's typical shape."""
    out = np.zeros((len(PROBES), N_CONTEXT), dtype=np.float32)
    col = {c: i for i, c in enumerate(CONTEXT_FEATURES)}
    for k, p in enumerate(PROBES):
        speed, move_arm, move_up, spin = PITCH_SHAPES[(p.group, lhp)]
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
        row[col["vs_lhp"]] = float(lhp)
        row[col["bats_left"]] = float(p.bats_left)
        row[col["gap"]] = PROBE_GAP
    return out


def _probs_to_features(probs: np.ndarray, lhp: np.ndarray) -> pd.DataFrame:
    """[rows, len(PROBES), N_OUTCOMES] probabilities -> ``PROBE_FEATURES`` per row.
    ``lhp``: each row's throwing hand, for the platoon gap (opposite-side batters minus
    same-side ones)."""
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
    # Both batter sides, equally weighted.
    for fam in FAMILIES:
        zone = pick(family=fam, in_zone=True, count="even")
        for kind in ("swing", "whiff", "hard", "barrel"):
            out[f"probe_zone_{kind}_{fam}"] = mean(kind, zone)
        out[f"probe_chase_swing_{fam}"] = mean(
            "swing", pick(family=fam, in_zone=False, count="even")
        )
    out["probe_chase_whiff"] = mean("whiff", pick(in_zone=False, count="even"))
    out["probe_two_strike_zone_whiff"] = mean("whiff", pick(in_zone=True, count="two_strike"))
    out["probe_two_strike_chase_swing"] = mean("swing", pick(in_zone=False, count="two_strike"))
    behind_fb = pick(family="fb", in_zone=True, count="ahead")  # the hitter is ahead
    out["probe_behind_zone_swing_fb"] = mean("swing", behind_fb)
    out["probe_behind_hard_fb"] = mean("hard", behind_fb)
    for kind, in_zone in (("whiff", True), ("hard", True), ("swing", False)):
        name = f"probe_platoon_{'zone' if in_zone else 'chase'}_{kind}"
        lefties = mean(kind, pick(in_zone=in_zone, count="even", bats_left=1))
        righties = mean(kind, pick(in_zone=in_zone, count="even", bats_left=0))
        # Opposite-side batters minus same-side ones: a righty's lefties - righties.
        out[name] = np.where(lhp == 1, righties - lefties, lefties - righties)
    return pd.DataFrame(out)


PROBE_FEATURES: tuple[str, ...] = tuple(
    _probs_to_features(np.full((1, len(PROBES), len(OUTCOMES)), 0.1), np.zeros(1)).columns
)
# Features that read contact quality: blank for models pretrained before Statcast's
# contact classes (``hitter_ros.probes.first_contact_season``).
CONTACT_FEATURES: tuple[str, ...] = tuple(
    f for f in PROBE_FEATURES if "_hard" in f or "_barrel" in f
)


def throwing_hand(store: PitchStore) -> np.ndarray:
    """Running sum of ``vs_lhp`` over the store's pitches (with a leading 0), so any
    history window's left-handed share is two lookups."""
    vs = store.feats[:, TOKEN_FEATURES.index("vs_lhp")].float().cpu().numpy()
    return np.concatenate([[0.0], np.cumsum(vs, dtype=np.float64)])


def probe_features(
    model: PretrainModel,
    store: PitchStore,
    player_ids: np.ndarray,
    as_of: pd.Series | np.ndarray,
    *,
    window: int,
    batch_size: int = 64,
    amp: bool = True,
    hand_sums: np.ndarray | None = None,
) -> pd.DataFrame:
    """``PROBE_FEATURES`` (raw probabilities, not standardized) for each (pitcher,
    as-of date), read off ``model``.

    History = his last ``window - 1`` pitches strictly before the as-of date; his hand is
    the majority hand over them. NaN for a row with no history. ``model`` must not have
    seen the rows' season (walk-forward). ``hand_sums``: :func:`throwing_hand` of
    ``store``, if already built.
    """
    dev = store.device
    model.eval()
    use_amp = amp and dev.type == "cuda"
    start, n = store.history(np.asarray(player_ids), to_days(pd.Series(as_of)), window - 1)
    has = n > 0
    sums = hand_sums if hand_sums is not None else throwing_hand(store)
    left_share = (sums[start + n] - sums[start]) / np.maximum(n, 1)
    lhp = (left_share >= 0.5).astype(np.float32)
    probs = np.full((len(start), len(PROBES), len(OUTCOMES)), np.nan, dtype=np.float32)
    rows = np.flatnonzero(has)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # The context layer is affine and the hand is 0 or 1: compute each hand's once.
            nxt_l = model.context(torch.as_tensor(probe_contexts(1), device=dev))
            nxt_r = model.context(torch.as_tensor(probe_contexts(0), device=dev))
        for b in range(0, len(rows), batch_size):
            idx = rows[b : b + batch_size]
            s_t = torch.as_tensor(start[idx], device=dev)
            n_t = torch.as_tensor(n[idx], device=dev)
            x, _ = store.gather(s_t, n_t, int(n[idx].max()))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                h = model.encoder(x.float())
                h = h[torch.arange(len(idx), device=dev), n_t - 1]  # after his last pitch
                left = torch.as_tensor(lhp[idx], device=dev)[:, None, None]
                nxt = left * nxt_l[None] + (1 - left) * nxt_r[None]
                logits = model.head(torch.cat([h[:, None].expand_as(nxt), nxt], dim=-1))
            probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    feats = _probs_to_features(np.nan_to_num(probs, nan=0.1), lhp)
    feats.loc[~has, :] = np.nan
    return feats


def reference_cohort(table: pd.DataFrame, season: int) -> np.ndarray:
    """Last season's pitchers (the pitcher table's pitcher-seasons): a group known before
    ``season`` starts, used to standardize that season's probe features."""
    return np.asarray(table.loc[table["season"] == season - 1, "player_id"].unique())


def probe_path(root: Path, run: str) -> Path:
    """Where ``scripts/build_pitcher_ros_probes.py`` writes a run's probe features."""
    return root / f"probes_{run}.parquet"


def load_probe_inputs(
    table: pd.DataFrame, root: Path, run: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """``PROBE_FEATURES`` from ``probes_<run>.parquet`` under ``root``, aligned to
    ``table``'s rows, and the file's build options."""
    return load_feature_file(
        table,
        probe_path(root, run),
        list(PROBE_FEATURES),
        f"run scripts/build_pitcher_ros_probes.py --run {run}",
    )


PROBE_KEYS = ROW_KEYS
