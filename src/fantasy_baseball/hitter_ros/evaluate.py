"""Score projections against what happened: per-stat raw error (#399's bar).

Error is the mean absolute error (MAE) of the projected rate vs. the actual rate, over a
fixed set of players that every compared projection covers:

* R, HR, RBI, SB: per-PA rate error x 600 -- "how many R/HR/RBI/SB off over 600 PA".
* AVG: batting-average error x 1000 -- in points (e.g. 25 = .025).

Every system is scored on the same players, so the numbers are comparable down a column.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fantasy_baseball.hitter_ros.features import TARGETS

logger = logging.getLogger(__name__)

SCALE = {"r": 600.0, "hr": 600.0, "rbi": 600.0, "sb": 600.0, "avg": 1000.0}


def rates_from_counts(df: pd.DataFrame) -> pd.DataFrame:
    """R/HR/RBI/SB per PA and AVG = H/AB from columns pa, ab, h, r, hr, rbi, sb."""
    pa = df["pa"].astype(float).where(df["pa"] > 0)
    ab = df["ab"].astype(float).where(df["ab"] > 0)
    out = pd.DataFrame({s: df[s].astype(float) / pa for s in ("r", "hr", "rbi", "sb")})
    out["avg"] = df["h"].astype(float) / ab
    return out


def load_fangraphs_hitters(path: Path) -> pd.DataFrame:
    """One FanGraphs hitter CSV -> rates (+ pa) indexed by MLBAM id."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw[raw["MLBAMID"].notna()]
    raw = raw.assign(MLBAMID=raw["MLBAMID"].astype(int))
    dupes = raw["MLBAMID"].duplicated(keep="first")
    if dupes.any():
        logger.warning("%s: %d duplicate MLBAM ids, keeping the first", path.name, dupes.sum())
        raw = raw[~dupes]
    counts = raw.rename(columns={c: c.lower() for c in ("PA", "AB", "H", "R", "HR", "RBI", "SB")})
    rates = rates_from_counts(counts)
    rates["pa"] = counts["pa"].astype(float)
    rates.index = pd.Index(raw["MLBAMID"], name="player_id")
    return rates


def load_systems(directory: Path) -> dict[str, pd.DataFrame]:
    """Every ``<system>-hitters*.csv`` in ``directory`` (not its subfolders), by system."""
    systems = {}
    for path in sorted(directory.glob("*-hitters*.csv")):
        systems[path.name.split("-hitters")[0]] = load_fangraphs_hitters(path)
    return systems


def blend(systems: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Plain average of the systems' rates over the players they all cover."""
    common = _common_index(systems.values())
    stacked = [s.loc[common, list(TARGETS)] for s in systems.values()]
    return sum(stacked[1:], stacked[0]) / len(stacked)


def _common_index(frames: Iterable[pd.DataFrame]) -> pd.Index:
    common: pd.Index | None = None
    for f in frames:
        common = f.index if common is None else common.intersection(f.index)
    assert common is not None, "no projections"
    return common


def score(
    projections: dict[str, pd.DataFrame], actual: pd.DataFrame, min_pa: float
) -> pd.DataFrame:
    """MAE table: one row per projection, one column per stat, plus ``n`` players.

    ``actual`` has rates plus ``pa`` (actual PA in the window); only players with
    ``pa >= min_pa`` that every projection covers are scored.
    """
    players = actual.index[actual["pa"] >= min_pa]
    players = players.intersection(_common_index(projections.values()))
    rows = {}
    for name, proj in projections.items():
        errors = (proj.loc[players, list(TARGETS)] - actual.loc[players, list(TARGETS)]).abs()
        row = {s: errors[s].mean() * SCALE[s] for s in TARGETS}
        row["n"] = len(players)
        rows[name] = row
    return pd.DataFrame(rows).T[[*TARGETS, "n"]]
