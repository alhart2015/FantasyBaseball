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

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.features import TARGETS, rates_from_counts

logger = logging.getLogger(__name__)

SCALE = {"r": 600.0, "hr": 600.0, "rbi": 600.0, "sb": 600.0, "avg": 1000.0}

# A full-season (preseason) projection has everyday players near 600+ PA. A file whose
# biggest projection is under this is a rest-of-season export saved in the wrong place.
PRESEASON_MIN_TOP_PA = 450


def load_fangraphs_hitters(path: Path) -> pd.DataFrame:
    """One FanGraphs hitter CSV -> rates (+ pa) indexed by MLBAM id."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw.assign(MLBAMID=pd.to_numeric(raw["MLBAMID"], errors="coerce"))
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


def load_systems(directory: Path, *, preseason: bool = False) -> dict[str, pd.DataFrame]:
    """Every ``<system>-hitters*.csv`` in ``directory`` (not its subfolders), by system.

    With ``preseason``, refuse a file that looks like a rest-of-season export (no
    projection reaches ``PRESEASON_MIN_TOP_PA``): scoring one against our week-0 rows
    would hand FanGraphs part of the season in hindsight.
    """
    systems = {}
    for path in sorted(directory.glob("*-hitters*.csv")):
        rates = load_fangraphs_hitters(path)
        top_pa = rates["pa"].max()
        if preseason and top_pa < PRESEASON_MIN_TOP_PA:
            raise ValueError(
                f"{path}: top projected PA is {top_pa:.0f}, so this looks like a "
                "rest-of-season export, not a preseason projection"
            )
        systems[path.name.split("-hitters")[0]] = rates
    return systems


def blend(systems: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Plain average of the systems' rates over the players they all cover."""
    common = _common_index(systems.values())
    stacked = [s.loc[common, list(TARGETS)] for s in systems.values()]
    return sum(stacked[1:], stacked[0]) / len(stacked)


def _common_index(frames: Iterable[pd.DataFrame]) -> pd.Index:
    """Players every frame has a value for, on every stat."""
    common: pd.Index | None = None
    for f in frames:
        have = f.dropna(subset=list(TARGETS)).index
        common = have if common is None else common.intersection(have)
    assert common is not None, "no projections"
    return common


def scored_players(
    projections: dict[str, pd.DataFrame], actual: pd.DataFrame, min_pa: float
) -> pd.DataFrame:
    """One row per (player, system, stat): projected and actual rate, and the scaled error.

    ``actual`` has rates plus ``pa`` (actual PA in the window); only players with
    ``pa >= min_pa`` that every projection covers are kept.
    """
    players = actual.index[actual["pa"] >= min_pa]
    players = players.intersection(_common_index([*projections.values(), actual]))
    parts = []
    for name, proj in projections.items():
        for s in TARGETS:
            projected = proj.loc[players, s].to_numpy()
            truth = actual.loc[players, s].to_numpy()
            parts.append(
                pd.DataFrame(
                    {
                        "player_id": players,
                        "system": name,
                        "stat": s,
                        "projected": projected,
                        "actual": truth,
                        "abs_err": abs(projected - truth) * SCALE[s],
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def mae_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Rows = systems (in first-seen order), columns = stats, plus ``n`` players."""
    order = list(dict.fromkeys(scored["system"]))
    table = scored.pivot_table(index="system", columns="stat", values="abs_err", aggfunc="mean")
    table["n"] = scored.groupby("system")["player_id"].nunique()
    return table.loc[order, [*TARGETS, "n"]]


def score(
    projections: dict[str, pd.DataFrame], actual: pd.DataFrame, min_pa: float
) -> pd.DataFrame:
    """MAE table: one row per projection, one column per stat, plus ``n`` players."""
    return mae_table(scored_players(projections, actual, min_pa))


def paired_bootstrap(
    scored: pd.DataFrame,
    a: str,
    b: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """MAE(a) - MAE(b) per stat, with a 95% interval from resampling players.

    Negative = ``a`` is better. Paired: each resample draws players, and both systems
    are scored on the same draw, so player-to-player luck cancels. An interval that
    crosses 0 means the data can't tell the two apart.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for s in TARGETS:
        rows = scored[scored["stat"] == s]
        wide = rows.pivot_table(index="player_id", columns="system", values="abs_err")
        diff = (wide[a] - wide[b]).to_numpy()
        draws = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        boot = diff[draws].mean(axis=1)
        out[s] = {
            "diff": diff.mean(),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
        }
    return pd.DataFrame(out).T


def spread(scored: pd.DataFrame) -> pd.DataFrame:
    """How spread out each system's projections are: SD across players, same scale as MAE."""
    scale = scored["stat"].map(SCALE)
    sd = (scored.assign(v=scored["projected"] * scale)).pivot_table(
        index="system", columns="stat", values="v", aggfunc="std"
    )
    actual = (scored.assign(v=scored["actual"] * scale)).drop_duplicates(["player_id", "stat"])
    sd.loc["(actual)"] = actual.groupby("stat")["v"].std()
    order = [*dict.fromkeys(scored["system"]), "(actual)"]
    return sd.loc[order, list(TARGETS)]
