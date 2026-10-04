"""Score projections against what happened: per-stat raw error (#399's bar), plus
scores that ignore the league's level (#424).

Raw error is the mean absolute error (MAE) of the projected rate vs. the actual rate,
over a fixed set of players that every compared projection covers:

* R, HR, RBI, SB: per-PA rate error x 600 -- "how many R/HR/RBI/SB off over 600 PA".
* AVG: batting-average error x 1000 -- in points (e.g. 25 = .025).

A roto league is relative: if the whole league's offense drops, every team drops
together. So three more scores ask "is X better than Y", not "25 or 30 HR":

* **Level-free MAE** (``lf_err``): each projection is rescaled so its PA-weighted mean
  over the scored players equals the actuals' mean, then MAE as above. A projection
  that is right about every player except for one league-wide factor scores 0.
* **Pairwise order accuracy**: over every pair of scored players, the share the
  projection orders the same way as what happened (a projected tie gets half credit;
  pairs that tied in reality are skipped). The weighted version counts each pair by
  the actual gap, so near-ties matter less.
* **Rank correlation** (Spearman) between projected and actual rates.

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
    """One row per (player, system, stat): projected and actual rate, actual PA, and the
    scaled raw and level-free errors.

    ``actual`` has rates plus ``pa`` (actual PA in the window); only players with
    ``pa >= min_pa`` that every projection covers are kept. Call it once per season or
    snapshot: the level-free error rescales over exactly these players.
    """
    players = actual.index[actual["pa"] >= min_pa]
    players = players.intersection(_common_index([*projections.values(), actual]))
    pa = actual.loc[players, "pa"].to_numpy(dtype=float)
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
                        "pa": pa,
                        "abs_err": abs(projected - truth) * SCALE[s],
                        "lf_err": level_free_error(projected, truth, pa) * SCALE[s],
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def level_free_error(projected: np.ndarray, actual: np.ndarray, pa: np.ndarray) -> np.ndarray:
    """|projected - actual| after scaling ``projected`` so its PA-weighted mean equals the
    actuals': both divided by their own mean, then put back in the actuals' units."""
    if len(projected) == 0:
        return np.zeros(0)
    proj_mean = np.average(projected, weights=pa)
    level = np.average(actual, weights=pa) / proj_mean if proj_mean else 1.0
    err: np.ndarray = np.abs(projected * level - actual)
    return err


def pairwise_accuracy(projected: np.ndarray, actual: np.ndarray, *, weighted: bool) -> float:
    """Share of player pairs the projection orders as the actuals did (see module doc)."""
    i, j = np.triu_indices(len(projected), k=1)
    gap = actual[i] - actual[j]
    keep = gap != 0
    if not keep.any():
        return float("nan")
    agree = np.sign(projected[i] - projected[j])[keep] * np.sign(gap[keep])
    credit = np.where(agree == 0, 0.5, (agree > 0).astype(float))
    return float(np.average(credit, weights=np.abs(gap[keep]) if weighted else None))


def spearman(projected: np.ndarray, actual: np.ndarray) -> float:
    """Rank correlation; NaN when either side is constant (e.g. ``league_avg``)."""
    p, a = pd.Series(projected).rank(), pd.Series(actual).rank()
    if p.nunique() < 2 or a.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(p, a)[0, 1])


# A scored frame may stack several seasons or snapshots; a player is scored once per
# unit, so these columns (when present) are part of what identifies a scored row.
UNIT_COLS = ("season", "snapshot")


def unit_key(scored: pd.DataFrame) -> list[str]:
    """Columns that identify one scored player-unit: player plus season/snapshot if present."""
    return ["player_id", *(c for c in UNIT_COLS if c in scored.columns)]


def mae_table(scored: pd.DataFrame, value: str = "abs_err") -> pd.DataFrame:
    """Rows = systems (in first-seen order), columns = stats, plus ``n`` scored player-units.
    ``value``: ``abs_err`` (raw MAE) or ``lf_err`` (level-free MAE)."""
    order = list(dict.fromkeys(scored["system"]))
    table = scored.pivot_table(index="system", columns="stat", values=value, aggfunc="mean")
    units = scored.drop_duplicates([*unit_key(scored), "system"])
    table["n"] = units.groupby("system").size()
    return table.loc[order, [*TARGETS, "n"]]


ORDER_METRICS = ("pairwise", "pairwise_w", "spearman")


def order_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Pairwise accuracy (plain and gap-weighted) and Spearman, one row per season or
    snapshot x system x stat. Pairs are only formed within one season or snapshot."""
    keys = [*(c for c in UNIT_COLS if c in scored.columns), "system", "stat"]
    rows = []
    for key, g in scored.groupby(keys, sort=False):
        p, a = g["projected"].to_numpy(), g["actual"].to_numpy()
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "pairwise": pairwise_accuracy(p, a, weighted=False),
                "pairwise_w": pairwise_accuracy(p, a, weighted=True),
                "spearman": spearman(p, a),
            }
        )
    return pd.DataFrame(rows)


def order_table(scored: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Systems x stats: ``metric`` (one of ``ORDER_METRICS``) averaged over the seasons or
    snapshots in ``scored``, each counting once. Pairwise accuracy is in percent."""
    if metric not in ORDER_METRICS:
        raise ValueError(f"unknown order metric {metric!r}")
    per_unit = order_scores(scored)
    table = per_unit.pivot_table(index="system", columns="stat", values=metric, aggfunc="mean")
    if metric != "spearman":
        table = table * 100
    order = list(dict.fromkeys(scored["system"]))
    return table.reindex(index=order, columns=list(TARGETS))


def paired_bootstrap(
    scored: pd.DataFrame,
    a: str,
    b: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    value: str = "abs_err",
) -> pd.DataFrame:
    """MAE(a) - MAE(b) per stat, with a 95% interval from resampling scored player-units.
    ``value``: ``abs_err`` (raw MAE) or ``lf_err`` (level-free MAE).

    Negative = ``a`` is better. Paired: each resample draws player-units (a player in a
    given season or snapshot), and both systems are scored on the same draw, so
    player-to-player luck cancels. An interval that crosses 0 means the data can't tell
    the two apart.
    """
    rng = np.random.default_rng(seed)
    key = unit_key(scored)
    out = {}
    for s in TARGETS:
        rows = scored[scored["stat"] == s]
        wide = rows.pivot_table(index=key, columns="system", values=value)
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
    """How spread out each system's projections are: SD across scored player-units, same
    scale as MAE. The ``(actual)`` row is the SD of the outcomes over the same units."""
    scaled = scored.assign(
        projected=scored["projected"] * scored["stat"].map(SCALE),
        actual=scored["actual"] * scored["stat"].map(SCALE),
    )
    sd = scaled.pivot_table(index="system", columns="stat", values="projected", aggfunc="std")
    outcomes = scaled.drop_duplicates([*unit_key(scored), "stat"])
    sd.loc["(actual)"] = outcomes.groupby("stat")["actual"].std()
    order = [*dict.fromkeys(scored["system"]), "(actual)"]
    return sd.loc[order, list(TARGETS)]
