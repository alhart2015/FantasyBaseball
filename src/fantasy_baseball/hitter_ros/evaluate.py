"""Score projections against what happened.

The **main score is MSE** (user call, #433; it replaced gap-weighted pairwise, #424):
squared error pooled over every scored player-unit, lower is better.

* R, HR, RBI, SB: on season totals over the PA the hitter actually got,
  ``(pa x (projected rate - actual rate))^2`` -- "how many R off, squared".
* AVG: in points, ``(1000 x (projected - actual))^2``, weighted by PA (the scored
  frames carry PA, not AB).

Pairwise accuracy and raw error (#399's original bar) are reported next to it.

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
  pairs that tied in reality are skipped). The gap-weighted version counts each pair by the actual gap, so near-ties matter less. Neither cares how far
  off the projected numbers were, only whether the order was right.
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
from fantasy_baseball.sgp.player_value import (
    REPLACEMENT_AVG,
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
)
from fantasy_baseball.utils.constants import DEFAULT_TEAM_AB, Category

logger = logging.getLogger(__name__)

SCALE = {"r": 600.0, "hr": 600.0, "rbi": 600.0, "sb": 600.0, "avg": 1000.0}

# A full-season (preseason) projection has everyday players near 600+ PA. A file whose
# biggest projection is under this is a rest-of-season export saved in the wrong place.
PRESEASON_MIN_TOP_PA = 450


def load_fangraphs_hitters(path: Path) -> pd.DataFrame:
    """One FanGraphs hitter CSV -> rates (+ pa, ab) indexed by MLBAM id."""
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
    rates["ab"] = counts["ab"].astype(float)
    rates.index = pd.Index(raw["MLBAMID"], name="player_id")
    return rates


_COUNTING = (("r", Category.R), ("hr", Category.HR), ("rbi", Category.RBI), ("sb", Category.SB))


def fantasy_value(projection: pd.DataFrame, denoms: dict[Category, float]) -> pd.Series:
    """Roto value (SGP) of a projection, one per player: R, HR, RBI and SB from rate x
    PA, AVG as marginal hits over a replacement hitter on ``ab`` at-bats
    (``sgp.player_value``). ``projection``: rates plus ``pa`` and ``ab``, as from
    :func:`load_fangraphs_hitters`."""
    pa = projection["pa"].astype(float)
    value = sum(
        (
            calculate_counting_sgp(projection[s].astype(float) * pa, denoms[cat])
            for s, cat in _COUNTING
        ),
        start=pd.Series(0.0, index=projection.index),
    )
    # The sgp helpers are plain arithmetic, so they work elementwise on Series too.
    avg = pd.Series(
        calculate_hitting_rate_sgp(
            player_avg=projection["avg"].astype(float),
            player_ab=projection["ab"].astype(float),
            replacement_avg=REPLACEMENT_AVG,
            sgp_denominator=denoms[Category.AVG],
            team_ab=DEFAULT_TEAM_AB,
        ),
        index=projection.index,
    )
    return (value + avg.fillna(0.0)).rename("value")


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
    credit = _pair_credit(projected, np.sign(gap[keep]), i[keep], j[keep])
    return float(np.average(credit, weights=np.abs(gap[keep]) if weighted else None))


def _pair_credit(
    projected: np.ndarray, gap_sign: np.ndarray, i: np.ndarray, j: np.ndarray
) -> np.ndarray:
    """1 where the projection orders pair (i, j) like the actuals, 0.5 on a projected
    tie, 0 where it is backwards."""
    agree = np.sign(projected[i] - projected[j]) * gap_sign
    credit: np.ndarray = np.where(agree == 0, 0.5, (agree > 0).astype(float))
    return credit


# Bootstrap draws handled per matrix product (memory: draws x player pairs floats).
_BOOT_CHUNK = 25


# A difference counts as real when ``sure`` reaches this (#457).
SURE_BAR = 0.95


def sure(draws: np.ndarray) -> float:
    """How sure bootstrap ``draws`` (a minus b, positive = a better) are that a and b
    really differ, signed by which is better: +0.95 = 95% sure a is better, -0.90 = 90%
    sure a is worse, 0 = no lean. It is 1 minus the two-sided p-value: the widest
    middle slice of the draws that leaves out 0 holds this share of them, so +/-0.95 is
    exactly where the 95% interval stops crossing 0. Read as is; no halving. Ties at 0
    count half to each side. It counts only the luck of which players happened to be
    scored, not seed-to-seed or season-to-season swings."""
    if not len(draws):
        return float("nan")
    a_better = float(np.mean((draws > 0) + 0.5 * (draws == 0)))
    return 2 * a_better - 1


def pairwise_bootstrap(
    scored: pd.DataFrame,
    a: str,
    b: str,
    *,
    weighted: bool = True,
    n_boot: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Pairwise accuracy (%) of ``a`` minus ``b`` per stat, with a 95% interval and
    ``sure`` (how sure ``a`` orders better (+) or worse (-)), from resampling players.
    Positive = ``a`` orders better. Paired: both systems are scored on each draw.

    ``scored`` may stack several seasons or snapshots (``UNIT_COLS``): the score is then
    the mean over them of each one's score, as in ``order_table`` (pairs never cross
    them; one with no untied pair is left out, and so, in a draw, is one where too few
    of its players were drawn to form an untied pair). Each draw resamples players once
    for all of them, so a player scored in several overlapping snapshots is one draw, not
    several independent ones -- otherwise the interval would be too narrow. Both systems
    must be scored in every season or snapshot (``order_table`` would average each over
    its own, a different set).

    A draw that picks player i ``c_i`` times and j ``c_j`` times holds their pair
    ``c_i * c_j`` times (a player paired with his own copy tied in reality, so it is
    skipped). So each draw's accuracy is a count-weighted average over the distinct
    pairs, which is computed for many draws at once as a matrix product."""
    rng = np.random.default_rng(seed)
    units = [c for c in UNIT_COLS if c in scored.columns]
    out = {}
    for s in TARGETS:
        rows = scored[scored["stat"] == s]
        players = pd.Index(sorted(rows["player_id"].unique()))
        n = len(players)
        counts = np.vstack(
            [np.ones(n), *(np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot))]
        )
        total = np.zeros(len(counts))
        n_units = np.zeros(len(counts))  # per draw: the units with an untied drawn pair
        for key, g in rows.groupby(units, sort=False) if units else [(None, rows)]:
            wide = g.pivot_table(index="player_id", columns="system", values="projected")
            if not {a, b} <= set(wide.columns):
                raise ValueError(f"{s} at {key}: both {a} and {b} must be scored in every unit")
            actual = g.drop_duplicates("player_id").set_index("player_id")["actual"]
            actual = actual.loc[wide.index].to_numpy()
            pid = players.get_indexer(wide.index)
            i, j = np.triu_indices(len(actual), k=1)
            gap = actual[i] - actual[j]
            keep = gap != 0
            if not keep.any():
                continue
            i, j, gap = i[keep], j[keep], gap[keep]
            weight = np.abs(gap) if weighted else np.ones_like(gap)
            # Per pair: weight x (credit of a - credit of b).
            edge = weight * (
                _pair_credit(wide[a].to_numpy(), np.sign(gap), i, j)
                - _pair_credit(wide[b].to_numpy(), np.sign(gap), i, j)
            )
            pi, pj = pid[i], pid[j]
            diffs = []
            for start in range(0, len(counts), _BOOT_CHUNK):
                c = counts[start : start + _BOOT_CHUNK]
                mult = c[:, pi] * c[:, pj]
                with np.errstate(invalid="ignore", divide="ignore"):
                    diffs.append(100 * (mult @ edge) / (mult @ weight))
            unit_diff = np.concatenate(diffs)  # NaN: a draw with no untied pair here
            ok = ~np.isnan(unit_diff)
            total[ok] += unit_diff[ok]
            n_units += ok
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(n_units > 0, total / n_units, np.nan)
        boot = d[1:][~np.isnan(d[1:])]
        out[s] = {
            "diff": float(d[0]),
            "lo": float(np.percentile(boot, 2.5)) if len(boot) else float("nan"),
            "hi": float(np.percentile(boot, 97.5)) if len(boot) else float("nan"),
            "sure": sure(boot),
        }
    return pd.DataFrame(out).T


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


MSE_COLUMNS = ("projected", "actual", "pa")


def has_mse_columns(scored: pd.DataFrame) -> bool:
    """Whether ``scored`` carries what MSE needs (very old test frames don't)."""
    return set(MSE_COLUMNS) <= set(scored.columns)


def squared_error(scored: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """``scored[keep]`` plus, per row, MSE's numerator ``sq`` and weight ``w`` (see module
    doc): R/HR/RBI/SB ``(pa x err)^2`` with weight 1; AVG ``pa x (1000 x err)^2`` with
    weight ``pa``. MSE over any set of rows = sum(sq) / sum(w). Rows keep their order on
    a fresh index (a stacked frame can repeat index labels)."""
    err = (scored["projected"] - scored["actual"]).to_numpy(dtype=float)
    pa = scored["pa"].to_numpy(dtype=float)
    is_avg = (scored["stat"] == "avg").to_numpy()
    return (
        scored[keep]
        .reset_index(drop=True)
        .assign(
            sq=np.where(is_avg, pa * (SCALE["avg"] * err) ** 2, (pa * err) ** 2),
            w=np.where(is_avg, pa, 1.0),
        )
    )


def mse_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Systems (in first-seen order) x stats: MSE pooled over every scored player-unit,
    so a season or snapshot with more hitters weighs more."""
    parts = squared_error(scored, ["system", "stat"])
    sums = parts.groupby(["system", "stat"])[["sq", "w"]].sum()
    table = (sums["sq"] / sums["w"]).unstack("stat")
    order = list(dict.fromkeys(scored["system"]))
    return table.reindex(index=order, columns=list(TARGETS))


def mse_bootstrap(
    scored: pd.DataFrame, a: str, b: str, *, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """MSE of ``a`` minus ``b`` per stat (negative = ``a`` better), with ``a``'s and
    ``b``'s own MSE, a 95% interval and ``sure`` (how sure ``a``'s MSE is lower (+) or
    higher (-)), from resampling players. Paired: both are scored on each draw. As in
    ``pairwise_bootstrap``, each draw redraws a player once for every season or snapshot
    he is in, so overlapping snapshots aren't independent evidence. Both systems must be
    scored on every player-unit either is."""
    rng = np.random.default_rng(seed)
    key = unit_key(scored)
    rows = scored[scored["system"].isin([a, b])]
    parts = squared_error(rows, [*key, "system", "stat"])
    out = {}
    for s in TARGETS:
        g = parts[parts["stat"] == s]
        wide = g.set_index([*key, "system"])[["sq", "w"]].unstack("system")
        if {a, b} - set(g["system"]) or wide.isna().to_numpy().any():
            raise ValueError(f"{s}: {a} and {b} must be scored on the same player-units")
        per = wide.groupby(level="player_id").sum()
        n = len(per)
        counts = np.vstack(
            [np.ones(n), *(np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot))]
        )
        mse = {
            x: (counts @ per[("sq", x)].to_numpy()) / (counts @ per[("w", x)].to_numpy())
            for x in (a, b)
        }
        d = mse[a] - mse[b]
        boot = d[1:]
        out[s] = {
            "diff": float(d[0]),
            "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)),
            "sure": sure(-boot),
            "a": float(mse[a][0]),
            "b": float(mse[b][0]),
        }
    return pd.DataFrame(out).T


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


def order_table(
    scored: pd.DataFrame, metric: str, per_unit: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Systems x stats: ``metric`` (one of ``ORDER_METRICS``) averaged over the seasons or
    snapshots in ``scored``, each counting once. Pairwise accuracy is in percent.
    ``per_unit``: ``order_scores(scored)`` if already computed (it is O(players^2))."""
    if metric not in ORDER_METRICS:
        raise ValueError(f"unknown order metric {metric!r}")
    if per_unit is None:
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
    """MAE(a) - MAE(b) per stat, with a 95% interval and ``sure`` (how sure ``a``'s MAE
    is lower (+) or higher (-)), from resampling scored player-units.
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
            "sure": sure(-boot),
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
