"""Blending ours with FanGraphs, per stat: ``w x ours + (1 - w) x FanGraphs``, and
picking ``w`` by MSE on actuals. Shared by pitchers (``pitcher_ros/final.py``) and
hitters (``hitter_ros/final.py``); each brings its own :class:`Scoring` (its stats and
how MSE weighs their errors).

A scored frame is ``pitcher_ros.evaluate`` / ``hitter_ros.evaluate``'s shape: one row per
player, unit (season / snapshot), system and stat, with ``projected``, ``actual`` and
the volume columns its ``squared_error`` reads. The fit uses systems ``ours`` and
``fg_blend``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fantasy_baseball.projection_eval import unit_key

# The weights a fit picks from.
STEP = 0.05
GRID = tuple(round(w, 2) for w in np.arange(0, 1.0001, STEP))
OURS, FG = "ours", "fg_blend"


@dataclass(frozen=True)
class Scoring:
    """How one player type is scored: its ``targets`` and ``squared_error(scored,
    keep)``, MSE's numerator ``sq`` and weight ``w`` per row (MSE = sum(sq) / sum(w);
    ``sq`` is a per-row weight times the squared rate miss)."""

    targets: Sequence[str]
    squared_error: Callable[[pd.DataFrame, list[str]], pd.DataFrame]

    def error_weight(self, rows: pd.DataFrame) -> np.ndarray:
        """Each row's weight on its squared rate miss in ``sq``: ``sq`` at a miss of
        exactly 1, so it is ``squared_error``'s own weighting, never a copy of it."""
        unit = rows.assign(projected=rows["actual"].astype(float) + 1.0)
        weight: np.ndarray = self.squared_error(unit, ["stat"])["sq"].to_numpy(float)
        return weight


def blend_rates(
    ours: pd.DataFrame, fg: pd.DataFrame, weights: Mapping[str, float], targets: Sequence[str]
) -> pd.DataFrame:
    """Per stat, ``weights[stat] x ours + (1 - weights[stat]) x fg``. Both are rates
    indexed by ``player_id`` with every stat in ``targets``. Every player in either
    comes back: one only one side projects (or projects a stat for) gets that side's
    rate."""
    missing = sorted(set(targets) - set(weights))
    if missing:
        raise ValueError(f"no weight for {missing}")
    bad = {s: w for s, w in weights.items() if not 0 <= w <= 1}
    if bad:
        raise ValueError(f"weights must be 0-1: {bad}")
    index = ours.index.union(fg.index)
    a = ours[list(targets)].reindex(index).astype(float)
    b = fg[list(targets)].reindex(index).astype(float)
    w = pd.Series(weights)[list(targets)]
    mixed = a * w + b * (1 - w)
    return mixed.fillna(a).fillna(b)


def wide(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per scored unit and stat: ``ours`` and ``fg_blend`` side by side, with
    the actual, volume columns and tags of the scored rows."""
    absent = {OURS, FG} - set(scored["system"])
    if absent:
        raise ValueError(f"scored frame has no {sorted(absent)} rows")
    key = [*unit_key(scored), "stat"]
    pair = scored[scored["system"].isin([OURS, FG])]
    if pair.duplicated([*key, "system"]).any():
        raise ValueError(f"more than one row per {key} and system")
    side = pair.pivot(index=key, columns="system", values="projected")
    lacking = side[[OURS, FG]].isna().sum()
    if lacking.any():
        raise ValueError(
            f"scored units with no projection from {lacking[lacking > 0].to_dict()}; "
            "both must score the same players"
        )
    rest = pair[pair["system"] == OURS].set_index(key).drop(columns=["system", "projected"])
    return side.join(rest, how="inner").reset_index()


def blended_scored(scored: pd.DataFrame, weights: pd.Series, name: str = "final") -> pd.DataFrame:
    """``scored`` plus rows for the blend, system ``name``: per row of ``ours``,
    ``w x ours + (1 - w) x fg_blend`` with ``w`` from ``weights``, indexed like the
    scored frame's unit key plus ``stat`` (a MultiIndex), or by ``stat`` alone. Ready
    for an ``mse_bootstrap`` against ``ours`` or ``fg_blend``."""
    rows_wide = wide(scored)
    key = list(weights.index.names)
    w = rows_wide.join(weights.rename("w"), on=key)["w"]
    if w.isna().any():
        raise ValueError(f"no weight for some rows (weights keyed by {key})")
    rows = rows_wide.assign(system=name, projected=w * rows_wide[OURS] + (1 - w) * rows_wide[FG])
    return pd.concat([scored, rows[scored.columns]], ignore_index=True)


def mse_at(rows: pd.DataFrame, w: float, scoring: Scoring) -> float:
    """MSE of ``w x ours + (1 - w) x fg_blend`` over ``wide`` rows."""
    s = rows.assign(system="blend", projected=w * rows[OURS] + (1 - w) * rows[FG])
    sq = scoring.squared_error(s, ["stat"])
    return float(sq["sq"].sum() / sq["w"].sum())


def best_weights(scored: pd.DataFrame, scoring: Scoring) -> pd.Series:
    """Per stat, the weight on ours in ``GRID`` with the lowest MSE over every scored
    row (``scoring.squared_error``). Ties go to the larger weight on ours.

    MSE is a parabola in the weight: with ``d = ours - fg``, ``e = actual - fg`` and
    MSE's error weights ``W``, it is lowest at ``sum(W d e) / sum(W d^2)``. So only the
    two grid points around that (clipped to 0-1) are scored."""
    rows_wide = wide(scored)
    out = {}
    for stat, rows in rows_wide.groupby("stat", sort=False):
        d = (rows[OURS] - rows[FG]).to_numpy(float)
        e = (rows["actual"] - rows[FG]).to_numpy(float)
        weight = scoring.error_weight(rows)
        denom = float((weight * d * d).sum())
        if denom == 0:  # ours and FanGraphs agree on every row: every weight ties
            out[stat] = GRID[-1]
            continue
        best = min(max(float((weight * d * e).sum()) / denom, 0.0), 1.0)
        lo = int(np.floor(best / STEP + 1e-9))
        near = [GRID[i] for i in (lo, lo + 1) if 0 <= i < len(GRID)]
        scores = {w: mse_at(rows, w, scoring) for w in near}
        out[stat] = min(near, key=lambda w: (scores[w], -w))
    return pd.Series(out, name="w").rename_axis("stat")


def held_out_weights(scored: pd.DataFrame, by: pd.Series, scoring: Scoring) -> pd.Series:
    """Per group of ``by`` (aligned with ``scored``: a season, a half of a season) and
    stat, the weights ``best_weights`` picks on the **other** groups' rows -- so scoring
    the blend with them never picks a weight on the rows it is scored on. Indexed by
    ``by``'s name and ``stat``."""
    if not by.index.equals(scored.index):
        raise ValueError("by must be aligned with scored")
    if by.isna().any():
        raise ValueError("by is blank on some rows; every row needs a group")
    groups = by.unique()
    if len(groups) < 2:
        raise ValueError("need two or more groups to hold one out")
    parts = {g: best_weights(scored[by != g], scoring) for g in groups}
    out = pd.concat(parts, names=[by.name if by.name is not None else "group", "stat"])
    return out.rename("w")
