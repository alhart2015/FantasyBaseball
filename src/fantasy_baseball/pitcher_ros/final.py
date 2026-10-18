"""The final pitcher projection: ours blended with FanGraphs (#499).

``ours`` (the pitcher net, ``model``) stays independent of FanGraphs: no FanGraphs
stats, shares or closer calls go into it, so it can be improved and judged on its own.
This is the separate step after it: per stat, ``w x ours + (1 - w) x FanGraphs``, where
FanGraphs is the plain average of its systems' rates. The weights differ before the
season and during it (``WEIGHTS``), because what FanGraphs knows that we don't differs:
before the season it is mostly the closer calls they take from news (a perfect "who
opens as closer" flag only tied FanGraphs' saves: #499), during it our net reads the
season so far better.

``WEIGHTS`` come from ``scripts/fit_pitcher_blend.py`` on run 035-saves-net, picked by
MSE over every scored pitcher on a 0.05 grid, with FanGraphs = the average of
``SYSTEMS[span]``:

* Preseason (2022-2026 pooled; Steamer + ZiPS, the systems on file every season):
  picked leaving each season out, the weights moved at most 0.10 from these, and the
  blend beat both ours and FanGraphs on K, W, SV and ERA over all pitchers
  (fantasy-relevant: K and SV beat both; ERA and WHIP tied ours).
* Mid-season (2026's 13 snapshots only, the one season with FanGraphs rest-of-season
  files; ATC + OOPSY + Steamer): **not validated**. They are fitted on the rows they
  are scored on. No honest held-out check exists yet: every snapshot's actuals run to
  the end of the same season, so any split of one season's snapshots shares games.
  In-sample, SV moved most (0.55); the rest are near all ours. Provisional: refit with
  a season held out once a second season of snapshots is stored.

The backtest scores only pitchers every system projects, so the weights were picked
on those. ``final_rates`` blends every pitcher anyone projects: FanGraphs is the
average of the systems that project him (``fg_average``), and a pitcher only one side
projects gets that side's rates.

Refit each season (the script prints the fitted weights next to these, and warns when
the run's FanGraphs systems differ from ``SYSTEMS``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from fantasy_baseball.pitcher_ros.evaluate import error_weight, squared_error
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.projection_eval import unit_key

PRESEASON = "preseason"
MIDSEASON = "midseason"
# Weight on ours per stat (1 = all ours, 0 = all FanGraphs).
WEIGHTS: dict[str, dict[str, float]] = {
    PRESEASON: {"k": 0.50, "w": 0.70, "sv": 0.30, "era": 0.50, "whip": 0.75},
    MIDSEASON: {"k": 0.85, "w": 1.00, "sv": 0.55, "era": 0.90, "whip": 0.95},
}
# The FanGraphs systems each span's WEIGHTS were fitted with (their average).
SYSTEMS: dict[str, tuple[str, ...]] = {
    PRESEASON: ("steamer", "zips"),
    MIDSEASON: ("atc", "oopsy", "steamer"),
}
# The weights a fit picks from.
STEP = 0.05
GRID = tuple(round(w, 2) for w in np.arange(0, 1.0001, STEP))
OURS, FG = "ours", "fg_blend"


def blend_rates(ours: pd.DataFrame, fg: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    """Per stat, ``weights[stat] x ours + (1 - weights[stat]) x fg``. Both are rates
    indexed by ``player_id`` with every stat in ``TARGETS`` (K, W, SV per batter faced;
    ERA, WHIP). Every pitcher in either comes back: one only side projects (or projects
    a stat for) gets that side's rate."""
    missing = sorted(set(TARGETS) - set(weights))
    if missing:
        raise ValueError(f"no weight for {missing}")
    bad = {s: w for s, w in weights.items() if not 0 <= w <= 1}
    if bad:
        raise ValueError(f"weights must be 0-1: {bad}")
    index = ours.index.union(fg.index)
    a = ours[list(TARGETS)].reindex(index).astype(float)
    b = fg[list(TARGETS)].reindex(index).astype(float)
    w = pd.Series(weights)[list(TARGETS)]
    mixed = a * w + b * (1 - w)
    return mixed.fillna(a).fillna(b)


def fg_average(systems: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per pitcher and stat, the plain average of the systems that project it (rates
    indexed by ``player_id``, every stat in ``TARGETS``). Unlike ``evaluate.blend``,
    which keeps only the pitchers every system covers, a pitcher one system misses
    still gets the others' average."""
    if not systems:
        raise ValueError("no FanGraphs systems")
    stacked = pd.concat([f[list(TARGETS)].astype(float) for f in systems.values()])
    return stacked.groupby(level=0).mean()


def final_rates(ours: pd.DataFrame, systems: Mapping[str, pd.DataFrame], span: str) -> pd.DataFrame:
    """The final projection: ``blend_rates`` of ours with ``fg_average(systems)`` at
    ``WEIGHTS[span]``. ``systems`` must be ``SYSTEMS[span]``, the ones the weights were
    fitted with: a different FanGraphs average needs its own fit."""
    if span not in WEIGHTS:
        raise ValueError(f"span must be one of {sorted(WEIGHTS)}, not {span!r}")
    if set(systems) != set(SYSTEMS[span]):
        raise ValueError(
            f"{span} weights were fitted with FanGraphs = {list(SYSTEMS[span])}, "
            f"got {sorted(systems)}; refit (scripts/fit_pitcher_blend.py) first"
        )
    return blend_rates(ours, fg_average(systems), WEIGHTS[span])


def _wide(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per scored unit and stat: ``ours`` and ``fg_blend`` side by side, with
    the actual, batters faced, innings and tags of the scored rows."""
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
            "both must score the same pitchers"
        )
    rest = pair[pair["system"] == OURS].set_index(key).drop(columns=["system", "projected"])
    return side.join(rest, how="inner").reset_index()


def blended_scored(scored: pd.DataFrame, weights: pd.Series, name: str = "final") -> pd.DataFrame:
    """``scored`` plus rows for the blend, system ``name``: per row of ``ours``,
    ``w x ours + (1 - w) x fg_blend`` with ``w`` from ``weights``, indexed like the
    scored frame's unit key plus ``stat`` (a MultiIndex), or by ``stat`` alone. Ready
    for ``evaluate.mse_bootstrap`` against ``ours`` or ``fg_blend``."""
    wide = _wide(scored)
    key = list(weights.index.names)
    w = wide.join(weights.rename("w"), on=key)["w"]
    if w.isna().any():
        raise ValueError(f"no weight for some rows (weights keyed by {key})")
    rows = wide.assign(system=name, projected=w * wide[OURS] + (1 - w) * wide[FG])
    return pd.concat([scored, rows[scored.columns]], ignore_index=True)


def _mse(rows: pd.DataFrame, w: float) -> float:
    s = rows.assign(system="blend", projected=w * rows[OURS] + (1 - w) * rows[FG])
    sq = squared_error(s, ["stat"])
    return float(sq["sq"].sum() / sq["w"].sum())


def best_weights(scored: pd.DataFrame) -> pd.Series:
    """Per stat, the weight on ours in ``GRID`` with the lowest MSE over every scored
    row (``evaluate.squared_error``: K/W/SV in season totals, ERA/WHIP innings-weighted).
    Ties go to the larger weight on ours.

    MSE is a parabola in the weight: with ``d = ours - fg``, ``e = actual - fg`` and
    MSE's error weights ``W``, it is lowest at ``sum(W d e) / sum(W d^2)``. So only the
    two grid points around that (clipped to 0-1) are scored."""
    wide = _wide(scored)
    out = {}
    for stat, rows in wide.groupby("stat", sort=False):
        d = (rows[OURS] - rows[FG]).to_numpy(float)
        e = (rows["actual"] - rows[FG]).to_numpy(float)
        weight = error_weight(rows)
        denom = float((weight * d * d).sum())
        if denom == 0:  # ours and FanGraphs agree on every row: every weight ties
            out[stat] = GRID[-1]
            continue
        best = min(max(float((weight * d * e).sum()) / denom, 0.0), 1.0)
        lo = int(np.floor(best / STEP + 1e-9))
        near = [GRID[i] for i in (lo, lo + 1) if 0 <= i < len(GRID)]
        scores = {w: _mse(rows, w) for w in near}
        out[stat] = min(near, key=lambda w: (scores[w], -w))
    return pd.Series(out, name="w").rename_axis("stat")


def held_out_weights(scored: pd.DataFrame, by: pd.Series) -> pd.Series:
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
    parts = {g: best_weights(scored[by != g]) for g in groups}
    out = pd.concat(parts, names=[by.name if by.name is not None else "group", "stat"])
    return out.rename("w")
