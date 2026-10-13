"""Scoring pieces shared by the hitter and pitcher projection backtests
(``hitter_ros``, ``pitcher_ros``): the FanGraphs blend, the paired MSE bootstrap and
how sure it is, and the fantasy-relevant set. Each side passes in its own stats and its
own squared error; what is here must mean the same thing for both.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

# A scored frame may stack several seasons or snapshots; a player is scored once per
# unit, so these columns (when present) are part of what identifies a scored row.
UNIT_COLS = ("season", "snapshot")


def unit_key(scored: pd.DataFrame) -> list[str]:
    """Columns that identify one scored player-unit: player plus season/snapshot if present."""
    return ["player_id", *(c for c in UNIT_COLS if c in scored.columns)]


def player_draws(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(1 + n_boot) x n: how many times each of ``n`` players is drawn. Row 0 is every
    player once (the actual score); each other row draws ``n`` players with replacement.
    Shared by the bootstraps so their intervals mean the same thing."""
    return np.vstack(
        [np.ones(n), *(np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot))]
    )


def sure(draws: np.ndarray) -> float:
    """How sure bootstrap ``draws`` (a minus b, positive = a better) are that a is
    better, from 0 to 1: the share of draws where a comes out ahead. 1 = sure a is
    better, 0 = sure a is worse, 0.5 = a coin flip. Ties at 0 count half. Just the
    number: no cutoff turns it into a verdict. It counts only the luck of which players
    happened to be scored, not seed-to-seed or season-to-season swings."""
    if not len(draws):
        return float("nan")
    return float(np.mean((draws > 0) + 0.5 * (draws == 0)))


def sure_text(value: float) -> str:
    """``evaluate.sure`` in words, on one 0-100 scale: "96% sure better", "50% sure
    better" (a coin flip), "4% sure better" (sure it's worse). A bootstrap can't show
    certainty, so the ends read ">99%" and "<1%"."""
    if pd.isna(value):
        return "n/a"
    if value > 0.99:
        pct = ">99%"
    elif value < 0.01:
        pct = "<1%"
    else:
        pct = f"{round(value * 100)}%"
    return f"{pct} sure better"


def common_index(frames: Iterable[pd.DataFrame], targets: Sequence[str]) -> pd.Index:
    """Players every frame has a value for, on every one of ``targets``."""
    common: pd.Index | None = None
    for f in frames:
        have = f.dropna(subset=list(targets)).index
        common = have if common is None else common.intersection(have)
    assert common is not None, "no projections"
    return common


def blend(systems: dict[str, pd.DataFrame], targets: Sequence[str]) -> pd.DataFrame:
    """Plain average of the systems' ``targets`` over the players they all cover."""
    common = common_index(systems.values(), targets)
    stacked = [s.loc[common, list(targets)] for s in systems.values()]
    return sum(stacked[1:], stacked[0]) / len(stacked)


def mse_bootstrap(
    parts: pd.DataFrame,
    key: list[str],
    a: str,
    b: str,
    targets: Sequence[str],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """MSE of ``a`` minus ``b`` per stat (negative = ``a`` better), with ``a``'s and
    ``b``'s own MSE, a 95% interval and ``sure`` (how sure ``a``'s MSE is lower, 0-1:
    :func:`sure`), from resampling players. Paired: both are scored on each draw.

    ``parts``: one row per ``key`` (player and season/snapshot) x ``system`` x ``stat``
    with MSE's numerator ``sq`` and weight ``w`` (each side's ``squared_error``), so MSE
    over any rows = sum(sq) / sum(w). Each draw redraws a player once for every season
    or snapshot he is in, so overlapping snapshots aren't independent evidence. Both
    systems must be scored on every player-unit either is."""
    rng = np.random.default_rng(seed)
    parts = parts[parts["system"].isin([a, b])]
    # One set of draws for every stat: the same players are redrawn in each.
    players = pd.Index(sorted(parts["player_id"].unique()))
    counts = player_draws(len(players), n_boot, rng)
    out = {}
    for s in targets:
        g = parts[parts["stat"] == s]
        wide = g.set_index([*key, "system"])[["sq", "w"]].unstack("system")
        if {a, b} - set(g["system"]) or wide.isna().to_numpy().any():
            raise ValueError(f"{s}: {a} and {b} must be scored on the same player-units")
        per = wide.groupby(level="player_id").sum().reindex(players, fill_value=0.0)
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


def systems_in_every_unit(scored: pd.DataFrame, unit: str) -> pd.DataFrame:
    """The rows of systems scored in every season (or snapshot, with ``unit``), so their
    scores are over the same units and compare down a column."""
    n_units = scored[unit].nunique()
    counts = scored.groupby("system")[unit].nunique()
    return scored[scored["system"].isin(counts.index[counts == n_units])]


def top_by_value(projected: Iterable[pd.Series], actual: pd.Series, top: int) -> pd.Index:
    """The fantasy-relevant players (#442): the ``top`` by value averaged over the
    ``projected`` systems that value each one, together with the ``top`` by ``actual``
    value over the scored span (so the set isn't picked on the compared systems' own
    output alone)."""
    mean = pd.concat(list(projected), axis=1).mean(axis=1)
    return mean.nlargest(top).index.union(actual.nlargest(top).index)
