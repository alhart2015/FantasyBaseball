"""Age corrections for pitchers (#503): our projected rates and the PT source's innings
both run too high for old pitchers, and the nets barely use the age they're given.

Held out 2022-2026, pitchers with 50+ IP the season before (so no selection on how the
season went): 36+ vs ages 27-35, ERA +0.30 worse than ours (100 sure), WHIP +0.06 (100),
K -4.2 per 180 IP (95), almost all starters (ERA +0.39, K -5.7); and 66% of Steamer's
innings vs 78% (100 sure).

Each correction is a **hinge**: nothing up to a knot age, then a straight line past it.
Per role (starter, reliever: the PT source's GS / G) and per stat, the knot is chosen
by leave-one-season-out loss on the training seasons, "no correction" among the
choices. The fit has an intercept -- the misses every age shares (our K is a little
low across the board, Steamer's innings high) -- but only the hinge part is applied,
so this corrects aging and nothing else.

* **Rates** (:func:`fit_rates`): a rate's miss (actual minus ours: per batter faced
  for K and W, per inning for ERA and WHIP) on the hinge, weighted by batters faced or
  innings. Applied as ours + slope x hinge.
* **Innings** (:func:`fit_innings`): actual / projected IP (0 if he didn't pitch, capped
  at ``RATIO_CAP``) on the hinge, as projected IP x (1 + slope / level x hinge), the
  share of his projection he loses for being that old. **Measured, not applied**: the
  projected IP is FanGraphs' (Steamer, Depth Charts), and FanGraphs' numbers are used
  as published -- corrections go to ours only. It is here for scoring a PT source and
  for a playing-time projection of our own.

Pure and I/O-free; ``scripts/backtest_pitcher_aging.py`` builds the rows and scores the
corrections held out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fantasy_baseball.pt_model.range_specs import STARTER_SHARE

ROLES = ("starter", "reliever")
# The rates that can be corrected, and what each is weighted by.
RATE_WEIGHT = {"k": "a_bf", "w": "a_bf", "era": "a_ip", "whip": "a_ip"}
# Which (role, rate) are corrected. Held out 2022-2026 (each season's correction fit
# on the others; scripts/backtest_pitcher_aging.py), weighted squared error vs ours,
# all / 36+ (how sure better): starters ERA -0.75% (90) / -7.0% (89), WHIP -1.9% (98) /
# -20.5% (98), W -0.4% (76) / -5.0% (79), K -0.0% (51) / -0.3% (52) -- even, but it
# moves the 36+ K miss (-8.0 per 180 IP) toward the others' (-6.2) on a gap 95 sure
# real; the 36+ ERA miss goes +0.31 -> +0.05. Relievers' weren't kept: K +0.6% (4) /
# +11.8% (0), W +0.2% (24), ERA and WHIP within 0.1% overall.
# Innings (measured on Steamer's IP, not applied -- see the module docstring): starters
# IP -2.4% (100) / -16.9% (100), relievers -0.9% (100) / -19.0% (100); 36+ starters'
# share of projected IP 0.69 -> 0.77 (ages 27-35: 0.74-0.82).
CORRECTED = frozenset({("starter", "k"), ("starter", "w"), ("starter", "era"), ("starter", "whip")})
KNOTS: tuple[float | None, ...] = (None, 32.0, 34.0, 36.0)
# An actual / projected IP ratio counts up to this (a reliever thrown into the rotation
# can pitch 5x his projection).
RATIO_CAP = 1.5
# A projection needs this many innings for its ratio to be read.
MIN_PROJECTED_IP = 20.0


# Past this share of the fit's ages a hinge stops growing: few pitchers pitch at 42+,
# and a straight line would give a 44-year-old a full run of ERA.
TOP_AGE_QUANTILE = 0.95


@dataclass(frozen=True)
class Hinge:
    """``slope`` x (min(age, ``top``) - ``knot``, at least 0) -- none when ``knot`` is
    None -- with the ``level`` (intercept) the fit shared with every age. ``top``: the
    oldest age it grows to (``TOP_AGE_QUANTILE`` of the fit's ages past the knot)."""

    knot: float | None
    slope: float
    level: float
    top: float = np.inf

    def of(self, age: np.ndarray) -> np.ndarray:
        a = np.asarray(age, dtype=float)
        if self.knot is None:
            return np.zeros_like(a)
        return np.asarray(self.slope * np.maximum(0.0, np.minimum(a, self.top) - self.knot))


def role(gs_share: pd.Series) -> pd.Series:
    """Each pitcher's role from his GS / G (NaN: none)."""
    out = pd.Series(
        np.where(gs_share >= STARTER_SHARE, "starter", "reliever"), index=gs_share.index
    )
    return out.where(gs_share.notna())


def _fit_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted least squares y = level + slope x; (slope, level)."""
    wsum = w.sum()
    xm, ym = (w * x).sum() / wsum, (w * y).sum() / wsum
    var = (w * (x - xm) ** 2).sum()
    slope = float((w * (x - xm) * (y - ym)).sum() / var) if var > 0 else 0.0
    return slope, float(ym - slope * xm)


def _fit_hinge(age: np.ndarray, y: np.ndarray, w: np.ndarray, knot: float | None) -> Hinge:
    if knot is None:
        return Hinge(None, 0.0, float(np.average(y, weights=w)))
    # The slope is read off the pitchers past the knot, so its reach is theirs.
    past = age[age > knot]
    top = float(np.quantile(past, TOP_AGE_QUANTILE)) if len(past) else np.inf
    slope, level = _fit_line(np.maximum(0.0, np.minimum(age, top) - knot), y, w)
    return Hinge(knot, slope, level, top)


def _choose(
    age: np.ndarray, y: np.ndarray, w: np.ndarray, season: np.ndarray, knots: Sequence[float | None]
) -> Hinge:
    """The knot with the lowest leave-one-season-out weighted squared error of
    ``y - (level + hinge)``, refit on every season."""
    seasons = np.unique(season)
    # With one season every knot's loss is 0 and "no correction" would win by default.
    if len(seasons) < 2:
        raise ValueError(
            f"choosing a knot needs rows from 2+ seasons, got {len(seasons)} ({len(y)} rows)"
        )
    best, best_loss = None, np.inf
    for knot in knots:
        loss = 0.0
        for s in seasons:
            train, test = season != s, season == s
            if not train.any() or not test.any():
                continue
            h = _fit_hinge(age[train], y[train], w[train], knot)
            loss += float((w[test] * (y[test] - h.level - h.of(age[test])) ** 2).sum())
        if loss < best_loss:
            best, best_loss = knot, loss
    return _fit_hinge(age, y, w, best)


def rate_misses(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each rate's miss, actual minus ours (NaN with no trials)."""
    bf = rows["a_bf"].to_numpy(dtype=float)
    ip = rows["a_ip"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "k": rows["a_k"].to_numpy(dtype=float) / bf - rows["p_k"].to_numpy(dtype=float),
            "w": rows["a_w"].to_numpy(dtype=float) / bf - rows["p_w"].to_numpy(dtype=float),
            "era": 9 * rows["a_er"].to_numpy(dtype=float) / ip
            - rows["p_era"].to_numpy(dtype=float),
            "whip": (rows["a_h"].to_numpy(dtype=float) + rows["a_bb"].to_numpy(dtype=float)) / ip
            - rows["p_whip"].to_numpy(dtype=float),
        }


def fit_rates(
    rows: pd.DataFrame,
    knots: Sequence[float | None] = KNOTS,
    corrected: Iterable[tuple[str, str]] = CORRECTED,
) -> dict[tuple[str, str], Hinge]:
    """Per (role, rate), the rate correction (none outside ``corrected``). ``rows``: one
    per pitcher-season with ``season``, ``age``, ``role``, our rates (``p_k``, ...) and
    the actuals (``a_bf``, ``a_ip``, ``a_k``, ``a_w``, ``a_er``, ``a_h``, ``a_bb``)."""
    miss = rate_misses(rows)
    keep = set(corrected)
    out = {}
    for r in ROLES:
        in_role = (rows["role"] == r).to_numpy()
        for stat, weight in RATE_WEIGHT.items():
            if (r, stat) not in keep:
                out[(r, stat)] = Hinge(None, 0.0, 0.0)
                continue
            w = rows[weight].to_numpy(dtype=float)
            ok = in_role & np.isfinite(miss[stat]) & (w > 0) & rows["age"].notna().to_numpy()
            if not ok.any():
                raise ValueError(f"no {r} rows with our {stat} and an actual to fit on")
            out[(r, stat)] = _choose(
                rows["age"].to_numpy(dtype=float)[ok],
                miss[stat][ok],
                w[ok],
                rows["season"].to_numpy()[ok],
                knots,
            )
    return out


def ip_ratio(rows: pd.DataFrame) -> np.ndarray:
    """Actual / projected IP, capped at ``RATIO_CAP`` (NaN under ``MIN_PROJECTED_IP``)."""
    proj = rows["proj_ip"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.minimum(rows["a_ip"].to_numpy(dtype=float) / proj, RATIO_CAP)
    return np.where(proj >= MIN_PROJECTED_IP, ratio, np.nan)


def fit_innings(rows: pd.DataFrame, knots: Sequence[float | None] = KNOTS) -> dict[str, Hinge]:
    """Per role, the innings correction. ``rows``: ``season``, ``age``, ``role``,
    ``proj_ip`` (the PT source's) and ``a_ip`` (0 if he didn't pitch)."""
    ratio = ip_ratio(rows)
    out = {}
    for r in ROLES:
        ok = (rows["role"] == r).to_numpy() & np.isfinite(ratio) & rows["age"].notna().to_numpy()
        if not ok.any():
            raise ValueError(f"no {r} rows with a projected IP ratio to fit on")
        out[r] = _choose(
            rows["age"].to_numpy(dtype=float)[ok],
            ratio[ok],
            np.ones(int(ok.sum())),
            rows["season"].to_numpy()[ok],
            knots,
        )
    return out


def adjust_rates(
    rates: pd.DataFrame, age: pd.Series, roles: pd.Series, fit: Mapping[tuple[str, str], Hinge]
) -> pd.DataFrame:
    """``rates`` (k, w per batter faced, era, whip) with each pitcher's age correction
    added; a pitcher with no age or role is left as he is. ERA, WHIP and the per-BF
    rates stay >= 0."""
    out = rates.copy()
    a = age.reindex(rates.index).to_numpy(dtype=float)
    rl = roles.reindex(rates.index)
    for (r, stat), h in fit.items():
        if stat not in out:
            continue
        mine = (rl == r).to_numpy() & np.isfinite(a)
        shift = np.where(mine, h.of(np.where(np.isfinite(a), a, 0.0)), 0.0)
        out[stat] = np.maximum(out[stat].to_numpy(dtype=float) + shift, 0.0)
    return out


def adjust_innings(
    ip: pd.Series, age: pd.Series, roles: pd.Series, fit: Mapping[str, Hinge]
) -> pd.Series:
    """Projected innings with each pitcher's age share taken off (``fit``: per role); a
    pitcher with no age or role is left as he is. Never below 0."""
    a = age.reindex(ip.index).to_numpy(dtype=float)
    rl = roles.reindex(ip.index)
    factor = np.ones(len(ip))
    for r, h in fit.items():
        mine = (rl == r).to_numpy() & np.isfinite(a)
        if h.level > 0:
            factor = np.where(mine, 1 + h.of(np.where(np.isfinite(a), a, 0.0)) / h.level, factor)
    return pd.Series(
        np.maximum(ip.to_numpy(dtype=float) * np.maximum(factor, 0.0), 0.0), index=ip.index
    )


def describe(fit: Mapping[object, Hinge], per: Iterable[object]) -> list[str]:
    """One line per correction, for reports."""
    lines = []
    for key in per:
        h = fit[key]
        if h.knot is None:
            lines.append(f"{key}: none")
        else:
            lines.append(
                f"{key}: from age {h.knot:g} to {h.top:.1f}, {h.slope:+.4g} a year (level {h.level:.4g})"
            )
    return lines
