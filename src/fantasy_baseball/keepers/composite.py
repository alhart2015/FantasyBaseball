"""Blend a season's observed value, true-talent skills and age into one keeper rank.

Pure and I/O-free like the rest of the normalization layer. Everything works in
PERCENTILE space within a player pool, so hitters and pitchers are ranked against
their own kind and the inputs -- roto value, rate stats, age -- become comparable
without inventing a common unit.

The weights below were fitted, not chosen. `scripts/keeper_rankings.py --backtest`
regenerates the study: features observed in season T against the SGP percentile
actually realized in T+1, fit on 2022->2023 and 2023->2024, held out on
2024->2025. Two findings drive the shape of this module.

**The weight surface is flat.** On the holdout the fitted blend beats using last
season's value alone by 0.017 (hitters, rho 0.689 vs 0.672) and 0.015 (pitchers,
0.470 vs 0.456). Treat these weights as "roughly this shape", not as tuned
constants; three seasons cannot separate 0.7 from 0.8.

**Skills and value predict DIFFERENT halves of next season, and the split is
reversed between hitters and pitchers.** Next-year roto value is mostly volume
(rho 0.95 hitters / 0.88 pitchers against next-year playing time), and last
season's value is the better predictor of volume. But split out next-year RATE:

    predictor -> next-year RATE      hitters   pitchers
    last season's roto value           0.514      0.342
    true-talent skills                 0.381      0.464

and inside the keeper tier (top 60 by last-year value) the pitcher column
separates completely -- skills 0.511, last-year value -0.015, stable across all
three seasons. Elite pitcher ERA carries almost no information about next
season's ERA; the skills carry most of it. Hitters go the other way, with value
ahead of skills in 3 of 3 seasons.

That is why :func:`regression_gap` exists and is worth more than the composite
for trade decisions: it is the part of a player's line his skills do not support.
"""

from __future__ import annotations

import pandas as pd

# (value, skill, age). Age is "younger is better" and is an adjustment, not a
# driver -- on its own it reaches rho 0.19 for hitters and 0.05 for pitchers.
FITTED_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "hitter": (0.7, 0.1, 0.2),
    "pitcher": (0.7, 0.2, 0.1),
}

HITTER_SKILLS = ("barrel_pct", "barrel_pa_pct", "xwoba", "xba", "wrc_plus")
PITCHER_SKILLS = ("era_minus", "fip", "k_pct", "swstr_pct", "whiff_pct", "csw_pct")
SKILL_COLUMNS: dict[str, tuple[str, ...]] = {
    "hitter": HITTER_SKILLS,
    "pitcher": PITCHER_SKILLS,
}
# Run prevention: a lower number is a better pitcher, so the percentile flips.
LOWER_IS_BETTER = frozenset({"era_minus", "fip"})


def percentile(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    """Rank `values` on 0-1 within the pool. NaN stays NaN rather than sorting last.

    A lower-is-better stat is ranked on the negated values rather than mirrored
    with ``1 - rank``. Mirroring is off by 1/n: ``rank(pct=True)`` runs over
    [1/n, 1], so the mirror runs over [0, 1-1/n] and the best player in a flipped
    stat tops out below 1.0. Averaging `era_minus` and `fip` against four
    unflipped stats would then hold the pitcher family down by a systematic 1/n.
    """
    ordered = values if higher_is_better else -values
    return ordered.rank(pct=True)


def skill_percentile(frame: pd.DataFrame, kind: str) -> pd.Series:
    """One skill percentile per player: the mean of each stat's own percentile.

    Equal weights WITHIN the family is deliberate. The backtest weights the three
    families against each other on three seasons; letting it also tune six
    within-family weights would fit noise. A player missing one stat is averaged
    over the rest rather than dropped.
    """
    columns = SKILL_COLUMNS[kind]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{kind} skills missing {missing}; got {sorted(frame.columns)}")
    parts = [percentile(frame[col], higher_is_better=col not in LOWER_IS_BETTER) for col in columns]
    return pd.concat(parts, axis=1).mean(axis=1)


def composite(
    value_pct: pd.Series,
    skill_pct: pd.Series,
    age_pct: pd.Series,
    kind: str,
    weights: tuple[float, float, float] | None = None,
) -> pd.Series:
    """Weighted blend of the three families, back on a 0-1 scale."""
    w_value, w_skill, w_age = weights if weights is not None else FITTED_WEIGHTS[kind]
    total = w_value + w_skill + w_age
    if total <= 0:
        raise ValueError(f"weights must sum to something positive, got {total}")
    blended = w_value * value_pct + w_skill * skill_pct + w_age * age_pct
    return blended / total


def regression_gap(value_pct: pd.Series, skill_pct: pd.Series) -> pd.Series:
    """How far a player's roto line ran ahead of the skills behind it.

    Positive means the production outran the peripherals -- a sell-high case, and
    for a top-tier pitcher a strong one, since last-year value carries no signal
    about next-year rate up there while the skills carry most of it. Negative is
    the buy-low mirror.
    """
    return value_pct - skill_pct
