"""Rank keeper candidates on four families: skill, luck, future and age.

Pure and I/O-free like the rest of the normalization layer. Everything works in
PERCENTILE space within a player pool, so hitters and pitchers rank against their
own kind and the inputs become comparable without inventing a common unit.

The families
------------
**skill** -- the peripherals. Contact quality and expected outcomes for hitters,
run prevention and pitch-level whiff/called-strike rates for pitchers. Two of the
inputs (`wrc_plus`, `era_minus`) are production-derived rather than expected, so
this is better read as "peripherals" than as clean true talent. Removing them was
tested and made pitchers strictly worse (0.464 -> 0.410 against next-year rate),
so they stay.

**luck** -- `value_pct - skill_pct`: the part of a player's roto line his
peripherals do not support. Note `value = skill + luck` exactly, so {skill, luck}
spans the same space as {skill, value}; this is a reparameterization chosen for
what it says, not a new model.

    Its weight is POSITIVE, which is not what the name suggests. Being "lucky"
    in year T predicts year T+1 because the gap also encodes role and
    durability: luck -> next-year PLAYING TIME is +0.38, while for pitchers
    luck -> next-year RATE is -0.04. Forcing a negative weight collapses the fit
    (rho 0.65 -> 0.13 for hitters). So luck is not a noise term to subtract; it
    is mostly a volume signal wearing a misleading name. For TRADE decisions
    read it the other way -- a large positive luck score is the sell-high case,
    because the rate half of it will not repeat.

**future** -- percentile of projected SGP from the out-year ZiPS files, blended
`FUTURE_BLEND` in favour of the nearer year.

**age** -- younger is better. Small but real; the only family that survives
purely as an adjustment.

Where the weights come from
---------------------------
`scripts/keeper_rankings.py --backtest`: features observed in season T against
the SGP percentile realized in T+1, fit on 2022->2023 and 2023->2024, held out on
2024->2025. Holdout Spearman:

                            hitters   pitchers
    shipped weights           0.709      0.495
    without future            0.689      0.470
    value + luck only         0.672      0.456

**The future weight is discounted for staleness, deliberately.** A FRESH
next-season projection (one that has seen season T) is the single strongest
predictor available -- 0.666 for hitters, 0.520 for pitchers, both beating
skill+luck -- and would earn a weight near 1.0-2.0. The out-year files are two
years forward from their information set: ZiPS 2027 was generated 2026-03-25 and
has never seen 2026. Measuring that same two-years-forward case historically
(ZiPS for year T predicting T+1) drops it to 0.523 / 0.347 alone and to a best
weight of ~0.4. That 0.4 is what ships. If a post-season ZiPS 2027 ever lands,
it is worth roughly triple this.

The fit surface is flat: the top four weight vectors are within 0.002 of each
other. `future` is set to 0.4 for both types because the fit cannot separate 0.2
from 0.4 for pitchers (delta 0.0001) and one value for both is the simpler model.
Read all of these as a shape, not as tuned constants -- two fit seasons cannot
justify a third decimal.
"""

from __future__ import annotations

import pandas as pd

# Weight per family, in (skill, luck, future, age) order. Not normalized;
# `composite` divides by the sum.
FITTED_WEIGHTS: dict[str, tuple[float, float, float, float]] = {
    "hitter": (1.0, 0.8, 0.4, 0.3),
    "pitcher": (1.0, 0.6, 0.4, 0.15),
}
FAMILIES: tuple[str, ...] = ("skill", "luck", "future", "age")

# Out-year projection blend, nearer year first. The two ZiPS out-years come from
# one 2026-03-25 model run and correlate 0.96, so the blend lands 0.995
# correlated with the nearer year alone -- this weighting expresses a preference
# rather than adding much information.
FUTURE_BLEND: tuple[float, float] = (2 / 3, 1 / 3)

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

    Equal weights WITHIN the family is deliberate. The backtest weights the four
    families against each other on two seasons; letting it also tune six
    within-family weights would fit noise. A player missing one stat is averaged
    over the rest rather than dropped.
    """
    columns = SKILL_COLUMNS[kind]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{kind} skills missing {missing}; got {sorted(frame.columns)}")
    parts = [percentile(frame[col], higher_is_better=col not in LOWER_IS_BETTER) for col in columns]
    return pd.concat(parts, axis=1).mean(axis=1)


def luck(value_pct: pd.Series, skill_pct: pd.Series) -> pd.Series:
    """The part of a roto line the peripherals do not support.

    Positive means production outran the peripherals. Carries a positive weight
    in the composite because it also encodes playing time, and reads as the
    sell-high signal for trades -- see the module docstring.
    """
    return value_pct - skill_pct


def future_percentile(near: pd.Series, far: pd.Series) -> pd.Series:
    """Percentile of the `FUTURE_BLEND`-weighted out-year projections.

    Blended on the projected-SGP scale before ranking, so the weighting applies
    to the projections themselves rather than to two separate rankings. A player
    the far year does not cover falls back to the near year alone rather than
    being dropped.
    """
    w_near, w_far = FUTURE_BLEND
    blended = w_near * near + w_far * far.reindex(near.index).fillna(near)
    return percentile(blended)


def composite(
    families: dict[str, pd.Series],
    kind: str,
    weights: tuple[float, float, float, float] | None = None,
) -> pd.Series:
    """Weighted blend of `FAMILIES`, back on a 0-1 scale.

    Missing families are treated as absent rather than as zero: their weight is
    dropped from the denominator, so a pool with no out-year projection still
    produces a comparable composite instead of one silently scaled down.
    """
    chosen = weights if weights is not None else FITTED_WEIGHTS[kind]
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise KeyError(f"unknown families {sorted(unknown)}; expected {list(FAMILIES)}")

    total = 0.0
    blended: pd.Series | None = None
    for name, weight in zip(FAMILIES, chosen, strict=True):
        series = families.get(name)
        if series is None or weight == 0:
            continue
        term = weight * series.fillna(series.mean())
        blended = term if blended is None else blended + term
        total += weight
    if blended is None or total <= 0:
        raise ValueError("no weighted family supplied; cannot form a composite")
    return blended / total
