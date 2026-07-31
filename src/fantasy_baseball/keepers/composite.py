"""Rank keeper candidates on five families: skill, luck, batted-ball, future, age.

Pure and I/O-free like the rest of the normalization layer. Everything works in
PERCENTILE space within a player pool, so hitters and pitchers rank against their
own kind and the inputs become comparable without inventing a common unit.

The families
------------
**skill** -- the peripherals. Contact quality and expected outcomes for hitters,
run prevention and pitch-level whiff/called-strike rates for pitchers. Two of the
inputs (`wrc_plus`, `era_minus`) are production-derived rather than expected, so
this is better read as "peripherals" than as clean true talent. Removing them was
tested and made pitchers strictly worse against next-year rate, so they stay --
`--study` reproduces that comparison.

**luck** -- `value_pct - skill_pct`: the part of a player's roto line his
peripherals do not support. Note `value = skill + luck` exactly, so {skill, luck}
spans the same space as {skill, value}; this is a reparameterization chosen for
what it says, not a new model.

    Its weight is POSITIVE, which is not what the name suggests. Being "lucky"
    in year T predicts year T+1 because the gap also encodes role and durability:
    it correlates strongly with next-year PLAYING TIME and, for pitchers, not at
    all with next-year RATE. Forcing a negative weight collapses the fit. So luck
    is not a noise term to subtract; it is mostly a volume signal wearing a
    misleading name. For TRADE decisions read it the other way -- a large positive
    luck score is the sell-high case, because the rate half will not repeat.
    `--study` prints all of these correlations.

**batted_ball** -- `avg - xba` for hitters, `fip - era` for pitchers (higher =
luckier): the specific rate overperformance the peripherals do not support. It
carries a NEGATIVE weight, and this is the point of the family. On its own the
measure predicts next-year value at ~0.05 (hitters) and ~0.00 (pitchers) -- pure
noise, and `--backtest` gives it exactly zero weight when it is the only
non-skill volume term. But it is positively correlated with `luck`, which bundles
the same batted-ball luck in with the real playing-time signal. Entering
`batted_ball` alongside `luck` lets the fit REWARD the gap (via luck) while
CLAWING BACK its batted-ball half (via a negative batted_ball weight), so an
everyday player who is merely running hot is no longer priced like one who earned
it. That is exactly the everyday-plus-lucky profile (Rafaela, Otto Lopez) the
single `luck` term used to oversell. The negative weight replicated across BOTH
pools independently; `--backtest` and `--study` are the evidence, #277 the history.

**future** -- percentile of projected SGP from the out-year ZiPS files, blended
`FUTURE_BLEND` in favour of the nearer year.

**age** -- younger is better. Small but real; the only family that survives
purely as an adjustment.

Where the weights come from
---------------------------
`scripts/keeper_rankings.py --backtest`: features observed in season T against the
SGP percentile realized in T+1, fit on the earlier transitions and held out on the
latest. It grid-searches each candidate family set and prints their holdout rho
with a fit-season noise floor -- read it there rather than from a cached copy here,
which drifts silently and already did once on this branch.

The shape those numbers support: skill leads, luck is close behind, batted_ball is
a small negative claw-back, and future (discounted for staleness, below) and age
are small terms close in size -- close enough that their order flips between the
pools. Read them as that shape and not as tuned constants -- the ranked table
`--backtest` prints separates its top candidates by less than it separates the two
fit seasons, so a third decimal here would be noise. The chosen set (keep `luck`,
add a negative `batted_ball`) was a bake-off against two alternatives -- adding a
raw playing-time family, and replacing `luck` with `batted_ball` outright -- both
of which lost the holdout; `--backtest` reproduces all three.

**The future weight is discounted for staleness, deliberately.** A FRESH
next-season projection (one that has seen season T) is the single strongest
predictor available and would earn a weight several times this. The out-year files
are two years forward from their information set: ZiPS 2027 was generated
2026-03-25 and has never seen 2026. `--study` measures that same
two-years-forward case historically, and the discounted `future` weight in
`FITTED_WEIGHTS` is what the bake-off's holdout settled on -- read the number
there, not here. If a post-season ZiPS 2027 ever lands, it is worth substantially
more.
"""

from __future__ import annotations

import pandas as pd

# The producer of these columns, so the contract is one literal and not two.
from fantasy_baseball.keepers.skills import HITTER_SKILLS, PITCHER_SKILLS

# Weight per family, aligned to `FAMILIES[kind]`. Not normalized; `composite`
# divides by the sum.
FITTED_WEIGHTS: dict[str, tuple[float, ...]] = {
    "hitter": (1.0, 0.8, -0.2, 0.2, 0.3),
    "pitcher": (1.0, 0.8, -0.2, 0.2, 0.15),
}
# The shipped family set per pool, aligned to FITTED_WEIGHTS. A dict, not one global
# tuple, because the pools carry separate weights and fits and a bake-off split
# verdict would be a legitimate outcome. `scripts/keeper_rankings.py --backtest` is
# where the set and weights are chosen; read it there.
FAMILIES: dict[str, tuple[str, ...]] = {
    "hitter": ("skill", "luck", "batted_ball", "future", "age"),
    "pitcher": ("skill", "luck", "batted_ball", "future", "age"),
}
# Every family the model knows how to blend. `family_order` selects a subset per
# pool; a name outside this set is a typo, not a silent no-op.
KNOWN_FAMILIES: frozenset[str] = frozenset({"skill", "luck", "pt", "batted_ball", "future", "age"})


def check_known_families(names: set[str]) -> None:
    """Raise a clear KeyError naming any family outside `KNOWN_FAMILIES`.

    Exported so `keeper_rankings.composite_pct` can reject a typo'd `family_order` with this
    same actionable message BEFORE it materializes `frame["{typo}_pct"]` (an opaque pandas
    KeyError). `composite` calls it too, for the direct callers `composite_pct` bypasses.
    """
    unknown = names - KNOWN_FAMILIES
    if unknown:
        raise KeyError(f"unknown families {sorted(unknown)}; expected {sorted(KNOWN_FAMILIES)}")


# Out-year projection blend, nearer year first. Both ZiPS out-years come from one
# 2026-03-25 model run, so they are near-duplicates and the blend is almost
# perfectly correlated with the nearer year alone: this weighting expresses a
# preference for the nearer year rather than adding information. `--study` prints
# both correlations per pool; no numbers here, since nothing would regenerate them.
FUTURE_BLEND: tuple[float, float] = (2 / 3, 1 / 3)

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

    Equal weights WITHIN the family is deliberate. The backtest weights the
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

    Positive means production outran them. Why that earns a POSITIVE weight, and
    why trades read it backwards, is argued once in the module docstring.
    """
    return value_pct - skill_pct


# Columns the batted-ball overperformance is measured from, per pool. NB "park-
# unadjusted" is not "park-neutral": the PRODUCED side (raw avg, era) is park-EXPOSED
# while the EXPECTED side (xba; fip, mostly) is not, so the gap keeps a repeating park
# signal, not pure luck -- a Coors bat outruns its xba every year, and batted_ball's
# negative weight then claws back some of that real, repeating park value along with
# the luck. Park-adjusting the produced side is a known, unimplemented refinement; the
# park-adjusted view already lives in `skill` (wrc_plus/era_minus).
BATTED_BALL_INPUTS: dict[str, tuple[str, str]] = {
    "hitter": ("avg", "xba"),
    "pitcher": ("fip", "era"),
}


def batted_ball(frame: pd.DataFrame, kind: str) -> pd.Series:
    """Rate overperformance the peripherals do not support (higher = luckier).

    `avg - xba` for hitters, `fip - era` for pitchers -- ERA below FIP means the
    pitcher outran his peripherals. This is the half of `luck` that regresses and
    should not be rewarded; parameterization B measures it directly instead of
    letting the `luck` catch-all proxy it. NaN in either input propagates.
    """
    produced, expected = BATTED_BALL_INPUTS[kind]
    missing = [c for c in (produced, expected) if c not in frame.columns]
    if missing:
        raise KeyError(f"{kind} batted_ball missing {missing}; got {sorted(frame.columns)}")
    return frame[produced] - frame[expected]


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
    weights: tuple[float, ...] | None = None,
    *,
    family_order: tuple[str, ...] | None = None,
    strict: bool = False,
) -> pd.Series:
    """Weighted blend of the pool's families, back on a 0-1 scale.

    `family_order` names the families to blend, in weight order; it defaults to the
    shipped `FAMILIES[kind]`. Missing families are treated as absent rather than as
    zero: their weight is dropped from the denominator, so a pool with no out-year
    projection still produces a comparable composite instead of one silently scaled
    down.

    `strict` governs the ALL-NaN case only: False DROPS an all-NaN family from the
    blend, True raises with a generic message. Which callers pass which is NOT the
    live-vs-offline split you might expect. The live board passes False but does NOT
    rely on the drop to survive an outage -- `keeper_rankings._require_mandatory_families`
    runs FIRST and fails loud (with a source-pointing message) on any all-NaN SHIPPED
    family, so composite's drop is only ever reached for a family that is legitimately
    absent, never for an xba/ZiPS outage. The offline fit/backtest pass True because
    their blended number is persisted as constants or decides the shipped model, so a
    silently-dropped family would corrupt the result. `strict` hardens the all-NaN path
    only -- a family entirely absent from `families` still drops -- but the strict
    callers materialize every column, so only all-NaN arises for them.
    """
    order = family_order if family_order is not None else FAMILIES[kind]
    chosen = weights if weights is not None else FITTED_WEIGHTS[kind]
    # A name outside the known universe -- in EITHER the supplied dict or the requested
    # `family_order` -- is a typo, not a silent no-op. An unknown `family_order` entry
    # would otherwise `.get()` to None below and drop its weighted slot with no error,
    # quietly shifting every downstream rank; catch it here as the comment promises.
    check_known_families(set(families) | set(order))
    # `family_order` and `weights` are index-aligned and default independently, so
    # supplying one without the other zips the shipped weights against a different
    # family set -- a silent mispairing `strict=True` can't catch when the lengths
    # happen to match. Reject it here so every caller is covered, not just the wrapper.
    if (family_order is None) != (weights is None):
        raise ValueError("pass family_order and weights together, or neither")

    # An empty pool (0 rows) blends to an empty result -- there is nothing to fail on,
    # and callers build intentional empty sub-pools (a not-yet-populated early-season
    # board, `--study`'s truncation to a tiny live pool) that expect an empty return to
    # skip, not a raise. Without this the all-NaN drop below would drop every (vacuously
    # all-NaN) family and hit the "no weighted family" guard. A NON-empty pool with an
    # all-NaN family is a different case the loop still handles (drop, or strict-raise).
    supplied = next((families[n] for n in order if families.get(n) is not None), None)
    if supplied is not None and supplied.empty:
        return pd.Series(dtype=float, index=supplied.index)

    total = 0.0
    blended: pd.Series | None = None
    for name, weight in zip(order, chosen, strict=True):
        series = families.get(name)
        if series is None or weight == 0:
            continue
        # An ALL-NaN family's mean is NaN, so `fillna(mean)` below would leave it NaN
        # and poison every player's blend (and `total > 0` would not catch it). The
        # live board drops it -- like a missing family -- to stay valid; a `strict`
        # caller fails loud rather than persist/decide on a silently-degraded blend.
        if series.isna().all():
            if strict:
                # `isna().all()` is vacuously True on an empty column too, so name which
                # it is -- an empty panel and an all-NaN family are different bugs.
                what = "empty" if len(series) == 0 else "entirely NaN"
                raise ValueError(f"family {name!r} is {what}; the blend cannot use it")
            continue
        term = weight * series.fillna(series.mean())
        blended = term if blended is None else blended + term
        total += weight
    if blended is None or total <= 0:
        raise ValueError("no weighted family supplied; cannot form a composite")
    return blended / total
