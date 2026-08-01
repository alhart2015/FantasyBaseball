"""Rank keeper candidates on families that each measure ONE named thing.

Hitters blend skill, speed, playing time, batted-ball, future and age; pitchers
the same minus speed. There is deliberately no residual family -- see "Why there
is no `luck` family" below, and #288 for the history.

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

**speed** -- SB value per plate appearance. HITTERS ONLY; there is no pitcher
analogue and `FAMILIES["pitcher"]` omits it. The peripherals carry no speed input
at all (`corr(SB rate, skill_pct)` is NEGATIVE, about -0.13), so without this
family the entire roto value of a player's legs is invisible to `skill` and lands
in whatever residual term exists. It is its own family and not a sixth
`SKILL_COLUMNS` entry on purpose: equal weighting within the skill family gives
speed 1/6 weight, which measurably fails to remove it from a residual (it moved
`corr(residual, SB rate)` only from +0.466 to +0.461). `--study` reproduces both
numbers.

**pt** -- playing time: the volume dimension, on its own axis. This is the
DURABILITY term. An injury-shortened elite-rate season is now scored as low
volume against a high rate, rather than being charged as a permanent demerit --
the Juan Soto case (359 PA on a 96th-percentile skill line) that motivated #288.
Same-season PT is also the most persistent thing in the whole table: it predicts
next-year PT at 0.607, above skill (0.464) and rate (0.462). Durability repeats
better than talent does.

Why there is no `luck` family
-----------------------------
`luck = value_pct - skill_pct` was a RESIDUAL, not a measurement: whatever roto
production the peripherals failed to explain. Because `value_pct` is a counting
total and `skill_pct` is a rate, its single largest component was the leftover
volume dimension. Measured against it (hitters, mean rho over three transitions):

    playing time            +0.662
    SB rate                 +0.466
    R rate                  +0.284
    batted-ball (avg-xba)   +0.251   <- the only part that is actually luck
    RBI rate                +0.031
    HR rate                 -0.083

The BABIP-type luck the name implies was the THIRD largest component. Its famously
counter-intuitive POSITIVE weight was therefore not paying for luck; it was paying
for durability and stolen bases, both of which repeat. Giving those two their own
families says the same thing without the false name.

Cleaning the residual rather than removing it does not work. Redefining it against
a rate (`rate_pct - skill_pct`, both PT-neutral) does purge playing time
(+0.662 -> +0.179), but what surfaces underneath is lineup context (R rate +0.562,
RBI +0.374) -- a team property that does not travel with a traded player -- and the
cleaned residual still predicts next season (0.279). It still holds signal we
cannot name, so there is no residual at all.

    For TRADE decisions the old sell-high read now comes from `batted_ball`
    directly: a large positive batted-ball score is the rate overperformance that
    will not repeat. That reading used to be smuggled through `luck`.

**batted_ball** -- `avg - xba` for hitters, `fip - era` for pitchers (higher =
luckier): the specific rate overperformance the peripherals do not support. This
is the direct measurement of the component `luck` was misnamed after. On its own
it predicts next-year value at ~0.05 (hitters) and ~0.00 (pitchers) -- pure noise
-- so it earns its place as a CLAW-BACK, not as a predictor: it subtracts the
part of a good rate that will regress, so an everyday player who is merely
running hot is not priced like one who earned it (Rafaela, Otto Lopez; #277).

    The shipped hitter weight DELIBERATELY OVERRIDES the grid. Left free, the
    fit zeroes this family once `durability` absorbs the volume signal -- and
    the 2026 board then promotes the two luckiest bats on it (Abrams bb=93 to
    rank 5, Otto bb=97 to rank 26), undoing exactly what #277 shipped to do.
    Pinning it to -0.2 costs 0.0092 holdout rho against a 0.0126 noise band --
    less than the panel can resolve -- and restores the demotion (Abrams 10,
    Otto 39, Rafaela 78). #277 set the precedent for this tiebreak by shipping C
    over a higher-scoring baseline on the same watchlist grounds. `--backtest`
    prints the unpinned fit, so the override stays visible rather than hiding
    in this constant.

    A zero weight would also silently disable this family's `strict` all-NaN
    guard -- `composite` skips zero-weight families BEFORE the fail-loud check,
    so a broken xba feed would let `--backtest` decide the model on a degraded
    blend instead of raising. That is a second, independent reason not to ship
    the grid's zero here.

    The pitcher pool needs no such override: pinning the same -0.2 there
    IMPROVES the holdout outright (0.4932 free -> 0.5018 pinned), so the grid's
    zero was the panel failing to resolve a real effect rather than evidence
    against one. The negative weight therefore still replicates across both
    pools as #277 found -- hitters by argument, pitchers on the number.

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

The shape those numbers support: skill leads, playing time is close behind, speed
is roughly half of skill, batted_ball is a small claw-back, and future (discounted
for staleness, below) and age are small terms close in size. Read them as that
shape and not as tuned constants -- the ranked table `--backtest` prints separates
its top candidates by less than it separates the two fit seasons, so a third
decimal here would be noise.

**This family set TIED the residual parameterization on the holdout; it did not
beat it, and that was accepted knowingly.** Candidates span 0.017 (hitters) and
0.016 (pitchers) against noise bands of 0.015-0.026 and 0.105-0.143, so the panel
cannot resolve them -- but every candidate clears the skill-only null floor
comfortably. A tie is the CORRECT outcome by construction: `pt` is precisely what
the old residual was already proxying, so this re-expresses the same information
under honest names rather than adding any. It ships on interpretability and on
having the lowest fit-season noise of any candidate in both pools, not on rho.
Any future predictive gain has to come from a durability estimator that beats raw
`pt` -- see #288's open questions. Do not re-litigate this from the rho column
alone; `--backtest` reproduces the whole table including the residual candidates.

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
    "hitter": (1.0, 0.4, 1.2, -0.2, 0.4, 0.45),
    "pitcher": (1.0, 0.6, -0.2, 0.4, 0.15),
}
# The shipped family set per pool, aligned to FITTED_WEIGHTS. A dict, not one global
# tuple, because the pools carry separate weights and fits and a bake-off split
# verdict would be a legitimate outcome. `scripts/keeper_rankings.py --backtest` is
# where the set and weights are chosen; read it there.
FAMILIES: dict[str, tuple[str, ...]] = {
    "hitter": ("skill", "speed", "durability", "batted_ball", "future", "age"),
    "pitcher": ("skill", "pt", "batted_ball", "future", "age"),
}
# Every family the model knows how to blend. `family_order` selects a subset per
# pool; a name outside this set is a typo, not a silent no-op.
KNOWN_FAMILIES: frozenset[str] = frozenset(
    {"skill", "speed", "luck", "pt", "durability", "batted_ball", "future", "age"}
)


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

    NOT a shipped family since #288 -- kept because `--backtest` still bakes the
    residual parameterizations off against the current one and `--study` prints
    what this term was actually made of. See the module docstring for why it was
    removed; do not reintroduce it to `FAMILIES` without reading that first.
    """
    return value_pct - skill_pct


# How much of `durability` is the current season. Not tuned here and not grid-
# searched: within-family shape is fixed by argument, between-family weight is what
# `--backtest` fits (same rule as `skill_percentile`'s equal weighting). 0.75 is
# ZiPS's OWN revealed recency preference -- regressing its out-year PA on the PA
# history it had seen recovers lag weights of 0.340 / 0.118, a 0.74 / 0.26 split.
DURABILITY_RECENCY = 0.75


def durability(current_pct: pd.Series, prior_pct: pd.Series) -> pd.Series:
    """Playing time WITH memory: this season blended with the one before it.

    Raw same-season PT cannot tell "durable for years, hurt this season" from
    "always fragile" -- it has no memory, and `MIN_PT` compresses the pool so an
    injury-shortened star lands mid-pack rather than at the bottom. That is the
    Juan Soto case (#288) and it is why this family exists rather than plain `pt`.

    Both inputs are WITHIN-SEASON percentiles, which is what makes seasons of
    different length comparable for free: an in-progress season ranks its players
    against each other exactly as a completed one does, so nothing needs prorating
    and no COVID-2020 rescale is required if that season is ever pulled in.

    A missing prior season falls back to the current one -- a rookie is judged on
    what he has shown, not charged for a career he has not had yet. The known cost:
    a veteran who missed ALL of the prior season does not appear in it either, so
    he gets the same benefit of the doubt. That errs toward not penalizing, which
    is the safer direction for a keeper board, but it is a real hole -- closing it
    needs a tenure signal from T-2. See #288.
    """
    return DURABILITY_RECENCY * current_pct + (1.0 - DURABILITY_RECENCY) * prior_pct.reindex(
        current_pct.index
    ).fillna(current_pct)


def speed(frame: pd.DataFrame) -> pd.Series:
    """SB value per plate appearance -- the speed skill the peripherals omit.

    Takes `sb_sgp` pre-computed rather than dividing SB by a denominator here, so
    this module stays free of the SGP denominators like the rest of the
    normalization layer. The SGP conversion is a division by a positive constant
    and this family is consumed through `percentile`, so it is rank-invariant --
    it buys unit consistency with the rest of the model, not a different ranking.

    Hitters only. `FAMILIES["pitcher"]` omits the family, so a pitcher pool never
    reaches here; the KeyError is for a caller that asks anyway.
    """
    missing = [c for c in ("sb_sgp", "pt") if c not in frame.columns]
    if missing:
        raise KeyError(f"speed missing {missing}; got {sorted(frame.columns)}")
    # `where` not a bare divide: a 0-PT row is undefined speed, and NaN is what
    # `composite` mean-fills to neutral. Dividing would emit inf and poison the
    # percentile for the whole pool.
    return frame["sb_sgp"] / frame["pt"].where(frame["pt"] > 0)


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
