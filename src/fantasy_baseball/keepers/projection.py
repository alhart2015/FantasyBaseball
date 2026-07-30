"""Turn a composite percentile into projected roto value, with an error bar.

The composite in `keepers.composite` is ordinal: it ranks a player within his own
pool and carries no units, so it cannot be compared across pools or differenced.
This module supplies the missing scale by regressing what players at a given
composite ACTUALLY went on to earn:

    E[SGP] = intercept + slope * composite

Regenerate every constant below with `scripts/keeper_rankings.py --fit`, which
prints them paste-ready. Two properties of the fit the model has to respect:

**It is monotone.** The band-level view of the same data had pitchers in the
90-95 composite band out-earning 95-100, which would mean a better composite
predicted less value. Linear and quadratic fits are both monotone increasing, so
that inversion is band noise and is deliberately not reproduced.
`test_projection.py` pins the monotonicity.

**The error grows with the composite**, so a single pooled SD would understate
uncertainty exactly where keeper decisions are made. SD is its own linear
function of the composite, fitted as E|residual| and scaled by sqrt(pi/2).

The error bars are wide on purpose -- one season of roto value is mostly not
predictable a year out, and a report that hides that invites false precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (intercept, slope) against composite on 0-1.
SGP_FIT: dict[str, tuple[float, float]] = {
    "hitter": (0.746, 12.619),
    "pitcher": (0.584, 8.337),
}
# (intercept, slope) for the residual standard deviation.
SGP_SD_FIT: dict[str, tuple[float, float]] = {
    "hitter": (2.676, 2.230),
    "pitcher": (1.977, 3.909),
}

# Empirical distribution of the STANDARDIZED residual, as quantiles at
# RESIDUAL_QUANTILE_LEVELS. Sampled from rather than assuming a normal because
# SGP is floored -- a pitcher who loses his job earns about zero, not minus
# fifteen -- so the left tail is far thinner than a normal and the right tail
# fatter. A normal would overstate downside and understate the upside tail, which
# is the half a "is he top three" question turns on. `test_projection.py` asserts
# the skew and both tail masses, so those claims stay honest without prose.
# Levels are probabilities on 0-1, matching `utils.constants.QUANTILE_LEVELS`.
RESIDUAL_QUANTILE_LEVELS = (
    0.0,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
    1.0,
)
STD_RESIDUAL_QUANTILES: dict[str, tuple[float, ...]] = {
    "hitter": (
        -2.598,
        -2.101,
        -1.776,
        -1.526,
        -1.217,
        -0.824,
        -0.551,
        -0.316,
        -0.084,
        0.189,
        0.495,
        0.850,
        1.371,
        1.768,
        2.153,
        2.468,
        4.068,
    ),
    "pitcher": (
        -1.672,
        -1.497,
        -1.407,
        -1.310,
        -1.153,
        -0.874,
        -0.624,
        -0.407,
        -0.178,
        0.125,
        0.446,
        0.818,
        1.362,
        1.812,
        2.296,
        2.908,
        5.244,
    ),
}


def expected_sgp(composite_pct: pd.Series, kind: str) -> pd.Series:
    """Projected next-season SGP for a player at this composite percentile."""
    intercept, slope = SGP_FIT[kind]
    return intercept + slope * composite_pct


def sgp_sd(composite_pct: pd.Series, kind: str) -> pd.Series:
    """Standard deviation of that projection for an INDIVIDUAL player.

    A predictive spread, not the standard error of a group mean -- roughly five
    times larger, and the one that matters when the question is "which of these
    two should I keep".
    """
    intercept, slope = SGP_SD_FIT[kind]
    return intercept + slope * composite_pct


def scarcity_floors(floors: dict[str, float]) -> dict[str, float]:
    """Replacement levels, mean-centred and with SP/RP collapsed.

    Two separate things, both about keeping this table commensurable with
    `expected_sgp`.

    **SP and RP are merged into one pitcher floor.** `expected_sgp` is fit on the
    pooled pitcher panel and has no role term, so it predicts the same value for a
    starter and a reliever at the same composite. Netting that against
    role-specific floors debits a difference the projection never credited, and
    measurement shows it lands the wrong way round: starters BEAT the pooled fit
    and relievers fall short of it, and the gap widens in the top composite decile
    -- exactly where keeper decisions are made -- while the shallower RP floor
    credits relievers on top of that. Compounded, that put the top of the pitcher
    board entirely on RP-classified arms who went on to earn less than the starters
    below them at every level of the score. Until the projection itself is
    role-aware, one floor is the only consistent choice. `--study` prints the
    per-role residual and its top-decile split.

    **The centring is a DISPLAY offset**, not a model decision, and worth being
    blunt about because the shape invites the opposite reading. Subtracting these
    gives `projection - level + mean(level)`, i.e. true VAR plus one constant
    shared by every position, so the ranking, every gap and every P(top-N) are
    identical to using the floors untouched. All it buys is a column that reads
    positive. It exists because the two quantities sit on different scales:
    `sgp.replacement`'s floors are draft-time full-season lines on which an ace
    projects ~20 SGP, while `expected_sgp` tops out near 13 (hitters) and 9
    (pitchers), so raw subtraction puts every pitcher below replacement -- a
    statement about the scales, not the pitchers. Centring makes it readable; no
    constant offset could repair it.

    What the floors MIGHT carry validly is the spread BETWEEN hitter positions,
    since a difference survives a change of scale where a level does not. Unlike
    the pitcher split above there is no measured bias to compound, so the spread is
    neither confirmed nor contradicted, and it is left alone on that basis.

    `--study` prints the evidence: credit against realized residual per position,
    plus the regression slope of one on the other. Read the "position known" slope,
    not the all-rows one -- the latter is dominated by players missing from the
    position map, who are routed to the harshest adjustment and who mostly left the
    league, so residual and credit co-move there for a survivorship reason with
    nothing to do with position scarcity.

    No numbers anywhere in this docstring, deliberately. Three successive attempts
    to state them here were each wrong in a different way, because nothing
    regenerated them; `--study` does.

    Still open regardless of that evidence: the spread was calibrated on the wider
    draft-time scale, so against this compressed one it may simply be too wide.
    """
    if not floors:
        return {}
    merged = dict(floors)
    pitcher_keys = [key for key in ("SP", "RP") if key in merged]
    if pitcher_keys:
        pooled = sum(merged[key] for key in pitcher_keys) / len(pitcher_keys)
        for key in pitcher_keys:
            merged[key] = pooled
    average = sum(merged.values()) / len(merged)
    return {position: level - average for position, level in merged.items()}


def sample_outcomes(
    means: pd.Series,
    sds: pd.Series,
    kinds: pd.Series,
    *,
    draws: int = 20_000,
    seed: int = 20260729,
) -> np.ndarray:
    """Simulate `draws` seasons, returning an array shaped (players, draws).

    Each player's outcome is `mean + sd * z`, with `z` drawn from his pool's
    empirical standardized-residual distribution by inverse-CDF interpolation.
    `kinds` supplies "hitter"/"pitcher" per player, since the two shapes differ.

    Draws are independent across players. A league-wide scoring shift would
    correlate them, but every question asked of these samples is a comparison
    WITHIN one roster, and a common component cancels in a difference.

    `seed` is fixed so a report does not reshuffle its own conclusions between
    runs; pass a different one to check stability.
    """
    rng = np.random.default_rng(seed)
    uniform = rng.random((len(means), draws))
    z = np.empty_like(uniform)
    for pool in set(kinds):
        rows = (kinds == pool).to_numpy()
        z[rows] = np.interp(uniform[rows], RESIDUAL_QUANTILE_LEVELS, STD_RESIDUAL_QUANTILES[pool])
    outcomes: np.ndarray = means.to_numpy()[:, None] + sds.to_numpy()[:, None] * z
    return outcomes


def probability_top_n(
    means: pd.Series,
    sds: pd.Series,
    kinds: pd.Series,
    top_n: int,
    *,
    draws: int = 20_000,
    seed: int = 20260729,
) -> pd.Series:
    """P(each player finishes among the `top_n` best of this set) next season.

    This is the keeper question -- "would I be right to keep him" -- and it is
    joint, not pairwise: it depends on who else is in the set, so the set must be
    exactly the players competing for the slots. Pass a roster, not a league pool.

    Probabilities sum to `top_n` by construction, which is a useful check: three
    slots distribute exactly three players' worth of confidence.
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    if top_n >= len(means):
        return pd.Series(1.0, index=means.index)
    outcomes = sample_outcomes(means, sds, kinds, draws=draws, seed=seed)
    # `argpartition` selects exactly `top_n` winners per simulated season, the
    # same idiom as `simulation._topk_indices`. Thresholding on the `top_n`-th
    # value instead would credit every player tied at the cut, breaking the
    # sum-to-`top_n` invariant -- only reachable at sd == 0, but the invariant is
    # documented and worth holding unconditionally.
    winners = np.argpartition(-outcomes, top_n - 1, axis=0)[:top_n]
    made_cut = np.zeros(outcomes.shape, dtype=bool)
    np.put_along_axis(made_cut, winners, True, axis=0)
    return pd.Series(made_cut.mean(axis=1), index=means.index)
