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
    "hitter": (0.767, 12.577),
    "pitcher": (0.538, 8.429),
}
# (intercept, slope) for the residual standard deviation.
SGP_SD_FIT: dict[str, tuple[float, float]] = {
    "hitter": (2.645, 2.298),
    "pitcher": (1.912, 4.076),
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
        -2.578,
        -2.034,
        -1.79,
        -1.584,
        -1.192,
        -0.841,
        -0.571,
        -0.32,
        -0.074,
        0.205,
        0.469,
        0.793,
        1.293,
        1.791,
        2.181,
        2.609,
        3.903,
    ),
    "pitcher": (
        -1.673,
        -1.472,
        -1.416,
        -1.286,
        -1.136,
        -0.866,
        -0.647,
        -0.413,
        -0.149,
        0.134,
        0.494,
        0.859,
        1.334,
        1.792,
        2.138,
        2.819,
        5.938,
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
