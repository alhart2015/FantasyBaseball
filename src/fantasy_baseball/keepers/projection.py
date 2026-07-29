"""Turn a composite percentile into projected roto value, with an error bar.

The composite in `keepers.composite` is ordinal: it ranks a player within his own
pool and carries no units, so it cannot be compared across pools or differenced.
This module supplies the missing scale by regressing what players at a given
composite ACTUALLY went on to earn.

Fitted over 2022->2023, 2023->2024 and 2024->2025 (n=977 hitters, 1070 pitchers),
target = realized next-season SGP, with a player who did not appear scoring 0:

    E[SGP] = intercept + slope * composite

Two properties of the fit that the model has to respect:

**It is monotone.** The band-level summary showed pitchers in the 90-95 composite
band out-earning the 95-100 band (11.17 vs 9.03, t~2.0), which would mean a better
composite predicted less value. A linear fit over the same data is monotone
increasing and a quadratic one is too, so that inversion is band noise -- 54
players per band -- and is deliberately not reproduced here.

**The error grows with the composite.** Residual SD runs 3.3 -> 5.6 for hitters
and 3.0 -> 5.5 for pitchers from the bottom of the pool to the top, so a single
pooled SD would understate uncertainty exactly where keeper decisions are made.
SD is therefore its own linear function of the composite, fitted as E|residual|
and scaled by sqrt(pi/2).

R^2 is 0.47 for hitters and 0.27 for pitchers. The error bars are wide on purpose:
one season of roto value is mostly not predictable a year out, and a report that
hides that invites false precision. Quadratic terms add ~0.015 R^2 and are left
out.
"""

from __future__ import annotations

import math

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
FIT_SEASONS = ("2022->2023", "2023->2024", "2024->2025")

# Empirical distribution of the STANDARDIZED residual, as quantiles at
# RESIDUAL_QUANTILE_GRID. Sampling from this rather than from a normal, because
# the residuals are not normal and the ways they differ both matter for a
# top-three probability:
#
#   pitchers: skew +0.98, excess kurtosis +1.48
#   P(z < -1.5): observed 0.009 for pitchers against a normal's 0.067
#   P(z > +1.5): observed 0.081             against            0.067
#
# SGP is bounded below -- a pitcher who loses his job earns about zero, which is
# not far under the mean for anyone mid-pack -- so the left tail is much thinner
# than a normal and the right tail fatter. A normal would overstate downside and
# understate the upside tail, which is the half a "is he top three" question
# actually turns on. 5.3% of hitters and 10.7% of pitchers land at exactly 0,
# and that mass is inside these quantiles rather than smoothed away.
RESIDUAL_QUANTILE_GRID = (0, 1, 2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97.5, 99, 100)
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

    This is a predictive spread, not the standard error of a group mean -- it is
    roughly five times larger, and it is the one that matters when the question
    is "which of these two should I keep".
    """
    intercept, slope = SGP_SD_FIT[kind]
    return intercept + slope * composite_pct


def scarcity_adjustments(floors: dict[str, float]) -> dict[str, float]:
    """Turn replacement LEVELS into mean-centred positional adjustments.

    `expected_sgp` cannot be differenced against `sgp.replacement`'s floors: those
    are draft-time full-season lines on which an ace projects ~20 SGP, while this
    module's projection is an empirical mean of realized next-season outcomes,
    regressed and injury-inclusive, topping out near 13 (hitters) and 9
    (pitchers). Subtracting one from the other put every pitcher below
    replacement, which said more about the two scales than about the pitchers.

    What the floors do carry validly is the SPREAD between positions -- catcher
    7.70 against 9.96 for OF/UTIL -- and a difference survives a change of scale
    that a level does not. Centring on the mean floor keeps exactly that spread
    and discards the absolute offset, so a catcher gains what his position is
    genuinely worth relative to an outfielder without importing the mismatch.

    Returned values add to a projection: positive means scarce.
    """
    if not floors:
        return {}
    average = sum(floors.values()) / len(floors)
    return {position: average - level for position, level in floors.items()}


def probability_better(mean_a: float, sd_a: float, mean_b: float, sd_b: float) -> float:
    """P(player A out-earns player B) next season.

    Normal approximation, treating the two outcomes as independent: the
    difference is then normal with variance sd_a^2 + sd_b^2. Independence is
    imperfect -- a league-wide offensive shift moves everyone together -- but
    that common component largely cancels in a difference, so the error is small
    relative to the spread it sits inside.

    Returns 0.5 when both spreads are degenerate, which is the honest answer for
    two players the model cannot separate at all.
    """
    spread = math.hypot(sd_a, sd_b)
    if spread <= 0:
        return 0.5
    return float(_standard_normal_cdf((mean_a - mean_b) / spread))


def _standard_normal_cdf(z: float) -> float:
    """Phi(z) via the error function -- no SciPy dependency for one call."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


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
    grid = np.asarray(RESIDUAL_QUANTILE_GRID, dtype=float) / 100.0
    uniform = rng.random((len(means), draws))
    z = np.empty_like(uniform)
    for pool in set(kinds):
        rows = (kinds == pool).to_numpy()
        z[rows] = np.interp(uniform[rows], grid, STD_RESIDUAL_QUANTILES[pool])
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
    outcomes = sample_outcomes(means, sds, kinds, draws=draws, seed=seed)
    if top_n >= len(means):
        return pd.Series(1.0, index=means.index)
    # Rank descending per simulated season; position < top_n means he made the cut.
    order = np.argsort(-outcomes, axis=0, kind="stable")
    placing = np.argsort(order, axis=0, kind="stable")
    return pd.Series((placing < top_n).mean(axis=1), index=means.index)


def probability_better_than_next(means: pd.Series, sds: pd.Series) -> pd.Series:
    """For a table ordered best-to-worst, P(each player beats the one below him).

    The last row has nobody below it and is NaN rather than 1.0. Pass the series
    already in display order; this does not sort, so a caller that re-sorts the
    table after calling this would silently pair the wrong players.
    """
    out = []
    values = list(zip(means.to_numpy(), sds.to_numpy(), strict=True))
    for index, (mean, sd) in enumerate(values):
        if index + 1 >= len(values):
            out.append(float("nan"))
            continue
        next_mean, next_sd = values[index + 1]
        out.append(probability_better(mean, sd, next_mean, next_sd))
    return pd.Series(out, index=means.index)
