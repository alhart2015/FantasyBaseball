"""Analytic per-category finish odds for the Category Bars view.

Given each team's projected mean and SD in one roto category, computes the
user team's probability of finishing 1st and top-3 in that category, plus a
count of opponents the user's +/-1 SD band clearly clears. Pure functions,
no I/O.

The probabilities integrate the user's projected normal distribution against
the opponents' normals via Gauss-Hermite quadrature, treating opponents as
independent given the user's draw -- each fantasy team's total is built from
a disjoint set of players. This is the same Gaussian model the chart's bands
come from, so the odds agree with what is drawn. Higher-is-better and inverse
(ERA/WHIP) categories are unified by negating the means for inverse cats (SD
is unchanged under negation), so every formula below reads "bigger wins".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erf, pi, sqrt

import numpy as np

# Gauss-Hermite nodes/weights (probabilists', weight exp(-x^2/2)) for
# E[g(X)], X ~ Normal. 24 nodes: the integrands (a product of normal CDFs and
# a Poisson-binomial CDF) are smooth and bounded, so this is far more than
# enough and still costs microseconds. Materialized to plain float tuples so
# numpy is touched only at import time.
_GH_N = 24
_gh_nodes, _gh_weights = np.polynomial.hermite_e.hermegauss(_GH_N)
_GH_NODES: tuple[float, ...] = tuple(float(x) for x in _gh_nodes)
_GH_WEIGHTS: tuple[float, ...] = tuple(float(x) for x in _gh_weights)
_SQRT_2PI = sqrt(2.0 * pi)


@dataclass
class CategoryOdds:
    """User-team odds for one category. Percentages are 0-100, unrounded."""

    first_pct: float
    top3_pct: float
    clear_wins: int
    opponents: int


def _prob_opp_above(x: float, mu: float, sd: float) -> float:
    """P(an opponent ~ N(mu, sd) exceeds the fixed value x)."""
    if sd == 0.0:
        # Degenerate (no projection uncertainty): step function, same
        # convention as scoring._prob_beats's combined-SD==0 branch (0.5 tie).
        if mu > x:
            return 1.0
        if mu < x:
            return 0.0
        return 0.5
    z = (x - mu) / sd
    cdf_below = 0.5 * (1.0 + erf(z / sqrt(2.0)))  # P(opp < x)
    return 1.0 - cdf_below


def _poisson_binomial_le2(qs: Sequence[float]) -> tuple[float, float]:
    """Return (P(k=0), P(k<=2)) for independent Bernoulli(q) opponents.

    k is the number of opponents that beat the user. Exact O(n^2) DP over the
    full pmf -- robust when a q is exactly 0 or 1.
    """
    pmf = [1.0]
    for q in qs:
        nxt = [0.0] * (len(pmf) + 1)
        for count, p in enumerate(pmf):
            nxt[count] += p * (1.0 - q)
            nxt[count + 1] += p * q
        pmf = nxt
    p0 = pmf[0]
    p_le2 = sum(pmf[:3])
    return p0, p_le2


def category_finish_odds(
    means: Sequence[float],
    sds: Sequence[float],
    user_index: int,
    *,
    higher_is_better: bool,
) -> CategoryOdds:
    """Analytic odds the user finishes 1st / top-3 in one roto category.

    ``means``/``sds`` are parallel per-team sequences; ``user_index`` selects
    the user. ``higher_is_better`` is False for ERA/WHIP. Percentages are
    0-100 (unrounded floats).
    """
    n = len(means)
    sign = 1.0 if higher_is_better else -1.0
    mu = [sign * m for m in means]
    mu_u = mu[user_index]
    sd_u = sds[user_index]
    opponents = [(mu[i], sds[i]) for i in range(n) if i != user_index]

    e_first = 0.0
    e_top3 = 0.0
    for node, weight in zip(_GH_NODES, _GH_WEIGHTS, strict=True):
        x = mu_u + sd_u * node
        qs = [_prob_opp_above(x, mu_o, sd_o) for (mu_o, sd_o) in opponents]
        p0, p_le2 = _poisson_binomial_le2(qs)
        w = weight / _SQRT_2PI
        e_first += w * p0
        e_top3 += w * p_le2

    lower_u = mu_u - sd_u
    clear_wins = sum(1 for (mu_o, sd_o) in opponents if lower_u > mu_o + sd_o)

    return CategoryOdds(
        first_pct=float(100.0 * e_first),
        top3_pct=float(100.0 * e_top3),
        clear_wins=clear_wins,
        opponents=len(opponents),
    )
