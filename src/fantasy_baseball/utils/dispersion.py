"""Resolve a STAT_DISPERSION value (scalar r or (mu_upper, r) bands) to per-element r."""

from __future__ import annotations

from typing import Any

import numpy as np

from fantasy_baseball.utils.constants import STAT_DISPERSION


def resolve_dispersion_r(value: float | list[tuple[float, float]], mu: Any) -> np.ndarray:
    """Per-element NegBin dispersion r for projected means mu.

    value is a scalar float (one r for all mu) or a list of (mu_upper, r) bands
    sorted ascending with the final mu_upper == inf. Each element takes the r of
    the first band whose mu_upper >= its mu (np.searchsorted, side="left").
    float("inf") as an r marks the Poisson floor and passes through unchanged.
    """
    mu = np.asarray(mu, dtype=float)
    if isinstance(value, (int, float)):
        return np.full(mu.shape, float(value))
    bounds = np.array([b for b, _ in value], dtype=float)
    rs = np.array([r for _, r in value], dtype=float)
    idx = np.clip(np.searchsorted(bounds, mu, side="left"), 0, len(rs) - 1)
    return np.asarray(rs[idx], dtype=float)


def negbin_variance_from_r(mu: Any, r: Any) -> np.ndarray:
    """Per-element NegBin performance variance ``mu + mu**2 / r`` from a resolved r.

    The single formula for per-stat performance dispersion, shared by
    :func:`negbin_perf_variance` (which resolves r from ``STAT_DISPERSION`` by
    stat key) and the MC's ``_negbin_copula_counts`` (which already holds the
    per-column r). An inf r (Poisson floor) yields ``var == mu``.
    """
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore"):
        overdispersion = np.where(np.isinf(r), 0.0, mu**2 / r)
    return np.asarray(mu + overdispersion, dtype=float)


def negbin_perf_variance(stat_key: str, mu: Any) -> np.ndarray:
    """Per-element NegBin performance variance ``mu + mu**2 / r``.

    r comes from ``resolve_dispersion_r(STAT_DISPERSION[stat_key], mu)``; an
    inf r (Poisson floor) yields ``var == mu``. Delegates the formula to
    :func:`negbin_variance_from_r` -- the single source of truth shared with the
    MC's ``_negbin_copula_counts``. Conditional on realized playing time (callers
    add the playing-time variance separately for counting stats).
    """
    mu = np.asarray(mu, dtype=float)
    r = resolve_dispersion_r(STAT_DISPERSION[stat_key], mu)
    return negbin_variance_from_r(mu, r)


def negbin_perf_cv(stat_key: str, mu: Any) -> np.ndarray:
    """Per-element performance CV ``sqrt(var)/mu == sqrt(1/mu + 1/r)`` (mu > 0).

    The multiplicative relative SD used by the pace color z-scores. Undefined at
    mu == 0 (callers guard expected/mu > 0).
    """
    mu = np.asarray(mu, dtype=float)
    var = negbin_perf_variance(stat_key, mu)
    return np.asarray(np.sqrt(var) / mu, dtype=float)
