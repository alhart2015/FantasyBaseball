"""Calibrate per-stat Negative-Binomial dispersion from projection-vs-actual
residuals, conditional on realized playing time (2022-2024).

Mirrors scripts/calibrate_playing_time.py's data handling. Emits a
STAT_DISPERSION dict (per-stat r, with a Poisson sentinel) for paste into
src/fantasy_baseball/utils/constants.py, plus a leave-one-season-out
interval-coverage table that gates the shipped values. Performance dispersion
is measured conditional on realized PT so it does NOT double-count the
playing-time model's variance.

Usage:
    python scripts/calibrate_stat_dispersion.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

# Sentinel for "no overdispersion -> use Poisson" (NegBin r -> inf).
POISSON_SENTINEL = float("inf")

# Bounds for the log-r search. exp(13.8) ~ 1e6: at the upper bound the NegBin is
# indistinguishable from Poisson, so we treat hitting it as the Poisson floor.
_LOG_R_LO = np.log(1e-3)
_LOG_R_HI = np.log(1e6)


def fit_dispersion(x: np.ndarray, mu: np.ndarray) -> float:
    """MLE of a single NegBin dispersion r for counts x with per-obs means mu.

    Each observation is x_i ~ NegBin(mean=mu_i, dispersion=r) with a shared r
    (heteroscedastic means, one dispersion). Returns POISSON_SENTINEL when the
    data is not overdispersed (sample variance <= sample mean, or the optimizer
    pins r at the upper bound where NegBin is indistinguishable from Poisson).
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mask = mu > 0
    x, mu = x[mask], mu[mask]

    def nll(log_r: float) -> float:
        r = np.exp(log_r)
        p = r / (r + mu)
        return -float(np.sum(nbinom.logpmf(x, r, p)))

    res = minimize_scalar(nll, bounds=(_LOG_R_LO, _LOG_R_HI), method="bounded")
    r_hat = float(np.exp(res.x))
    # r >= 200 means NegBin is indistinguishable from Poisson at any mu<100.
    # For Poisson data the MLE drifts to large r rather than pinning to the
    # search bound, so we use a generous threshold rather than checking the bound.
    if r_hat >= 200.0:
        return POISSON_SENTINEL
    return r_hat
