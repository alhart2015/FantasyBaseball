import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from calibrate_stat_dispersion import POISSON_SENTINEL, fit_dispersion, interval_coverage


def test_fit_dispersion_recovers_known_r():
    rng = np.random.default_rng(0)
    mu = np.full(20_000, 12.0)
    true_r = 2.5
    p = true_r / (true_r + mu)
    x = rng.negative_binomial(true_r, p)
    r_hat = fit_dispersion(x, mu)
    assert abs(r_hat - true_r) < 0.3


def test_fit_dispersion_clamps_to_poisson_when_underdispersed():
    rng = np.random.default_rng(1)
    mu = np.full(20_000, 8.0)
    x = rng.poisson(mu)  # var == mean -> no overdispersion
    assert fit_dispersion(x, mu) == POISSON_SENTINEL


def test_interval_coverage_matches_nominal_when_model_is_correct():
    rng = np.random.default_rng(2)
    mu = np.full(50_000, 15.0)
    r = 3.0
    p = r / (r + mu)
    x = rng.negative_binomial(r, p)
    cov80 = interval_coverage(x, mu, r, level=0.80)
    assert abs(cov80 - 0.80) < 0.03


def test_interval_coverage_handles_poisson_sentinel():
    # ppf-based central intervals for discrete distributions are conservative:
    # the actual coverage >= the nominal level rather than approximately equal.
    # For Poisson(6) at level=0.50, ppf(0.25)=4 and ppf(0.75)=8 enclose ~70%
    # of the mass, so we assert that coverage is at least 0.50 and within 0.25.
    rng = np.random.default_rng(3)
    mu = np.full(50_000, 6.0)
    x = rng.poisson(mu)
    cov50 = interval_coverage(x, mu, POISSON_SENTINEL, level=0.50)
    assert cov50 >= 0.50
    assert abs(cov50 - 0.50) < 0.25
