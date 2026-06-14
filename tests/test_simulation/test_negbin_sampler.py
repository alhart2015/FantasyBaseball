import numpy as np

from fantasy_baseball.simulation import _negbin_copula_counts


def test_mean_is_unbiased_and_no_zero_spike():
    rng = np.random.default_rng(0)
    mu = np.full(200_000, 19.0)  # Cruz-like SB
    r = np.full(200_000, 2.2)
    z = rng.standard_normal(200_000)
    x = _negbin_copula_counts(mu, r, z, fraction_remaining=1.0)
    assert abs(x.mean() - 19.0) / 19.0 < 0.01  # no upward bias
    assert np.mean(x == 0) < 0.02  # no ~9% zero-spike
    assert np.all(x >= 0)
    assert np.all(x == np.floor(x))  # integer draws


def test_finite_under_extreme_latent():
    mu = np.array([19.0, 4.0])
    r = np.array([2.2, 3.8])
    z = np.array([40.0, -40.0])  # Phi(z) -> 1 / 0
    x = _negbin_copula_counts(mu, r, z, fraction_remaining=1.0)
    assert np.all(np.isfinite(x))


def test_fraction_remaining_scales_variance_linearly():
    rng = np.random.default_rng(1)
    mu = np.full(300_000, 30.0)
    r = np.full(300_000, 2.2)
    z = rng.standard_normal(300_000)
    var_full = (_negbin_copula_counts(mu, r, z, 1.0)).var()
    var_half = (_negbin_copula_counts(mu, r, z, 0.5)).var()
    assert abs(var_half / var_full - 0.5) < 0.06  # supra-floor regime


def test_poisson_sentinel_draws_poisson():
    rng = np.random.default_rng(2)
    mu = np.full(100_000, 8.0)
    r = np.full(100_000, np.inf)
    z = rng.standard_normal(100_000)
    x = _negbin_copula_counts(mu, r, z, fraction_remaining=1.0)
    assert abs(x.var() / x.mean() - 1.0) < 0.05  # Poisson: var == mean
