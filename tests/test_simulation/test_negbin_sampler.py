import numpy as np

from fantasy_baseball.simulation import PITCHER_CORR_MATRIX, _negbin_copula_counts
from fantasy_baseball.utils.constants import PITCHER_CORR_STATS


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


def test_copula_preserves_sign_of_correlation():
    rng = np.random.default_rng(7)
    n = 100_000
    z = rng.multivariate_normal(np.zeros(len(PITCHER_CORR_STATS)), PITCHER_CORR_MATRIX, size=n)
    i_sv = PITCHER_CORR_STATS.index("sv")
    i_er = PITCHER_CORR_STATS.index("er")
    mu_sv = np.full(n, 25.0)
    r_sv = np.full(n, 2.0)
    mu_er = np.full(n, 65.0)
    r_er = np.full(n, 4.0)
    sv = _negbin_copula_counts(mu_sv, r_sv, z[:, i_sv], 1.0)
    er = _negbin_copula_counts(mu_er, r_er, z[:, i_er], 1.0)
    # PITCHER_CORRELATION has sv vs er at -0.341 -> realized correlation negative.
    assert np.corrcoef(sv, er)[0, 1] < -0.1


def test_finite_r_reproduces_full_variance_at_fraction_one():
    rng = np.random.default_rng(8)
    mu = np.full(400_000, 25.0)
    r = np.full(400_000, 3.0)
    z = rng.standard_normal(400_000)
    x = _negbin_copula_counts(mu, r, z, fraction_remaining=1.0)
    expected_var = 25.0 + 25.0**2 / 3.0  # mu + mu^2/r at fraction=1 (r_eff==r)
    assert abs(x.var() / expected_var - 1.0) < 0.03
    assert abs(x.mean() - 25.0) < 0.1
