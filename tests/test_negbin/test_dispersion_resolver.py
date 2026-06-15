import numpy as np

from fantasy_baseball.utils.dispersion import resolve_dispersion_r


def test_scalar_broadcasts():
    out = resolve_dispersion_r(3.0, np.array([1.0, 50.0, 200.0]))
    assert np.allclose(out, [3.0, 3.0, 3.0])


def test_inf_scalar_passes_through():
    out = resolve_dispersion_r(float("inf"), np.array([5.0, 20.0]))
    assert np.all(np.isinf(out))


def test_banded_lookup_picks_first_band_with_upper_ge_mu():
    bands = [(5.0, 0.6), (12.0, 1.5), (float("inf"), 3.5)]
    out = resolve_dispersion_r(bands, np.array([3.0, 5.0, 5.1, 12.0, 100.0]))
    # mu<=5 -> 0.6; 5<mu<=12 -> 1.5; mu>12 -> 3.5
    assert np.allclose(out, [0.6, 0.6, 1.5, 1.5, 3.5])


def test_banded_with_poisson_floor_band():
    # A band whose r is the Poisson sentinel passes inf through for that range.
    bands = [(3.0, float("inf")), (float("inf"), 2.0)]
    out = resolve_dispersion_r(bands, np.array([1.0, 10.0]))
    assert np.isinf(out[0]) and np.isclose(out[1], 2.0)


def test_negbin_perf_variance_matches_mu_plus_mu2_over_r():
    import numpy as np

    from fantasy_baseball.utils.dispersion import negbin_perf_variance

    # k is scalar r=109.134 in STAT_DISPERSION; var = mu + mu^2/r
    mu = np.array([10.0, 100.0])
    out = negbin_perf_variance("k", mu)
    expected = mu + mu**2 / 109.134
    assert np.allclose(out, expected)


def test_negbin_perf_variance_poisson_floor_is_mu():
    import numpy as np

    from fantasy_baseball.utils.dispersion import negbin_perf_variance

    # h is float("inf") (Poisson) -> var == mu
    out = negbin_perf_variance("h", np.array([0.0, 50.0, 150.0]))
    assert np.allclose(out, [0.0, 50.0, 150.0])


def test_negbin_perf_variance_banded_uses_resolved_r():
    import numpy as np

    from fantasy_baseball.utils.constants import STAT_DISPERSION
    from fantasy_baseball.utils.dispersion import negbin_perf_variance, resolve_dispersion_r

    mu = np.array([2.0, 30.0])  # sb is banded
    r = resolve_dispersion_r(STAT_DISPERSION["sb"], mu)
    assert np.allclose(negbin_perf_variance("sb", mu), mu + mu**2 / r)


def test_negbin_perf_cv_is_sqrt_var_over_mu():
    import numpy as np

    from fantasy_baseball.utils.dispersion import negbin_perf_cv, negbin_perf_variance

    mu = np.array([5.0, 40.0])
    cv = negbin_perf_cv("sv", mu)
    assert np.allclose(cv, np.sqrt(negbin_perf_variance("sv", mu)) / mu)
    # Poisson floor: cv == sqrt(1/mu)
    assert np.allclose(negbin_perf_cv("h", mu), np.sqrt(1.0 / mu))


def test_negbin_variance_from_r_inf_and_finite_branches():
    import numpy as np

    from fantasy_baseball.utils.dispersion import negbin_variance_from_r

    # Mixed inf (Poisson floor -> var == mu) and finite r (var == mu + mu^2/r) in
    # one array -- the exact shape the MC passes directly (per-column r, some inf).
    # This pins the primitive's inf branch at its own boundary; the analytic
    # callers only reach it transitively on the finite-r path.
    mu = np.array([10.0, 20.0])
    r = np.array([np.inf, 5.0])
    out = negbin_variance_from_r(mu, r)
    assert np.allclose(out, [10.0, 20.0 + 20.0**2 / 5.0])
    # Scalar edges: mu=0 -> 0 (no NaN from 0/inf); finite-r scalar formula holds.
    assert np.isclose(float(negbin_variance_from_r(0.0, np.inf)), 0.0)
    assert np.isclose(float(negbin_variance_from_r(30.0, 7.0)), 30.0 + 30.0**2 / 7.0)
