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
