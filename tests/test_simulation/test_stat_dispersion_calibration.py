import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from calibrate_stat_dispersion import POISSON_SENTINEL, fit_dispersion


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
