"""Tests for the closer role-switch mixture (SV variance). See
docs/superpowers/specs/2026-07-10-closer-role-mixture-design.md.
"""

import numpy as np

from fantasy_baseball.sgp import closer_mixture as cm
from fantasy_baseball.utils.constants import STAT_DISPERSION


def _brute_force_var(s, q, a_m, a_s, r, n=2_000_000, seed=0):
    """Variance of the true generative process NegBin(s*X, r) over the role draw X."""
    rng = np.random.default_rng(seed)
    mult = np.where(rng.random(n) < q, a_m, a_s)
    m = s * mult
    lam = rng.gamma(r, m / r)  # NegBin(mean=m, disp=r) via gamma-poisson
    return float(rng.poisson(lam).var())


def test_closed_form_matches_generative():
    r = STAT_DISPERSION["sv"]
    s, q, a_m, a_s = 30.0, 0.55, 1.0 / 0.55, 0.0  # a_s=0 => a_m=1/q (mean-1)
    within = q * cm.nb_var(s * a_m) + (1 - q) * cm.nb_var(s * a_s)
    between = s**2 * q * (1 - q) * (a_m - a_s) ** 2
    closed = float(within + between)
    assert abs(closed - _brute_force_var(s, q, a_m, a_s, r)) / closed < 0.01
    naive = float(cm.nb_var(s)) + between  # WRONG single-mean form
    assert (closed - naive) / closed > 0.02  # closed keeps the ~2.4% between/r cross-term


def test_variance_nonnegative_and_vectorized():
    v = cm.sv_role_variance(np.array([0.0, 1.0, 8.0, 22.0, 40.0]))
    assert v.shape == (5,) and np.all(v >= 0)


def test_provisional_components_mean_one():
    q, a_m, a_s = cm._components(np.array([5.0, 30.0]))
    assert np.allclose(q * a_m + (1 - q) * a_s, 1.0)


def test_components_reflect_constant(monkeypatch):
    """_components must read constants.SV_ROLE_MIXTURE live (not an import-bound copy)."""
    monkeypatch.setattr(
        cm.constants, "SV_ROLE_MIXTURE", {"q_logit": [0.0, 0.1], "f_logit": [-2.0, 0.0]}
    )
    q0, _, _ = cm._components(0.0)
    q40, _, _ = cm._components(40.0)
    assert abs(float(q0) - 0.5) < 1e-6  # sigmoid(0)=0.5
    assert float(q40) > 0.95  # sigmoid(4)=0.982 -- proves it reads the coeffs
    q, a_m, a_s = cm._components(np.array([0.0, 40.0]))
    assert np.allclose(q * a_m + (1 - q) * a_s, 1.0)  # mean-1 still holds


def test_role_multiplier_draw_2d_moments():
    rng = np.random.default_rng(1)
    s2d = np.full((40_000, 1), 30.0)  # 2-D: between-variance lives ACROSS rows
    for frac in (1.0, 0.5, 0.25):
        x = cm.role_multiplier_draw(s2d, rng, fraction_remaining=frac)
        assert x.shape == s2d.shape
        assert abs(x.mean() - 1.0) < 0.02  # E[X']=1
        assert np.all(x >= 0)
    x1 = cm.role_multiplier_draw(np.full((200_000, 1), 30.0), np.random.default_rng(2), 1.0)
    xh = cm.role_multiplier_draw(np.full((200_000, 1), 30.0), np.random.default_rng(2), 0.5)
    q, a_m, a_s = cm._components(np.asarray(30.0))
    theo = float(q * (1 - q) * (a_m - a_s) ** 2)  # Var(X) at s=30
    assert theo > 1e-3  # scaffold is non-degenerate
    # the 2-D draw reproduces Var(X); a 1-D broadcast (the F1 bug) would give ~0
    assert abs(x1.var() - theo) / theo < 0.1
    assert abs(xh.var() / x1.var() - 0.5) < 0.05  # Var scales ~ frac
