"""Tests for the closer role-switch mixture (SV variance). See
docs/superpowers/specs/2026-07-10-closer-role-mixture-design.md.
"""

import numpy as np

from fantasy_baseball.sgp import closer_mixture as cm
from fantasy_baseball.utils.constants import STAT_DISPERSION


def _brute_force_var(s, p, a, r, n=2_000_000, seed=0):
    """Variance of the true generative process NegBin(s*X, r) over the categorical X."""
    rng = np.random.default_rng(seed)
    comp = rng.choice(len(p), size=n, p=p)
    m = s * np.asarray(a)[comp]
    lam = rng.gamma(r, m / r)  # NegBin(mean=m, disp=r) via gamma-poisson
    return float(rng.poisson(lam).var())


def test_closed_form_matches_generative():
    r = STAT_DISPERSION["sv"]
    s = 30.0
    p, a = cm._components(np.asarray(s))  # (K,), (K,) at this s
    closed = float(cm.sv_role_variance(s))
    brute = _brute_force_var(s, p, a, r)
    assert abs(closed - brute) / closed < 0.01
    # the naive single-mean form omits the between/r cross-term; the closed form must
    # exceed it by EXACTLY between/r (the verified identity), a non-trivial slice.
    ex2 = float(np.sum(p * a * a))
    between = s**2 * (ex2 - 1.0)
    naive = float(cm.nb_var(s)) + between
    assert abs((closed - naive) - between / float(r)) < 1e-6 * closed
    assert between / float(r) > 0.01 * closed


def test_variance_nonnegative_and_vectorized():
    v = cm.sv_role_variance(np.array([0.0, 1.0, 8.0, 22.0, 40.0]))
    assert v.shape == (5,) and np.all(v >= 0)


def test_components_mean_one():
    p, a = cm._components(np.array([0.5, 5.0, 18.0, 35.0]))
    assert np.allclose(p.sum(axis=-1), 1.0)  # probabilities normalize
    assert np.allclose(np.sum(p * a, axis=-1), 1.0)  # mean-1
    assert np.all(a >= 0)


def test_components_reflect_constant(monkeypatch):
    """_components must read constants.SV_ROLE_MIXTURE live (not an import-bound copy)."""
    monkeypatch.setattr(
        cm.constants,
        "SV_ROLE_MIXTURE",
        {"p_logits": [[2.0, 0.0], [0.0, 0.0]], "w_logits": [[0.0, 0.0], [0.0, 0.0]]},
    )
    p, a = cm._components(np.asarray(0.0))
    # component 0 has logit 2.0 (others 0) -> softmax favors it; shares are uniform 1/3
    assert p[0] > p[1] and p[0] > p[2]
    assert np.allclose(np.sum(p * a), 1.0)  # mean-1 regardless of coeffs


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
    p, a = cm._components(np.asarray(30.0))
    theo = float(np.sum(p * a * a) - 1.0)  # Var(X) at s=30
    assert theo > 1e-3  # scaffold is non-degenerate
    # the 2-D draw reproduces Var(X); a 1-D broadcast (the F1 bug) would give ~0
    assert abs(x1.var() - theo) / theo < 0.1
    assert abs(xh.var() / x1.var() - 0.5) < 0.05  # Var scales ~ frac
