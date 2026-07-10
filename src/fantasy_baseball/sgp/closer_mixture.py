"""Bimodal role-switch mixture for saves (SV) variance. Single source of truth
for the SV dispersion shared by ERoto (scoring.py) and the MC (simulation.py).

X is a mean-1 multiplier on a pitcher's SV mean:
  X = a_m w.p. q(s), else a_s w.p. 1-q(s);  q*a_m + (1-q)*a_s = 1  (E[X]=1).
FULL-SEASON variance (law of total variance over role AND NegBin), r fixed at
STAT_DISPERSION['sv']:
  within  = q*nb_var(s*a_m) + (1-q)*nb_var(s*a_s)
  between = s^2 * q*(1-q) * (a_m - a_s)^2
so within + between == negbin_perf_variance(s) + between*(1 + 1/r) (the between/r
cross-term the naive single-mean form omits, ~2.4% for a 30-SV closer).

In-season scaling is applied EXTERNALLY and uniformly -- ERoto via build_team_sds
(sd_scale = sqrt(frac)), the MC via the copula (within) plus role_multiplier_draw's
X' shrink (between) -- so sv_role_variance is FULL-SEASON and takes no frac param.

Known limitations (second-order, do not touch the marginal SV variance target):
SV pulled out of the shared MC ``scales`` loses its playing-time co-movement with
W/K/ER; the role Bernoulli is independent of the copula so job-loss does not
co-move with high ER; handcuffed closers' save anti-correlation is not modeled.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from fantasy_baseball.utils import constants
from fantasy_baseball.utils.constants import STAT_DISPERSION

_R: float = cast(float, STAT_DISPERSION["sv"])


def _components(s: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(q, a_m, a_s) for projected SV ``s``, mean-1 and non-negative by construction.

    Parameterized by two logistic curves: the modal probability ``q(s)`` and the
    surprise component's share ``f(s)`` of the unit mean. Then ``a_s = f/(1-q)`` and
    ``a_m = (1-f)/q`` -- so ``q*a_m + (1-q)*a_s == 1`` exactly and both multipliers are
    >= 0 for any ``q, f`` in (0,1). (A direct ``a_s(s)`` curve could force a negative
    ``a_m`` at low ``s`` where the rare vault multiplier is large.)

    Reads ``constants.SV_ROLE_MIXTURE`` module-qualified each call so the calibration
    (Task 5) regeneration and test monkeypatches both take effect.
    """
    s = np.asarray(s, dtype=float)
    b0, b1 = constants.SV_ROLE_MIXTURE["q_logit"]
    g0, g1 = constants.SV_ROLE_MIXTURE["f_logit"]
    q = np.clip(1.0 / (1.0 + np.exp(-(b0 + b1 * s))), 1e-3, 1 - 1e-3)
    f = np.clip(1.0 / (1.0 + np.exp(-(g0 + g1 * s))), 1e-9, 1 - 1e-6)
    a_s = f / (1.0 - q)
    a_m = (1.0 - f) / q
    return q, a_m, a_s


def nb_var(m: Any) -> np.ndarray:
    """Per-element NegBin performance variance ``m + m^2/r`` at the fixed SV ``r``."""
    m = np.asarray(m, dtype=float)
    return np.asarray(m + m * m / _R, dtype=float)


def sv_role_variance(s: Any) -> Any:
    """Full-season SV variance ``within + between`` keyed and meaned on projected SV ``s``.

    Returns a float for scalar ``s``, else an ndarray. In-season scaling is applied
    externally (see module docstring); this is the full-season term.
    """
    s = np.asarray(s, dtype=float)
    q, a_m, a_s = _components(s)
    within = q * nb_var(s * a_m) + (1.0 - q) * nb_var(s * a_s)
    between = s**2 * q * (1.0 - q) * (a_m - a_s) ** 2
    out = np.asarray(within + between, dtype=float)
    return float(out) if out.ndim == 0 else out


def role_multiplier_draw(
    s: Any, rng: np.random.Generator, fraction_remaining: float = 1.0
) -> np.ndarray:
    """Per-draw mean-1 SV multiplier ``X'`` the same shape as ``s`` (the raw projected SV).

    The caller broadcasts ``s`` to the shape it wants sampled -- the MC passes 2-D
    ``(n_iter, n_players)`` so the role varies per iteration (the source of the
    between-component variance). ``X' = 1 + sqrt(frac)*(X - 1)`` gives ``E[X']=1``,
    ``Var(X') = frac*Var(X)``, ``X' >= 0``.
    """
    s = np.asarray(s, dtype=float)
    q, a_m, a_s = _components(s)
    x = np.where(rng.random(s.shape) < q, a_m, a_s)
    x_prime = 1.0 + np.sqrt(fraction_remaining) * (x - 1.0)
    return np.asarray(np.maximum(x_prime, 0.0), dtype=float)
