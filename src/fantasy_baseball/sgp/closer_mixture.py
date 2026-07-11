"""Bimodal/multimodal role-switch mixture for saves (SV) variance. Single source of
truth for the SV dispersion shared by ERoto (scoring.py) and the MC (simulation.py).

X is a mean-1 multiplier on a pitcher's SV mean, a K-component mixture keyed on
projected SV ``s``:
  X = a_k  with prob p_k(s);   sum_k p_k = 1,   sum_k p_k a_k = 1  (E[X]=1).
Parameterized by two K-way softmaxes over ``s`` -- the component probabilities p(s)
and their shares w(s) of the unit mean -- with a_k = w_k / p_k. Then
``sum p_k a_k = sum w_k = 1`` exactly and every a_k >= 0, for any softmax outputs.
(A direct multiplier curve could force a negative multiplier at low ``s`` where the
rare vault component is large.)

FULL-SEASON variance (law of total variance over role AND NegBin), r fixed at
STAT_DISPERSION['sv']:
  within  = sum_k p_k * nb_var(s*a_k)
  between = s^2 * (sum_k p_k a_k^2 - 1) = s^2 * (sum_k w_k^2/p_k - 1)
so within + between == negbin_perf_variance(s) + between*(1 + 1/r) (the between/r
cross-term the naive single-mean form omits).

In-season scaling is applied EXTERNALLY and uniformly -- ERoto via build_team_sds
(sd_scale = sqrt(frac)), the MC via the copula (within) plus role_multiplier_draw's
X' shrink (between) -- so sv_role_variance is FULL-SEASON and takes no frac param.

Known limitations (second-order, do not touch the marginal SV variance target):
SV pulled out of the shared MC ``scales`` loses its playing-time co-movement with
W/K/ER; the role draw is independent of the copula so job-loss does not co-move with
high ER; handcuffed closers' save anti-correlation is not modeled.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from fantasy_baseball.utils import constants
from fantasy_baseball.utils.constants import STAT_DISPERSION
from fantasy_baseball.utils.dispersion import negbin_variance_from_r

_R: float = cast(float, STAT_DISPERSION["sv"])


def _softmax_curves(s: np.ndarray, logit_curves: list[list[float]]) -> np.ndarray:
    """K-way softmax over ``s``. ``logit_curves`` holds K-1 free ``[b0, b1]`` logit
    lines; a zero logit is appended for the reference component. Returns shape
    ``(*s.shape, K)``."""
    logits = [b0 + b1 * s for b0, b1 in logit_curves]
    logits.append(np.zeros_like(s))
    stacked = np.stack(logits, axis=-1)
    stacked = stacked - stacked.max(axis=-1, keepdims=True)
    e = np.exp(stacked)
    return np.asarray(e / e.sum(axis=-1, keepdims=True), dtype=float)


def _components(s: Any) -> tuple[np.ndarray, np.ndarray]:
    """(p, a) each shape ``(*s.shape, K)`` -- component probabilities and multipliers,
    mean-1 (``sum p*a = 1``) and non-negative by construction.

    Reads ``constants.SV_ROLE_MIXTURE`` module-qualified each call so the calibration
    (Task 5) regeneration and test monkeypatches both take effect.
    """
    s = np.asarray(s, dtype=float)
    p = _softmax_curves(s, constants.SV_ROLE_MIXTURE["p_logits"])
    w = _softmax_curves(s, constants.SV_ROLE_MIXTURE["w_logits"])
    a = w / p
    return p, a


def nb_var(m: Any) -> np.ndarray:
    """Per-element NegBin performance variance ``m + m^2/r`` at the fixed SV ``r``.

    Delegates to the shared ``negbin_variance_from_r`` (single source of truth for the
    NegBin variance formula, also used by the MC's ``_negbin_copula_counts``)."""
    return negbin_variance_from_r(m, _R)


def sv_role_variance(s: Any) -> Any:
    """Full-season SV variance ``within + between`` keyed and meaned on projected SV ``s``.

    Returns a float for scalar ``s``, else an ndarray. In-season scaling is applied
    externally (see module docstring); this is the full-season term.
    """
    s = np.asarray(s, dtype=float)
    p, a = _components(s)
    mu = s[..., None] * a
    within = np.sum(p * nb_var(mu), axis=-1)
    ex2 = np.sum(p * a * a, axis=-1)
    between = s**2 * (ex2 - 1.0)
    out = np.asarray(within + between, dtype=float)
    return float(out) if out.ndim == 0 else out


def role_multiplier_draw(
    s: Any,
    rng: np.random.Generator,
    fraction_remaining: float = 1.0,
    n_iter: int | None = None,
) -> np.ndarray:
    """Per-draw mean-1 SV multiplier ``X'`` keyed on the raw projected SV ``s`` (1-D).

    Returns shape ``s.shape`` when ``n_iter`` is None (scalar MC path), else
    ``(n_iter, *s.shape)`` -- the role drawn independently per iteration (the source of
    the between-component variance). Components depend only on ``s``, so they are computed
    once; only the uniform draw and the categorical index take the full shape.
    ``X' = 1 + sqrt(frac)*(X - 1)`` gives ``E[X']=1``, ``Var(X') = frac*Var(X)``, ``X' >= 0``.
    """
    s = np.asarray(s, dtype=float)
    p, a = _components(s)  # (*s.shape, K), computed once on the unique projections
    cum = np.cumsum(p, axis=-1)
    shape = s.shape if n_iter is None else (n_iter, *s.shape)
    idx = (rng.random(shape)[..., None] < cum).argmax(axis=-1)
    x = np.take_along_axis(np.broadcast_to(a, (*idx.shape, a.shape[-1])), idx[..., None], -1)[
        ..., 0
    ]
    x_prime = 1.0 + np.sqrt(fraction_remaining) * (x - 1.0)
    return np.asarray(np.maximum(x_prime, 0.0), dtype=float)
