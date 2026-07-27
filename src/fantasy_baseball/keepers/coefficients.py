"""The shipped output of the #266 calibration study.

These are the numbers increment 2 applies. They live in code rather than only in
the finding document so the production path imports them instead of retyping a
markdown table, and so `tests/test_keepers/test_coefficients.py` can hold them
against the study's own CSV output.

Full derivation and every caveat: `docs/superpowers/keeper-calibration-finding-2026-07-27.md`.
Regenerate the CSVs with `python scripts/keeper_calibration.py`.

Three things a caller must not get wrong:

  * **The coefficients are conditional on `n0`.** `k` and the shrink enter the
    fold multiplicatively, and the study found `k` scales almost exactly
    inversely with `n0` while held-out error does not move. Only the product
    `k * w` is identified. Using these `k` values with a different `n0` is
    meaningless (finding B.5).
  * **Apply `fold.gate_ramp`, not `fold.gate_mask`.** The hard gate is a 78.7%
    (hitter) / 44.6% (pitcher) cliff across two plate appearances, because the
    playing-time term is unshrunk. `gate_mask` selects fit-sample rows; the ramp
    is what production applies (finding B.7).
  * **`pa` and `k_ip` are endpoint fallbacks, not fitted values.** Both lost to
    `k=1` on held-out error under the pre-registered rule. For `pa` the mechanism
    is understood and unresolved -- see finding B.3, the largest open item.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class FoldPolicy:
    """Everything the production fold needs for one player type."""

    coefficients: Mapping[str, float]
    """Per-column transfer coefficient `k`, conditional on `n0`."""

    n0: float
    """Shrink constant: the playing time at which observation and projection get
    equal weight, in `w = n / (n + n0)`."""

    gate: float
    """Realized playing time below which a player is not folded at all."""

    ramp_width: float
    """Width of the linear on-ramp above `gate`, for `fold.gate_ramp`."""


# Fitted 2026-07-27 over pairs 2022->2023, 2023->2024, 2024->2025.
# `pa` is the k=1 fallback (fitted 0.646, CI [0.584, 0.708]); see finding B.3.
HITTER = FoldPolicy(
    coefficients=MappingProxyType(
        {
            "hr_pa": 0.494,
            "r_pa": 0.531,
            "rbi_pa": 0.532,
            "sb_pa": 0.637,  # provisional -- widest CI, 2023 rules break (B.4)
            "h_ab": 0.428,
            "ab_pa": 0.687,
            "pa": 1.000,  # fallback:k=1
        }
    ),
    n0=200.0,
    gate=100.0,
    ramp_width=100.0,
)

# `k_ip` is the k=1 fallback (fitted 0.970, CI [0.866, 1.073]); the fitted value
# and the endpoint are 0.4% apart, so strikeout rate carries forward in full.
PITCHER = FoldPolicy(
    coefficients=MappingProxyType(
        {
            "k_ip": 1.000,  # fallback:k=1
            "w_ip": 0.491,
            "er_ip": 0.343,
            "bb_ip": 0.697,
            "h_ip": 0.385,
            "ip": 0.631,
        }
    ),
    n0=50.0,
    gate=50.0,
    ramp_width=50.0,
)

POLICIES = MappingProxyType({"hitter": HITTER, "pitcher": PITCHER})
