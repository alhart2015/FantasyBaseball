"""Display helpers for the deltaRoto confidence band.

Single source of truth for the P(helps) verdict and the +/- label,
so every surface (roster audit, trade, compare, lineup) colors and
formats bands identically.
"""

from __future__ import annotations

from typing import Literal


def band_class(p_positive: float) -> Literal["real", "coin-flip", "downgrade"]:
    """Verdict from a deltaRoto band, keyed on P(helps) thresholds.

    real      -- P(helps) > 75%: the swap is a genuine upgrade with high confidence
    downgrade -- P(helps) < 25%: the swap hurts more often than it helps
    coin-flip -- 25% <= P(helps) <= 75%: not enough signal to call it either way

    Thresholds are strict (> 0.75, < 0.25), so p_positive == 0.75 and
    p_positive == 0.25 both fall into coin-flip.
    """
    if p_positive > 0.75:
        return "real"
    if p_positive < 0.25:
        return "downgrade"
    return "coin-flip"
