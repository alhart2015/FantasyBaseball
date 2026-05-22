"""Display helpers for the deltaRoto confidence band.

Single source of truth for the crosses-zero verdict and the +/- label,
so every surface (roster audit, trade, compare, lineup) colors and
formats bands identically.
"""

from __future__ import annotations

from typing import Literal


def band_class(mean: float, sd: float) -> Literal["real", "coin-flip", "downgrade"]:
    """Verdict from a deltaRoto band, keyed on whether +/-1 SD crosses zero.

    real      -- band entirely above zero (mean - sd > 0, ~P(help) >= 84%)
    downgrade -- band entirely below zero (mean + sd < 0)
    coin-flip -- band straddles zero
    """
    if mean - sd > 0:
        return "real"
    if mean + sd < 0:
        return "downgrade"
    return "coin-flip"
