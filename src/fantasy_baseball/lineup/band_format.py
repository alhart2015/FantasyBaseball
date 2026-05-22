"""Display helpers for the deltaRoto confidence band.

Single source of truth for the crosses-zero verdict and the +/- label,
so every surface (roster audit, trade, compare, lineup) colors and
formats bands identically.
"""

from __future__ import annotations


def band_class(mean: float, sd: float) -> str:
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


def band_label(mean: float, sd: float) -> str:
    """ASCII band string, e.g. ``+1.9 +/- 2.3``."""
    return f"{mean:+.1f} +/- {sd:.1f}"
