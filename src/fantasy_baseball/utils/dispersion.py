"""Resolve a STAT_DISPERSION value (scalar r or (mu_upper, r) bands) to per-element r."""

from __future__ import annotations

from typing import Any

import numpy as np


def resolve_dispersion_r(value: float | list[tuple[float, float]], mu: Any) -> np.ndarray:
    """Per-element NegBin dispersion r for projected means mu.

    value is a scalar float (one r for all mu) or a list of (mu_upper, r) bands
    sorted ascending with the final mu_upper == inf. Each element takes the r of
    the first band whose mu_upper >= its mu (np.searchsorted, side="left").
    float("inf") as an r marks the Poisson floor and passes through unchanged.
    """
    mu = np.asarray(mu, dtype=float)
    if isinstance(value, (int, float)):
        return np.full(mu.shape, float(value))
    bounds = np.array([b for b, _ in value], dtype=float)
    rs = np.array([r for _, r in value], dtype=float)
    idx = np.clip(np.searchsorted(bounds, mu, side="left"), 0, len(rs) - 1)
    return np.asarray(rs[idx], dtype=float)
