"""Compact outcome-distribution curves for the Monte Carlo standings ridgeline.

``run_ros_monte_carlo`` retains per-iteration samples transiently; these helpers
collapse them into small KDE curves (continuous metrics) and exact PMFs (discrete
category roto-points) that are cheap to cache and ship to the browser. Every
output is plain Python floats -- JSON-serializable, no numpy types in the payload.
"""

from typing import Any, cast

import numpy as np

# Grid resolution and the metric-relative bandwidth floor. Tunable; see the spec
# "Open questions" -- these are visual knobs, not data-contract values.
GRID_POINTS = 60
BW_FLOOR_FRACTION = 0.01  # floor = 1% of a metric's pooled (post-sentinel) range


def _clean_samples(samples: Any, sentinel: float | None) -> np.ndarray:
    """Drop non-finite values (and an optional sentinel) from a sample array.

    ERA/WHIP carry a ``99.0`` zero-IP sentinel from the batch simulation; left in,
    it would stretch the grid and paint a phantom tail near 99.
    """
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if sentinel is not None:
        arr = arr[arr != sentinel]
    return arr


def _silverman_bandwidth(samples: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth; 0.0 for <2 points or zero spread."""
    n = samples.size
    if n < 2:
        return 0.0
    std = float(np.std(samples, ddof=1))
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = float(q75 - q25)
    spread = min(std, iqr / 1.349) if iqr > 0 else std
    return 0.9 * spread * n ** (-0.2)


def _gaussian_kde_curve(samples: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian KDE of ``samples`` sampled on ``grid``, normalized to integrate ~1.

    ``bw <= 0`` (a degenerate, zero-variance team) collapses to a single spike at
    the median rather than dividing by zero. Defensive: ``build_continuous_metric``
    always applies a positive metric-relative floor, so the real callers never pass
    ``bw <= 0`` -- this guard only protects direct/standalone use of the helper.
    """
    if bw <= 0.0:
        y = np.zeros_like(grid)
        y[int(np.argmin(np.abs(grid - float(np.median(samples)))))] = 1.0
        return cast(np.ndarray, y)
    z = (grid[:, None] - samples[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (samples.size * bw * np.sqrt(2.0 * np.pi))
    area = float(np.trapezoid(dens, grid))  # np.trapz removed in numpy 2.0+
    if area > 0.0:
        dens = dens / area
    return cast(np.ndarray, dens)


def build_continuous_metric(team_samples: dict[str, Any], sentinel: float | None = None) -> dict:
    """Build a shared-grid KDE ridgeline payload for one continuous metric.

    ``team_samples`` is ``{team: sample_array}``. Returns
    ``{"x": [...], "teams": {team: {"y": [...], "median": float}}}`` where ``x`` is
    one grid shared by every team (so ridgeline rows are horizontally comparable),
    bandwidth is per-team (Silverman with a metric-relative floor), and the grid is
    padded by ``3 * bw_max`` so no team's tails clip. Teams with no usable samples
    are omitted.
    """
    cleaned = {}
    for name, raw in team_samples.items():
        arr = _clean_samples(raw, sentinel)
        if arr.size > 0:
            cleaned[name] = arr
    if not cleaned:
        return {"x": [], "teams": {}}

    pooled = np.concatenate(list(cleaned.values()))
    lo = float(pooled.min())
    hi = float(pooled.max())
    span = hi - lo
    bw_floor = max(1e-9, BW_FLOOR_FRACTION * span)
    bws = {name: max(_silverman_bandwidth(arr), bw_floor) for name, arr in cleaned.items()}
    bw_max = max(bws.values())
    grid = np.linspace(lo - 3.0 * bw_max, hi + 3.0 * bw_max, GRID_POINTS)

    teams = {}
    for name, arr in cleaned.items():
        y = _gaussian_kde_curve(arr, grid, bws[name])
        teams[name] = {
            "y": [float(v) for v in y],
            "median": float(np.median(arr)),
        }
    return {"x": [float(v) for v in grid], "teams": teams}


def build_discrete_metric(team_samples: dict[str, Any]) -> dict:
    """Build a shared-support PMF ridgeline payload for one discrete metric.

    ``team_samples`` is ``{team: point_value_array}`` (category roto points, which
    are half-integers under tie-splitting). Returns
    ``{"x": [...], "teams": {team: {"p": [...], "mean": float}}}`` where ``x`` is
    the sorted union of distinct point values observed across ALL teams and each
    team's ``p`` is aligned to that shared ``x`` (0 at unobserved values), so a
    ridgeline can stack the rows on one axis. Teams with no samples are omitted.
    """
    cleaned = {}
    for name, raw in team_samples.items():
        arr = _clean_samples(raw, None)
        if arr.size > 0:
            # Snap to the nearest 0.5 so tie-split values compare exactly.
            cleaned[name] = np.round(arr * 2.0) / 2.0
    if not cleaned:
        return {"x": [], "teams": {}}

    support = np.unique(np.concatenate(list(cleaned.values())))
    teams = {}
    for name, arr in cleaned.items():
        counts = np.array([np.count_nonzero(arr == v) for v in support], dtype=float)
        p = counts / counts.sum()
        teams[name] = {
            "p": [float(v) for v in p],
            "mean": float(np.sum(support * p)),
        }
    return {"x": [float(v) for v in support], "teams": teams}


# ERA/WHIP carry a 99.0 zero-IP sentinel from simulate_remaining_season_batch.
_SENTINEL_CATS = {"ERA", "WHIP"}
_SENTINEL_VALUE = 99.0


def build_distributions(
    all_totals: dict[str, list[float]],
    batch: dict[str, dict[str, np.ndarray]],
    all_cat_pts: dict[str, dict[str, list[float]]],
    cats: list[str],
    user_team: str,
) -> dict:
    """Assemble the full ``distributions`` payload from the MC's transient arrays.

    - ``overall``: KDE of each team's total roto points (``all_totals``).
    - ``category_totals``: KDE of each team's raw stat total per category
      (``batch``); ERA/WHIP drop the 99.0 sentinel.
    - ``category_points``: exact PMF of each team's roto points per category
      (``all_cat_pts``).
    ``user_team`` is carried through for the formatter to mark ``is_user``.
    """
    overall = build_continuous_metric(
        {name: np.asarray(v, dtype=float) for name, v in all_totals.items()}
    )

    category_totals = {}
    category_points = {}
    for cat in cats:
        sentinel = _SENTINEL_VALUE if cat in _SENTINEL_CATS else None
        category_totals[cat] = build_continuous_metric(
            {name: batch[name][cat] for name in batch}, sentinel=sentinel
        )
        category_points[cat] = build_discrete_metric(
            {name: all_cat_pts[name][cat] for name in all_cat_pts}
        )

    return {
        "overall": overall,
        "category_totals": category_totals,
        "category_points": category_points,
        "user_team": user_team,
    }
