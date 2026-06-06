"""Playing-time model lookup: projected volume -> (mean_scale, cv_pt).

Single source of truth for how realized PA/IP deviates from projection,
shared by the Monte Carlo sampler (``simulation._apply_variance``) and ERoto
(``scoring.project_team_stats`` / ``project_team_sds``). The curves themselves
are calibrated in ``scripts/calibrate_playing_time.py`` and stored in
``constants.PLAYING_TIME_CURVES``; see those for the method and caveats.

``mean_scale`` is the multiplicative haircut on projected counting stats;
``cv_pt`` is the SD of actual/projected playing time at that projected volume.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import (
    PLAYING_TIME_CURVES,
    PLAYING_TIME_SHAPE,
    QUANTILE_LEVELS,
    STARTER_IP_THRESHOLD,
)


def _curve_key(player_type: PlayerType | str, volume: float) -> str:
    """Pick the curve. Pitcher role is IP-based (no GS field at deployment)."""
    if player_type == PlayerType.HITTER:
        return "hitters"
    return "SP" if volume >= STARTER_IP_THRESHOLD else "RP"


def _interp_xy(xs: list[float], ys: list[float], x: float) -> float:
    """Piecewise-linear interpolation of ``ys`` over ``xs``, clamped at ends.

    ``xs`` is ascending. NaN ``x`` (bad/missing data) -> lowest band, the
    conservative end. Must come first: every comparison against NaN is False,
    which would otherwise fall through to the highest (best) band.
    """
    if x != x or x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if x <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span == 0:
                return ys[i]
            t = (x - xs[i]) / span
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _interp(points: list[dict[str, float]], volume: float, field: str) -> float:
    """Piecewise-linear interpolation of ``field`` over ``vol``, clamped at ends.

    ``points`` is the curve for one (type, role), sorted ascending by ``vol``.
    """
    return _interp_xy([p["vol"] for p in points], [p[field] for p in points], volume)


def playing_time_params(player_type: PlayerType | str, volume: float) -> tuple[float, float]:
    """Return (mean_scale, cv_pt) for a player's projected volume.

    ``volume`` is projected PA for hitters, projected IP for pitchers.
    """
    points = PLAYING_TIME_CURVES[_curve_key(player_type, volume)]
    return _interp(points, volume, "mean_scale"), _interp(points, volume, "cv_pt")


def playing_time_shape(player_type: PlayerType | str, volume: float) -> list[float]:
    """Return the standardized-z ladder (one z per QUANTILE_LEVELS entry) for a volume.

    The ladder carries only the SHAPE of realized/projected playing time (skew +
    bounded tails); the caller applies ``mean_scale``/``cv_pt`` as location/scale.
    Interpolated band-to-band on the same volume axis as ``playing_time_params``.
    """
    points = PLAYING_TIME_SHAPE[_curve_key(player_type, volume)]
    vols = [cast(float, p["vol"]) for p in points]
    ladders = [cast("list[float]", p["z"]) for p in points]
    return [_interp_xy(vols, [lad[j] for lad in ladders], volume) for j in range(len(ladders[0]))]


def scale_from_uniform(
    mean_scale: float,
    cv_pt: float,
    z_ladder: list[float],
    u: float,
    fraction_remaining: float,
) -> float:
    """Realized PA/IP multiplier for a single uniform draw ``u`` in [0, 1].

    Maps ``u`` through the empirical standardized-z ladder, then locates/scales
    by the (fraction_remaining-damped) curve moments:

        eff_mean = 1 - (1 - mean_scale) * fraction_remaining
        eff_sd   = cv_pt * sqrt(fraction_remaining)
        scale    = max(0, eff_mean + z(u) * eff_sd)

    ``u`` outside ``[QUANTILE_LEVELS[0], QUANTILE_LEVELS[-1]]`` clamps to the
    p01/p99 ends -- the realistic injury floor and over-performance ceiling. At
    ``fraction_remaining == 0`` nothing is left to play, so the result is exactly
    ``1.0`` (projected) for every draw.
    """
    eff_mean = 1.0 - (1.0 - mean_scale) * fraction_remaining
    eff_sd = cv_pt * (fraction_remaining**0.5)
    z = float(np.interp(u, QUANTILE_LEVELS, z_ladder))
    return float(max(0.0, eff_mean + z * eff_sd))
