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
    ROS_PLAYING_TIME_QUANTILES,
    ROS_PT_LEVELS,
    role_from_ip,
)


def _curve_key(player_type: PlayerType | str, volume: float) -> str:
    """Pick the curve. Pitcher role is IP-based (no GS field at deployment)."""
    if player_type == PlayerType.HITTER:
        return "hitters"
    return role_from_ip(volume)


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


def ros_playing_time_quantiles(
    player_type: PlayerType | str, volume: float, fraction_remaining: float
) -> list[float]:
    """Quantiles (at ``ROS_PT_LEVELS``) of realized / projected REST-OF-SEASON volume.

    The in-season counterpart of ``playing_time_params`` + ``playing_time_shape``
    (issue #393): over a short window a player is mostly healthy or out for the rest
    of it, so outcomes pile up near 0 and 1 in a way the full-season curve cannot
    represent. ``volume`` is FULL-SEASON PA / IP (it picks the SP/RP role, like the
    full-season curve). Interpolated linearly between the fitted horizons and
    clamped at the ends.

    SHAPE vs LEVEL: the table gives the shape (the near-0 / near-1 split over a
    short window). It was fitted against a pace-based stand-in for the projection,
    which runs ~10-25% less optimistic about innings than real projections, so its
    level is rescaled to the full-season curve's ``mean_scale`` -- fitted against
    real projections (2022-2025) and so carrying their optimism. Multiplying keeps
    the mass at zero intact.
    """
    points = ROS_PLAYING_TIME_QUANTILES[_curve_key(player_type, volume)]
    fs = [cast(float, p["f"]) for p in points]
    qs = [cast("list[float]", p["q"]) for p in points]
    q = [_interp_xy(fs, [qq[j] for qq in qs], fraction_remaining) for j in range(len(qs[0]))]
    shape_mean = float(np.interp(_MEAN_GRID, ROS_PT_LEVELS, q).mean())
    if shape_mean <= 0:
        return q
    target, _ = playing_time_params(player_type, volume)
    return [v * target / shape_mean for v in q]


# Fine grid on (0, 1) for the mean of a piecewise-linear quantile function.
_MEAN_GRID = np.linspace(0.0, 1.0, 1001)


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


def playing_time_moments(
    mean_scale: float,
    cv_pt: float,
    fraction_remaining: float,
) -> tuple[float, float]:
    """The fraction_remaining-damped (eff_mean, eff_sd) of the realized-PT scale.

    Single source of truth for the location-scale mapping shared by the scalar
    ``scale_from_uniform`` and the batched MC (``_apply_variance_batch``):

        eff_mean = 1 - (1 - mean_scale) * fraction_remaining
        eff_sd   = cv_pt * sqrt(fraction_remaining)

    Over a partial season only the remaining playing time is at risk, so both the
    haircut and the spread shrink with ``fraction_remaining`` (at 0, eff_mean == 1
    and eff_sd == 0, pinning every draw to projected).
    """
    eff_mean = 1.0 - (1.0 - mean_scale) * fraction_remaining
    eff_sd = cv_pt * (fraction_remaining**0.5)
    return eff_mean, eff_sd


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
    eff_mean, eff_sd = playing_time_moments(mean_scale, cv_pt, fraction_remaining)
    z = float(np.interp(u, QUANTILE_LEVELS, z_ladder))
    return float(max(0.0, eff_mean + z * eff_sd))
