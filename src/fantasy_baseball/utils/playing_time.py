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

from itertools import pairwise

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import (
    PLAYING_TIME_CURVES,
    STARTER_IP_THRESHOLD,
)


def _curve_key(player_type: PlayerType | str, volume: float) -> str:
    """Pick the curve. Pitcher role is IP-based (no GS field at deployment)."""
    if player_type == PlayerType.HITTER:
        return "hitters"
    return "SP" if volume >= STARTER_IP_THRESHOLD else "RP"


def _interp(points: list[dict[str, float]], volume: float, field: str) -> float:
    """Piecewise-linear interpolation of ``field`` over ``vol``, clamped at ends.

    ``points`` is the curve for one (type, role), sorted ascending by ``vol``.
    """
    if volume <= points[0]["vol"]:
        return points[0][field]
    if volume >= points[-1]["vol"]:
        return points[-1][field]
    for lo, hi in pairwise(points):
        if volume <= hi["vol"]:
            span = hi["vol"] - lo["vol"]
            t = (volume - lo["vol"]) / span
            return lo[field] + t * (hi[field] - lo[field])
    return points[-1][field]


def playing_time_params(player_type: PlayerType | str, volume: float) -> tuple[float, float]:
    """Return (mean_scale, cv_pt) for a player's projected volume.

    ``volume`` is projected PA for hitters, projected IP for pitchers.
    """
    points = PLAYING_TIME_CURVES[_curve_key(player_type, volume)]
    return _interp(points, volume, "mean_scale"), _interp(points, volume, "cv_pt")
