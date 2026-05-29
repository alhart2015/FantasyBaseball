"""Shared pitcher-swap math used by displacement (scoring.py) and the stash
board (lineup/stash_value.py).

The two surfaces ask the same question: when one pitcher takes another
pitcher's slot share, how much of the target's ROS is consumed? The legacy
answer was a direct IP swap, which is correct SP-to-SP (both share the same
"IP-per-active-game" rate) but wrong cross-role (an SP returning at 60 IP
does NOT consume 60 IP of an RP's ROS -- the RP throws 1 IP per appearance,
not 6).

The shared formula here uses each pitcher's preseason IP as a per-pitcher
"rate denominator." The candidate's ``ros_ip / preseason_ip`` is the fraction
of the season they'll spend in an active slot (their slot-share). Multiplied
by the target's preseason IP, this gives the IP the target would have thrown
during the same time window -- the right discount IP regardless of role.

Falls back to a direct IP swap when either preseason field is missing, so
preseason-naive data paths (older fixtures, draft scripts) still work.
"""

from __future__ import annotations

from fantasy_baseball.models.player import PitcherStats, Player
from fantasy_baseball.utils.constants import safe_float as _safe


def _ros_ip(p: Player) -> float:
    ros = p.rest_of_season
    if isinstance(ros, PitcherStats):
        return _safe(ros.ip)
    return 0.0


def _preseason_ip(p: Player) -> float:
    pre = p.preseason
    if isinstance(pre, PitcherStats):
        return _safe(pre.ip)
    return 0.0


def swap_window_ip(candidate: Player, target: Player) -> float:
    """IP window of ``target`` that ``candidate`` consumes when taking the
    target's slot share.

    Uses preseason-IP proration when both pitchers have preseason data:

        window = target.preseason.ip * (candidate.ros.ip / candidate.preseason.ip)

    Falls back to ``candidate.ros.ip`` (direct IP swap, legacy) when either
    side lacks preseason. Returns 0.0 if the candidate has no ROS IP.
    """
    cand_ros = _ros_ip(candidate)
    if cand_ros <= 0.0:
        return 0.0
    cand_pre = _preseason_ip(candidate)
    tgt_pre = _preseason_ip(target)
    if cand_pre <= 0.0 or tgt_pre <= 0.0:
        return cand_ros
    return tgt_pre * (cand_ros / cand_pre)


def discount_factor(target_ros_ip: float, window: float) -> float:
    """Scale factor to apply to ``target``'s ROS stats so they reflect the
    portion NOT consumed by the candidate's slot share.

    Returns 0.0 when the window meets or exceeds the target's ROS IP (full
    swap-out) or when ``target_ros_ip`` is non-positive. Otherwise returns
    ``max(0, target_ros_ip - window) / target_ros_ip``.
    """
    if target_ros_ip <= 0.0:
        return 0.0
    return max(0.0, target_ros_ip - window) / target_ros_ip
