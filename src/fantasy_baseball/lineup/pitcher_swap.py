"""Shared pitcher-swap math used by displacement (scoring.py) and the stash
board (lineup/stash_value.py).

The two surfaces ask the same question: when one pitcher takes another
pitcher's slot share, how much of the target's REMAINING (ROS) workload is
displaced?

The model is a time-share, not an IP-for-IP trade. Rosters are set weekly and
a slot cannot be platooned within a week, so the candidate's slot-share is the
fraction of the REMAINING weeks it holds the slot:

    slot_share = candidate.ros.ip / (candidate.preseason.ip * fraction_remaining)

The denominator is the candidate's *remaining-season* healthy workload, not its
full-season preseason IP -- the already-elapsed part of the season is gone for
everyone, so dividing by the full season would mis-read a pitcher who is
healthy for the whole rest of the year as a part-timer. An arm back now whose
ROS already equals a healthy remainder (e.g. 33 of a ~35-IP healthy remainder)
has slot-share ~1.0 and is active almost every remaining week; one who will
miss half the rest of the season has slot-share ~0.5.

That same fraction of the *target's ROS* is displaced: the worst pitcher keeps
``1 - slot_share`` of what it had left to pitch. Role-agnostic by construction
-- a returning reliever displacing a starter takes ``slot_share`` of the
starter's remaining innings, no preseason-IP-of-the-target conversion required.

Only the candidate's preseason IP is needed (to compute the slot-share); the
target's preseason IP is irrelevant. Falls back to a direct IP swap when the
candidate lacks preseason (preseason-naive data paths -- older fixtures,
draft-script dicts). The slot-share is clamped to ``[0, 1]`` so a junk-tiny
candidate preseason (e.g. a same-name-collision projection row) cannot amplify
the displacement past a full swap-out.
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


def swap_window_ip(candidate: Player, target: Player, *, fraction_remaining: float = 1.0) -> float:
    """IP window of ``target``'s ROS that ``candidate`` displaces when taking
    the target's slot share.

        healthy_remainder = candidate.preseason.ip * fraction_remaining
        slot_share        = min(1.0, candidate.ros.ip / healthy_remainder)
        window            = target.ros.ip * slot_share

    So ``discount_factor(target.ros.ip, window) == 1 - slot_share``: the worst
    pitcher keeps ``1 - slot_share`` of its remaining workload.

    ``slot_share`` is the fraction of the REMAINING season the candidate is
    active -- i.e. the share of remaining weeks it holds the slot, given that
    rosters are set weekly and a slot cannot be platooned within a week. The
    denominator is the candidate's *remaining-season* healthy workload
    (``preseason.ip * fraction_remaining``), NOT its full-season preseason IP:
    the already-elapsed part of the season is gone for everyone, so dividing by
    the full season would mis-read a pitcher who is healthy for the whole rest
    of the year as a part-timer. A returner whose ROS already equals a healthy
    remainder (back now) gets slot_share ~= 1.0 and displaces ~all of the worst
    pitcher's ROS; one who will miss half the rest of the year gets ~0.5.

    Only the candidate's preseason IP is consulted (the slot-share
    denominator); the target's preseason IP is not used. ``slot_share`` is
    clamped to ``[0, 1]`` so a junk-tiny candidate preseason (e.g. a
    same-name-collision projection row, or a Yahoo IL stash with a near-zero
    preseason line) cannot push the window past the target's full ROS. Falls
    back to ``candidate.ros.ip`` (legacy direct-IP swap) when the candidate
    lacks usable preseason data. Returns 0.0 if the candidate has no ROS IP.
    """
    cand_ros = _ros_ip(candidate)
    if cand_ros <= 0.0:
        return 0.0
    healthy_remainder = _preseason_ip(candidate) * fraction_remaining
    if healthy_remainder <= 0.0:
        return cand_ros
    slot_share = min(1.0, cand_ros / healthy_remainder)
    return _ros_ip(target) * slot_share


def discount_factor(target_ros_ip: float, window: float) -> float:
    """Scale factor to apply to ``target``'s ROS stats so they reflect the
    portion NOT consumed by the candidate's slot share.

    Returns 0.0 when the window meets or exceeds the target's ROS IP (full
    swap-out) or when ``target_ros_ip`` is non-positive. Returns 1.0 when
    ``window`` is non-positive (no consumption). Otherwise returns
    ``max(0, target_ros_ip - window) / target_ros_ip``.

    The negative-window clamp guards against future callers whose arithmetic
    may go negative; the function contract excludes amplification (return > 1).
    """
    if target_ros_ip <= 0.0:
        return 0.0
    if window <= 0.0:
        return 1.0
    return max(0.0, target_ros_ip - window) / target_ros_ip
