"""Unit tests for shared pitcher-swap helpers.

These functions are used by both the displacement model in scoring.py and the
stash board in lineup/stash_value.py. They define the IP window over which a
displacement target is discounted when a candidate pitcher takes their slot
share, and the resulting scale factor on the target's ROS stats.
"""

from __future__ import annotations

import pytest

from fantasy_baseball.lineup.pitcher_swap import discount_factor, swap_window_ip
from fantasy_baseball.models.player import PitcherStats, Player, PlayerType


def _pitcher(name: str, *, ros_ip: float, preseason_ip: float | None = None) -> Player:
    """Test-only constructor: a pitcher with ROS IP (and optionally preseason IP).

    All non-IP stats default to zero so tests focus on the IP arithmetic.
    """
    ros = PitcherStats(ip=ros_ip, w=0, k=0, sv=0, er=0, bb=0, h_allowed=0, era=0, whip=0)
    pre = (
        PitcherStats(ip=preseason_ip, w=0, k=0, sv=0, er=0, bb=0, h_allowed=0, era=0, whip=0)
        if preseason_ip is not None
        else None
    )
    return Player(name=name, player_type=PlayerType.PITCHER, rest_of_season=ros, preseason=pre)


def test_swap_window_at_full_season_scales_target_ros_by_workload_ratio():
    """With ``fraction_remaining`` defaulting to 1.0 (season start: a healthy
    remainder equals the full preseason), the slot-share is ``ros/preseason``
    and the window is that fraction of the TARGET's ROS -- measured against the
    target's remaining IP, not its preseason/full-season IP, so the discount is
    exactly ``1 - s`` regardless of the target's role.

    Mid-season the denominator shrinks to the remaining-season workload -- see
    ``test_swap_window_slot_share_uses_remaining_season_not_full_season``.
    """
    candidate = _pitcher("Cand", ros_ip=120, preseason_ip=200)  # 60% of a full season
    worst = _pitcher("Worst", ros_ip=115.0, preseason_ip=180.0)
    s = 120.0 / 200.0
    assert swap_window_ip(candidate, worst) == pytest.approx(115.0 * s)
    # End-to-end: the target keeps 1 - s of its ROS.
    assert discount_factor(115.0, swap_window_ip(candidate, worst)) == pytest.approx(1.0 - s)


def test_swap_window_slot_share_uses_remaining_season_not_full_season():
    """Slot-share is the candidate's share of the REMAINING season, not the
    full season: ``s = ros / (preseason * fraction_remaining)``. The already-
    elapsed part of the season is gone for everyone and must NOT count as the
    returner 'missing' time -- otherwise a pitcher who is healthy for the whole
    rest of the year looks like a part-timer.

    Hader: 33.2 ROS IP, 54.6 preseason IP, ~35% of the season already elapsed
    (fraction_remaining=0.649). Healthy remainder = 54.6 * 0.649 = 35.4 IP, so
    his slot-share is 33.2/35.4 = ~0.94 -- active nearly every remaining week,
    displacing ~94% of the worst pitcher's ROS (NOT the 60% that ros/preseason
    alone would imply by double-counting the elapsed season).
    """
    hader = _pitcher("Hader", ros_ip=33.2, preseason_ip=54.6)
    worst = _pitcher("Worst", ros_ip=115.0, preseason_ip=180.0)
    s = 33.2 / (54.6 * 0.649)
    window = swap_window_ip(hader, worst, fraction_remaining=0.649)
    assert window == pytest.approx(115.0 * s)
    assert discount_factor(115.0, window) == pytest.approx(1.0 - s)


def test_swap_window_clamps_fraction_remaining_above_one():
    """fraction_remaining can exceed 1.0 before opening day (compute_fraction_
    remaining does not clamp its upper bound). The slot-share denominator must
    treat that as a whole season remaining -- a >1.0 value must NOT inflate the
    denominator and under-displace. fr=1.3 yields the same window as fr=1.0.
    """
    candidate = _pitcher("Cand", ros_ip=120, preseason_ip=200)
    target = _pitcher("Target", ros_ip=130, preseason_ip=200)
    assert swap_window_ip(candidate, target, fraction_remaining=1.3) == swap_window_ip(
        candidate, target, fraction_remaining=1.0
    )


def test_swap_window_uses_target_ros_not_target_preseason():
    """The window is the TARGET's remaining (ROS) IP scaled by the candidate's
    slot-share; the target's preseason IP is irrelevant. Candidate at 60/200
    preseason (slot-share 0.30) displaces 0.30 of the target's 130 ROS IP.
    (Previously this multiplied the target's 200 preseason IP and returned 60.)
    """
    candidate = _pitcher("Cand", ros_ip=60, preseason_ip=200)
    target = _pitcher("Target", ros_ip=130, preseason_ip=200)
    assert swap_window_ip(candidate, target) == 130.0 * (60.0 / 200.0)  # 39.0


def test_swap_window_cross_role_sp_displacing_rp():
    """SP returning at 60/200 preseason (slot-share 0.30) displaces 0.30 of the
    reliever's 25 ROS IP = 7.5 IP -- the RP keeps 70% of its remaining work.
    Role-agnostic: the discount is exactly ``1 - slot_share``.
    """
    starter = _pitcher("Webb", ros_ip=60, preseason_ip=200)
    reliever = _pitcher("Closer", ros_ip=25, preseason_ip=65)
    assert swap_window_ip(starter, reliever) == 25.0 * (60.0 / 200.0)  # 7.5
    assert discount_factor(25.0, swap_window_ip(starter, reliever)) == pytest.approx(1.0 - 0.30)


def test_swap_window_cross_role_rp_displacing_sp():
    """An RP returning at 20/65 preseason (slot-share ~0.31) displaces ~0.31 of
    a starter's 130 ROS IP = 40 IP. The returning reliever's own low IP does
    NOT cap how much of the starter's remaining workload it frees up -- the
    slot is consumed by time, not by innings.
    """
    reliever = _pitcher("Closer", ros_ip=20, preseason_ip=65)
    starter = _pitcher("SP", ros_ip=130, preseason_ip=200)
    assert swap_window_ip(reliever, starter) == pytest.approx(130.0 * (20.0 / 65.0))  # 40.0


def test_swap_window_clamps_slot_share_when_candidate_preseason_is_junk():
    """A same-name-collision projection row can hand a real returner a junk
    preseason line (the audit's Mason Miller had preseason IP = 1.7). The
    slot-share ``min(1.0, 33 / 1.7)`` clamps to 1.0, so the window is the
    target's full ROS (a clean full swap-out) rather than an amplified
    >ROS window. Defensive only -- the real fix is matching the correct
    same-name projection (see ``match_roster_to_projections``).
    """
    junk = _pitcher("Mason Miller (wrong row)", ros_ip=33.0, preseason_ip=1.7)
    target = _pitcher("Target", ros_ip=115.0, preseason_ip=180.0)
    assert swap_window_ip(junk, target) == 115.0  # clamped to a full swap-out
    assert discount_factor(115.0, swap_window_ip(junk, target)) == 0.0


def test_swap_window_falls_back_to_direct_ip_when_candidate_lacks_preseason():
    """A candidate without preseason data (older fixture, draft-script dict)
    gets the legacy direct-IP swap. Better to be slightly wrong cross-role
    than to crash on missing data."""
    candidate = _pitcher("NoPre", ros_ip=60)  # preseason=None
    target = _pitcher("Target", ros_ip=130, preseason_ip=200)
    assert swap_window_ip(candidate, target) == 60.0


def test_swap_window_ignores_missing_target_preseason():
    """The new model never reads the target's preseason IP, so a target with no
    preseason data still gets a correct ROS-based window -- no fallback. (This
    case previously fell back to the candidate's 60 ROS IP.)"""
    candidate = _pitcher("Cand", ros_ip=60, preseason_ip=200)
    target = _pitcher("NoPre", ros_ip=130)  # preseason=None
    assert swap_window_ip(candidate, target) == 130.0 * (60.0 / 200.0)  # 39.0


def test_swap_window_zero_when_candidate_has_no_ros():
    candidate = _pitcher("Hurt", ros_ip=0, preseason_ip=200)
    target = _pitcher("Target", ros_ip=130, preseason_ip=200)
    assert swap_window_ip(candidate, target) == 0.0


def test_discount_factor_partial_swap():
    """Window smaller than target ROS -> scale factor between 0 and 1."""
    assert discount_factor(target_ros_ip=130.0, window=60.0) == (130.0 - 60.0) / 130.0


def test_discount_factor_full_swap_clamps_to_zero():
    """Window >= target ROS -> target fully wiped."""
    assert discount_factor(target_ros_ip=25.0, window=60.0) == 0.0


def test_discount_factor_zero_window_keeps_target_full():
    assert discount_factor(target_ros_ip=130.0, window=0.0) == 1.0


def test_discount_factor_zero_target_returns_zero():
    """No ROS IP -> no way to scale -> 0 (caller should skip these targets)."""
    assert discount_factor(target_ros_ip=0.0, window=10.0) == 0.0


def test_discount_factor_negative_window_clamps_to_full():
    """A negative window must not amplify (return > 1.0). Callers in
    scoring.py and stash_value.py may pass raw arithmetic that goes negative;
    the function contract excludes amplification."""
    assert discount_factor(target_ros_ip=100.0, window=-10.0) == 1.0


def test_swap_window_same_role_scales_target_ros_by_candidate_slot_share():
    """Same-role: the window depends only on the candidate's slot-share and the
    TARGET's ROS IP. A candidate at 60/200 preseason (slot-share 0.30) on a
    target with 120 ROS IP -> 36 IP, independent of the target's 180 preseason
    IP. (The old proration multiplied the target's preseason IP and gave 54.)
    """
    candidate = _pitcher("Cand", ros_ip=60, preseason_ip=200)
    target = _pitcher("Target", ros_ip=120, preseason_ip=180)
    assert swap_window_ip(candidate, target) == 120.0 * (60.0 / 200.0)  # 36.0
