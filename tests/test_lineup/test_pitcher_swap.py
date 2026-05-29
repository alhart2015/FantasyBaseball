"""Unit tests for shared pitcher-swap helpers.

These functions are used by both the displacement model in scoring.py and the
stash board in lineup/stash_value.py. They define the IP window over which a
displacement target is discounted when a candidate pitcher takes their slot
share, and the resulting scale factor on the target's ROS stats.
"""

from __future__ import annotations

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


def test_swap_window_same_role_with_preseason_matches_direct_ip():
    """SP-to-SP: when both have preseason IP and similar ratios, the window
    approximates the candidate's ROS IP -- i.e. the legacy direct-IP swap is a
    special case of the preseason-proration formula."""
    candidate = _pitcher("Cand", ros_ip=60, preseason_ip=200)
    target = _pitcher("Target", ros_ip=130, preseason_ip=200)
    # 200 * (60 / 200) = 60 IP -- exactly the direct-IP swap.
    assert swap_window_ip(candidate, target) == 60.0


def test_swap_window_cross_role_uses_preseason_proration():
    """SP-to-RP: a starter returning at 60 IP (30% of his 200 IP preseason)
    consumes 30% of the reliever's preseason IP, NOT 60 IP of his ROS.

    Without proration the RP would be fully wiped (60 > RP's ROS); with
    proration only the RP's slot-share over the candidate's window is taken.
    """
    starter = _pitcher("Webb", ros_ip=60, preseason_ip=200)
    reliever = _pitcher("Closer", ros_ip=25, preseason_ip=65)
    # 65 * (60 / 200) = 19.5 IP
    assert swap_window_ip(starter, reliever) == 19.5


def test_swap_window_rp_returning_displaces_sp_window():
    """Symmetric case: an RP returning at 20 IP (~30% of his 65 IP preseason)
    consumes 30% of a starter's preseason IP -- much MORE IP than the RP
    himself will throw, but the right amount for the SP to lose."""
    reliever = _pitcher("Closer", ros_ip=20, preseason_ip=65)
    starter = _pitcher("SP", ros_ip=130, preseason_ip=200)
    # 200 * (20 / 65) = 61.538...
    assert swap_window_ip(reliever, starter) == 200.0 * (20.0 / 65.0)


def test_swap_window_falls_back_to_direct_ip_when_candidate_lacks_preseason():
    """A candidate without preseason data (older fixture, draft-script dict)
    gets the legacy direct-IP swap. Better to be slightly wrong cross-role
    than to crash on missing data."""
    candidate = _pitcher("NoPre", ros_ip=60)  # preseason=None
    target = _pitcher("Target", ros_ip=130, preseason_ip=200)
    assert swap_window_ip(candidate, target) == 60.0


def test_swap_window_falls_back_to_direct_ip_when_target_lacks_preseason():
    candidate = _pitcher("Cand", ros_ip=60, preseason_ip=200)
    target = _pitcher("NoPre", ros_ip=130)  # preseason=None
    assert swap_window_ip(candidate, target) == 60.0


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
