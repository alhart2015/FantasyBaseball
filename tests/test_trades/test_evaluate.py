from datetime import date

import pytest

from fantasy_baseball.models.player import HitterStats, PitcherStats, Player
from fantasy_baseball.models.standings import (
    CategoryStats,
    Standings,
    StandingsEntry,
)
from fantasy_baseball.trades.evaluate import (
    aggregate_player_stats,
    apply_swap_delta,
    compute_roto_points,
    compute_roto_points_by_cat,
    compute_trade_impact,
)
from fantasy_baseball.utils.constants import Category


def _standings(teams: list[tuple[str, dict]]) -> Standings:
    """Build a typed Standings from ``[(name, stats_dict), ...]``."""
    return Standings(
        effective_date=date(2026, 4, 15),
        entries=[
            StandingsEntry(
                team_name=name,
                team_key="",
                rank=0,
                stats=CategoryStats.from_dict(stats),
            )
            for name, stats in teams
        ],
    )


STANDINGS = _standings(
    [
        (
            "Team A",
            {
                "R": 900,
                "HR": 250,
                "RBI": 880,
                "SB": 150,
                "AVG": 0.265,
                "W": 80,
                "K": 1300,
                "SV": 80,
                "ERA": 3.50,
                "WHIP": 1.15,
            },
        ),
        (
            "Team B",
            {
                "R": 850,
                "HR": 280,
                "RBI": 900,
                "SB": 120,
                "AVG": 0.255,
                "W": 85,
                "K": 1400,
                "SV": 60,
                "ERA": 3.80,
                "WHIP": 1.20,
            },
        ),
        (
            "Team C",
            {
                "R": 800,
                "HR": 260,
                "RBI": 850,
                "SB": 180,
                "AVG": 0.250,
                "W": 75,
                "K": 1200,
                "SV": 90,
                "ERA": 3.30,
                "WHIP": 1.10,
            },
        ),
    ]
)


def test_compute_roto_points():
    points = compute_roto_points(STANDINGS)
    # Team A: R=3, HR=1, RBI=2, SB=2, AVG=3, W=2, K=2, SV=2, ERA=2, WHIP=2 = 21
    assert points["Team A"] == 21
    assert points["Team C"] == 19


def test_compute_trade_impact():
    hart_loses_ros = {
        "R": 50,
        "HR": 30,
        "RBI": 60,
        "SB": 20,
        "AVG": 0.280,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 0,
        "WHIP": 0,
        "ab": 400,
        "ip": 0,
    }
    hart_gains_ros = {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AVG": 0,
        "W": 5,
        "K": 100,
        "SV": 30,
        "ERA": 3.00,
        "WHIP": 1.05,
        "ab": 0,
        "ip": 150,
    }
    opp_loses_ros = hart_gains_ros
    opp_gains_ros = hart_loses_ros

    result = compute_trade_impact(
        standings=STANDINGS,
        hart_name="Team A",
        opp_name="Team B",
        hart_loses_ros=hart_loses_ros,
        hart_gains_ros=hart_gains_ros,
        opp_loses_ros=opp_loses_ros,
        opp_gains_ros=opp_gains_ros,
    )
    assert "hart_delta" in result
    assert "opp_delta" in result
    assert "hart_cat_deltas" in result
    assert "opp_cat_deltas" in result
    assert isinstance(result["hart_delta"], (int, float))


def test_trade_impact_zero_for_identical_players():
    same = {
        "R": 50,
        "HR": 20,
        "RBI": 50,
        "SB": 10,
        "AVG": 0.260,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 0,
        "WHIP": 0,
        "ab": 400,
        "ip": 0,
    }
    result = compute_trade_impact(
        standings=STANDINGS,
        hart_name="Team A",
        opp_name="Team B",
        hart_loses_ros=same,
        hart_gains_ros=same,
        opp_loses_ros=same,
        opp_gains_ros=same,
    )
    assert result["hart_delta"] == 0
    assert result["opp_delta"] == 0


def test_compute_roto_points_by_cat_missing_stats():
    """Teams missing some stat categories should get default values, not crash."""
    # CategoryStats defaults fill missing keys: 0 for counting stats and
    # AVG, 99.0 for ERA/WHIP. That makes "No Pitching" rank last in the
    # inverse categories (higher is worse).
    standings = _standings(
        [
            (
                "Full",
                {
                    "R": 100,
                    "HR": 30,
                    "RBI": 90,
                    "SB": 20,
                    "AVG": 0.260,
                    "W": 10,
                    "K": 150,
                    "SV": 10,
                    "ERA": 3.50,
                    "WHIP": 1.15,
                },
            ),
            (
                "No Pitching",
                {"R": 80, "HR": 25, "RBI": 85, "SB": 15, "AVG": 0.250, "W": 0, "K": 0, "SV": 0},
            ),
            # ERA and WHIP missing entirely for "No Pitching"
        ]
    )
    result = compute_roto_points_by_cat(standings)
    # Should not crash, and every team should have all 10 categories
    assert Category.ERA in result["No Pitching"]
    assert Category.WHIP in result["No Pitching"]
    assert len(result["Full"]) == 10
    assert len(result["No Pitching"]) == 10
    # "No Pitching" should rank last in ERA/WHIP (got default 99.0)
    assert result["Full"][Category.ERA] > result["No Pitching"][Category.ERA]
    assert result["Full"][Category.WHIP] > result["No Pitching"][Category.WHIP]


def test_aggregate_two_hitters_sums_counts_and_weights_avg():
    h1 = Player(
        name="A",
        player_type="hitter",
        positions=["OF"],
        rest_of_season=HitterStats(pa=600, ab=500, h=150, r=80, hr=25, rbi=70, sb=10, avg=0.300),
    )
    h2 = Player(
        name="B",
        player_type="hitter",
        positions=["2B"],
        rest_of_season=HitterStats(pa=500, ab=400, h=100, r=50, hr=10, rbi=40, sb=5, avg=0.250),
    )
    agg = aggregate_player_stats([h1, h2])
    assert agg["R"] == 130
    assert agg["HR"] == 35
    assert agg["ab"] == 900
    assert abs(agg["AVG"] - 250 / 900) < 1e-9
    assert agg["ip"] == 0


def test_aggregate_two_pitchers_weights_era_and_whip():
    p1 = Player(
        name="P1",
        player_type="pitcher",
        positions=["P"],
        rest_of_season=PitcherStats(
            ip=100, w=8, k=100, sv=0, era=3.60, whip=1.20, er=40, bb=30, h_allowed=90
        ),
    )
    p2 = Player(
        name="P2",
        player_type="pitcher",
        positions=["P"],
        rest_of_season=PitcherStats(
            ip=50, w=3, k=60, sv=20, era=2.70, whip=1.00, er=15, bb=10, h_allowed=40
        ),
    )
    agg = aggregate_player_stats([p1, p2])
    assert agg["W"] == 11
    assert agg["K"] == 160
    assert agg["SV"] == 20
    assert agg["ip"] == 150
    assert abs(agg["ERA"] - 3.30) < 1e-6
    assert abs(agg["WHIP"] - 170 / 150) < 1e-6


def test_aggregate_empty_list_returns_zeros():
    agg = aggregate_player_stats([])
    assert agg == {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AVG": 0.0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 0.0,
        "WHIP": 0.0,
        "ab": 0,
        "ip": 0,
    }


def test_swap_delta_uses_ros_only_not_full_season():
    """A swap of cold-YTD Soto-archetype for hot-YTD Cruz-archetype should
    score by ROS-remaining only, not by full-season totals that double-count
    YTD already locked into team standings.

    Setup: Hart's projected end-of-season R=900 (CategoryStats baseline).
    Soto-archetype: 3 R YTD, 87 R remaining (full-season would be 90).
    Cruz-archetype: 19 R YTD, 68 R remaining (full-season would be 87).
    Swapping Cruz out for Soto should bump Hart's projected R by +19
    (Soto's 87 in vs Cruz's 68 out), NOT +3 (the full-season diff that
    would double-count YTD already locked in).
    """
    current = {
        "R": 900.0,
        "HR": 200.0,
        "RBI": 800.0,
        "SB": 100.0,
        "AVG": 0.260,
        "W": 80.0,
        "K": 1300.0,
        "SV": 40.0,
        "ERA": 3.80,
        "WHIP": 1.20,
    }
    cruz_ros_only = {
        "R": 68,
        "HR": 22,
        "RBI": 64,
        "SB": 7,
        "AVG": 0.255,
        "ab": 400,
        "ip": 0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 0,
        "WHIP": 0,
    }
    soto_ros_only = {
        "R": 87,
        "HR": 29,
        "RBI": 79,
        "SB": 14,
        "AVG": 0.290,
        "ab": 432,
        "ip": 0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 0,
        "WHIP": 0,
    }

    after = apply_swap_delta(current, loses_ros=cruz_ros_only, gains_ros=soto_ros_only)
    assert after["R"] == 900.0 - 68 + 87
    assert after["R"] - current["R"] == 19  # NOT 3 (which would be full-season diff)


def test_swap_delta_avg_respects_caller_supplied_team_ab():
    """Mid-season team_ab is much smaller than the legacy 5500 constant.

    Pre-PR-110, the hardcoded _TEAM_AB=5500 was a reasonable proxy for the
    user's full-season AB total. After the team-YTD refactor, the user_row's
    AVG reflects team_YTD + ROS -- mid-season totals are ~3000-4500 AB.
    Backing out current_hits with 5500 over-counts hits by 20-50%.

    Setup: user's current AVG is .280 with a real team_ab of 4000 (a
    mid-season number). Swap a 100 AB / .200 AVG bat OUT for a 100 AB /
    .400 AVG bat IN. The marginal hit shift is +20 (40 in - 20 out) over
    a denominator that stays at 4000 (loses_ab and gains_ab cancel).
    Expected post-swap AVG is (.280 * 4000 + 20) / 4000 = .285.

    With the bug (defaults to 5500): current_hits = .280 * 5500 = 1540,
    new_hits = 1540 + 20 = 1560, new_ab = 5500, new AVG = .28364.
    """
    current = {
        "R": 600.0,
        "HR": 150.0,
        "RBI": 600.0,
        "SB": 80.0,
        "AVG": 0.280,
        "W": 60.0,
        "K": 900.0,
        "SV": 30.0,
        "ERA": 3.80,
        "WHIP": 1.20,
    }
    loses_ros = {
        "R": 10,
        "HR": 3,
        "RBI": 10,
        "SB": 2,
        "AVG": 0.200,
        "ab": 100,
        "ip": 0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 0,
        "WHIP": 0,
    }
    gains_ros = {
        "R": 20,
        "HR": 8,
        "RBI": 20,
        "SB": 4,
        "AVG": 0.400,
        "ab": 100,
        "ip": 0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 0,
        "WHIP": 0,
    }

    # Legacy path: defaults to hardcoded 5500 -- biased.
    after_legacy = apply_swap_delta(current, loses_ros=loses_ros, gains_ros=gains_ros)

    # Fixed path: caller passes the team's real AB (mid-season ~ 4000).
    after_fixed = apply_swap_delta(
        current,
        loses_ros=loses_ros,
        gains_ros=gains_ros,
        team_ab=4000,
        team_ip=1000,
    )

    # Mid-season truth: hits shifted by +20 (40 in - 20 out) over ~4000 AB.
    expected_avg = (0.280 * 4000 + (0.400 * 100 - 0.200 * 100)) / 4000
    assert after_fixed["AVG"] == pytest.approx(expected_avg, abs=1e-9)
    # And: the fixed value differs from the legacy 5500-default value.
    assert after_fixed["AVG"] != pytest.approx(after_legacy["AVG"], abs=1e-9)


def test_swap_delta_era_respects_caller_supplied_team_ip():
    """Mid-season team_ip is much smaller than the legacy 1450 constant.

    Same logic as the AB test: hardcoded 1450 over-states the denominator
    so swap deltas in ERA/WHIP are mis-weighted mid-season. With the
    caller passing the real team_ip (e.g. ~900 mid-season), the rate
    math reflects the actual team baseline.
    """
    current = {
        "R": 600.0,
        "HR": 150.0,
        "RBI": 600.0,
        "SB": 80.0,
        "AVG": 0.265,
        "W": 60.0,
        "K": 900.0,
        "SV": 30.0,
        "ERA": 3.80,
        "WHIP": 1.20,
    }
    loses_ros = {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AVG": 0,
        "ab": 0,
        "ip": 50,
        "W": 3,
        "K": 50,
        "SV": 0,
        "ERA": 5.00,
        "WHIP": 1.40,
    }
    gains_ros = {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AVG": 0,
        "ab": 0,
        "ip": 50,
        "W": 4,
        "K": 60,
        "SV": 0,
        "ERA": 3.00,
        "WHIP": 1.00,
    }

    after_legacy = apply_swap_delta(current, loses_ros=loses_ros, gains_ros=gains_ros)
    after_fixed = apply_swap_delta(
        current,
        loses_ros=loses_ros,
        gains_ros=gains_ros,
        team_ab=4000,
        team_ip=900,
    )

    # Truth with real team_ip=900: ER shifts by (3.00*50 - 5.00*50)/9 = -100/9.
    # current_er = 3.80 * 900 / 9 = 380, new_er = 380 - 100/9, new_ip = 900.
    # The 50 IP cancel because loses_ip == gains_ip.
    expected_era = 9.0 * (3.80 * 900 / 9.0 + (3.00 * 50 - 5.00 * 50) / 9.0) / 900.0
    assert after_fixed["ERA"] == pytest.approx(expected_era, abs=1e-9)
    # WHIP similarly: bh shift = (1.00*50 - 1.40*50) over 900 IP.
    expected_whip = (1.20 * 900 + (1.00 * 50 - 1.40 * 50)) / 900.0
    assert after_fixed["WHIP"] == pytest.approx(expected_whip, abs=1e-9)
    # The fixed values differ from the legacy-default values.
    assert after_fixed["ERA"] != pytest.approx(after_legacy["ERA"], abs=1e-9)
    assert after_fixed["WHIP"] != pytest.approx(after_legacy["WHIP"], abs=1e-9)
