"""Tests for Monte Carlo simulation functions (ROS extensions)."""

import numpy as np
import pytest

from fantasy_baseball.simulation import (
    run_ros_monte_carlo,
    simulate_remaining_season,
)


def _make_hitter(name, r=80, hr=25, rbi=80, sb=10, h=150, ab=550):
    """Create a minimal hitter dict for testing."""
    return {
        "name": name,
        "player_type": "hitter",
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "h": h,
        "ab": ab,
    }


def _make_pitcher(name, w=10, k=150, sv=0, ip=180, er=70, bb=50, h_allowed=150):
    """Create a minimal pitcher dict for testing."""
    return {
        "name": name,
        "player_type": "pitcher",
        "w": w,
        "k": k,
        "sv": sv,
        "ip": ip,
        "er": er,
        "bb": bb,
        "h_allowed": h_allowed,
    }


def _make_closer(name, w=3, k=60, sv=30, ip=65, er=20, bb=20, h_allowed=55):
    """Create a minimal closer dict for testing."""
    return _make_pitcher(name, w=w, k=k, sv=sv, ip=ip, er=er, bb=bb, h_allowed=h_allowed)


def _build_two_team_rosters():
    """Build a 2-team roster dict with enough players for h_slots=3, p_slots=2."""
    return {
        "Team A": [
            _make_hitter("H1", r=90, hr=30, rbi=100, sb=15, h=160, ab=550),
            _make_hitter("H2", r=70, hr=20, rbi=70, sb=8, h=140, ab=520),
            _make_hitter("H3", r=60, hr=15, rbi=55, sb=5, h=130, ab=500),
            _make_pitcher("P1", w=12, k=180, sv=0, ip=190, er=65, bb=45, h_allowed=160),
            _make_closer("C1", w=3, k=70, sv=35, ip=65, er=18, bb=18, h_allowed=50),
        ],
        "Team B": [
            _make_hitter("H4", r=85, hr=28, rbi=90, sb=12, h=155, ab=540),
            _make_hitter("H5", r=65, hr=18, rbi=65, sb=20, h=135, ab=510),
            _make_hitter("H6", r=55, hr=12, rbi=50, sb=3, h=125, ab=490),
            _make_pitcher("P2", w=10, k=160, sv=0, ip=175, er=70, bb=50, h_allowed=155),
            _make_closer("C2", w=2, k=55, sv=28, ip=60, er=22, bb=22, h_allowed=55),
        ],
    }


def _build_actual_standings():
    """Build actual YTD standings for 2 teams (midseason-like values)."""
    return {
        "Team A": {
            "R": 350, "HR": 100, "RBI": 340, "SB": 40,
            "AVG": 0.265, "W": 35, "K": 600, "SV": 25,
            "ERA": 3.80, "WHIP": 1.20,
        },
        "Team B": {
            "R": 320, "HR": 90, "RBI": 310, "SB": 50,
            "AVG": 0.255, "W": 30, "K": 550, "SV": 20,
            "ERA": 4.10, "WHIP": 1.28,
        },
    }


# ---------------------------------------------------------------------------
# Task 5 tests: simulate_remaining_season
# ---------------------------------------------------------------------------


class TestSimulateRemainingSeason:
    """Tests for simulate_remaining_season()."""

    def test_blends_actuals(self):
        """Counting stats should be greater than actuals (actual + simulated)
        and rate stats should be in valid ranges."""
        rosters = _build_two_team_rosters()
        actuals = _build_actual_standings()
        rng = np.random.default_rng(42)

        team_stats, injuries = simulate_remaining_season(
            actuals, rosters, fraction_remaining=0.5, rng=rng,
            h_slots=3, p_slots=2,
        )

        for team in ["Team A", "Team B"]:
            stats = team_stats[team]
            act = actuals[team]

            # Counting stats: final should be >= actual (sim adds non-negative)
            for cat in ["R", "HR", "RBI", "SB", "W", "K", "SV"]:
                assert stats[cat] >= act[cat], (
                    f"{team} {cat}: final {stats[cat]} < actual {act[cat]}"
                )

            # AVG should be in a valid range
            assert 0.150 < stats["AVG"] < 0.350, (
                f"{team} AVG out of range: {stats['AVG']}"
            )

            # ERA should be in a valid range
            assert 1.0 < stats["ERA"] < 8.0, (
                f"{team} ERA out of range: {stats['ERA']}"
            )

            # WHIP should be in a valid range
            assert 0.8 < stats["WHIP"] < 2.0, (
                f"{team} WHIP out of range: {stats['WHIP']}"
            )

        # Both teams should be in injuries dict
        assert "Team A" in injuries
        assert "Team B" in injuries

    def test_zero_remaining(self):
        """When fraction_remaining=0, result must equal actuals exactly."""
        rosters = _build_two_team_rosters()
        actuals = _build_actual_standings()

        rng = np.random.default_rng(99)
        team_stats, injuries = simulate_remaining_season(
            actuals, rosters, fraction_remaining=0.0, rng=rng,
            h_slots=3, p_slots=2,
        )

        for team in ["Team A", "Team B"]:
            act = actuals[team]
            result = team_stats[team]
            for cat in ["R", "HR", "RBI", "SB", "W", "K", "SV"]:
                assert result[cat] == pytest.approx(act[cat]), (
                    f"{team} {cat}: expected {act[cat]} at fraction_remaining=0, "
                    f"got {result[cat]}"
                )
            assert injuries[team] == []


# ---------------------------------------------------------------------------
# Task 6 tests: run_ros_monte_carlo
# ---------------------------------------------------------------------------


class TestRunRosMonteCarlo:
    """Tests for run_ros_monte_carlo()."""

    def test_returns_expected_format(self):
        """Verify return has team_results and category_risk with expected keys."""
        rosters = _build_two_team_rosters()
        actuals = _build_actual_standings()

        result = run_ros_monte_carlo(
            team_rosters=rosters,
            actual_standings=actuals,
            fraction_remaining=0.5,
            h_slots=3,
            p_slots=2,
            user_team_name="Team A",
            n_iterations=100,
            seed=42,
        )

        # Top-level keys
        assert "team_results" in result
        assert "category_risk" in result

        # team_results should have both teams
        tr = result["team_results"]
        assert "Team A" in tr
        assert "Team B" in tr

        # Each team result should have the expected keys
        expected_team_keys = {"median_pts", "p10", "p90", "first_pct", "top3_pct"}
        for team in ["Team A", "Team B"]:
            assert set(tr[team].keys()) == expected_team_keys, (
                f"{team} keys: {set(tr[team].keys())} != {expected_team_keys}"
            )
            # Sanity: median_pts should be positive (roto points)
            assert tr[team]["median_pts"] > 0
            # first_pct and top3_pct are percentages 0-100
            assert 0 <= tr[team]["first_pct"] <= 100
            assert 0 <= tr[team]["top3_pct"] <= 100

        # With only 2 teams, first_pct should sum to ~100
        total_first = sum(tr[t]["first_pct"] for t in tr)
        assert abs(total_first - 100.0) < 0.1, f"first_pct sum: {total_first}"

        # category_risk should have all 10 categories
        cr = result["category_risk"]
        expected_cats = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "ERA", "WHIP", "SV"]
        for cat in expected_cats:
            assert cat in cr, f"Missing category: {cat}"
            expected_cat_keys = {"median_pts", "p10", "p90", "top3_pct", "bot3_pct"}
            assert set(cr[cat].keys()) == expected_cat_keys, (
                f"{cat} keys: {set(cr[cat].keys())} != {expected_cat_keys}"
            )
