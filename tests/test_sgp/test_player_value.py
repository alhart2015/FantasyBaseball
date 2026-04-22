import pandas as pd
import pytest

from fantasy_baseball.models.player import HitterStats, PitcherStats
from fantasy_baseball.sgp.player_value import (
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
    calculate_player_sgp,
)


class TestCountingSgp:
    def test_hr_sgp(self):
        assert calculate_counting_sgp(45, 9.0) == pytest.approx(5.0)

    def test_zero_stat(self):
        assert calculate_counting_sgp(0, 9.0) == pytest.approx(0.0)

    def test_saves(self):
        assert calculate_counting_sgp(40, 7.0) == pytest.approx(5.714, abs=0.001)


class TestHittingRateSgp:
    def test_avg_marginal_hits(self):
        sgp = calculate_hitting_rate_sgp(
            player_avg=0.291,
            player_ab=550,
            replacement_avg=0.250,
            sgp_denominator=0.005,
            team_ab=5500,
        )
        expected = (0.291 - 0.250) * 550 / (0.005 * 5500)
        assert sgp == pytest.approx(expected)

    def test_below_replacement_avg(self):
        sgp = calculate_hitting_rate_sgp(
            player_avg=0.220,
            player_ab=400,
            replacement_avg=0.250,
            sgp_denominator=0.005,
            team_ab=5500,
        )
        assert sgp < 0


class TestPitchingRateSgp:
    def test_era_marginal(self):
        sgp = calculate_pitching_rate_sgp(
            player_rate=3.15,
            player_ip=200,
            replacement_rate=4.50,
            sgp_denominator=0.15,
            team_ip=1400,
            innings_divisor=9,
        )
        expected = (4.50 - 3.15) * 200 / 9 / (0.15 * 1400 / 9)
        assert sgp == pytest.approx(expected)

    def test_whip_marginal(self):
        sgp = calculate_pitching_rate_sgp(
            player_rate=1.05,
            player_ip=200,
            replacement_rate=1.35,
            sgp_denominator=0.015,
            team_ip=1400,
            innings_divisor=1,
        )
        expected = (1.35 - 1.05) * 200 / (0.015 * 1400)
        assert sgp == pytest.approx(expected)

    def test_bad_era_is_negative(self):
        sgp = calculate_pitching_rate_sgp(
            player_rate=5.50,
            player_ip=150,
            replacement_rate=4.50,
            sgp_denominator=0.15,
            team_ip=1400,
            innings_divisor=9,
        )
        assert sgp < 0


class TestCalculatePlayerSgp:
    def test_hitter_total_sgp(self):
        player = pd.Series(
            {
                "name": "Aaron Judge",
                "player_type": "hitter",
                "r": 110,
                "hr": 45,
                "rbi": 120,
                "sb": 5,
                "avg": 0.291,
                "ab": 550,
                "h": 160,
            }
        )
        sgp = calculate_player_sgp(player, team_ab=5500, team_ip=1400)
        assert sgp > 0
        assert sgp > 5.0

    def test_pitcher_total_sgp(self):
        player = pd.Series(
            {
                "name": "Gerrit Cole",
                "player_type": "pitcher",
                "w": 15,
                "k": 240,
                "sv": 0,
                "era": 3.15,
                "whip": 1.05,
                "ip": 200,
                "er": 70,
                "bb": 56,
                "h_allowed": 154,
            }
        )
        sgp = calculate_player_sgp(player, team_ab=5500, team_ip=1400)
        assert sgp > 0

    def test_nan_hitter_stats_produce_finite_sgp(self):
        """NaN stats must not propagate — treat as 0."""
        player = pd.Series(
            {
                "name": "Bad Data",
                "player_type": "hitter",
                "r": float("nan"),
                "hr": 20,
                "rbi": None,
                "sb": 5,
                "avg": float("nan"),
                "ab": 500,
                "h": 100,
            }
        )
        sgp = calculate_player_sgp(player)
        assert sgp == sgp  # not NaN
        assert isinstance(sgp, float)

    def test_nan_pitcher_stats_produce_finite_sgp(self):
        player = pd.Series(
            {
                "name": "Bad Pitcher",
                "player_type": "pitcher",
                "w": float("nan"),
                "k": 100,
                "sv": None,
                "era": float("nan"),
                "whip": 1.20,
                "ip": 150,
            }
        )
        sgp = calculate_player_sgp(player)
        assert sgp == sgp  # not NaN
        assert isinstance(sgp, float)


class TestCalculatePlayerSgpDataclass:
    def test_hitter_stats_matches_series(self):
        """HitterStats input produces same result as equivalent pd.Series."""
        stats = HitterStats(pa=650, ab=550, h=160, r=110, hr=45, rbi=120, sb=5, avg=0.291)
        series = pd.Series(
            {
                "player_type": "hitter",
                "r": 110,
                "hr": 45,
                "rbi": 120,
                "sb": 5,
                "avg": 0.291,
                "ab": 550,
                "h": 160,
            }
        )
        sgp_dc = calculate_player_sgp(stats)
        sgp_series = calculate_player_sgp(series)
        assert sgp_dc == pytest.approx(sgp_series)

    def test_pitcher_stats_matches_series(self):
        """PitcherStats input produces same result as equivalent pd.Series."""
        stats = PitcherStats(
            ip=200, w=15, k=240, sv=0, er=70, bb=56, h_allowed=154, era=3.15, whip=1.05
        )
        series = pd.Series(
            {
                "player_type": "pitcher",
                "w": 15,
                "k": 240,
                "sv": 0,
                "era": 3.15,
                "whip": 1.05,
                "ip": 200,
            }
        )
        sgp_dc = calculate_player_sgp(stats)
        sgp_series = calculate_player_sgp(series)
        assert sgp_dc == pytest.approx(sgp_series)

    def test_hitter_stats_positive(self):
        stats = HitterStats(pa=650, ab=550, h=160, r=110, hr=45, rbi=120, sb=5, avg=0.291)
        assert calculate_player_sgp(stats) > 5.0

    def test_pitcher_stats_positive(self):
        stats = PitcherStats(
            ip=200, w=15, k=240, sv=0, er=70, bb=56, h_allowed=154, era=3.15, whip=1.05
        )
        assert calculate_player_sgp(stats) > 0
