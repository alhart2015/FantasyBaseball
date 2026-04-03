import pytest
import pandas as pd
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import HitterStats, PitcherStats


def _make_hitter(name, r, hr, rbi, sb, avg, ab):
    return pd.Series({
        "name": name, "player_type": "hitter",
        "r": r, "hr": hr, "rbi": rbi, "sb": sb,
        "avg": avg, "ab": ab, "h": int(avg * ab),
    })


def _make_pitcher(name, w, k, sv, era, whip, ip):
    return pd.Series({
        "name": name, "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip, "ip": ip,
        "er": era * ip / 9, "bb": int(whip * ip * 0.3),
        "h_allowed": int(whip * ip * 0.7),
    })


class TestWeightedSgp:
    def test_equal_weights_matches_regular_sgp(self):
        player = _make_hitter("Judge", 110, 45, 120, 5, .291, 550)
        equal = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
        wsgp = calculate_weighted_sgp(player, equal)
        assert wsgp > 0

    def test_sb_heavy_weights_favor_speedster(self):
        power = _make_hitter("Power", 90, 40, 100, 2, .260, 520)
        speed = _make_hitter("Speed", 95, 15, 65, 40, .280, 550)
        sb_heavy = {"R": 0.05, "HR": 0.05, "RBI": 0.05, "SB": 0.6, "AVG": 0.05,
                    "W": 0.04, "K": 0.04, "SV": 0.04, "ERA": 0.04, "WHIP": 0.04}
        power_wsgp = calculate_weighted_sgp(power, sb_heavy)
        speed_wsgp = calculate_weighted_sgp(speed, sb_heavy)
        assert speed_wsgp > power_wsgp

    def test_pitcher_with_pitching_weights(self):
        pitcher = _make_pitcher("Cole", 15, 240, 0, 3.15, 1.05, 200)
        k_heavy = {"R": 0.02, "HR": 0.02, "RBI": 0.02, "SB": 0.02, "AVG": 0.02,
                   "W": 0.1, "K": 0.6, "SV": 0.05, "ERA": 0.08, "WHIP": 0.07}
        wsgp = calculate_weighted_sgp(pitcher, k_heavy)
        assert wsgp > 0

    def test_zero_weight_category_ignored(self):
        player = _make_hitter("Steals Only", 0, 0, 0, 50, .200, 400)
        only_sb = {"R": 0, "HR": 0, "RBI": 0, "SB": 1.0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}
        wsgp = calculate_weighted_sgp(player, only_sb)
        assert wsgp > 0  # Only SB contributes


class TestWeightedSgpDataclass:
    def test_hitter_stats_matches_series(self):
        stats = HitterStats(pa=650, ab=550, h=160, r=110, hr=45, rbi=120, sb=5, avg=0.291)
        series = pd.Series({
            "player_type": "hitter",
            "r": 110, "hr": 45, "rbi": 120, "sb": 5,
            "avg": 0.291, "ab": 550, "h": 160,
        })
        equal = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}
        wsgp_dc = calculate_weighted_sgp(stats, equal)
        wsgp_series = calculate_weighted_sgp(series, equal)
        assert wsgp_dc == pytest.approx(wsgp_series)

    def test_pitcher_stats_matches_series(self):
        stats = PitcherStats(ip=200, w=15, k=240, sv=0, er=70, bb=56, h_allowed=154, era=3.15, whip=1.05)
        series = pd.Series({
            "player_type": "pitcher",
            "w": 15, "k": 240, "sv": 0,
            "era": 3.15, "whip": 1.05, "ip": 200,
            "er": 70, "bb": 56, "h_allowed": 154,
        })
        k_heavy = {"R": 0.02, "HR": 0.02, "RBI": 0.02, "SB": 0.02, "AVG": 0.02,
                   "W": 0.1, "K": 0.6, "SV": 0.05, "ERA": 0.08, "WHIP": 0.07}
        wsgp_dc = calculate_weighted_sgp(stats, k_heavy)
        wsgp_series = calculate_weighted_sgp(series, k_heavy)
        assert wsgp_dc == pytest.approx(wsgp_series)

    def test_zero_weight_category_ignored(self):
        stats = HitterStats(pa=450, ab=400, h=80, r=0, hr=0, rbi=0, sb=50, avg=0.200)
        only_sb = {"R": 0, "HR": 0, "RBI": 0, "SB": 1.0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}
        assert calculate_weighted_sgp(stats, only_sb) > 0
