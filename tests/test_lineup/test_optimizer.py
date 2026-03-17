import pytest
import pandas as pd
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup


def _make_hitter(name, positions, r, hr, rbi, sb, avg, ab):
    return pd.Series({
        "name": name, "positions": positions, "player_type": "hitter",
        "r": r, "hr": hr, "rbi": rbi, "sb": sb,
        "avg": avg, "ab": ab, "h": int(avg * ab),
    })


def _make_pitcher(name, positions, w, k, sv, era, whip, ip):
    return pd.Series({
        "name": name, "positions": positions, "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip, "ip": ip,
        "er": era * ip / 9, "bb": int(whip * ip * 0.3),
        "h_allowed": int(whip * ip * 0.7),
    })


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}


class TestOptimizeHitterLineup:
    def test_assigns_all_starters(self):
        hitters = [
            _make_hitter("C1", ["C"], 60, 20, 65, 2, .250, 450),
            _make_hitter("1B1", ["1B"], 80, 30, 95, 3, .270, 520),
            _make_hitter("2B1", ["2B"], 75, 18, 70, 12, .265, 510),
            _make_hitter("3B1", ["3B"], 70, 25, 80, 5, .260, 500),
            _make_hitter("SS1", ["SS"], 85, 22, 75, 25, .275, 540),
            _make_hitter("OF1", ["OF"], 100, 35, 100, 8, .280, 550),
            _make_hitter("OF2", ["OF"], 90, 28, 85, 15, .275, 530),
            _make_hitter("OF3", ["OF"], 80, 22, 70, 20, .270, 510),
            _make_hitter("OF4", ["OF"], 75, 18, 65, 10, .265, 490),
            _make_hitter("UTIL1", ["1B", "OF"], 85, 32, 90, 2, .260, 520),
            _make_hitter("UTIL2", ["DH"], 70, 20, 75, 1, .255, 480),
            _make_hitter("BN1", ["OF"], 50, 10, 40, 5, .245, 350),
        ]
        lineup = optimize_hitter_lineup(hitters, EQUAL_LEVERAGE)
        assert "C" in lineup
        assert "1B" in lineup
        assert lineup["C"] == "C1"

    def test_multi_position_player_optimal_slot(self):
        hitters = [
            _make_hitter("Multi", ["SS", "2B"], 90, 25, 80, 20, .280, 540),
            _make_hitter("SS Only", ["SS"], 70, 15, 60, 10, .260, 480),
            _make_hitter("2B Only", ["2B"], 65, 12, 55, 8, .255, 470),
        ]
        lineup = optimize_hitter_lineup(hitters, EQUAL_LEVERAGE)
        assert lineup.get("SS") is not None or lineup.get("2B") is not None

    def test_bench_players_identified(self):
        hitters = [
            _make_hitter("Star", ["OF"], 110, 45, 120, 5, .291, 550),
            _make_hitter("Scrub", ["OF"], 30, 5, 20, 1, .220, 200),
        ]
        lineup = optimize_hitter_lineup(hitters, EQUAL_LEVERAGE)
        starters = set(lineup.values())
        assert "Star" in starters


class TestOptimizePitcherLineup:
    def test_starts_top_pitchers(self):
        pitchers = [
            _make_pitcher("Ace", ["SP"], 15, 240, 0, 3.00, 1.05, 200),
            _make_pitcher("Mid", ["SP"], 10, 160, 0, 3.80, 1.20, 170),
            _make_pitcher("Bad", ["SP"], 5, 80, 0, 5.00, 1.45, 100),
            _make_pitcher("Closer", ["RP"], 3, 60, 35, 2.50, 1.00, 65),
        ]
        starters, bench = optimize_pitcher_lineup(pitchers, EQUAL_LEVERAGE, slots=3)
        starter_names = [p["name"] for p in starters]
        assert "Ace" in starter_names
        assert "Closer" in starter_names
        assert len(starters) == 3

    def test_respects_slot_count(self):
        pitchers = [
            _make_pitcher(f"P{i}", ["SP"], 10, 150, 0, 3.50, 1.15, 170)
            for i in range(12)
        ]
        starters, bench = optimize_pitcher_lineup(pitchers, EQUAL_LEVERAGE, slots=9)
        assert len(starters) == 9
        assert len(bench) == 3
