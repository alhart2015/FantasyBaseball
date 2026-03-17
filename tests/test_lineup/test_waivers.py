import pytest
import pandas as pd
from fantasy_baseball.lineup.waivers import evaluate_pickup, scan_waivers


def _make_player(name, player_type, **stats):
    data = {"name": name, "player_type": player_type}
    data.update(stats)
    return pd.Series(data)


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}


class TestEvaluatePickup:
    def test_better_player_has_positive_gain(self):
        add = _make_player("Good", "hitter", r=90, hr=30, rbi=85, sb=15, avg=.280, ab=540, h=151)
        drop = _make_player("Bad", "hitter", r=40, hr=8, rbi=30, sb=2, avg=.230, ab=300, h=69)
        result = evaluate_pickup(add, drop, EQUAL_LEVERAGE)
        assert result["sgp_gain"] > 0
        assert result["add"] == "Good"
        assert result["drop"] == "Bad"

    def test_worse_player_has_negative_gain(self):
        add = _make_player("Bad", "hitter", r=40, hr=8, rbi=30, sb=2, avg=.230, ab=300, h=69)
        drop = _make_player("Good", "hitter", r=90, hr=30, rbi=85, sb=15, avg=.280, ab=540, h=151)
        result = evaluate_pickup(add, drop, EQUAL_LEVERAGE)
        assert result["sgp_gain"] < 0

    def test_returns_category_breakdown(self):
        add = _make_player("Steals", "hitter", r=70, hr=10, rbi=50, sb=40, avg=.270, ab=500, h=135)
        drop = _make_player("Power", "hitter", r=70, hr=30, rbi=80, sb=2, avg=.250, ab=500, h=125)
        result = evaluate_pickup(add, drop, EQUAL_LEVERAGE)
        assert "categories" in result
        assert result["categories"]["SB"] > 0
        assert result["categories"]["HR"] < 0


class TestScanWaivers:
    def test_returns_ranked_recommendations(self):
        # Roster: one weak hitter
        roster = [
            _make_player("Weak", "hitter", r=30, hr=5, rbi=20, sb=1, avg=.220, ab=300, h=66,
                         positions=["OF"], best_position="OF"),
        ]
        # Free agents: two better hitters
        free_agents = [
            _make_player("Better", "hitter", r=70, hr=20, rbi=60, sb=10, avg=.270, ab=500, h=135,
                         positions=["OF"], best_position="OF"),
            _make_player("Best", "hitter", r=90, hr=30, rbi=80, sb=15, avg=.280, ab=540, h=151,
                         positions=["OF"], best_position="OF"),
        ]
        results = scan_waivers(roster, free_agents, EQUAL_LEVERAGE)
        assert len(results) > 0
        assert all(r["sgp_gain"] > 0 for r in results)
        # Should be sorted best-first
        assert results[0]["sgp_gain"] >= results[-1]["sgp_gain"]

    def test_no_recommendations_when_roster_is_better(self):
        roster = [
            _make_player("Star", "hitter", r=110, hr=45, rbi=120, sb=20, avg=.300, ab=550, h=165,
                         positions=["OF"], best_position="OF"),
        ]
        free_agents = [
            _make_player("Scrub", "hitter", r=30, hr=5, rbi=20, sb=1, avg=.220, ab=300, h=66,
                         positions=["OF"], best_position="OF"),
        ]
        results = scan_waivers(roster, free_agents, EQUAL_LEVERAGE)
        assert len(results) == 0  # No positive-gain pickups

    def test_empty_free_agents(self):
        roster = [
            _make_player("Player", "hitter", r=70, hr=20, rbi=60, sb=10, avg=.270, ab=500, h=135,
                         positions=["OF"], best_position="OF"),
        ]
        results = scan_waivers(roster, [], EQUAL_LEVERAGE)
        assert results == []
