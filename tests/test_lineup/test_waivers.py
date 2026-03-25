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

    def test_open_slots_recommends_pure_adds(self):
        """When there are open roster slots, recommend free agents without drops."""
        roster = [
            _make_player("Current", "hitter", r=70, hr=20, rbi=60, sb=10, avg=.270, ab=500, h=135,
                         positions=["OF"], best_position="OF"),
        ]
        free_agents = [
            _make_player("Available", "hitter", r=80, hr=25, rbi=70, sb=12, avg=.275, ab=520, h=143,
                         positions=["1B"], best_position="1B"),
        ]
        results = scan_waivers(roster, free_agents, EQUAL_LEVERAGE, open_hitter_slots=1)
        assert len(results) >= 1
        pure_adds = [r for r in results if r["drop"].startswith("(empty")]
        assert len(pure_adds) >= 1
        assert pure_adds[0]["add"] == "Available"

    def test_open_slots_with_empty_roster(self):
        """Open slots should work even with an empty matched roster."""
        free_agents = [
            _make_player("FreeAgent", "hitter", r=80, hr=25, rbi=70, sb=12, avg=.275, ab=520, h=143,
                         positions=["OF"], best_position="OF"),
        ]
        results = scan_waivers([], free_agents, EQUAL_LEVERAGE, open_bench_slots=2)
        assert len(results) >= 1
        assert results[0]["drop"].startswith("(empty")

    def test_skips_drop_that_leaves_position_hole(self):
        """Don't recommend dropping the only 1B if the add can't play 1B."""
        roster = [
            _make_player("Only1B", "hitter", r=40, hr=8, rbi=30, sb=2, avg=.230, ab=300, h=69,
                         positions=["1B", "Util"]),
            _make_player("GoodOF", "hitter", r=90, hr=30, rbi=80, sb=15, avg=.280, ab=540, h=151,
                         positions=["OF", "Util"]),
        ]
        # Free agent SS is "better" than Only1B but can't play 1B
        free_agents = [
            _make_player("BetterSS", "hitter", r=70, hr=20, rbi=60, sb=10, avg=.270, ab=500, h=135,
                         positions=["SS", "Util"]),
        ]
        roster_slots = {"1B": 1, "OF": 1, "UTIL": 1}
        results = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                               roster_slots=roster_slots)
        # Should NOT recommend dropping Only1B since BetterSS can't play 1B
        drop_names = [r["drop"] for r in results]
        assert "Only1B" not in drop_names

    def test_typed_slots_only_fill_matching_type(self):
        """Pitcher open slots should only be filled by pitchers, not hitters."""
        roster = [
            _make_player("Hitter", "hitter", r=70, hr=20, rbi=60, sb=10, avg=.270, ab=500, h=135,
                         positions=["OF"], best_position="OF"),
        ]
        hitter_fa = _make_player("BigBat", "hitter", r=90, hr=30, rbi=80, sb=15, avg=.280,
                                 ab=540, h=151, positions=["1B"], best_position="1B")
        pitcher_fa = _make_player("Ace", "pitcher", w=12, k=180, sv=0, era=3.20, whip=1.10,
                                  ip=180, er=64, bb=50, h_allowed=150, gs=30, g=30,
                                  positions=["SP"], best_position="SP")
        results = scan_waivers(roster, [hitter_fa, pitcher_fa], EQUAL_LEVERAGE,
                               open_pitcher_slots=2)
        pure_adds = [r for r in results if r["drop"].startswith("(empty")]
        # Only the pitcher should fill the pitcher slot
        assert all(r["add"] == "Ace" for r in pure_adds)
