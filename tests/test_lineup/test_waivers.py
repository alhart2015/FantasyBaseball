import pytest
import pandas as pd
from fantasy_baseball.lineup.waivers import evaluate_pickup, scan_waivers, _compute_team_wsgp, _build_lineup_summary


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
        results = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                               roster_slots={"OF": 1, "P": 0, "BN": 0, "IL": 0})
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


ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}


class TestComputeTeamWsgp:
    def test_returns_total_and_lineups(self):
        """_compute_team_wsgp returns total wSGP of assigned starters plus lineup dicts."""
        roster = [
            _make_player("Hitter A", "hitter", r=80, hr=25, rbi=75, sb=10, avg=.270, ab=500, h=135,
                         positions=["1B"]),
            _make_player("Hitter B", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["OF"]),
            _make_player("Pitcher A", "pitcher", w=12, k=180, sv=0, ip=180, er=60, bb=50, h_allowed=150,
                         era=3.00, whip=1.11, positions=["SP"]),
        ]
        result = _compute_team_wsgp(roster, EQUAL_LEVERAGE, ROSTER_SLOTS)
        assert "total_wsgp" in result
        assert "hitter_lineup" in result
        assert "pitcher_starters" in result
        assert "player_wsgp" in result
        assert result["total_wsgp"] > 0
        assert isinstance(result["hitter_lineup"], dict)
        assert isinstance(result["pitcher_starters"], list)

    def test_unassigned_players_dont_count(self):
        """Players who can't be assigned to any slot contribute 0 to total."""
        roster = [
            _make_player("Catcher 1", "hitter", r=50, hr=15, rbi=50, sb=2, avg=.260, ab=400, h=104,
                         positions=["C"]),
            _make_player("Catcher 2", "hitter", r=40, hr=10, rbi=35, sb=1, avg=.240, ab=350, h=84,
                         positions=["C"]),
        ]
        slots = {"C": 1, "P": 0, "BN": 1, "IL": 0}
        result = _compute_team_wsgp(roster, EQUAL_LEVERAGE, slots)
        assert len(result["hitter_lineup"]) == 1


class TestBuildLineupSummary:
    def test_basic_lineup(self):
        """Builds lineup list from optimizer output."""
        hitter_lineup = {"C": "Player A", "1B": "Player B", "OF": "Player C"}
        pitcher_starters = [{"name": "Pitcher X", "wsgp": 2.0}]
        player_wsgp = {"Player A": 1.5, "Player B": 2.0, "Player C": 1.0, "Pitcher X": 2.0}
        all_players = ["Player A", "Player B", "Player C", "Player D", "Pitcher X", "Pitcher Y"]

        result = _build_lineup_summary(hitter_lineup, pitcher_starters, player_wsgp, all_players)
        assert len(result) > 0
        player_a = next(e for e in result if e["name"] == "Player A")
        assert player_a["slot"] == "C"
        assert player_a["wsgp"] == 1.5

    def test_strips_slot_suffixes(self):
        """OF_2, UTIL_3 etc. are stripped to base slot name for display."""
        hitter_lineup = {"OF": "Player A", "OF_2": "Player B"}
        pitcher_starters = []
        player_wsgp = {"Player A": 1.0, "Player B": 0.8}

        result = _build_lineup_summary(hitter_lineup, pitcher_starters, player_wsgp, ["Player A", "Player B"])
        slots = [e["slot"] for e in result if e["name"] in ("Player A", "Player B")]
        assert all(s == "OF" for s in slots)

    def test_bench_players_flagged(self):
        """Players not in any optimizer output are marked as bench."""
        hitter_lineup = {"C": "Starter"}
        pitcher_starters = []
        player_wsgp = {"Starter": 2.0, "Benched": 0.5}

        result = _build_lineup_summary(hitter_lineup, pitcher_starters, player_wsgp, ["Starter", "Benched"])
        benched = next(e for e in result if e["name"] == "Benched")
        assert benched["slot"] == "BN"


class TestScanWaiversReoptimize:
    def test_cross_position_swap_recommended(self):
        """Can pick up a 3B and drop a benched OF if roster reshuffles to cover.

        Good SS is eligible at SS, 3B, and OF.  With slots SS/1B/3B, Weak OF has
        no active slot and sits on the bench.  Adding Great 3B (3B only) fills the
        3B slot while Weak OF is dropped — a cross-position swap that the
        whole-roster re-optimiser can evaluate but the old same-type logic could not.
        """
        roster = [
            _make_player("Good SS", "hitter", r=80, hr=20, rbi=70, sb=15, avg=.280, ab=500, h=140,
                         positions=["SS", "3B", "OF"]),
            _make_player("Weak OF", "hitter", r=40, hr=5, rbi=25, sb=2, avg=.230, ab=300, h=69,
                         positions=["OF"]),
            _make_player("Decent 1B", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["1B"]),
        ]
        free_agents = [
            _make_player("Great 3B", "hitter", r=90, hr=30, rbi=85, sb=10, avg=.275, ab=540, h=148,
                         positions=["3B"]),
        ]
        # No OF slot — Weak OF is currently benched.  Adding Great 3B fills 3B.
        slots = {"SS": 1, "1B": 1, "3B": 1, "P": 0, "BN": 1, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        assert len(result) >= 1
        assert result[0]["add"] == "Great 3B"
        assert result[0]["drop"] == "Weak OF"
        assert result[0]["sgp_gain"] > 0
        assert "add_positions" in result[0]
        assert "drop_positions" in result[0]

    def test_position_infeasible_swap_skipped(self):
        """Can't drop the only C if no one else can play C."""
        roster = [
            _make_player("Only C", "hitter", r=50, hr=12, rbi=45, sb=3, avg=.250, ab=400, h=100,
                         positions=["C"]),
            _make_player("OF Guy", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["OF"]),
        ]
        free_agents = [
            _make_player("Great OF", "hitter", r=90, hr=30, rbi=85, sb=15, avg=.280, ab=540, h=151,
                         positions=["OF"]),
        ]
        slots = {"C": 1, "OF": 1, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        assert len(result) >= 1
        assert result[0]["add"] == "Great OF"
        assert result[0]["drop"] == "OF Guy"
        for r in result:
            assert r["drop"] != "Only C"

    def test_includes_lineup_before_after(self):
        """Recommendations include before/after lineup data for expanded card."""
        roster = [
            _make_player("Starter", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["OF"]),
        ]
        free_agents = [
            _make_player("Better", "hitter", r=80, hr=25, rbi=75, sb=10, avg=.270, ab=500, h=135,
                         positions=["OF"]),
        ]
        slots = {"OF": 1, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        assert len(result) >= 1
        assert "lineup_before" in result[0]
        assert "lineup_after" in result[0]
        before_names = [e["name"] for e in result[0]["lineup_before"]]
        after_names = [e["name"] for e in result[0]["lineup_after"]]
        assert "Starter" in before_names
        assert "Better" in after_names

    def test_best_drop_per_fa(self):
        """For each FA, only the best drop candidate is kept."""
        roster = [
            _make_player("OK OF", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["OF"]),
            _make_player("Bad OF", "hitter", r=30, hr=5, rbi=20, sb=1, avg=.220, ab=300, h=66,
                         positions=["OF"]),
        ]
        free_agents = [
            _make_player("Good OF", "hitter", r=80, hr=25, rbi=75, sb=10, avg=.270, ab=500, h=135,
                         positions=["OF"]),
        ]
        slots = {"OF": 2, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        good_of_recs = [r for r in result if r["add"] == "Good OF"]
        assert len(good_of_recs) == 1
        assert good_of_recs[0]["drop"] == "Bad OF"

    def test_wsgp_floor_prunes_bad_fas(self):
        """FAs below the wSGP floor are skipped."""
        roster = [
            _make_player("Decent", "hitter", r=70, hr=20, rbi=65, sb=8, avg=.265, ab=480, h=127,
                         positions=["1B"]),
        ]
        free_agents = [
            _make_player("Terrible", "hitter", r=10, hr=1, rbi=5, sb=0, avg=.180, ab=100, h=18,
                         positions=["1B"]),
        ]
        slots = {"1B": 1, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        assert len(result) == 0
