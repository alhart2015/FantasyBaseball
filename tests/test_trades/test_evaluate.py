import pytest
from fantasy_baseball.models.player import Player, HitterStats, PitcherStats
from fantasy_baseball.sgp.rankings import rank_key
from fantasy_baseball.trades.evaluate import (
    compute_roto_points,
    compute_roto_points_by_cat,
    compute_trade_impact,
    find_trades,
)

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]

STANDINGS = [
    {"name": "Team A", "stats": {"R": 900, "HR": 250, "RBI": 880, "SB": 150,
     "AVG": .265, "W": 80, "K": 1300, "SV": 80, "ERA": 3.50, "WHIP": 1.15}},
    {"name": "Team B", "stats": {"R": 850, "HR": 280, "RBI": 900, "SB": 120,
     "AVG": .255, "W": 85, "K": 1400, "SV": 60, "ERA": 3.80, "WHIP": 1.20}},
    {"name": "Team C", "stats": {"R": 800, "HR": 260, "RBI": 850, "SB": 180,
     "AVG": .250, "W": 75, "K": 1200, "SV": 90, "ERA": 3.30, "WHIP": 1.10}},
]


def test_compute_roto_points():
    points = compute_roto_points(STANDINGS)
    # Team A: R=3, HR=1, RBI=2, SB=2, AVG=3, W=2, K=2, SV=2, ERA=2, WHIP=2 = 21
    assert points["Team A"] == 21
    assert points["Team C"] == 19


def test_compute_trade_impact():
    hart_loses_ros = {"R": 50, "HR": 30, "RBI": 60, "SB": 20, "AVG": .280,
                      "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0,
                      "ab": 400, "ip": 0}
    hart_gains_ros = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                      "W": 5, "K": 100, "SV": 30, "ERA": 3.00, "WHIP": 1.05,
                      "ab": 0, "ip": 150}
    opp_loses_ros = hart_gains_ros
    opp_gains_ros = hart_loses_ros

    result = compute_trade_impact(
        standings=STANDINGS, hart_name="Team A", opp_name="Team B",
        hart_loses_ros=hart_loses_ros, hart_gains_ros=hart_gains_ros,
        opp_loses_ros=opp_loses_ros, opp_gains_ros=opp_gains_ros,
    )
    assert "hart_delta" in result
    assert "opp_delta" in result
    assert "hart_cat_deltas" in result
    assert "opp_cat_deltas" in result
    assert isinstance(result["hart_delta"], (int, float))


def test_trade_impact_zero_for_identical_players():
    same = {"R": 50, "HR": 20, "RBI": 50, "SB": 10, "AVG": .260,
            "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0,
            "ab": 400, "ip": 0}
    result = compute_trade_impact(
        standings=STANDINGS, hart_name="Team A", opp_name="Team B",
        hart_loses_ros=same, hart_gains_ros=same,
        opp_loses_ros=same, opp_gains_ros=same,
    )
    assert result["hart_delta"] == 0
    assert result["opp_delta"] == 0


ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
                "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}

SAMPLE_STANDINGS = [
    {"name": "Hart", "team_key": "t.1", "rank": 3,
     "stats": {"R": 900, "HR": 280, "RBI": 880, "SB": 120,
               "AVG": .260, "W": 80, "K": 1300, "SV": 80, "ERA": 3.50, "WHIP": 1.15}},
    {"name": "Rival", "team_key": "t.2", "rank": 5,
     "stats": {"R": 850, "HR": 250, "RBI": 870, "SB": 180,
               "AVG": .255, "W": 85, "K": 1400, "SV": 40, "ERA": 3.80, "WHIP": 1.20}},
    {"name": "Rival A", "team_key": "t.3", "rank": 4,
     "stats": {"R": 870, "HR": 260, "RBI": 860, "SB": 140,
               "AVG": .258, "W": 82, "K": 1350, "SV": 50, "ERA": 3.60, "WHIP": 1.18}},
    {"name": "Rival B", "team_key": "t.4", "rank": 6,
     "stats": {"R": 830, "HR": 240, "RBI": 850, "SB": 160,
               "AVG": .252, "W": 78, "K": 1250, "SV": 45, "ERA": 3.70, "WHIP": 1.22}},
    {"name": "Rival C", "team_key": "t.5", "rank": 7,
     "stats": {"R": 820, "HR": 230, "RBI": 840, "SB": 170,
               "AVG": .248, "W": 76, "K": 1200, "SV": 55, "ERA": 3.90, "WHIP": 1.25}},
]


def test_compute_roto_points_by_cat_missing_stats():
    """Teams missing some stat categories should get default values, not crash."""
    standings = [
        {"name": "Full", "stats": {"R": 100, "HR": 30, "RBI": 90, "SB": 20,
         "AVG": .260, "W": 10, "K": 150, "SV": 10, "ERA": 3.50, "WHIP": 1.15}},
        {"name": "No Pitching", "stats": {"R": 80, "HR": 25, "RBI": 85, "SB": 15,
         "AVG": .250, "W": 0, "K": 0, "SV": 0}},
        # ERA and WHIP missing entirely for "No Pitching"
    ]
    result = compute_roto_points_by_cat(standings)
    # Should not crash, and every team should have all 10 categories
    assert "ERA" in result["No Pitching"]
    assert "WHIP" in result["No Pitching"]
    assert len(result["Full"]) == 10
    assert len(result["No Pitching"]) == 10
    # "No Pitching" should rank last in ERA/WHIP (got default 99.0)
    assert result["Full"]["ERA"] > result["No Pitching"]["ERA"]
    assert result["Full"]["WHIP"] > result["No Pitching"]["WHIP"]


def test_find_trades_returns_ranked_list():
    hart_roster = [
        Player(name="Slugger", player_type="hitter", positions=["OF"],
               ros=HitterStats(pa=570, ab=520, h=140, r=80, hr=35, rbi=90, sb=5, avg=.270)),
        Player(name="Speedy", player_type="hitter", positions=["SS"],
               ros=HitterStats(pa=550, ab=500, h=130, r=70, hr=10, rbi=50, sb=40, avg=.260)),
    ]
    opp_rosters = {
        "Rival": [
            Player(name="Closer", player_type="pitcher", positions=["RP"],
                   ros=PitcherStats(ip=65, w=3, k=60, sv=30, era=2.80, whip=1.00,
                                    er=20, bb=15, h_allowed=50)),
            Player(name="Stealer", player_type="hitter", positions=["OF"],
                   ros=HitterStats(pa=560, ab=510, h=135, r=75, hr=8, rbi=45, sb=45, avg=.265)),
        ],
    }
    leverage_by_team = {
        "Hart": {"R": .1, "HR": .05, "RBI": .1, "SB": .15, "AVG": .1,
                 "W": .1, "K": .1, "SV": .15, "ERA": .1, "WHIP": .05},
        "Rival": {"R": .1, "HR": .15, "RBI": .1, "SB": .05, "AVG": .1,
                  "W": .1, "K": .1, "SV": .1, "ERA": .1, "WHIP": .1},
    }
    rankings = {
        rank_key("Slugger", "hitter"): 30,
        rank_key("Speedy", "hitter"): 40,
        rank_key("Closer", "pitcher"): 35,
        rank_key("Stealer", "hitter"): 38,
    }

    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team=leverage_by_team,
        roster_slots=ROSTER_SLOTS, rankings=rankings, max_results=5,
    )
    assert isinstance(trades, list)
    if trades:
        t = trades[0]
        assert "send" in t
        assert "receive" in t
        assert "opponent" in t
        assert "hart_delta" in t
        assert "hart_wsgp_gain" in t
        assert "send_rank" in t
        assert "receive_rank" in t


# Shared fixtures for perception-based trade tests
_EQUAL_LEVERAGE = {cat: 0.1 for cat in ALL_CATS}

def _make_hitter(name, positions, r=70, hr=20, rbi=65, sb=8, avg=.270, ab=500):
    h = int(avg * ab)
    return Player(name=name, player_type="hitter", positions=positions,
                  ros=HitterStats(pa=int(ab * 1.15), ab=ab, h=h,
                                  r=r, hr=hr, rbi=rbi, sb=sb, avg=avg))

def _make_pitcher(name, positions, ip=150, w=9, k=140, sv=0, era=3.80, whip=1.25):
    er = int(era * ip / 9)
    bb = int((whip * ip - ip * 0.8) / 1)  # rough estimate
    h_allowed = int(whip * ip - bb)
    return Player(name=name, player_type="pitcher", positions=positions,
                  ros=PitcherStats(ip=ip, w=w, k=k, sv=sv, era=era, whip=whip,
                                   er=er, bb=bb, h_allowed=h_allowed))


def test_rank_filter_accepts_within_threshold():
    """Trade where send_rank - receive_rank = 5 should be accepted."""
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
    opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
    rankings = {
        rank_key("Hart OF", "hitter"): 55,
        rank_key("Opp OF", "hitter"): 50,
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert any(t["send"] == "Hart OF" and t["receive"] == "Opp OF" for t in trades)


def test_rank_filter_rejects_beyond_threshold():
    """Trade where send_rank - receive_rank = 6 should be rejected."""
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
    opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
    rankings = {
        rank_key("Hart OF", "hitter"): 56,
        rank_key("Opp OF", "hitter"): 50,
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert not any(t["send"] == "Hart OF" and t["receive"] == "Opp OF" for t in trades)


def test_rank_filter_accepts_sending_better_ranked():
    """Sending a better-ranked player (negative gap) should always be accepted."""
    hart_roster = [_make_hitter("Hart Star", ["OF"], hr=30, sb=3)]
    opp_rosters = {"Rival": [_make_hitter("Opp Guy", ["OF"], hr=10, sb=30)]}
    rankings = {
        rank_key("Hart Star", "hitter"): 20,
        rank_key("Opp Guy", "hitter"): 50,
    }
    leverage = {"Hart": {"R": .1, "HR": .05, "RBI": .1, "SB": .2, "AVG": .1,
                         "W": .1, "K": .1, "SV": .1, "ERA": .05, "WHIP": .05},
                "Rival": _EQUAL_LEVERAGE}
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team=leverage,
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert any(t["send"] == "Hart Star" and t["receive"] == "Opp Guy" for t in trades)


def test_rejects_trade_with_no_wsgp_gain():
    """Trade must have positive hart_wsgp_gain even if ranking looks fair."""
    hart_roster = [_make_hitter("Hart Star", ["OF"], r=100, hr=40, rbi=110, sb=20, avg=.300)]
    opp_rosters = {"Rival": [_make_hitter("Opp Scrub", ["OF"], r=40, hr=5, rbi=30, sb=2, avg=.220)]}
    rankings = {
        rank_key("Hart Star", "hitter"): 10,
        rank_key("Opp Scrub", "hitter"): 12,
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert not any(t["send"] == "Hart Star" for t in trades)


def test_sort_by_wsgp_gain_descending():
    """Trades should be sorted by hart_wsgp_gain descending."""
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=20, sb=3)]
    opp_rosters = {
        "Rival A": [_make_hitter("Opp A", ["OF"], hr=18, sb=10)],
        "Rival B": [_make_hitter("Opp B", ["OF"], hr=18, sb=25)],
        "Rival C": [_make_hitter("Opp C", ["OF"], hr=18, sb=18)],
    }
    rankings = {
        rank_key("Hart OF", "hitter"): 50,
        rank_key("Opp A", "hitter"): 48,
        rank_key("Opp B", "hitter"): 49,
        rank_key("Opp C", "hitter"): 47,
    }
    leverage = {"Hart": {"R": .05, "HR": .05, "RBI": .05, "SB": .3, "AVG": .05,
                         "W": .1, "K": .1, "SV": .1, "ERA": .1, "WHIP": .1},
                "Rival A": _EQUAL_LEVERAGE, "Rival B": _EQUAL_LEVERAGE,
                "Rival C": _EQUAL_LEVERAGE}
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team=leverage,
        roster_slots=ROSTER_SLOTS, rankings=rankings, max_results=10,
    )
    gains = [t["hart_wsgp_gain"] for t in trades]
    assert gains == sorted(gains, reverse=True)


def test_sort_tiebreaker_by_rank_generosity():
    """Trades with equal wSGP gain should prefer sending better-ranked player."""
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=20, sb=5)]
    opp_rosters = {
        "Rival A": [_make_hitter("Opp A", ["OF"], hr=20, sb=5)],
        "Rival B": [_make_hitter("Opp B", ["OF"], hr=20, sb=5)],
    }
    rankings = {
        rank_key("Hart OF", "hitter"): 50,
        rank_key("Opp A", "hitter"): 52,  # gap = -2 (sending better)
        rank_key("Opp B", "hitter"): 48,  # gap = +2 (sending worse)
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE,
                                                      "Rival A": _EQUAL_LEVERAGE,
                                                      "Rival B": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    if len(trades) >= 2:
        assert trades[0]["receive"] == "Opp A"


def test_roster_legality_still_enforced():
    """A swap that violates position coverage is rejected even if ranking is fair."""
    hart_roster = [_make_hitter("Hart C", ["C"])]
    opp_rosters = {"Rival": [_make_pitcher("Opp SP", ["SP"])]}
    rankings = {
        rank_key("Hart C", "hitter"): 50,
        rank_key("Opp SP", "pitcher"): 50,
    }
    slots = {"C": 1, "P": 0, "BN": 0, "IL": 0}
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=slots, rankings=rankings,
    )
    assert len(trades) == 0


def test_trades_include_rank_data():
    """Each trade result should include send_rank and receive_rank."""
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
    opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
    rankings = {
        rank_key("Hart OF", "hitter"): 55,
        rank_key("Opp OF", "hitter"): 50,
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert len(trades) > 0
    t = trades[0]
    assert t["send_rank"] == 55
    assert t["receive_rank"] == 50
