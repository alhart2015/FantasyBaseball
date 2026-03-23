import pytest
from fantasy_baseball.trades.evaluate import (
    compute_roto_points,
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
]


def test_find_trades_returns_ranked_list():
    hart_roster = [
        {"name": "Slugger", "player_type": "hitter", "positions": ["OF"],
         "r": 80, "hr": 35, "rbi": 90, "sb": 5, "avg": .270, "h": 140, "ab": 520, "pa": 570},
        {"name": "Speedy", "player_type": "hitter", "positions": ["SS"],
         "r": 70, "hr": 10, "rbi": 50, "sb": 40, "avg": .260, "h": 130, "ab": 500, "pa": 550},
    ]
    opp_rosters = {
        "Rival": [
            {"name": "Closer", "player_type": "pitcher", "positions": ["RP"],
             "w": 3, "k": 60, "sv": 30, "era": 2.80, "whip": 1.00, "ip": 65,
             "er": 20, "bb": 15, "h_allowed": 50},
            {"name": "Stealer", "player_type": "hitter", "positions": ["OF"],
             "r": 75, "hr": 8, "rbi": 45, "sb": 45, "avg": .265, "h": 135, "ab": 510, "pa": 560},
        ],
    }
    leverage_by_team = {
        "Hart": {"R": .1, "HR": .05, "RBI": .1, "SB": .15, "AVG": .1,
                 "W": .1, "K": .1, "SV": .15, "ERA": .1, "WHIP": .05},
        "Rival": {"R": .1, "HR": .15, "RBI": .1, "SB": .05, "AVG": .1,
                  "W": .1, "K": .1, "SV": .1, "ERA": .1, "WHIP": .1},
    }

    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team=leverage_by_team,
        roster_slots=ROSTER_SLOTS, max_results=5,
    )
    assert isinstance(trades, list)
    if trades:
        t = trades[0]
        assert "send" in t
        assert "receive" in t
        assert "opponent" in t
        assert "hart_delta" in t
        assert "opp_delta" in t
        assert "hart_wsgp_gain" in t
