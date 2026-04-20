from fantasy_baseball.models.player import HitterStats, PitcherStats, Player
from fantasy_baseball.trades.evaluate import (
    aggregate_player_stats,
    compute_roto_points,
    compute_roto_points_by_cat,
    compute_trade_impact,
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


def test_aggregate_two_hitters_sums_counts_and_weights_avg():
    h1 = Player(name="A", player_type="hitter", positions=["OF"],
                rest_of_season=HitterStats(pa=600, ab=500, h=150,
                                            r=80, hr=25, rbi=70, sb=10, avg=0.300))
    h2 = Player(name="B", player_type="hitter", positions=["2B"],
                rest_of_season=HitterStats(pa=500, ab=400, h=100,
                                            r=50, hr=10, rbi=40, sb=5, avg=0.250))
    agg = aggregate_player_stats([h1, h2])
    assert agg["R"] == 130
    assert agg["HR"] == 35
    assert agg["ab"] == 900
    assert abs(agg["AVG"] - 250/900) < 1e-9
    assert agg["ip"] == 0


def test_aggregate_two_pitchers_weights_era_and_whip():
    p1 = Player(name="P1", player_type="pitcher", positions=["P"],
                rest_of_season=PitcherStats(ip=100, w=8, k=100, sv=0,
                                             era=3.60, whip=1.20,
                                             er=40, bb=30, h_allowed=90))
    p2 = Player(name="P2", player_type="pitcher", positions=["P"],
                rest_of_season=PitcherStats(ip=50, w=3, k=60, sv=20,
                                             era=2.70, whip=1.00,
                                             er=15, bb=10, h_allowed=40))
    agg = aggregate_player_stats([p1, p2])
    assert agg["W"] == 11
    assert agg["K"] == 160
    assert agg["SV"] == 20
    assert agg["ip"] == 150
    assert abs(agg["ERA"] - 3.30) < 1e-6
    assert abs(agg["WHIP"] - 170/150) < 1e-6


def test_aggregate_empty_list_returns_zeros():
    agg = aggregate_player_stats([])
    assert agg == {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 0.0, "WHIP": 0.0,
                   "ab": 0, "ip": 0}

