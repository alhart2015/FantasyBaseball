import pytest
from fantasy_baseball.lineup.leverage import calculate_leverage


def _make_standings():
    """10 teams with standings data. User team is rank 5."""
    return [
        {"name": "Team 1", "rank": 1, "stats": {"R": 500, "HR": 150, "RBI": 480, "SB": 90, "AVG": 0.275, "W": 55, "K": 850, "SV": 55, "ERA": 3.40, "WHIP": 1.15}},
        {"name": "Team 2", "rank": 2, "stats": {"R": 490, "HR": 145, "RBI": 470, "SB": 85, "AVG": 0.272, "W": 52, "K": 830, "SV": 50, "ERA": 3.50, "WHIP": 1.18}},
        {"name": "Team 3", "rank": 3, "stats": {"R": 475, "HR": 140, "RBI": 455, "SB": 80, "AVG": 0.270, "W": 50, "K": 810, "SV": 48, "ERA": 3.60, "WHIP": 1.20}},
        {"name": "Team 4", "rank": 4, "stats": {"R": 460, "HR": 135, "RBI": 445, "SB": 75, "AVG": 0.268, "W": 48, "K": 790, "SV": 45, "ERA": 3.70, "WHIP": 1.22}},
        {"name": "User Team", "rank": 5, "stats": {"R": 450, "HR": 130, "RBI": 430, "SB": 50, "AVG": 0.265, "W": 45, "K": 770, "SV": 40, "ERA": 3.80, "WHIP": 1.25}},
        {"name": "Team 6", "rank": 6, "stats": {"R": 430, "HR": 120, "RBI": 410, "SB": 45, "AVG": 0.260, "W": 42, "K": 740, "SV": 35, "ERA": 3.95, "WHIP": 1.28}},
        {"name": "Team 7", "rank": 7, "stats": {"R": 420, "HR": 115, "RBI": 400, "SB": 40, "AVG": 0.258, "W": 40, "K": 720, "SV": 30, "ERA": 4.10, "WHIP": 1.30}},
        {"name": "Team 8", "rank": 8, "stats": {"R": 400, "HR": 105, "RBI": 380, "SB": 35, "AVG": 0.252, "W": 35, "K": 690, "SV": 25, "ERA": 4.30, "WHIP": 1.35}},
        {"name": "Team 9", "rank": 9, "stats": {"R": 380, "HR": 95, "RBI": 360, "SB": 30, "AVG": 0.248, "W": 32, "K": 660, "SV": 20, "ERA": 4.50, "WHIP": 1.40}},
        {"name": "Team 10", "rank": 10, "stats": {"R": 350, "HR": 80, "RBI": 330, "SB": 20, "AVG": 0.240, "W": 28, "K": 620, "SV": 15, "ERA": 4.80, "WHIP": 1.48}},
    ]


class TestCalculateLeverage:
    def test_returns_all_categories(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        assert "R" in leverage
        assert "HR" in leverage
        assert "ERA" in leverage
        assert len(leverage) == 10

    def test_all_weights_positive(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        for cat, weight in leverage.items():
            assert weight >= 0, f"{cat} has negative weight"

    def test_weights_sum_to_one(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_small_gap_gets_high_leverage(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        assert leverage["R"] > leverage["SB"]

    def test_inverse_stats_correct_direction(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        assert leverage["ERA"] > 0

    def test_last_place_team_has_leverage(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "Team 10")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=0.01)
