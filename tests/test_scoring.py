import pytest
from fantasy_baseball.scoring import project_team_stats, score_roto, ALL_CATS


class TestProjectTeamStats:
    def test_pitcher_with_util_position_contributes_pitching_stats(self):
        """Regression: pitchers whose Yahoo positions include Util must still
        contribute pitching stats, not be silently dropped or misrouted as
        hitters.  This bug caused 20+ point swings in projected standings
        when Gerrit Cole, Shane Bieber, etc. had their pitching zeroed out.
        """
        roster = [
            {"name": "Hitter A", "player_type": "hitter",
             "r": 100, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 550,
             "positions": ["OF", "Util"]},
            {"name": "Pitcher With Util", "player_type": "pitcher",
             "w": 15, "k": 200, "sv": 0, "ip": 180, "er": 60, "bb": 50,
             "h_allowed": 150, "positions": ["P", "Util"]},
        ]
        stats = project_team_stats(roster)
        assert stats["W"] == 15
        assert stats["K"] == 200
        assert stats["ERA"] == pytest.approx(60 * 9 / 180)

    def test_hitter_and_pitcher_both_counted(self):
        roster = [
            {"name": "Hitter", "player_type": "hitter",
             "r": 80, "hr": 25, "rbi": 70, "sb": 5, "h": 130, "ab": 500},
            {"name": "Pitcher", "player_type": "pitcher",
             "w": 10, "k": 150, "sv": 30, "ip": 60, "er": 20, "bb": 15,
             "h_allowed": 50},
        ]
        stats = project_team_stats(roster)
        assert stats["R"] == 80
        assert stats["HR"] == 25
        assert stats["W"] == 10
        assert stats["SV"] == 30
        assert stats["AVG"] == pytest.approx(130 / 500)
        assert stats["ERA"] == pytest.approx(20 * 9 / 60)
        assert stats["WHIP"] == pytest.approx((15 + 50) / 60)

    def test_empty_roster(self):
        stats = project_team_stats([])
        assert stats["R"] == 0
        assert stats["AVG"] == 0
        assert stats["ERA"] == 99
        assert stats["WHIP"] == 99

    def test_pitchers_only(self):
        roster = [
            {"name": "SP", "player_type": "pitcher",
             "w": 12, "k": 180, "sv": 0, "ip": 200, "er": 70, "bb": 50,
             "h_allowed": 170},
        ]
        stats = project_team_stats(roster)
        assert stats["R"] == 0
        assert stats["AVG"] == 0
        assert stats["W"] == 12

    def test_hitters_only(self):
        roster = [
            {"name": "H", "player_type": "hitter",
             "r": 90, "hr": 35, "rbi": 100, "sb": 15, "h": 160, "ab": 580},
        ]
        stats = project_team_stats(roster)
        assert stats["W"] == 0
        assert stats["ERA"] == 99
        assert stats["R"] == 90


class TestScoreRoto:
    def test_two_teams_simple(self):
        stats = {
            "A": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
            "B": {"R": 800, "HR": 200, "RBI": 750, "SB": 80, "AVG": 0.260,
                   "W": 70, "K": 1100, "SV": 40, "ERA": 4.00, "WHIP": 1.25},
        }
        roto = score_roto(stats)
        assert roto["A"]["total"] == 20  # wins every category
        assert roto["B"]["total"] == 10

    def test_fractional_tiebreaker(self):
        stats = {
            "A": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
            "B": {"R": 900, "HR": 250, "RBI": 850, "SB": 100, "AVG": 0.270,
                   "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.15},
        }
        roto = score_roto(stats)
        # Tied in everything — both get 1.5 per cat (avg of 1 and 2)
        assert roto["A"]["total"] == pytest.approx(15.0)
        assert roto["B"]["total"] == pytest.approx(15.0)

    def test_inverse_stats_lower_is_better(self):
        stats = {
            "A": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 3.00, "WHIP": 1.10},
            "B": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 4.50, "WHIP": 1.30},
        }
        roto = score_roto(stats)
        assert roto["A"]["ERA_pts"] == 2  # lower ERA = better = more points
        assert roto["B"]["ERA_pts"] == 1

    def test_all_categories_present(self):
        stats = {
            "A": {c: 1 for c in ALL_CATS},
        }
        roto = score_roto(stats)
        for c in ALL_CATS:
            assert f"{c}_pts" in roto["A"]
        assert "total" in roto["A"]
