import pytest
from fantasy_baseball.lineup.player_classification import classify_roster
from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats


def _hitter(name, wsgp, status=""):
    p = Player(name=name, player_type=PlayerType.HITTER,
               positions=["OF", "Util"],
               ros=HitterStats(pa=600, ab=550, h=150, r=80, hr=25, rbi=80, sb=10, avg=0.273),
               wsgp=wsgp, status=status)
    return p


def _pitcher(name, wsgp, status=""):
    p = Player(name=name, player_type=PlayerType.PITCHER,
               positions=["P"],
               ros=PitcherStats(ip=180, w=12, k=180, sv=0, er=60, bb=50, h_allowed=150, era=3.00, whip=1.11),
               wsgp=wsgp, status=status)
    return p


class TestClassifyRoster:
    def test_core_high_rank_high_wsgp(self):
        """Top-ranked player with above-median wSGP is core."""
        roster = [
            _hitter("Star", 2.0),
            _hitter("Average", 1.0),
            _hitter("Weak", 0.5),
        ]
        rankings = {"star::hitter": 10, "average::hitter": 80, "weak::hitter": 200}
        result = classify_roster(roster, rankings)
        assert result["Star"] == "core"

    def test_trade_candidate_high_rank_low_wsgp(self):
        """Top-ranked player with below-median wSGP is trade candidate."""
        roster = [
            _hitter("Misfit", 0.3),
            _hitter("Filler1", 1.5),
            _hitter("Filler2", 1.2),
        ]
        rankings = {"misfit::hitter": 25, "filler1::hitter": 60, "filler2::hitter": 70}
        result = classify_roster(roster, rankings)
        assert result["Misfit"] == "trade_candidate"

    def test_role_player_low_rank_high_wsgp(self):
        """Low-ranked player with above-median wSGP is role player."""
        roster = [
            _hitter("Niche", 1.8),
            _hitter("Star", 1.6),
            _hitter("Weak", 0.3),
        ]
        rankings = {"niche::hitter": 180, "star::hitter": 5, "weak::hitter": 250}
        result = classify_roster(roster, rankings)
        assert result["Niche"] == "role_player"

    def test_droppable_low_rank_low_wsgp(self):
        """Low-ranked player with below-median wSGP is droppable."""
        roster = [
            _hitter("Scrub", 0.2),
            _hitter("Star", 2.0),
            _hitter("Average", 1.0),
        ]
        rankings = {"scrub::hitter": 250, "star::hitter": 5, "average::hitter": 80}
        result = classify_roster(roster, rankings)
        assert result["Scrub"] == "droppable"

    def test_il_player_excluded_from_median(self):
        """IL players don't affect the median but still get classified."""
        roster = [
            _hitter("Active1", 2.0),
            _hitter("Active2", 1.0),
            _hitter("Active3", 0.5),
            _hitter("ILStar", 0.0, status="IL"),
        ]
        rankings = {
            "active1::hitter": 20, "active2::hitter": 50,
            "active3::hitter": 200, "ilstar::hitter": 15,
        }
        result = classify_roster(roster, rankings)
        assert result["ILStar"] == "trade_candidate"

    def test_unranked_player_treated_as_low_sgp(self):
        """Player missing from rankings is treated as rank > threshold."""
        roster = [
            _hitter("Unknown", 1.5),
            _hitter("Star", 2.0),
            _hitter("Weak", 0.3),
        ]
        rankings = {"star::hitter": 5, "weak::hitter": 200}
        result = classify_roster(roster, rankings)
        assert result["Unknown"] == "droppable"

    def test_pitcher_classification(self):
        """Pitchers use pitcher rank keys."""
        roster = [
            _pitcher("Ace", 1.8),
            _pitcher("Middle", 1.0),
            _pitcher("Scrub", 0.3),
        ]
        rankings = {"ace::pitcher": 5, "middle::pitcher": 90, "scrub::pitcher": 200}
        result = classify_roster(roster, rankings)
        assert result["Ace"] == "core"
        assert result["Middle"] == "trade_candidate"
        assert result["Scrub"] == "droppable"

    def test_custom_threshold(self):
        """Custom rosterable_threshold changes the SGP cutoff."""
        roster = [
            _hitter("Borderline", 0.5),
            _hitter("Star", 2.0),
            _hitter("Other", 1.0),
        ]
        rankings = {"borderline::hitter": 80, "star::hitter": 5, "other::hitter": 60}
        result_50 = classify_roster(roster, rankings, rosterable_threshold=50)
        result_130 = classify_roster(roster, rankings, rosterable_threshold=130)
        assert result_50["Borderline"] == "droppable"
        assert result_130["Borderline"] == "trade_candidate"
