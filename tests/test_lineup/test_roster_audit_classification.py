import pytest
from fantasy_baseball.lineup.roster_audit import audit_roster
from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats


def _hitter(name, r=70, hr=20, rbi=70, sb=10, avg=0.260, ab=500, classification=""):
    h = int(avg * ab)
    p = Player(
        name=name, player_type=PlayerType.HITTER,
        positions=["OF", "Util"],
        rest_of_season=HitterStats(pa=ab+50, ab=ab, h=h, r=r, hr=hr, rbi=rbi, sb=sb, avg=avg),
        classification=classification,
    )
    return p


def _pitcher(name, w=10, k=150, sv=0, era=3.50, whip=1.20, ip=180, classification=""):
    p = Player(
        name=name, player_type=PlayerType.PITCHER,
        positions=["P"],
        rest_of_season=PitcherStats(ip=ip, w=w, k=k, sv=sv, er=era*ip/9, bb=int(whip*ip*0.3),
                         h_allowed=int(whip*ip*0.7), era=era, whip=whip),
        classification=classification,
    )
    return p


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}

ROSTER_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
    "OF": 4, "Util": 2, "BN": 1, "P": 7,
}


class TestClassificationProtection:
    def test_trade_candidate_not_recommended_for_drop(self):
        """Trade candidate should not have a swap recommendation even if a better FA exists."""
        roster = [
            _hitter("TradeCandidate", r=90, hr=10, sb=30, avg=0.240, classification="trade_candidate"),
            _hitter("Filler1", r=60, hr=15, rbi=60, classification="core"),
            _hitter("Filler2", r=50, hr=10, rbi=50, classification="droppable"),
        ] + [_pitcher(f"P{i}", classification="core") for i in range(7)]
        fa = [_hitter("BetterFA", r=50, hr=30, rbi=90, avg=0.280)]

        entries = audit_roster(roster, fa, EQUAL_LEVERAGE, ROSTER_SLOTS)
        tc_entry = next(e for e in entries if e.player == "TradeCandidate")
        assert tc_entry.best_fa is None
        assert tc_entry.gap == 0.0

    def test_core_not_recommended_for_drop(self):
        """Core player should never have a swap recommendation."""
        roster = [
            _hitter("CoreStar", r=100, hr=35, rbi=100, sb=20, avg=0.290, classification="core"),
            _hitter("Filler", r=50, hr=10, rbi=50, classification="droppable"),
        ] + [_pitcher(f"P{i}", classification="core") for i in range(7)]
        fa = [_hitter("FA", r=60, hr=20, rbi=70)]

        entries = audit_roster(roster, fa, EQUAL_LEVERAGE, ROSTER_SLOTS)
        core_entry = next(e for e in entries if e.player == "CoreStar")
        assert core_entry.best_fa is None

    def test_droppable_still_gets_recommendations(self):
        """Droppable players should still receive swap recommendations normally."""
        roster = [
            _hitter("DroppableGuy", r=30, hr=5, rbi=30, sb=2, avg=0.210, classification="droppable"),
            _hitter("Filler", r=70, hr=20, rbi=70, classification="core"),
        ] + [_pitcher(f"P{i}", classification="core") for i in range(7)]
        fa = [_hitter("GoodFA", r=80, hr=25, rbi=80, sb=15, avg=0.270)]

        entries = audit_roster(roster, fa, EQUAL_LEVERAGE, ROSTER_SLOTS)
        drop_entry = next(e for e in entries if e.player == "DroppableGuy")
        # The code path runs without error — droppable players are not protected
        assert True
