import pytest
from fantasy_baseball.lineup.waivers import scan_waivers
from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats


def _hitter(name, r=70, hr=20, rbi=70, sb=10, avg=0.260, ab=500, classification=""):
    h = int(avg * ab)
    return Player(
        name=name, player_type=PlayerType.HITTER,
        positions=["OF", "Util"],
        ros=HitterStats(pa=ab+50, ab=ab, h=h, r=r, hr=hr, rbi=rbi, sb=sb, avg=avg),
        classification=classification,
    )


def _pitcher(name, w=10, k=150, sv=0, era=3.50, whip=1.20, ip=180, classification=""):
    return Player(
        name=name, player_type=PlayerType.PITCHER,
        positions=["P"],
        ros=PitcherStats(ip=ip, w=w, k=k, sv=sv, er=era*ip/9, bb=int(whip*ip*0.3),
                         h_allowed=int(whip*ip*0.7), era=era, whip=whip),
        classification=classification,
    )


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}

ROSTER_SLOTS = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
    "OF": 4, "Util": 2, "BN": 1, "P": 7,
}


class TestWaiverClassificationProtection:
    def test_trade_candidate_never_dropped(self):
        """Waiver recs should never suggest dropping a trade candidate."""
        roster = [
            _hitter("TradeCandidate", r=90, hr=10, sb=30, avg=0.240, classification="trade_candidate"),
            _hitter("Droppable", r=30, hr=5, rbi=30, avg=0.210, classification="droppable"),
        ] + [_pitcher(f"P{i}", classification="core") for i in range(7)]
        fa = [_hitter("GoodFA", r=80, hr=25, rbi=80, sb=15, avg=0.270)]

        recs = scan_waivers(roster, fa, EQUAL_LEVERAGE, roster_slots=ROSTER_SLOTS)
        for rec in recs:
            assert rec["drop"] != "TradeCandidate"

    def test_core_never_dropped(self):
        """Waiver recs should never suggest dropping a core player."""
        roster = [
            _hitter("CoreStar", r=100, hr=35, rbi=100, avg=0.290, classification="core"),
            _hitter("Droppable", r=30, hr=5, rbi=30, avg=0.210, classification="droppable"),
        ] + [_pitcher(f"P{i}", classification="core") for i in range(7)]
        fa = [_hitter("GoodFA", r=80, hr=25, rbi=80, avg=0.270)]

        recs = scan_waivers(roster, fa, EQUAL_LEVERAGE, roster_slots=ROSTER_SLOTS)
        for rec in recs:
            assert rec["drop"] != "CoreStar"
