from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats
from fantasy_baseball.lineup.roster_audit import audit_roster


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}

ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "UTIL": 1, "P": 3, "BN": 2, "IL": 0}


def _hitter(name, positions, **stats):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=positions,
        ros=HitterStats(
            pa=int(stats.get("ab", 500) * 1.15),
            ab=stats.get("ab", 500), h=stats.get("h", 130),
            r=stats.get("r", 70), hr=stats.get("hr", 20),
            rbi=stats.get("rbi", 70), sb=stats.get("sb", 5),
            avg=stats.get("avg", 0.260),
        ),
    )


def _pitcher(name, positions, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=positions,
        ros=PitcherStats(
            ip=stats.get("ip", 60.0), w=stats.get("w", 3.0),
            k=stats.get("k", 60.0), sv=stats.get("sv", 0.0),
            er=stats.get("er", 20.0), bb=stats.get("bb", 20.0),
            h_allowed=stats.get("h_allowed", 50.0),
            era=stats.get("era", 3.00), whip=stats.get("whip", 1.17),
        ),
    )


class TestAuditRoster:
    def test_identifies_upgrade_available(self):
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher("Decent SP", ["SP"], ip=180, w=12, k=180, era=3.50, whip=1.20,
                     er=70, bb=40, h_allowed=176),
            _pitcher("Decent SP2", ["SP"], ip=170, w=10, k=160, era=3.60, whip=1.22,
                     er=68, bb=40, h_allowed=167),
            _pitcher("Decent RP", ["RP"], ip=60, w=3, k=60, era=3.00, whip=1.17,
                     sv=20, er=20, bb=20, h_allowed=50),
        ]
        free_agents = [
            _hitter("Better OF", ["OF"], r=80, hr=28, rbi=85, sb=12, avg=0.280, ab=550, h=154),
        ]
        results = audit_roster(roster, free_agents, EQUAL_LEVERAGE, ROSTER_SLOTS)

        # Should have an entry for every roster player
        assert len(results) == len(roster)

        # The weak OF should have an upgrade identified
        weak_entry = next(e for e in results if e["player"] == "Weak OF")
        assert weak_entry["best_fa"] == "Better OF"
        assert weak_entry["gap"] > 0

    def test_shows_no_better_option(self):
        roster = [
            _hitter("Star OF", ["OF"], r=100, hr=40, rbi=110, sb=20, avg=0.300, ab=550, h=165),
        ]
        free_agents = [
            _hitter("Scrub", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
        ]
        results = audit_roster(roster, free_agents, EQUAL_LEVERAGE,
                               {"OF": 1, "P": 0, "BN": 0, "IL": 0})
        assert len(results) == 1
        assert results[0]["best_fa"] is None
        assert results[0]["gap"] == 0.0

    def test_sorted_by_gap_descending(self):
        roster = [
            _hitter("OK 1B", ["1B"], r=60, hr=15, rbi=55, sb=3, avg=0.255, ab=480, h=122),
            _hitter("Bad OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
        ]
        free_agents = [
            _hitter("Good 1B", ["1B"], r=75, hr=22, rbi=70, sb=5, avg=0.270, ab=520, h=140),
            _hitter("Great OF", ["OF"], r=90, hr=30, rbi=85, sb=10, avg=0.285, ab=550, h=157),
        ]
        results = audit_roster(roster, free_agents, EQUAL_LEVERAGE,
                               {"1B": 1, "OF": 1, "P": 0, "BN": 0, "IL": 0})
        gaps = [e["gap"] for e in results]
        assert gaps == sorted(gaps, reverse=True)

    def test_empty_free_agents_all_no_upgrade(self):
        roster = [
            _hitter("Solo", ["OF"], r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, h=135),
        ]
        results = audit_roster(roster, [], EQUAL_LEVERAGE,
                               {"OF": 1, "P": 0, "BN": 0, "IL": 0})
        assert len(results) == 1
        assert results[0]["best_fa"] is None
        assert results[0]["gap"] == 0.0

    def test_cross_type_swap_pitcher_slot(self):
        """A starter could replace a weak reliever if it produces more team wSGP."""
        roster = [
            _hitter("Hitter", ["OF"], r=80, hr=25, rbi=80, sb=10, avg=0.275, ab=540, h=149),
            _pitcher("Bad RP", ["RP"], ip=30, w=1, k=20, sv=2, era=5.50, whip=1.60,
                     er=18, bb=15, h_allowed=33),
            _pitcher("OK SP", ["SP"], ip=150, w=9, k=140, era=3.80, whip=1.25,
                     er=63, bb=40, h_allowed=148),
        ]
        free_agents = [
            _pitcher("Good SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                     er=64, bb=30, h_allowed=168),
        ]
        results = audit_roster(roster, free_agents, EQUAL_LEVERAGE,
                               {"OF": 1, "P": 2, "BN": 1, "IL": 0})
        # The bad RP should have the Good SP as best_fa (cross-type upgrade)
        bad_rp_entry = next(e for e in results if e["player"] == "Bad RP")
        assert bad_rp_entry["best_fa"] == "Good SP"
        assert bad_rp_entry["gap"] > 0
