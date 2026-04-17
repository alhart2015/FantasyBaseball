from itertools import combinations
import pytest
from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats
from fantasy_baseball.models.positions import Position
from fantasy_baseball.lineup.optimizer import (
    HitterAssignment, optimize_hitter_lineup_roto,
)


CATEGORIES = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]


def _hitter(name, positions, r=70, hr=20, rbi=70, sb=10, h=120, ab=450, pa=500):
    avg = (h / ab) if ab else 0
    return Player(
        name=name, player_type=PlayerType.HITTER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=HitterStats(pa=pa, ab=ab, h=h, r=r, hr=hr, rbi=rbi, sb=sb, avg=avg),
        selected_position=Position.parse(positions[0]),
    )


def _pitcher(name, positions, ip=180, w=12, k=180, sv=0, era=3.50, whip=1.20):
    return Player(
        name=name, player_type=PlayerType.PITCHER,
        positions=[Position.parse(p) for p in positions],
        rest_of_season=PitcherStats(
            ip=ip, w=w, k=k, sv=sv, era=era, whip=whip,
            er=era * ip / 9, bb=int(whip * ip * 0.3), h_allowed=int(whip * ip * 0.7),
        ),
        selected_position=Position.parse(positions[0]),
    )


def _zero_stats():
    return {c: 0.0 for c in CATEGORIES}


def _standing(name: str, **overrides) -> dict:
    stats = _zero_stats()
    stats.update(overrides)
    return {"name": name, "team_key": "", "rank": 0, "stats": stats}


SMALL_ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "UTIL": 1, "BN": 2, "P": 9, "IL": 0}


class TestBasic:
    def test_fills_all_required_slots(self):
        hitters = [
            _hitter("C1", ["C"]),
            _hitter("1B1", ["1B"]),
            _hitter("2B1", ["2B"]),
            _hitter("3B1", ["3B"]),
            _hitter("SS1", ["SS"]),
            _hitter("OF1", ["OF"]),
            _hitter("OF2", ["OF"]),
            _hitter("OF3", ["OF"]),
            _hitter("OF4", ["OF"]),
            _hitter("UTIL1", ["1B", "OF"]),
        ]
        standings = [
            _standing("Us", R=0, HR=0, RBI=0, SB=0, AVG=0),
            _standing("Rival", R=1, HR=1, RBI=1, SB=1, AVG=0),
        ]
        lineup = optimize_hitter_lineup_roto(
            hitters=hitters, full_roster=hitters,
            projected_standings=standings, team_name="Us",
            roster_slots=SMALL_ROSTER_SLOTS,
        )
        assert isinstance(lineup, list)
        assert len(lineup) == 9  # C, 1B, 2B, 3B, SS, OF, OF, OF, UTIL
        slot_counts: dict = {}
        for a in lineup:
            assert isinstance(a, HitterAssignment)
            slot_counts[a.slot] = slot_counts.get(a.slot, 0) + 1
        assert slot_counts[Position.C] == 1
        assert slot_counts[Position.OF] == 3
        assert slot_counts[Position.UTIL] == 1


class TestERotoMaximization:
    def test_picks_hitter_who_lifts_category_boundary(self):
        """Scenario: 2 teams + us. We're tied with Rival on SB.
        Hitter A has more HR but we already dominate HR.
        Hitter B has more SB and starting him flips the SB tiebreak.
        Lineup must start Hitter B even though A has higher overall stats.
        """
        a = _hitter("A", ["OF"], r=80, hr=40, rbi=90, sb=5, h=120, ab=450)
        b = _hitter("B", ["OF"], r=70, hr=20, rbi=70, sb=25, h=120, ab=450)
        c = _hitter("C", ["OF"], r=60, hr=18, rbi=60, sb=8, h=120, ab=450)

        slots = {"OF": 1, "BN": 2, "P": 9, "IL": 0}
        full = [a, b, c]

        standings = [
            _standing("Us", R=0, HR=100, RBI=0, SB=0, AVG=0),   # overwritten by loop
            _standing("Rival", R=0, HR=30, RBI=0, SB=20, AVG=0),
            _standing("Third", R=0, HR=20, RBI=0, SB=15, AVG=0),
        ]
        lineup = optimize_hitter_lineup_roto(
            hitters=full, full_roster=full,
            projected_standings=standings, team_name="Us",
            roster_slots=slots,
        )
        assert len(lineup) == 1
        assert lineup[0].name == "B"

    def test_roto_delta_non_negative_for_every_starter(self):
        hitters = [
            _hitter("A", ["OF"], r=80, hr=25),
            _hitter("B", ["OF"], r=70, hr=20),
            _hitter("C", ["OF"], r=60, hr=15),
        ]
        slots = {"OF": 2, "BN": 1, "P": 9, "IL": 0}
        standings = [
            _standing("Us", R=0, HR=0),
            _standing("Rival", R=1, HR=1),
        ]
        lineup = optimize_hitter_lineup_roto(
            hitters=hitters, full_roster=hitters,
            projected_standings=standings, team_name="Us",
            roster_slots=slots,
        )
        for a in lineup:
            assert a.roto_delta >= 0

    def test_feasibility_drives_selection_when_top_subset_infeasible(self):
        """3 catchers, 1 C slot + 1 OF slot: top-two by stats might both be
        catchers, but only 1 can fill C. The optimizer must pick a feasible
        subset (1 C + 1 OF-eligible), not an infeasible one."""
        c1 = _hitter("C1", ["C"], r=90, hr=30, rbi=90)
        c2 = _hitter("C2", ["C"], r=85, hr=28, rbi=85)
        o1 = _hitter("O1", ["OF"], r=60, hr=15, rbi=50)
        hitters = [c1, c2, o1]
        slots = {"C": 1, "OF": 1, "BN": 1, "P": 9, "IL": 0}
        standings = [_standing("Us"), _standing("Rival", R=1, HR=1, RBI=1)]
        lineup = optimize_hitter_lineup_roto(
            hitters=hitters, full_roster=hitters,
            projected_standings=standings, team_name="Us",
            roster_slots=slots,
        )
        slots_assigned = {a.slot for a in lineup}
        assert Position.C in slots_assigned
        assert Position.OF in slots_assigned
        names = {a.name for a in lineup}
        assert "O1" in names  # only OF-eligible, must start

    def test_il_hitters_excluded_from_optimization(self):
        il = _hitter("IL_Guy", ["OF"], r=100, hr=40, rbi=120)
        il.selected_position = Position.IL
        active = [
            _hitter("A", ["OF"], r=70, hr=20),
            _hitter("B", ["OF"], r=60, hr=15),
        ]
        slots = {"OF": 1, "BN": 1, "P": 9, "IL": 0}
        standings = [_standing("Us"), _standing("Rival", R=1, HR=1)]
        lineup = optimize_hitter_lineup_roto(
            hitters=active, full_roster=active + [il],
            projected_standings=standings, team_name="Us",
            roster_slots=slots,
        )
        names = {a.name for a in lineup}
        assert "IL_Guy" not in names
