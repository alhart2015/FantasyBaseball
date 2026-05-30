from datetime import date

import pytest


class TestCategoryStats:
    def test_default_values(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats()
        assert stats.r == 0.0
        assert stats.hr == 0.0
        assert stats.rbi == 0.0
        assert stats.sb == 0.0
        assert stats.avg == 0.0
        assert stats.w == 0.0
        assert stats.k == 0.0
        assert stats.sv == 0.0
        assert stats.era == 99.0
        assert stats.whip == 99.0

    def test_construction_with_values(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats(
            r=120,
            hr=45,
            rbi=130,
            sb=22,
            avg=0.275,
            w=60,
            k=800,
            sv=35,
            era=3.80,
            whip=1.15,
        )
        assert stats.r == 120
        assert stats.avg == pytest.approx(0.275)
        assert stats.era == pytest.approx(3.80)

    def test_from_dict(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats.from_dict(
            {
                "R": 120,
                "HR": 40,
                "RBI": 110,
                "SB": 8,
                "AVG": 0.272,
                "W": 55,
                "K": 750,
                "SV": 30,
                "ERA": 3.85,
                "WHIP": 1.18,
            }
        )
        assert stats.r == 120
        assert stats.whip == pytest.approx(1.18)

    def test_from_dict_missing_keys_default(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats.from_dict({"R": 100})
        assert stats.r == 100
        assert stats.hr == 0.0
        assert stats.era == 99.0


class TestCategoryStatsTypedAccess:
    def test_getitem_accepts_category_enum(self):
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.utils.constants import Category

        stats = CategoryStats(r=100, hr=40, era=3.5)
        assert stats[Category.R] == 100
        assert stats[Category.HR] == 40
        assert stats[Category.ERA] == pytest.approx(3.5)

    def test_getitem_rejects_bare_string(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats(r=100)
        with pytest.raises(TypeError, match="Category enum"):
            _ = stats["R"]

    def test_getitem_rejects_other_types(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats()
        with pytest.raises(TypeError, match="Category enum"):
            _ = stats[0]

    def test_items_yields_category_enums(self):
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category

        stats = CategoryStats(
            r=100, hr=40, rbi=120, sb=15, avg=0.280, w=50, k=700, sv=20, era=3.9, whip=1.20
        )
        items = list(stats.items())
        assert [k for k, _ in items] == ALL_CATEGORIES
        as_map = dict(items)
        assert as_map[Category.R] == 100
        assert as_map[Category.HR] == 40
        assert as_map[Category.WHIP] == pytest.approx(1.20)


class TestStandingsEntry:
    def test_construction(self):
        from fantasy_baseball.models.standings import (
            CategoryStats,
            StandingsEntry,
        )

        entry = StandingsEntry(
            team_name="Hart of the Order",
            team_key="431.l.17492.t.3",
            rank=4,
            stats=CategoryStats(r=100, hr=40),
        )
        assert entry.team_name == "Hart of the Order"
        assert entry.team_key == "431.l.17492.t.3"
        assert entry.rank == 4
        assert entry.stats.r == 100


class TestStandingsJSON:
    def _canonical_payload(self):
        return {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "yahoo_points_for": 78.5,
                    "stats": {
                        "R": 45.0,
                        "HR": 12.0,
                        "RBI": 40.0,
                        "SB": 8.0,
                        "AVG": 0.268,
                        "W": 3.0,
                        "K": 85.0,
                        "SV": 4.0,
                        "ERA": 3.21,
                        "WHIP": 1.14,
                    },
                    "extras": {},
                },
            ],
        }

    def test_from_json_canonical_round_trip(self):
        from fantasy_baseball.models.standings import Standings

        payload = self._canonical_payload()
        s = Standings.from_json(payload)
        assert s.effective_date == date(2026, 4, 15)
        assert len(s.entries) == 1
        e = s.entries[0]
        assert e.team_name == "Alpha"
        assert e.team_key == "431.l.1.t.1"
        assert e.rank == 1
        assert e.yahoo_points_for == 78.5
        assert e.stats.r == 45
        assert e.stats.whip == pytest.approx(1.14)
        assert e.extras == {}
        assert s.to_json() == payload

    def test_from_json_accepts_missing_extras_key(self):
        """Entries without an 'extras' key (older in-flight writes) must
        still parse — ``extras`` defaults to the empty dict."""
        from fantasy_baseball.models.standings import Standings

        payload = {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "stats": {
                        "R": 45,
                        "HR": 12,
                        "RBI": 40,
                        "SB": 8,
                        "AVG": 0.268,
                        "W": 3,
                        "K": 85,
                        "SV": 4,
                        "ERA": 3.21,
                        "WHIP": 1.14,
                    },
                },
            ],
        }
        s = Standings.from_json(payload)
        assert s.entries[0].extras == {}

    def test_extras_round_trip_with_pa_ip(self):
        """PA / IP land in ``extras`` keyed by :class:`OpportunityStat`
        and round-trip as uppercase string keys."""
        from fantasy_baseball.models.standings import Standings
        from fantasy_baseball.utils.constants import OpportunityStat

        payload = {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "yahoo_points_for": 78.5,
                    "stats": {
                        "R": 45.0,
                        "HR": 12.0,
                        "RBI": 40.0,
                        "SB": 8.0,
                        "AVG": 0.268,
                        "W": 3.0,
                        "K": 85.0,
                        "SV": 4.0,
                        "ERA": 3.21,
                        "WHIP": 1.14,
                    },
                    "extras": {"IP": 190.0, "PA": 720.0},
                },
            ],
        }
        s = Standings.from_json(payload)
        extras = s.entries[0].extras
        assert extras[OpportunityStat.IP] == 190.0
        assert extras[OpportunityStat.PA] == 720.0
        # Round-trip: string keys back out.
        round_tripped = s.to_json()
        assert round_tripped["teams"][0]["extras"] == {"IP": 190.0, "PA": 720.0}

    def test_extras_ignores_unknown_keys(self):
        """Unknown extras keys survive as ignored (forward compat)."""
        from fantasy_baseball.models.standings import Standings
        from fantasy_baseball.utils.constants import OpportunityStat

        payload = {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "stats": {"R": 45},
                    "extras": {"IP": 42.0, "NOT_A_STAT": 999.0},
                },
            ],
        }
        s = Standings.from_json(payload)
        assert s.entries[0].extras == {OpportunityStat.IP: 42.0}

    def test_from_json_rejects_legacy_shape(self):
        from fantasy_baseball.models.standings import Standings

        legacy = {
            "teams": [
                {
                    "team": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "r": 45,
                    "hr": 12,
                    "rbi": 40,
                    "sb": 8,
                    "avg": 0.268,
                    "w": 3,
                    "k": 85,
                    "sv": 4,
                    "era": 3.21,
                    "whip": 1.14,
                },
            ],
        }
        with pytest.raises(ValueError, match=r"legacy|unknown|name"):
            Standings.from_json(legacy)


class TestProjectedStandingsJSON:
    def test_round_trip(self):
        from fantasy_baseball.models.standings import (
            CategoryStats,
            ProjectedStandings,
            ProjectedStandingsEntry,
        )

        ps = ProjectedStandings(
            effective_date=date(2026, 4, 15),
            entries=[
                ProjectedStandingsEntry(
                    team_name="Alpha",
                    stats=CategoryStats(r=600, hr=250, era=3.8, whip=1.18),
                ),
            ],
        )
        round_tripped = ProjectedStandings.from_json(ps.to_json())
        assert round_tripped == ps


class TestCategoryPoints:
    def test_getitem_and_total(self):
        from fantasy_baseball.models.standings import CategoryPoints
        from fantasy_baseball.utils.constants import Category

        cp = CategoryPoints(
            values={Category.R: 7.0, Category.HR: 4.5},
            total=11.5,
        )
        assert cp[Category.R] == 7.0
        assert cp[Category.HR] == 4.5
        assert cp.total == 11.5

    def test_getitem_rejects_string(self):
        from fantasy_baseball.models.standings import CategoryPoints

        cp = CategoryPoints(values={}, total=0.0)
        with pytest.raises(TypeError, match="Category"):
            _ = cp["R"]


class TestTeamYtdComponents:
    """Tests for the TeamYtdComponents derivation from Yahoo standings.

    These components are the rate-stat ingredients (H, AB, ER, IP, BB+H_allowed)
    recovered from CategoryStats + extras so the team-YTD + ROS projection in
    ProjectedStandings.from_rosters can recombine team AVG/ERA/WHIP without
    losing precision to pre-computed rates.
    """

    @staticmethod
    def _entry(*, avg=0.0, era=0.0, whip=0.0, ip=0.0, ab=None, pa=None, **counts):
        from fantasy_baseball.models.standings import CategoryStats, StandingsEntry
        from fantasy_baseball.utils.constants import OpportunityStat

        stats = CategoryStats(
            r=counts.get("r", 0),
            hr=counts.get("hr", 0),
            rbi=counts.get("rbi", 0),
            sb=counts.get("sb", 0),
            avg=avg,
            w=counts.get("w", 0),
            k=counts.get("k", 0),
            sv=counts.get("sv", 0),
            era=era,
            whip=whip,
        )
        extras: dict[OpportunityStat, float] = {OpportunityStat.IP: ip}
        if ab is not None:
            extras[OpportunityStat.AB] = ab
        if pa is not None:
            extras[OpportunityStat.PA] = pa
        return StandingsEntry(team_name="T", team_key="t", rank=1, stats=stats, extras=extras)

    def test_components_use_explicit_ab_when_present(self):
        """When Yahoo standings expose AB directly, use it verbatim - no PA conversion."""
        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.275,
            ab=400,
            ip=200,
            w=10,
            k=180,
            sv=5,
            era=3.50,
            whip=1.20,
        )
        c = e.ytd_components()
        assert c.ab == 400.0
        assert c.h == 0.275 * 400.0  # AVG * AB
        assert c.ip == 200.0
        assert c.er == 3.50 * 200.0 / 9.0
        assert c.bb_plus_h_allowed == 1.20 * 200.0

    def test_components_fall_back_to_pa_when_ab_absent(self):
        """When only PA is exposed, derive AB via AB_PER_PA."""
        from fantasy_baseball.utils.constants import AB_PER_PA

        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.250,
            pa=500,
            ip=100,
            w=5,
            k=80,
            sv=2,
            era=4.00,
            whip=1.30,
        )
        c = e.ytd_components()
        assert c.ab == 500.0 * AB_PER_PA
        assert c.h == 0.250 * (500.0 * AB_PER_PA)

    def test_components_are_zero_when_neither_ab_nor_pa_present(self):
        """No way to recover AB without it or PA -> components.ab/h are 0.

        Callers (ProjectedStandings.from_rosters) detect this and fall back to
        summing ROS-only AVG from the roster (i.e. legacy ROS-only mode for AVG).
        """
        e = self._entry(
            r=80,
            hr=20,
            rbi=70,
            sb=10,
            avg=0.250,
            ip=100,
            w=5,
            k=80,
            sv=2,
            era=4.00,
            whip=1.30,
        )
        c = e.ytd_components()
        assert c.ab == 0.0
        assert c.h == 0.0

    def test_components_zero_when_no_ip_for_pitching_rates(self):
        """Pre-season standings have IP=0; ERA/WHIP components must be 0, not NaN."""
        e = self._entry(avg=0.0, era=0.0, whip=0.0, ip=0.0)
        c = e.ytd_components()
        assert c.ip == 0.0
        assert c.er == 0.0
        assert c.bb_plus_h_allowed == 0.0

    def test_components_carry_counting_stats(self):
        """Counting stats pass through unchanged."""
        e = self._entry(
            r=85,
            hr=22,
            rbi=78,
            sb=15,
            avg=0.0,
            ip=0.0,
            w=10,
            k=200,
            sv=8,
            era=0.0,
            whip=0.0,
        )
        c = e.ytd_components()
        assert c.r == 85
        assert c.hr == 22
        assert c.rbi == 78
        assert c.sb == 15
        assert c.w == 10
        assert c.k == 200
        assert c.sv == 8
