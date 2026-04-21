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
            r=120, hr=45, rbi=130, sb=22, avg=0.275,
            w=60, k=800, sv=35, era=3.80, whip=1.15,
        )
        assert stats.r == 120
        assert stats.avg == pytest.approx(0.275)
        assert stats.era == pytest.approx(3.80)

    def test_from_dict(self):
        from fantasy_baseball.models.standings import CategoryStats
        stats = CategoryStats.from_dict({
            "R": 120, "HR": 40, "RBI": 110, "SB": 8, "AVG": 0.272,
            "W": 55, "K": 750, "SV": 30, "ERA": 3.85, "WHIP": 1.18,
        })
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

        stats = CategoryStats(r=100, hr=40, rbi=120, sb=15, avg=0.280,
                              w=50, k=700, sv=20, era=3.9, whip=1.20)
        items = list(stats.items())
        assert [k for k, _ in items] == ALL_CATEGORIES
        as_map = dict(items)
        assert as_map[Category.R] == 100
        assert as_map[Category.HR] == 40
        assert as_map[Category.WHIP] == pytest.approx(1.20)


from datetime import date


class TestStandingsEntry:
    def test_construction(self):
        from fantasy_baseball.models.standings import (
            CategoryStats, StandingsEntry,
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
                        "R": 45, "HR": 12, "RBI": 40, "SB": 8, "AVG": 0.268,
                        "W": 3, "K": 85, "SV": 4, "ERA": 3.21, "WHIP": 1.14,
                    },
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
        assert s.to_json() == payload

    def test_from_json_rejects_legacy_shape(self):
        from fantasy_baseball.models.standings import Standings
        legacy = {
            "teams": [
                {
                    "team": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "r": 45, "hr": 12, "rbi": 40, "sb": 8, "avg": 0.268,
                    "w": 3, "k": 85, "sv": 4, "era": 3.21, "whip": 1.14,
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
