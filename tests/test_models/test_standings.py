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

    def test_getitem_compat_uppercase(self):
        """Dict-compat access using uppercase category names (for migration)."""
        from fantasy_baseball.models.standings import CategoryStats
        stats = CategoryStats(r=100, hr=40, era=3.5)
        assert stats["R"] == 100
        assert stats["HR"] == 40
        assert stats["ERA"] == pytest.approx(3.5)

    def test_get_with_default(self):
        from fantasy_baseball.models.standings import CategoryStats
        stats = CategoryStats(r=100)
        assert stats.get("R") == 100
        assert stats.get("UNKNOWN", 42) == 42

    def test_items_yields_all_categories(self):
        from fantasy_baseball.models.standings import CategoryStats
        stats = CategoryStats(r=100, hr=40, rbi=120, sb=15, avg=0.280,
                              w=50, k=700, sv=20, era=3.9, whip=1.20)
        d = dict(stats.items())
        assert d == {
            "R": 100, "HR": 40, "RBI": 120, "SB": 15, "AVG": pytest.approx(0.280),
            "W": 50, "K": 700, "SV": 20, "ERA": pytest.approx(3.9), "WHIP": pytest.approx(1.20),
        }

    def test_getitem_unknown_raises(self):
        from fantasy_baseball.models.standings import CategoryStats
        stats = CategoryStats()
        with pytest.raises(KeyError):
            _ = stats["UNKNOWN"]

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


class TestStandingsSnapshot:
    def test_empty_snapshot(self):
        from fantasy_baseball.models.standings import StandingsSnapshot
        snap = StandingsSnapshot(effective_date=date(2026, 4, 14), entries=[])
        assert snap.effective_date == date(2026, 4, 14)
        assert snap.entries == []

    def test_by_team_lookup(self):
        from fantasy_baseball.models.standings import (
            CategoryStats, StandingsEntry, StandingsSnapshot,
        )
        e1 = StandingsEntry("Hart of the Order", "k1", 1, CategoryStats(r=120))
        e2 = StandingsEntry("Rivals", "k2", 2, CategoryStats(r=100))
        snap = StandingsSnapshot(date(2026, 4, 14), [e1, e2])

        lookup = snap.by_team()
        assert lookup["Hart of the Order"] is e1
        assert lookup["Rivals"] is e2

    def test_by_team_duplicate_names_raises(self):
        from fantasy_baseball.models.standings import (
            CategoryStats, StandingsEntry, StandingsSnapshot,
        )
        e1 = StandingsEntry("Dupe", "k1", 1, CategoryStats())
        e2 = StandingsEntry("Dupe", "k2", 2, CategoryStats())
        snap = StandingsSnapshot(date(2026, 4, 14), [e1, e2])
        with pytest.raises(ValueError, match="duplicate team"):
            snap.by_team()
