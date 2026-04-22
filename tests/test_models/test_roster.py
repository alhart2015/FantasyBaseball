from datetime import date


class TestRosterEntry:
    def test_construction(self):
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.roster import RosterEntry

        entry = RosterEntry(
            name="Ivan Herrera",
            positions=[Position.C, Position.UTIL],
            selected_position=Position.C,
            status="",
            yahoo_id="12345",
        )
        assert entry.name == "Ivan Herrera"
        assert entry.positions == [Position.C, Position.UTIL]
        assert entry.selected_position is Position.C

    def test_default_status_and_yahoo_id(self):
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.roster import RosterEntry

        entry = RosterEntry(
            name="X",
            positions=[Position.OF],
            selected_position=Position.OF,
        )
        assert entry.status == ""
        assert entry.yahoo_id == ""


class TestRoster:
    def _make(self, *entries):
        from fantasy_baseball.models.roster import Roster

        return Roster(effective_date=date(2026, 4, 14), entries=list(entries))

    def _entry(self, name, slot, eligible=None):
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.roster import RosterEntry

        slot_p = Position.parse(slot) if isinstance(slot, str) else slot
        eligible_list = (
            [Position.parse(p) if isinstance(p, str) else p for p in eligible]
            if eligible is not None
            else [slot_p]
        )
        return RosterEntry(
            name=name,
            positions=eligible_list,
            selected_position=slot_p,
        )

    def test_empty_roster(self):
        roster = self._make()
        assert len(roster) == 0
        assert list(roster) == []
        assert roster.names() == set()

    def test_names(self):
        roster = self._make(
            self._entry("Ivan Herrera", "C"),
            self._entry("Juan Soto", "OF"),
        )
        assert roster.names() == {"Ivan Herrera", "Juan Soto"}

    def test_len_and_iter(self):
        roster = self._make(
            self._entry("A", "C"),
            self._entry("B", "OF"),
            self._entry("C", "OF"),
        )
        assert len(roster) == 3
        entries = list(roster)
        assert [e.name for e in entries] == ["A", "B", "C"]

    def test_by_slot_groups_by_selected_position(self):
        from fantasy_baseball.models.positions import Position

        roster = self._make(
            self._entry("A", "C"),
            self._entry("B", "OF"),
            self._entry("C", "OF"),
            self._entry("D", "BN"),
        )
        grouped = roster.by_slot()
        assert set(grouped.keys()) == {Position.C, Position.OF, Position.BN}
        assert [e.name for e in grouped[Position.OF]] == ["B", "C"]
        assert [e.name for e in grouped[Position.C]] == ["A"]

    def test_by_slot_returns_empty_dict_for_empty_roster(self):
        roster = self._make()
        assert roster.by_slot() == {}
