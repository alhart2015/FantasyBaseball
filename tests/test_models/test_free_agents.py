from datetime import date


class TestFreeAgentPoolBasics:
    def test_construction(self):
        from fantasy_baseball.models.free_agents import FreeAgentPool
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.roster import RosterEntry

        entries = [
            RosterEntry(
                name="Available Guy",
                positions=[Position.OF, Position.UTIL],
                selected_position=Position.OF,
            ),
        ]
        pool = FreeAgentPool(effective_date=date(2026, 4, 14), entries=entries)
        assert pool.effective_date == date(2026, 4, 14)
        assert len(pool.entries) == 1
        assert pool.entries[0].name == "Available Guy"

    def test_empty_pool(self):
        from fantasy_baseball.models.free_agents import FreeAgentPool

        pool = FreeAgentPool(effective_date=date(2026, 4, 14), entries=[])
        assert list(pool) == []
        assert len(pool) == 0

    def test_iter_and_len(self):
        from fantasy_baseball.models.free_agents import FreeAgentPool
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.roster import RosterEntry

        entries = [
            RosterEntry(name=f"P{i}", positions=[Position.OF], selected_position=Position.OF)
            for i in range(3)
        ]
        pool = FreeAgentPool(date(2026, 4, 14), entries)
        assert len(pool) == 3
        assert [e.name for e in pool] == ["P0", "P1", "P2"]

    def test_names(self):
        from fantasy_baseball.models.free_agents import FreeAgentPool
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.models.roster import RosterEntry

        entries = [
            RosterEntry(name="Alpha", positions=[Position.C], selected_position=Position.C),
            RosterEntry(name="Beta", positions=[Position.OF], selected_position=Position.OF),
        ]
        pool = FreeAgentPool(date(2026, 4, 14), entries)
        assert pool.names() == {"Alpha", "Beta"}


class TestFreeAgentPoolFromYahooParsed:
    """Tests the static parser that converts raw Yahoo output to entries.

    Kept separate from ``from_yahoo`` (which hits the network) so we can
    exercise the conversion logic without mocking the Yahoo client.
    """

    def test_parse_normalizes_positions(self):
        from fantasy_baseball.models.free_agents import FreeAgentPool
        from fantasy_baseball.models.positions import Position

        raw = [
            {"name": "Util Guy", "positions": ["OF", "Util"], "player_id": "42", "status": ""},
            {"name": "Starter", "positions": ["SP"], "player_id": "43", "status": ""},
        ]
        entries = FreeAgentPool._parse_yahoo_entries(raw)
        assert len(entries) == 2
        assert entries[0].name == "Util Guy"
        assert entries[0].positions == [Position.OF, Position.UTIL]
        # Free agents don't have a slot assignment — default to BN
        assert entries[0].selected_position is Position.BN
        assert entries[0].yahoo_id == "42"

    def test_parse_skips_unknown_positions(self):
        """Unknown positions in FA data are dropped rather than raising.

        Yahoo occasionally returns weird tokens for free agents (NA,
        blank). We tolerate these rather than crash the pool load —
        the stakes are lower than a roster snapshot because the pool
        is transient.
        """
        from fantasy_baseball.models.free_agents import FreeAgentPool

        raw = [
            {"name": "Guy A", "positions": ["OF", "NA"], "player_id": "1"},
            {"name": "Guy B", "positions": ["XYZ"], "player_id": "2"},
        ]
        entries = FreeAgentPool._parse_yahoo_entries(raw)
        # Guy A keeps OF (dropped NA), Guy B has zero eligible positions
        # and is skipped entirely
        assert len(entries) == 1
        assert entries[0].name == "Guy A"

    def test_parse_skips_entry_with_no_positions(self):
        from fantasy_baseball.models.free_agents import FreeAgentPool

        raw = [{"name": "No Positions", "positions": [], "player_id": "1"}]
        entries = FreeAgentPool._parse_yahoo_entries(raw)
        assert entries == []
