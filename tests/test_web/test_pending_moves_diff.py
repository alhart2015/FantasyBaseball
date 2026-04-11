"""Tests for _compute_pending_moves_diff helper."""


class TestPendingMovesDiff:
    def _roster(self, *players):
        """Build a minimal roster list matching parse_roster's output."""
        return [
            {
                "name": name,
                "positions": positions,
                "selected_position": slot,
                "player_id": "",
                "status": "",
            }
            for name, positions, slot in players
        ]

    def test_no_diff_returns_empty_list(self):
        from fantasy_baseball.web.season_data import _compute_pending_moves_diff

        roster = self._roster(
            ("Ivan Herrera", ["C", "Util"], "C"),
            ("Juan Soto", ["OF", "Util"], "OF"),
        )
        result = _compute_pending_moves_diff(roster, roster, "Hart of the Order", "k-hart")
        assert result == []

    def test_add_only(self):
        from fantasy_baseball.web.season_data import _compute_pending_moves_diff

        today = self._roster(
            ("Ivan Herrera", ["C", "Util"], "C"),
        )
        future = self._roster(
            ("Ivan Herrera", ["C", "Util"], "C"),
            ("Marcus Semien", ["2B", "Util"], "2B"),
        )

        result = _compute_pending_moves_diff(today, future, "Hart", "k-hart")

        assert len(result) == 1
        move = result[0]
        assert move["team"] == "Hart"
        assert move["team_key"] == "k-hart"
        assert move["adds"] == [{"name": "Marcus Semien", "positions": ["2B", "Util"]}]
        assert move["drops"] == []

    def test_drop_only(self):
        from fantasy_baseball.web.season_data import _compute_pending_moves_diff

        today = self._roster(
            ("Ivan Herrera", ["C", "Util"], "C"),
            ("Otto Lopez", ["2B", "SS"], "BN"),
        )
        future = self._roster(
            ("Ivan Herrera", ["C", "Util"], "C"),
        )

        result = _compute_pending_moves_diff(today, future, "Hart", "k-hart")

        assert len(result) == 1
        move = result[0]
        assert move["adds"] == []
        assert move["drops"] == [{"name": "Otto Lopez", "positions": ["2B", "SS"]}]

    def test_add_and_drop_bundled_in_one_move(self):
        """A single move entry carries all adds + all drops.

        The UI template iterates move.adds and move.drops separately,
        so packing everything into one move keeps the banner compact
        (matches the existing multi-add/drop rendering).
        """
        from fantasy_baseball.web.season_data import _compute_pending_moves_diff

        today = self._roster(
            ("Ivan Herrera", ["C", "Util"], "C"),
            ("Otto Lopez", ["2B", "SS"], "BN"),
            ("Clayton Beeter", ["SP", "RP"], "BN"),
        )
        future = self._roster(
            ("Ivan Herrera", ["C", "Util"], "C"),
            ("Marcus Semien", ["2B", "Util"], "2B"),
            ("Michael Wacha", ["SP"], "BN"),
        )

        result = _compute_pending_moves_diff(today, future, "Hart", "k-hart")

        assert len(result) == 1
        move = result[0]
        add_names = {a["name"] for a in move["adds"]}
        drop_names = {d["name"] for d in move["drops"]}
        assert add_names == {"Marcus Semien", "Michael Wacha"}
        assert drop_names == {"Otto Lopez", "Clayton Beeter"}

    def test_diff_uses_normalized_names(self):
        """Yahoo sometimes returns the same player with different casing /
        accent encodings across endpoints. The diff should not produce
        spurious adds/drops for a player whose name only differs in
        accent."""
        from fantasy_baseball.web.season_data import _compute_pending_moves_diff

        today = self._roster(("Julio Rodríguez", ["OF"], "OF"))
        future = self._roster(("Julio Rodriguez", ["OF"], "OF"))

        result = _compute_pending_moves_diff(today, future, "Hart", "k-hart")

        assert result == []

    def test_empty_rosters_return_empty(self):
        from fantasy_baseball.web.season_data import _compute_pending_moves_diff

        result = _compute_pending_moves_diff([], [], "Hart", "k-hart")
        assert result == []
