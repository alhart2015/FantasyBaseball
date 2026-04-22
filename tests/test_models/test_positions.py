import pytest


class TestPositionEnum:
    def test_all_expected_values_present(self):
        from fantasy_baseball.models.positions import Position

        assert Position.C.value == "C"
        assert Position.FIRST_BASE.value == "1B"
        assert Position.SECOND_BASE.value == "2B"
        assert Position.THIRD_BASE.value == "3B"
        assert Position.SS.value == "SS"
        assert Position.IF.value == "IF"
        assert Position.OF.value == "OF"
        assert Position.DH.value == "DH"
        assert Position.UTIL.value == "UTIL"
        assert Position.P.value == "P"
        assert Position.SP.value == "SP"
        assert Position.RP.value == "RP"
        assert Position.BN.value == "BN"
        assert Position.IL.value == "IL"
        assert Position.IL_PLUS.value == "IL+"
        assert Position.DL.value == "DL"
        assert Position.DL_PLUS.value == "DL+"

    def test_is_strenum(self):
        """Position values compare equal to their string for compat."""
        from fantasy_baseball.models.positions import Position

        assert Position.UTIL == "UTIL"
        assert Position.OF == "OF"


class TestPositionParse:
    def test_parse_canonical(self):
        from fantasy_baseball.models.positions import Position

        assert Position.parse("OF") is Position.OF
        assert Position.parse("UTIL") is Position.UTIL
        assert Position.parse("1B") is Position.FIRST_BASE

    def test_parse_normalizes_yahoo_util(self):
        """Yahoo returns 'Util'; parser normalizes to Position.UTIL."""
        from fantasy_baseball.models.positions import Position

        assert Position.parse("Util") is Position.UTIL

    def test_parse_normalizes_lowercase_mixed_case(self):
        from fantasy_baseball.models.positions import Position

        assert Position.parse("of") is Position.OF
        assert Position.parse("1b") is Position.FIRST_BASE
        assert Position.parse("bn") is Position.BN

    def test_parse_strips_whitespace(self):
        from fantasy_baseball.models.positions import Position

        assert Position.parse("  OF  ") is Position.OF

    def test_parse_preserves_plus_suffix(self):
        from fantasy_baseball.models.positions import Position

        assert Position.parse("IL+") is Position.IL_PLUS
        assert Position.parse("il+") is Position.IL_PLUS

    def test_parse_unknown_raises(self):
        from fantasy_baseball.models.positions import Position

        with pytest.raises(ValueError, match="Unknown position"):
            Position.parse("QB")

    def test_parse_empty_raises(self):
        from fantasy_baseball.models.positions import Position

        with pytest.raises(ValueError, match="Unknown position"):
            Position.parse("")

    def test_parse_list(self):
        from fantasy_baseball.models.positions import Position

        result = Position.parse_list("OF, Util, 1B")
        assert result == [Position.OF, Position.UTIL, Position.FIRST_BASE]

    def test_parse_list_empty_string(self):
        from fantasy_baseball.models.positions import Position

        assert Position.parse_list("") == []
        assert Position.parse_list(None) == []

    def test_parse_strips_trailing_digits_numbered_slot(self):
        """Historical JSON roster snapshots use "OF2", "BN3", "P5" to
        disambiguate multiple same-named slots. Position.parse must
        collapse them to the base position so League.from_redis can load
        legacy rows from weekly_rosters_history without crashing.

        Production bug landed 2026-04-12 with "Unknown position: 'BN2'"
        when League.from_redis hit a historical row.
        """
        from fantasy_baseball.models.positions import Position

        # The exact value from the production crash
        assert Position.parse("BN2") is Position.BN

        # Other common numbered slots from legacy snapshots
        assert Position.parse("OF2") is Position.OF
        assert Position.parse("OF3") is Position.OF
        assert Position.parse("OF4") is Position.OF
        assert Position.parse("P2") is Position.P
        assert Position.parse("P5") is Position.P
        assert Position.parse("BN3") is Position.BN
        assert Position.parse("UTIL2") is Position.UTIL
        assert Position.parse("IF2") is Position.IF
        assert Position.parse("IL2") is Position.IL

    def test_parse_handles_multi_digit_slot_suffix(self):
        """Leagues with many pitcher slots can have P9, P10, P11..."""
        from fantasy_baseball.models.positions import Position

        assert Position.parse("P10") is Position.P
        assert Position.parse("OF10") is Position.OF

    def test_parse_preserves_leading_digit_positions(self):
        """Regression guard: 1B/2B/3B have digits at the START, not end.

        The trailing-digit stripper must not touch them.
        """
        from fantasy_baseball.models.positions import Position

        assert Position.parse("1B") is Position.FIRST_BASE
        assert Position.parse("2B") is Position.SECOND_BASE
        assert Position.parse("3B") is Position.THIRD_BASE

    def test_parse_numbered_slot_case_insensitive(self):
        """Numbered slot stripping works on mixed-case input."""
        from fantasy_baseball.models.positions import Position

        assert Position.parse("bn2") is Position.BN
        assert Position.parse("Of2") is Position.OF

    def test_parse_list_handles_numbered_slots(self):
        """The DB loader passes comma-joined position strings through
        parse_list. Historical positions columns may contain numbered
        slots too."""
        from fantasy_baseball.models.positions import Position

        result = Position.parse_list("OF2, Util, BN3")
        assert result == [Position.OF, Position.UTIL, Position.BN]

    def test_parse_all_digits_raises(self):
        """A bare integer string isn't a position at all."""
        from fantasy_baseball.models.positions import Position

        with pytest.raises(ValueError, match="Unknown position"):
            Position.parse("42")


class TestPositionSets:
    def test_hitter_eligible_contains_all_hitter_positions(self):
        from fantasy_baseball.models.positions import (
            HITTER_ELIGIBLE,
            Position,
        )

        assert Position.C in HITTER_ELIGIBLE
        assert Position.FIRST_BASE in HITTER_ELIGIBLE
        assert Position.OF in HITTER_ELIGIBLE
        assert Position.UTIL in HITTER_ELIGIBLE
        assert Position.DH in HITTER_ELIGIBLE
        assert Position.IF in HITTER_ELIGIBLE

    def test_hitter_eligible_excludes_pitcher_positions(self):
        from fantasy_baseball.models.positions import (
            HITTER_ELIGIBLE,
            Position,
        )

        assert Position.P not in HITTER_ELIGIBLE
        assert Position.SP not in HITTER_ELIGIBLE
        assert Position.RP not in HITTER_ELIGIBLE

    def test_pitcher_eligible(self):
        from fantasy_baseball.models.positions import (
            PITCHER_ELIGIBLE,
            Position,
        )

        assert frozenset({Position.P, Position.SP, Position.RP}) == PITCHER_ELIGIBLE

    def test_bench_slots(self):
        from fantasy_baseball.models.positions import (
            BENCH_SLOTS,
            Position,
        )

        assert Position.BN in BENCH_SLOTS
        assert Position.IL in BENCH_SLOTS
        assert Position.IL_PLUS in BENCH_SLOTS
        assert Position.DL in BENCH_SLOTS
        assert Position.DL_PLUS in BENCH_SLOTS
        assert Position.OF not in BENCH_SLOTS

    def test_il_slots(self):
        from fantasy_baseball.models.positions import (
            IL_SLOTS,
            Position,
        )

        assert Position.IL in IL_SLOTS
        assert Position.IL_PLUS in IL_SLOTS
        assert Position.DL in IL_SLOTS
        assert Position.DL_PLUS in IL_SLOTS
        assert Position.BN not in IL_SLOTS


class TestInteropWithUtilsPositions:
    def test_utils_hitter_positions_contains_enum_members(self):
        """Legacy HITTER_POSITIONS set accepts enum values."""
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.utils.positions import HITTER_POSITIONS

        assert Position.C in HITTER_POSITIONS
        assert Position.OF in HITTER_POSITIONS
        assert Position.UTIL in HITTER_POSITIONS
        assert Position.P not in HITTER_POSITIONS

    def test_can_fill_slot_accepts_enum_args(self):
        """can_fill_slot works with Position enum values."""
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.utils.positions import can_fill_slot

        # Outfielder can fill OF and UTIL
        assert can_fill_slot([Position.OF], Position.OF)
        assert can_fill_slot([Position.OF], Position.UTIL)
        assert can_fill_slot([Position.OF], Position.BN)
        # Outfielder cannot fill C
        assert not can_fill_slot([Position.OF], Position.C)
        # Infielder can fill IF
        assert can_fill_slot([Position.FIRST_BASE], Position.IF)

    def test_can_fill_slot_accepts_string_args_backward_compat(self):
        """Legacy string call sites still work because Position is StrEnum."""
        from fantasy_baseball.utils.positions import can_fill_slot

        assert can_fill_slot(["OF"], "OF")
        assert can_fill_slot(["OF"], "UTIL")

    def test_is_pitcher_empty_string_returns_false(self):
        """Unfilled Yahoo slots pass selected_position="". Must not raise."""
        from fantasy_baseball.utils.positions import is_hitter, is_pitcher

        assert is_pitcher([""]) is False
        assert is_hitter([""]) is False

    def test_can_fill_slot_empty_slot_returns_false(self):
        """An empty slot string can't be filled by anything."""
        from fantasy_baseball.utils.positions import can_fill_slot

        assert can_fill_slot(["OF"], "") is False
        assert can_fill_slot(["OF"], None) is False

    def test_can_fill_slot_ignores_empty_entries_in_player_positions(self):
        """Empty strings in player_positions are skipped."""
        from fantasy_baseball.utils.positions import can_fill_slot

        assert can_fill_slot(["OF", ""], "OF") is True
        assert can_fill_slot(["", "OF"], "UTIL") is True

    def test_is_hitter_mixed_empty_and_valid(self):
        from fantasy_baseball.utils.positions import is_hitter, is_pitcher

        assert is_hitter(["", "OF"]) is True
        assert is_pitcher(["", "SP"]) is True
