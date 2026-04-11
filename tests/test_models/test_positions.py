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


class TestPositionSets:
    def test_hitter_eligible_contains_all_hitter_positions(self):
        from fantasy_baseball.models.positions import (
            HITTER_ELIGIBLE, Position,
        )
        assert Position.C in HITTER_ELIGIBLE
        assert Position.FIRST_BASE in HITTER_ELIGIBLE
        assert Position.OF in HITTER_ELIGIBLE
        assert Position.UTIL in HITTER_ELIGIBLE
        assert Position.DH in HITTER_ELIGIBLE
        assert Position.IF in HITTER_ELIGIBLE

    def test_hitter_eligible_excludes_pitcher_positions(self):
        from fantasy_baseball.models.positions import (
            HITTER_ELIGIBLE, Position,
        )
        assert Position.P not in HITTER_ELIGIBLE
        assert Position.SP not in HITTER_ELIGIBLE
        assert Position.RP not in HITTER_ELIGIBLE

    def test_pitcher_eligible(self):
        from fantasy_baseball.models.positions import (
            PITCHER_ELIGIBLE, Position,
        )
        assert PITCHER_ELIGIBLE == frozenset({Position.P, Position.SP, Position.RP})

    def test_bench_slots(self):
        from fantasy_baseball.models.positions import (
            BENCH_SLOTS, Position,
        )
        assert Position.BN in BENCH_SLOTS
        assert Position.IL in BENCH_SLOTS
        assert Position.IL_PLUS in BENCH_SLOTS
        assert Position.DL in BENCH_SLOTS
        assert Position.DL_PLUS in BENCH_SLOTS
        assert Position.OF not in BENCH_SLOTS

    def test_il_slots(self):
        from fantasy_baseball.models.positions import (
            IL_SLOTS, Position,
        )
        assert Position.IL in IL_SLOTS
        assert Position.IL_PLUS in IL_SLOTS
        assert Position.DL in IL_SLOTS
        assert Position.DL_PLUS in IL_SLOTS
        assert Position.BN not in IL_SLOTS
