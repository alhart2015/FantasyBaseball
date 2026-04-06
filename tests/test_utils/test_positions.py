import pytest
from fantasy_baseball.utils.positions import can_fill_slot, can_cover_slots, is_hitter, is_pitcher


class TestCanFillSlot:
    def test_catcher_fills_c(self):
        assert can_fill_slot(["C"], "C") is True

    def test_catcher_cannot_fill_1b(self):
        assert can_fill_slot(["C"], "1B") is False

    def test_shortstop_fills_if(self):
        assert can_fill_slot(["SS"], "IF") is True

    def test_catcher_cannot_fill_if(self):
        assert can_fill_slot(["C"], "IF") is False

    def test_outfielder_fills_of(self):
        assert can_fill_slot(["OF"], "OF") is True

    def test_any_hitter_fills_util(self):
        assert can_fill_slot(["C"], "UTIL") is True
        assert can_fill_slot(["OF"], "UTIL") is True
        assert can_fill_slot(["1B", "OF"], "UTIL") is True

    def test_pitcher_cannot_fill_util(self):
        assert can_fill_slot(["SP"], "UTIL") is False
        assert can_fill_slot(["RP"], "UTIL") is False

    def test_pitcher_fills_p(self):
        assert can_fill_slot(["SP"], "P") is True
        assert can_fill_slot(["RP"], "P") is True
        assert can_fill_slot(["P"], "P") is True

    def test_multi_position_player(self):
        assert can_fill_slot(["SS", "2B"], "SS") is True
        assert can_fill_slot(["SS", "2B"], "2B") is True
        assert can_fill_slot(["SS", "2B"], "IF") is True
        assert can_fill_slot(["SS", "2B"], "UTIL") is True
        assert can_fill_slot(["SS", "2B"], "OF") is False

    def test_any_hitter_cannot_fill_p(self):
        assert can_fill_slot(["1B"], "P") is False

    def test_bench_and_il_accept_anyone(self):
        assert can_fill_slot(["C"], "BN") is True
        assert can_fill_slot(["SP"], "BN") is True
        assert can_fill_slot(["OF"], "IL") is True
        assert can_fill_slot(["RP"], "IL") is True


class TestCanCoverSlots:
    def test_exact_coverage(self):
        players = [["C"], ["1B"], ["OF"]]
        slots = {"C": 1, "1B": 1, "OF": 1}
        assert can_cover_slots(players, slots) is True

    def test_missing_position(self):
        """No one can play 1B — coverage fails."""
        players = [["C"], ["SS"], ["OF"]]
        slots = {"C": 1, "1B": 1, "OF": 1}
        assert can_cover_slots(players, slots) is False

    def test_multi_position_enables_coverage(self):
        """Multi-position player can shift to let everyone fit."""
        players = [["1B", "3B"], ["3B"], ["OF"]]
        slots = {"1B": 1, "3B": 1, "OF": 1}
        assert can_cover_slots(players, slots) is True

    def test_util_absorbs_overflow(self):
        players = [["C"], ["1B"], ["OF"], ["OF"]]
        slots = {"C": 1, "1B": 1, "OF": 1, "UTIL": 1}
        assert can_cover_slots(players, slots) is True

    def test_too_few_players(self):
        players = [["C"]]
        slots = {"C": 1, "1B": 1}
        assert can_cover_slots(players, slots) is False

    def test_ignores_pitcher_slots(self):
        """P, BN, IL slots are skipped — only hitter slots checked."""
        players = [["C"]]
        slots = {"C": 1, "P": 9, "BN": 2, "IL": 2}
        assert can_cover_slots(players, slots) is True


class TestIsHitter:
    def test_catcher_is_hitter(self):
        assert is_hitter(["C"]) is True

    def test_outfielder_is_hitter(self):
        assert is_hitter(["OF"]) is True

    def test_pitcher_is_not_hitter(self):
        assert is_hitter(["SP"]) is False
        assert is_hitter(["RP"]) is False

    def test_two_way_player(self):
        assert is_hitter(["DH", "SP"]) is True

    def test_util_is_hitter(self):
        assert is_hitter(["Util"]) is True


class TestIsPitcher:
    def test_sp_is_pitcher(self):
        assert is_pitcher(["SP"]) is True

    def test_rp_is_pitcher(self):
        assert is_pitcher(["RP"]) is True

    def test_hitter_is_not_pitcher(self):
        assert is_pitcher(["1B"]) is False

    def test_two_way_player(self):
        assert is_pitcher(["DH", "SP"]) is True
