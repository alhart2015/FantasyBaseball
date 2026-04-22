from dataclasses import FrozenInstanceError

import pytest

from fantasy_baseball.draft.roster_state import RosterState
from fantasy_baseball.models.positions import BENCH_SLOTS, Position


class TestFromDicts:
    def test_string_keys_are_parsed(self):
        state = RosterState.from_dicts(
            filled={"OF": 2, "SS": 1},
            capacity={"OF": 5, "SS": 1, "BN": 3},
        )
        assert state.filled == {Position.OF: 2, Position.SS: 1}
        assert state.capacity == {
            Position.OF: 5,
            Position.SS: 1,
            Position.BN: 3,
        }

    def test_position_keys_are_preserved(self):
        state = RosterState.from_dicts(
            filled={Position.OF: 2},
            capacity={Position.OF: 5, Position.BN: 2},
        )
        assert state.filled == {Position.OF: 2}
        assert state.capacity == {Position.OF: 5, Position.BN: 2}

    def test_mixed_keys_normalize(self):
        state = RosterState.from_dicts(
            filled={"OF": 1, Position.SS: 1},
            capacity={Position.OF: 3, "SS": 1},
        )
        assert state.filled == {Position.OF: 1, Position.SS: 1}
        assert state.capacity == {Position.OF: 3, Position.SS: 1}

    def test_yahoo_casing_parsed(self):
        """Yahoo-style 'Util' / 'util' still land on Position.UTIL."""
        state = RosterState.from_dicts(
            filled={"util": 0},
            capacity={"Util": 1},
        )
        assert state.capacity == {Position.UTIL: 1}

    def test_frozen(self):
        state = RosterState.from_dicts({}, {"OF": 5})
        with pytest.raises(FrozenInstanceError):
            state.filled = {Position.OF: 1}  # type: ignore[misc]


class TestOpenSlots:
    def test_excludes_il_by_default(self):
        state = RosterState.from_dicts(
            filled={},
            capacity={"OF": 5, "IL": 3, "IL+": 1},
        )
        assert Position.IL not in state.open_slots()
        assert Position.IL_PLUS not in state.open_slots()
        assert state.open_slots()[Position.OF] == 5

    def test_bench_included_by_default(self):
        """BN is a valid draft destination, so open_slots() includes it."""
        state = RosterState.from_dicts(
            filled={},
            capacity={"OF": 2, "BN": 5},
        )
        assert state.open_slots()[Position.BN] == 5

    def test_full_slot_excluded(self):
        state = RosterState.from_dicts(
            filled={"OF": 5},
            capacity={"OF": 5, "SS": 1},
        )
        open_ = state.open_slots()
        assert Position.OF not in open_
        assert open_[Position.SS] == 1

    def test_counts_remaining_capacity(self):
        state = RosterState.from_dicts(
            filled={"OF": 2},
            capacity={"OF": 5},
        )
        assert state.open_slots()[Position.OF] == 3

    def test_missing_filled_entry_treated_as_zero(self):
        state = RosterState.from_dicts(
            filled={},
            capacity={"OF": 5},
        )
        assert state.open_slots()[Position.OF] == 5

    def test_custom_exclude_overrides_default(self):
        state = RosterState.from_dicts(
            filled={},
            capacity={"OF": 5, "BN": 3, "IL": 1},
        )
        open_ = state.open_slots(exclude=BENCH_SLOTS)
        assert Position.BN not in open_
        assert Position.IL not in open_
        assert open_[Position.OF] == 5


class TestUnfilledStarterSlots:
    def test_excludes_bn_and_il(self):
        state = RosterState.from_dicts(
            filled={},
            capacity={"OF": 5, "BN": 3, "IL": 1, "IL+": 1},
        )
        unfilled = state.unfilled_starter_slots()
        assert Position.BN not in unfilled
        assert Position.IL not in unfilled
        assert Position.IL_PLUS not in unfilled
        assert Position.OF in unfilled

    def test_full_starter_slot_excluded(self):
        state = RosterState.from_dicts(
            filled={"OF": 5},
            capacity={"OF": 5, "SS": 1},
        )
        assert state.unfilled_starter_slots() == {Position.SS}

    def test_empty_when_all_starters_full(self):
        state = RosterState.from_dicts(
            filled={"OF": 5, "SS": 1},
            capacity={"OF": 5, "SS": 1, "BN": 3},
        )
        assert state.unfilled_starter_slots() == set()


class TestAnySlotOpenFor:
    def test_true_when_position_matches_open_slot(self):
        state = RosterState.from_dicts(
            filled={},
            capacity={"OF": 5},
        )
        assert state.any_slot_open_for(["OF"]) is True
        assert state.any_slot_open_for([Position.OF]) is True

    def test_false_when_all_slots_full(self):
        state = RosterState.from_dicts(
            filled={"OF": 5, "BN": 3},
            capacity={"OF": 5, "BN": 3},
        )
        assert state.any_slot_open_for(["OF"]) is False

    def test_bench_catches_when_starter_slots_full(self):
        """If OF is full but BN is open, an OF player can still be rostered."""
        state = RosterState.from_dicts(
            filled={"OF": 5},
            capacity={"OF": 5, "BN": 3},
        )
        assert state.any_slot_open_for(["OF"]) is True

    def test_util_flex_catches_hitter_when_specific_slot_full(self):
        state = RosterState.from_dicts(
            filled={"OF": 5},
            capacity={"OF": 5, "UTIL": 1},
        )
        assert state.any_slot_open_for(["OF"]) is True

    def test_il_alone_does_not_roster_player(self):
        """IL is excluded from open_slots, so a player with only IL room cannot be rostered."""
        state = RosterState.from_dicts(
            filled={"OF": 5, "BN": 3},
            capacity={"OF": 5, "BN": 3, "IL": 1},
        )
        assert state.any_slot_open_for(["OF"]) is False

    def test_false_when_position_incompatible(self):
        state = RosterState.from_dicts(
            filled={},
            capacity={"OF": 5},
        )
        # A pitcher with only SP eligibility cannot fill an OF slot
        assert state.any_slot_open_for(["SP"]) is False

    def test_handles_iterator_input(self):
        """Accepts any Iterable, not just list."""
        state = RosterState.from_dicts(
            filled={},
            capacity={"OF": 5},
        )
        assert state.any_slot_open_for(iter([Position.OF])) is True
