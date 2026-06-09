from fantasy_baseball.draft.recommend import RankedPick
from fantasy_baseball.draft.strategy import (
    CLOSER_DEADLINE_ROUND,
    FOUR_CLOSERS_DEADLINES,
    OVERLAYS,
    THREE_CLOSERS_DEADLINES,
    TWO_CLOSERS_DEADLINES,
)
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position


def _pick(name, score, pos=Position.OF):
    return RankedPick(
        player_id=name,
        name=name,
        positions=[pos],
        player_type=PlayerType.HITTER,
        score=score,
        metrics={"immediate_delta": score},
    )


def _closer(name, score, sv):
    return RankedPick(
        player_id=name,
        name=name,
        positions=[Position.RP],
        player_type=PlayerType.PITCHER,
        score=score,
        metrics={"immediate_delta": score},
        per_category={"SV": sv},
    )


def test_default_overlay_defers_to_slot_gate():
    # default applies NO constraint -- it returns None so recommend()'s
    # select_from_ranked makes the position-aware greedy pick (verdict winner).
    ranked = [_pick("A", 5.0), _pick("B", 3.0)]
    assert OVERLAYS["default"](ranked, roster_state=None, config=None) is None


# ---------------------------------------------------------------------------
# nonzero_sv overlay
# ---------------------------------------------------------------------------


def test_nonzero_sv_skips_zero_save_relievers_for_closer_slot():
    # Middle reliever scores higher but has SV=0 -- overlay must skip it.
    ranked = [_closer("MiddleReliever", 9.0, 0.0), _closer("Closer", 4.0, 30.0)]
    chosen = OVERLAYS["nonzero_sv"](
        ranked,
        roster_state=None,
        config=None,
        current_round=CLOSER_DEADLINE_ROUND,
        closer_count=0,
    )
    assert chosen is not None
    assert chosen.name == "Closer"


def test_nonzero_sv_defers_before_deadline():
    # Before the deadline, even with no closer, should defer (return None).
    ranked = [_closer("MiddleReliever", 9.0, 0.0), _closer("Closer", 4.0, 30.0)]
    result = OVERLAYS["nonzero_sv"](
        ranked,
        roster_state=None,
        config=None,
        current_round=CLOSER_DEADLINE_ROUND - 1,
        closer_count=0,
    )
    assert result is None


def test_nonzero_sv_defers_when_closer_already_drafted():
    # Already have a closer -- no forcing needed regardless of round.
    ranked = [_closer("AnotherCloser", 5.0, 25.0)]
    result = OVERLAYS["nonzero_sv"](
        ranked,
        roster_state=None,
        config=None,
        current_round=CLOSER_DEADLINE_ROUND + 5,
        closer_count=1,
    )
    assert result is None


def test_nonzero_sv_sv_zero_treated_as_zero_not_falsy():
    # SV=0.0 must NOT be treated as missing -- the MR with SV=0.0 must be skipped.
    ranked = [_closer("MR", 10.0, 0.0)]
    result = OVERLAYS["nonzero_sv"](
        ranked,
        roster_state=None,
        config=None,
        current_round=CLOSER_DEADLINE_ROUND,
        closer_count=0,
    )
    # No real closer available -- overlay returns None (can't force what's not there).
    assert result is None


# ---------------------------------------------------------------------------
# two_closers overlay
# ---------------------------------------------------------------------------


def test_two_closers_forces_first_closer_at_first_deadline():
    deadline = TWO_CLOSERS_DEADLINES[0]  # 8
    ranked = [_closer("C1", 5.0, 35.0), _closer("C2", 3.0, 28.0)]
    chosen = OVERLAYS["two_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    assert chosen is not None and chosen.name == "C1"


def test_two_closers_defers_before_first_deadline():
    deadline = TWO_CLOSERS_DEADLINES[0]  # 8
    ranked = [_closer("C1", 5.0, 35.0)]
    result = OVERLAYS["two_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline - 1,
        closer_count=0,
    )
    assert result is None


def test_two_closers_forces_second_closer_at_second_deadline():
    deadline = TWO_CLOSERS_DEADLINES[1]  # 14
    ranked = [_closer("C2", 4.0, 22.0)]
    chosen = OVERLAYS["two_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=1,
    )
    assert chosen is not None and chosen.name == "C2"


def test_two_closers_defers_when_target_met():
    # Already have 2 closers -- no forcing at any round.
    ranked = [_closer("C3", 6.0, 30.0)]
    result = OVERLAYS["two_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=20,
        closer_count=2,
    )
    assert result is None


# ---------------------------------------------------------------------------
# three_closers overlay
# ---------------------------------------------------------------------------


def test_three_closers_forces_first_closer_at_first_deadline():
    deadline = THREE_CLOSERS_DEADLINES[0]  # 5
    ranked = [_closer("C1", 7.0, 40.0), _closer("MR", 9.0, 0.0)]
    chosen = OVERLAYS["three_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    # MR has higher score but SV=0 -- must pick C1
    assert chosen is not None and chosen.name == "C1"


def test_three_closers_defers_before_deadline():
    deadline = THREE_CLOSERS_DEADLINES[0]  # 5
    ranked = [_closer("C1", 7.0, 40.0)]
    result = OVERLAYS["three_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline - 1,
        closer_count=0,
    )
    assert result is None


def test_three_closers_defers_when_target_met():
    ranked = [_closer("C4", 6.0, 30.0)]
    result = OVERLAYS["three_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=20,
        closer_count=3,
    )
    assert result is None


# ---------------------------------------------------------------------------
# four_closers overlay
# ---------------------------------------------------------------------------


def test_four_closers_forces_first_closer_at_first_deadline():
    deadline = FOUR_CLOSERS_DEADLINES[0]  # 5
    ranked = [_closer("C1", 6.0, 35.0)]
    chosen = OVERLAYS["four_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    assert chosen is not None and chosen.name == "C1"


def test_four_closers_forces_fourth_closer_at_last_deadline():
    deadline = FOUR_CLOSERS_DEADLINES[3]  # 16
    ranked = [_closer("C4", 3.0, 21.0)]
    chosen = OVERLAYS["four_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=3,
    )
    assert chosen is not None and chosen.name == "C4"


def test_four_closers_defers_when_target_met():
    ranked = [_closer("C5", 8.0, 45.0)]
    result = OVERLAYS["four_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=20,
        closer_count=4,
    )
    assert result is None
