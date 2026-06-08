"""``_choose_rec`` -- pick_rank (field variance) + position_aware (draft-time
"fill a starter slot, bench last") selection used by the draft simulator.

Both kwargs default off, so ``pick_default`` keeps taking the top rosterable
recommendation. With them set, a sim can pull the 2nd/3rd choice for opponent
variance and refuse bench-bound picks while a starter slot is still open.
"""

from types import SimpleNamespace
from unittest.mock import patch

from fantasy_baseball.draft.recommender import Recommendation
from fantasy_baseball.draft.strategy import _choose_rec


def _rec(name, positions, score):
    return Recommendation(
        name=name,
        var=score,
        score=score,
        best_position=positions[0],
        positions=positions,
        player_type="hitter",
    )


_TRACKER = SimpleNamespace(user_roster_ids=[])
_CFG = SimpleNamespace(roster_slots={"OF": 1, "C": 1, "BN": 2})


def test_default_returns_top_rec():
    recs = [_rec("A", ["OF"], 10), _rec("B", ["OF"], 9)]
    assert _choose_rec(recs, _TRACKER, None, _CFG).name == "A"


def test_pick_rank_selects_nth_choice():
    recs = [_rec("A", ["OF"], 10), _rec("B", ["OF"], 9), _rec("C", ["OF"], 8)]
    assert _choose_rec(recs, _TRACKER, None, _CFG, pick_rank=1).name == "B"
    assert _choose_rec(recs, _TRACKER, None, _CFG, pick_rank=2).name == "C"


def test_pick_rank_clamps_to_last():
    recs = [_rec("A", ["OF"], 10), _rec("B", ["OF"], 9)]
    assert _choose_rec(recs, _TRACKER, None, _CFG, pick_rank=9).name == "B"


def test_position_aware_skips_bench_only_pick():
    # OF starter full -> only the C starter is open. The top rec is another OF
    # (could only go to the bench); the 2nd fills the open C starter. With
    # position_aware on, the catcher must be taken.
    recs = [_rec("ExtraOF", ["OF"], 10), _rec("Catcher", ["C"], 5)]
    with patch("fantasy_baseball.draft.strategy.get_filled_positions", return_value={"OF": 1}):
        chosen = _choose_rec(recs, _TRACKER, None, _CFG, position_aware=True)
    assert chosen.name == "Catcher"


def test_position_aware_off_takes_top_even_if_bench_bound():
    recs = [_rec("ExtraOF", ["OF"], 10), _rec("Catcher", ["C"], 5)]
    with patch("fantasy_baseball.draft.strategy.get_filled_positions", return_value={"OF": 1}):
        chosen = _choose_rec(recs, _TRACKER, None, _CFG)  # position_aware defaults off
    assert chosen.name == "ExtraOF"
