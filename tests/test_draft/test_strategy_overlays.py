from fantasy_baseball.draft.recommend import RankedPick
from fantasy_baseball.draft.strategy import OVERLAYS
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.player import PlayerType


def _pick(name, score, pos=Position.OF):
    return RankedPick(player_id=name, name=name, positions=[pos],
                      player_type=PlayerType.HITTER, score=score,
                      metrics={"immediate_delta": score})


def test_default_overlay_defers_to_slot_gate():
    # default applies NO constraint -- it returns None so recommend()'s
    # select_from_ranked makes the position-aware greedy pick (verdict winner).
    ranked = [_pick("A", 5.0), _pick("B", 3.0)]
    assert OVERLAYS["default"](ranked, roster_state=None, config=None) is None
