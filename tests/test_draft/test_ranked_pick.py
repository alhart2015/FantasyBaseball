from fantasy_baseball.draft.recommend import RankedPick
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position


def test_ranked_pick_holds_core_fields_and_defaults():
    rp = RankedPick(
        player_id="123",
        name="Test Player",
        positions=[Position.SS, Position.OF],
        player_type=PlayerType.HITTER,
        score=4.2,
    )
    assert rp.score == 4.2
    assert rp.metrics == {}
    assert rp.per_category == {}
    assert rp.note == ""
    assert rp.need_flag is False


def test_position_strings_serializes_enum_values():
    rp = RankedPick(
        player_id="1",
        name="P",
        positions=[Position.SS, Position.OF],
        player_type=PlayerType.HITTER,
        score=0.0,
    )
    assert rp.position_strings() == ["SS", "OF"]
