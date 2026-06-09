from fantasy_baseball.draft.recommend import RankedPick, from_recommendation
from fantasy_baseball.draft.recommender import Recommendation
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


def test_from_recommendation_maps_var_to_score():
    rec = Recommendation(
        name="Slugger",
        var=6.5,
        score=6.5,
        best_position="OF",
        positions=["OF"],
        player_type=PlayerType.HITTER,
        need_flag=True,
        note="need OF",
    )
    rp = from_recommendation(rec, player_id="999")
    assert rp.score == 6.5
    assert rp.metrics == {"var": 6.5}
    assert rp.name == "Slugger"
    assert rp.need_flag is True
    assert rp.note == "need OF"
    assert rp.position_strings() == ["OF"]
