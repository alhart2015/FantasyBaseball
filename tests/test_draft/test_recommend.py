import pytest

from fantasy_baseball.draft.recommend import RecommendContext, rank_for_mode


def test_rank_for_mode_deltaroto_immediate_returns_ranked_picks(deltaroto_ctx):
    picks = rank_for_mode(deltaroto_ctx(scoring_mode="deltaroto_immediate"))
    assert picks, "expected at least one ranked pick"
    assert picks[0].score == picks[0].metrics["immediate_delta"]
    scores = [p.metrics["immediate_delta"] for p in picks]
    assert scores == sorted(scores, reverse=True)


def test_rank_for_mode_vopn_sorts_by_vopn(deltaroto_ctx):
    picks = rank_for_mode(deltaroto_ctx(scoring_mode="deltaroto_vopn"))
    assert picks[0].score == picks[0].metrics["value_of_picking_now"]
    vopn = [p.metrics["value_of_picking_now"] for p in picks]
    assert vopn == sorted(vopn, reverse=True)


def test_rank_for_mode_deltaroto_requires_inputs():
    ctx = RecommendContext(
        scoring_mode="deltaroto_immediate", team_name="X", picks_until_next=8, inputs=None
    )
    with pytest.raises(ValueError, match="requires inputs"):
        rank_for_mode(ctx)
