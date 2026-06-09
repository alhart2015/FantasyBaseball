import pytest

from fantasy_baseball.draft.recommend import RecommendContext, rank_for_mode, recommend


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


@pytest.mark.parametrize("mode", ["var", "vona"])
def test_rank_for_mode_var_vona_scores_present(varvona_ctx, mode):
    picks = rank_for_mode(varvona_ctx(scoring_mode=mode))
    assert picks
    assert picks[0].metrics[mode] == picks[0].score


def test_rank_for_mode_var_vona_requires_board():
    ctx = RecommendContext(scoring_mode="var", team_name="X", picks_until_next=8, board=None)
    with pytest.raises(ValueError, match="requires board"):
        rank_for_mode(ctx)


def test_rank_for_mode_rejects_unknown_mode(varvona_ctx):
    with pytest.raises(ValueError, match="unknown scoring_mode"):
        rank_for_mode(varvona_ctx(scoring_mode="nope"))


def test_recommend_deltaroto_default_picks_top_immediate(deltaroto_ctx):
    chosen = recommend(
        deltaroto_ctx(scoring_mode="deltaroto_immediate"),
        strategy="default",
        open_starters=set(),
    )
    assert chosen is not None
    assert chosen.score == chosen.metrics["immediate_delta"]


@pytest.mark.parametrize("mode", ["var", "vona"])
def test_recommend_var_vona_runs_through_same_seam(varvona_ctx, mode):
    # Proves recommend() serves all four modes through one entry (spec sec 2).
    chosen = recommend(varvona_ctx(scoring_mode=mode), strategy="default", open_starters=set())
    assert chosen is not None
    assert chosen.score == chosen.metrics[mode]


def test_recommend_forwards_overlay_kwargs(deltaroto_ctx, monkeypatch):
    """recommend() must forward **overlay_kwargs to the overlay.

    Closer-family overlays read current_round and closer_count from kwargs.
    Without forwarding, kwargs.get("current_round", 0) always returns 0 and
    the overlay is permanently inert (the bug this test guards against).
    """
    from fantasy_baseball.draft import strategy as strat

    seen = {}

    def spy(ranked, *, roster_state=None, config=None, **kwargs):
        seen.update(kwargs)
        return None  # defer, so recommend() still returns a real pick

    monkeypatch.setitem(strat.OVERLAYS, "spy", spy)
    chosen = recommend(
        deltaroto_ctx(scoring_mode="deltaroto_immediate"),
        strategy="spy",
        open_starters=set(),
        current_round=14,
        closer_count=0,
    )
    assert seen == {"current_round": 14, "closer_count": 0}
    assert chosen is not None  # defer path still produced a pick
