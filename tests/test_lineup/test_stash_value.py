from fantasy_baseball.lineup.stash_value import StashScore, StashResult


def test_stash_score_to_dict_shape():
    s = StashScore(
        name="Blake Snell",
        player_type="pitcher",
        status="IL15",
        owned=False,
        gain=4.2,
        cost=0.0,
        stash_value=4.2,
        band={"mean": 4.2, "sd": 1.1, "p_positive": 0.91, "verdict": "real"},
        recommended_drop=None,
    )
    d = s.to_dict()
    assert d["name"] == "Blake Snell"
    assert d["stash_value"] == 4.2
    assert d["band"]["verdict"] == "real"
    assert d["recommended_drop"] is None


def test_stash_result_to_dict_shape():
    r = StashResult(open_il_slots=1, cutline_rank=2, candidates=[], warning=None)
    d = r.to_dict()
    assert d["open_il_slots"] == 1
    assert d["cutline_rank"] == 2
    assert d["candidates"] == []
    assert d["warning"] is None
