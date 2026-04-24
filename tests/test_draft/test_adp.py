import pandas as pd


def test_blend_adp_across_systems_uses_mean():
    from fantasy_baseball.draft.adp import blend_adp

    per_system = {
        "steamer": pd.DataFrame(
            {
                "player_id": ["a::hitter", "b::hitter"],
                "adp": [5.0, 100.0],
            }
        ),
        "atc": pd.DataFrame(
            {
                "player_id": ["a::hitter", "b::hitter"],
                "adp": [7.0, 110.0],
            }
        ),
    }
    blended = blend_adp(per_system)
    assert blended["a::hitter"] == 6.0
    assert blended["b::hitter"] == 105.0


def test_blend_adp_skips_missing():
    from fantasy_baseball.draft.adp import blend_adp

    per_system = {
        "steamer": pd.DataFrame(
            {
                "player_id": ["a::hitter"],
                "adp": [5.0],
            }
        ),
        "atc": pd.DataFrame(
            {
                "player_id": ["a::hitter", "b::hitter"],
                "adp": [7.0, 110.0],
            }
        ),
    }
    blended = blend_adp(per_system)
    assert blended["a::hitter"] == 6.0
    assert blended["b::hitter"] == 110.0


def test_fallback_adp_for_unknown_player():
    from fantasy_baseball.draft.adp import ADPTable

    table = ADPTable(adp={"a::hitter": 10.0}, fallback_offset=1000.0)
    assert table.get("a::hitter") == 10.0
    assert table.get("unknown::hitter") > 1000.0
