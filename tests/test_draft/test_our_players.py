import pandas as pd

from fantasy_baseball.draft import our_players


def test_an_id_that_maps_two_ways_is_dropped_not_guessed():
    raw = pd.DataFrame(
        {"fg_id": ["a", "a", "b", "c", "d", "e"], "mlbam_id": [1, 2, 3, 3, 4.0, None]}
    )
    assert our_players.fg_to_mlbam(raw).to_dict() == {"d": 4}


def test_ours_is_looked_for_in_the_type_s_own_folder_and_missing_is_none(caplog):
    assert our_players.ours_path("hitter", 2027).parent.name == "hitter_ros"
    assert our_players.ours_path("pitcher", 2027).parent.name == "pitcher_ros"
    assert our_players.load_ours("hitter", 2027) is None  # tests never see real files
    assert "build_our_hitters.py" in caplog.text
