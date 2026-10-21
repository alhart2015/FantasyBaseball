import pandas as pd
import pytest

from fantasy_baseball.draft import our_pitchers as op
from fantasy_baseball.pitcher_ros.final import BOARD, WEIGHTS


def _fg():
    # Three board pitchers: 1 and 2 ours projects, 3 it doesn't.
    return pd.DataFrame(
        {
            "fg_id": ["a", "b", "c"],
            "name": ["One", "Two", "Three"],
            "ip": [180.0, 60.0, 100.0],
            "k": [200.0, 70.0, 90.0],
            "w": [12.0, 3.0, 6.0],
            "sv": [0.0, 30.0, 0.0],
            "er": [70.0, 20.0, 50.0],
            "era": [3.5, 3.0, 4.5],
            "h_allowed": [150.0, 45.0, 95.0],
            "bb": [50.0, 20.0, 35.0],
            "whip": [200 / 180, 65 / 60, 1.3],
            "bf": [735.0, 262.0, 430.0],
        }
    )


def _ours():
    return pd.DataFrame(
        {
            "k": [0.30, 0.20],
            "w": [0.02, 0.01],
            "sv": [0.0, 0.15],
            "era": [3.0, 4.0],
            "whip": [1.0, 1.3],
        },
        index=pd.Index([10, 20], name="player_id"),
    )


def test_ours_reshapes_the_rates_and_fangraphs_keeps_the_playing_time():
    fg = _fg()
    out = op.blend_board_pitchers(fg, _ours(), pd.Series({"a": 10, "b": 20}))
    w = WEIGHTS[BOARD]
    bf = 735.0  # FanGraphs' own batters faced, not 3 x IP + H + BB (740)
    k = w["k"] * 0.30 + (1 - w["k"]) * 200 / bf
    assert out.loc[0, "k"] == pytest.approx(k * bf)
    era = w["era"] * 3.0 + (1 - w["era"]) * 3.5
    assert out.loc[0, "era"] == pytest.approx(era) and out.loc[0, "er"] == pytest.approx(era * 20)
    whip = w["whip"] * 1.0 + (1 - w["whip"]) * 200 / 180
    assert out.loc[0, "whip"] == pytest.approx(whip)
    # Hits and walks carry the blended WHIP, split as FanGraphs split them.
    assert out.loc[0, "h_allowed"] + out.loc[0, "bb"] == pytest.approx(whip * 180)
    assert out.loc[0, "h_allowed"] / out.loc[0, "bb"] == pytest.approx(150 / 50)
    assert out["ip"].tolist() == fg["ip"].tolist() and out["name"].tolist() == fg["name"].tolist()
    # A pitcher ours doesn't project keeps the FanGraphs line, and the input is untouched.
    pd.testing.assert_series_equal(
        out.loc[2], fg.loc[2].astype(out.loc[2].dtype), check_names=False
    )
    pd.testing.assert_frame_equal(fg, _fg())


def test_an_unmatched_board_is_fangraphs_as_is():
    out = op.blend_board_pitchers(_fg(), _ours(), pd.Series({"x": 10}))
    pd.testing.assert_frame_equal(out, _fg())


def test_two_board_rows_on_one_mlbam_id_are_refused():
    with pytest.raises(ValueError, match="share an MLBAM id"):
        op.blend_board_pitchers(_fg(), _ours(), pd.Series({"a": 10, "b": 10}))


def test_batters_faced_is_fangraphs_tbf_and_the_estimate_only_where_it_is_missing(caplog):
    fg = _fg().assign(bf=[735.0, None, 0.0])
    assert op.batters_faced(fg).tolist() == [735.0, 180 + 45 + 20, 300 + 95 + 35]
    assert "2 FanGraphs blend lines have no bf" in caplog.text
    old = _fg().drop(columns="bf")  # a blend stored before bf was
    assert op.batters_faced(old).tolist() == [740.0, 245.0, 430.0]
    assert "build_db.py" in caplog.text
