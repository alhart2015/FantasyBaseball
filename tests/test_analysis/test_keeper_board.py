import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.analysis import keeper_board as kb
from fantasy_baseball.utils.constants import Category

DENOMS = {
    Category.R: 20.0,
    Category.HR: 7.0,
    Category.RBI: 20.0,
    Category.SB: 6.0,
    Category.AVG: 0.004,
    Category.W: 3.0,
    Category.K: 30.0,
    Category.SV: 7.0,
    Category.ERA: 0.1,
    Category.WHIP: 0.015,
}


def _hit_rates(ids):
    return pd.DataFrame(
        {"r": 0.14, "hr": 0.05, "rbi": 0.14, "sb": 0.02, "avg": 0.27}, index=pd.Index(ids)
    )


def test_hitter_lines_are_rates_times_playing_time_and_default_to_outfield():
    pt = pd.DataFrame({"pa": [600.0, 0.0], "ab": [540.0, 0.0]}, index=[1, 2])
    board = kb.hitter_board(_hit_rates([1, 2]), pt, {1: ["SS", "UTIL"]}, DENOMS)
    assert board.index.tolist() == [1]  # no playing time: no line
    assert board.loc[1, "hr"] == pytest.approx(30.0) and board.loc[1, "avg"] == 0.27
    assert board.loc[1, "positions"] == ["SS"]
    other = kb.hitter_board(
        _hit_rates([3]), pd.DataFrame({"pa": [600.0], "ab": [540.0]}, index=[3]), {}, DENOMS
    )
    assert other.loc[3, "positions"] == [kb.DEFAULT_HIT_SLOT]


def test_relievers_are_capped_at_70_innings_and_starters_are_not():
    rates = pd.DataFrame(
        {"k": 0.28, "w": 0.04, "sv": 0.1, "era": 3.0, "whip": 1.1}, index=pd.Index([1, 2, 3])
    )
    pt = pd.DataFrame(
        {"ip": [104.0, 180.0, 60.0], "bf": [440.0, 750.0, 250.0], "gs_share": [0.1, 1.0, np.nan]},
        index=[1, 2, 3],
    )
    board = kb.pitcher_board(rates, pt, DENOMS)
    assert board.index.tolist() == [1, 2]  # no role: no line
    assert board.loc[1, "role"] == "RP" and board.loc[1, "ip"] == pytest.approx(70.0)
    assert board.loc[1, "sv"] == pytest.approx(0.1 * 440 * 70 / 104)
    assert board.loc[2, "role"] == "SP" and board.loc[2, "ip"] == 180.0


def _board():
    return pd.DataFrame(
        {
            "kind": ["hitter", "pitcher", "hitter", "hitter"],
            "positions": [["OF"], ["P"], ["3B"], ["SS"]],
            "sgp": [15.0, 12.0, 9.0, 8.0],
        },
        index=pd.Index([10, 10, 20, 21]),  # 10 is a two-way player
    )


def test_matching_finds_both_halves_of_a_two_way_player_and_settles_shared_names():
    names = {10: "Shohei Ohtani", 20: "Max Muncy", 21: "Max Muncy"}
    rostered = [
        kb.Rostered("A", "Shohei Ohtani", "hitter", ("UTIL",), "LAD"),
        kb.Rostered("A", "Shohei Ohtani", "pitcher", ("P",), "LAD"),
        kb.Rostered("B", "Max Muncy", "hitter", ("3B",), "LAD"),
        kb.Rostered("C", "Max Muncy", "hitter", ("SS",), "NYM"),  # neither is a Met
        kb.Rostered("C", "Nobody", "pitcher", ("P",), "SEA"),
    ]
    matched, problems = kb.match_rostered(rostered, _board(), names, {20: "LAD", 21: "ATH"})
    assert matched["sgp"].tolist() == [15.0, 12.0, 9.0]
    assert matched["team"].tolist() == ["A", "A", "B"] and matched["player_id"].tolist() == [
        10,
        10,
        20,
    ]
    # A hitter's roster slots are his own eligibility; UTIL alone isn't one, so his board's stand.
    assert matched["positions"].tolist() == [["OF"], ["P"], ["3B"]]
    assert (
        len(problems) == 2
        and "no unique MLB-team match" in problems[0]
        and "no projection" in problems[1]
    )


def test_var_is_sgp_over_the_replacement_at_his_best_position():
    board = pd.DataFrame(
        {
            "kind": ["hitter"] * 3 + ["pitcher"] * 3,
            "positions": [["OF"], ["OF"], ["OF"], ["P"], ["P"], ["P"]],
            "sgp": [14.0, 12.0, 10.0, 11.0, 9.0, 7.0],
            "name": list("abcdef"),
        },
        index=pd.Index(range(6)),
    )
    levels, repl = kb.replacement(board, {"OF": 1, "P": 1})
    assert levels["OF"] == 12.0 and levels["P"] == 9.0 and repl["OF"]["name"] == "b"
    assert kb.var([14.0, 11.0], [["OF"], ["P"]], levels) == [pytest.approx(2.0), pytest.approx(2.0)]
