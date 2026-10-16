"""Pitch tokens follow the hitter who saw each pitch or the pitcher who threw it (#492)."""

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.hitter_ros.pitch_tokens import TOKEN_FEATURES, build_pitch_tokens

HITTER_A, HITTER_B, PITCHER_X, PITCHER_Y = 1, 2, 7, 8


def _pitch(day, pk, ab, n, batter, pitcher, description):
    return {
        "batter": batter,
        "pitcher": pitcher,
        "game_date": pd.Timestamp(day),
        "game_pk": pk,
        "game_type": "R",
        "at_bat_number": ab,
        "pitch_number": n,
        "pitch_type": "FF",
        "release_speed": 95.0,
        "pfx_x": -0.5,
        "pfx_z": 1.3,
        "release_spin_rate": 2300.0,
        "plate_x": 0.0,
        "plate_z": 2.5,
        "sz_top": 3.5,
        "sz_bot": 1.5,
        "balls": 0,
        "strikes": n - 1,
        "outs_when_up": 0,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "p_throws": "R",
        "stand": "R",
        "description": description,
        "type": "S",
        "launch_speed_angle": None,
        "launch_speed": None,
        "launch_angle": None,
    }


@pytest.fixture
def store(tmp_path):
    rows = [
        # Game 1: X faces A then B.
        _pitch("2024-05-01", 1, 1, 1, HITTER_A, PITCHER_X, "called_strike"),
        _pitch("2024-05-01", 1, 2, 1, HITTER_B, PITCHER_X, "swinging_strike"),
        # Game 2, four days later: Y faces A.
        _pitch("2024-05-05", 2, 1, 1, HITTER_A, PITCHER_Y, "swinging_strike"),
    ]
    path = tmp_path / "pitches" / "season=2024" / "chunk.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return tmp_path


def test_pitcher_tokens_follow_the_pitcher_with_the_same_columns(store):
    hitter = build_pitch_tokens(store)
    pitcher = build_pitch_tokens(store, side="pitcher")
    assert list(hitter.columns) == list(pitcher.columns)
    assert set(TOKEN_FEATURES) <= set(pitcher.columns)
    # The same three pitches, grouped by who threw them.
    assert pitcher["player_id"].tolist() == [PITCHER_X, PITCHER_X, PITCHER_Y]
    assert hitter["player_id"].tolist() == [HITTER_A, HITTER_A, HITTER_B]
    assert sorted(pitcher["outcome"]) == sorted(hitter["outcome"])
    # gap: days since the same player's previous pitch. A saw both games; X and Y one.
    assert hitter["gap"].tolist()[1] > 0
    assert (pitcher["gap"] == 0).all()


def test_an_unknown_side_is_refused(store):
    with pytest.raises(ValueError, match="unknown side"):
        build_pitch_tokens(store, side="catcher")


def test_players_split_into_two_fixed_halves():
    pytest.importorskip("torch")
    from fantasy_baseball.hitter_ros.pretrain import player_half

    ids = np.arange(400_000, 700_000, 7)
    half = player_half(ids)
    assert set(np.unique(half)) == {0, 1}
    assert abs(half.mean() - 0.5) < 0.01
    # The same id lands in the same half however it is passed in.
    np.testing.assert_array_equal(player_half(pd.Series(ids)), half)
    np.testing.assert_array_equal(player_half(ids[::-1]), half[::-1])
    # Neighbouring ids are not split by parity or by blocks.
    assert 0.3 < (half[1:] != half[:-1]).mean() < 0.7


def test_the_side_is_recorded_so_the_two_files_are_told_apart():
    from fantasy_baseball.hitter_ros.pitch_tokens import token_options, token_side

    # Same pitches, same layout: only the recorded side tells a pitcher file from a hitter's.
    assert token_options("statcast") == {"zone": "statcast"}  # hitter files, as before
    pitcher = token_options("statcast", "pitcher")
    assert pitcher != token_options("statcast")
    assert token_side(pitcher) == "pitcher"
    assert token_side(token_options("fixed")) == token_side({}) == "hitter"
    with pytest.raises(ValueError, match="unknown side"):
        token_options("statcast", "catcher")
