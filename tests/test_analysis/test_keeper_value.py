import pandas as pd

from fantasy_baseball.analysis import keeper_value as kv
from fantasy_baseball.draft.board import build_board_from_frames
from fantasy_baseball.models.player import PlayerType


def _tiny_scale_and_board():
    # build_board_from_frames scores every row via calculate_player_sgp, which
    # dispatches on player_type -- so the input frames must carry it (real frames
    # get it from parse_*_csv / get_blended_projections).
    hitters = pd.DataFrame([
        {"name": "Star Bat", "r": 100, "hr": 35, "rbi": 100, "sb": 15, "ab": 550, "h": 165, "avg": 0.300, "player_type": PlayerType.HITTER},
        {"name": "Meh Bat", "r": 60, "hr": 12, "rbi": 55, "sb": 5, "ab": 480, "h": 120, "avg": 0.250, "player_type": PlayerType.HITTER},
    ])
    pitchers = pd.DataFrame([
        {"name": "Ace Arm", "w": 15, "k": 220, "sv": 0, "ip": 190, "era": 3.10, "whip": 1.05, "player_type": PlayerType.PITCHER},
        {"name": "Closer Guy", "w": 4, "k": 90, "sv": 35, "ip": 65, "era": 2.70, "whip": 1.00, "player_type": PlayerType.PITCHER},
    ])
    positions = {"Star Bat": ["OF"], "Meh Bat": ["2B"], "Ace Arm": ["SP"], "Closer Guy": ["RP"]}
    board, scale = build_board_from_frames(hitters, pitchers, positions)
    return board, scale


def test_clamp_ratio_clamps_to_band():
    band = (0.25, 2.5)
    assert kv._clamp_ratio(10.0, 2.0, band, kv.EPS) == 2.5   # 5.0 -> clamp hi
    assert kv._clamp_ratio(1.0, 10.0, band, kv.EPS) == 0.25  # 0.1 -> clamp lo
    assert kv._clamp_ratio(3.0, 4.0, band, kv.EPS) == 0.75   # in-band


def test_clamp_ratio_none_on_tiny_denominator():
    assert kv._clamp_ratio(5.0, 0.0, (0.25, 2.5), kv.EPS) is None


def test_scale_line_scales_scored_fields_and_keeps_flat_on_none():
    anchor = {"r": 100.0, "hr": 30.0, "rbi": 90.0, "sb": 10.0, "ab": 500.0, "avg": 0.280}
    zips_base = {"r": 90.0, "hr": 25.0, "rbi": 80.0, "sb": 0.0, "ab": 450.0, "avg": 0.270}
    zips_y = {"r": 99.0, "hr": 20.0, "rbi": 88.0, "sb": 5.0, "ab": 441.0, "avg": 0.2565}
    out = kv._scale_line(anchor, zips_base, zips_y, "hitter", (0.25, 2.5), kv.EPS)
    assert out["r"] == 100.0 * (99.0 / 90.0)         # 1.10
    assert out["hr"] == 30.0 * (20.0 / 25.0)          # 0.80
    assert round(out["avg"], 4) == round(0.280 * (0.2565 / 0.270), 4)  # rate scaled directly
    assert out["sb"] == 10.0                          # zips_base sb == 0 -> ratio None -> flat


def test_value_of_line_matches_board_var():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    line = row.to_dict()
    v = kv._value_of_line(line, list(row["positions"]), row["player_type"], scale)
    assert abs(v - float(row["var"])) < 1e-9
