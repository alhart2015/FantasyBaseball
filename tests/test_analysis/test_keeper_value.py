import pandas as pd

from fantasy_baseball.analysis import keeper_value as kv
from fantasy_baseball.draft.board import build_board_from_frames
from fantasy_baseball.models.player import PlayerType


def _tiny_scale_and_board():
    # build_board_from_frames scores every row via calculate_player_sgp, which
    # dispatches on player_type -- so the input frames must carry it (real frames
    # get it from parse_*_csv / get_blended_projections).
    hitters = pd.DataFrame(
        [
            {
                "name": "Star Bat",
                "r": 100,
                "hr": 35,
                "rbi": 100,
                "sb": 15,
                "ab": 550,
                "h": 165,
                "avg": 0.300,
                "player_type": PlayerType.HITTER,
            },
            {
                "name": "Meh Bat",
                "r": 60,
                "hr": 12,
                "rbi": 55,
                "sb": 5,
                "ab": 480,
                "h": 120,
                "avg": 0.250,
                "player_type": PlayerType.HITTER,
            },
        ]
    )
    pitchers = pd.DataFrame(
        [
            {
                "name": "Ace Arm",
                "w": 15,
                "k": 220,
                "sv": 0,
                "ip": 190,
                "era": 3.10,
                "whip": 1.05,
                "player_type": PlayerType.PITCHER,
            },
            {
                "name": "Closer Guy",
                "w": 4,
                "k": 90,
                "sv": 35,
                "ip": 65,
                "era": 2.70,
                "whip": 1.00,
                "player_type": PlayerType.PITCHER,
            },
        ]
    )
    positions = {"Star Bat": ["OF"], "Meh Bat": ["2B"], "Ace Arm": ["SP"], "Closer Guy": ["RP"]}
    board, scale = build_board_from_frames(hitters, pitchers, positions)
    return board, scale


def test_clamp_ratio_clamps_to_band():
    band = (0.25, 2.5)
    assert kv._clamp_ratio(10.0, 2.0, band, kv.EPS) == 2.5  # 5.0 -> clamp hi
    assert kv._clamp_ratio(1.0, 10.0, band, kv.EPS) == 0.25  # 0.1 -> clamp lo
    assert kv._clamp_ratio(3.0, 4.0, band, kv.EPS) == 0.75  # in-band


def test_clamp_ratio_none_on_tiny_denominator():
    assert kv._clamp_ratio(5.0, 0.0, (0.25, 2.5), kv.EPS) is None


def test_scale_line_scales_scored_fields_and_keeps_flat_on_none():
    anchor = {"r": 100.0, "hr": 30.0, "rbi": 90.0, "sb": 10.0, "ab": 500.0, "avg": 0.280}
    zips_base = {"r": 90.0, "hr": 25.0, "rbi": 80.0, "sb": 0.0, "ab": 450.0, "avg": 0.270}
    zips_y = {"r": 99.0, "hr": 20.0, "rbi": 88.0, "sb": 5.0, "ab": 441.0, "avg": 0.2565}
    out = kv._scale_line(anchor, zips_base, zips_y, "hitter", (0.25, 2.5), kv.EPS)
    assert out["r"] == 100.0 * (99.0 / 90.0)  # 1.10
    assert out["hr"] == 30.0 * (20.0 / 25.0)  # 0.80
    assert round(out["avg"], 4) == round(0.280 * (0.2565 / 0.270), 4)  # rate scaled directly
    assert out["sb"] == 10.0  # zips_base sb == 0 -> ratio None -> flat


def test_scale_line_holds_flat_on_missing_or_nan_numerator():
    # A blank/NaN or absent out-year cell carries no aging signal -> hold flat,
    # symmetric with a missing base (NOT floored to the band low of 0.25).
    anchor = {"r": 100.0, "hr": 30.0, "ab": 500.0}
    zips_base = {"r": 100.0, "hr": 30.0, "ab": 500.0}
    zips_y = {"r": 110.0, "hr": float("nan")}  # hr NaN, ab absent
    out = kv._scale_line(anchor, zips_base, zips_y, "hitter", (0.25, 2.5), kv.EPS)
    assert abs(out["r"] - 110.0) < 1e-9  # 100 * (110/100)
    assert out["hr"] == 30.0  # NaN numerator -> flat, not 30*0.25
    assert out["ab"] == 500.0  # absent numerator -> flat


def test_value_of_line_matches_board_var():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    line = row.to_dict()
    v = kv._value_of_line(line, list(row["positions"]), row["player_type"], scale)
    assert abs(v - float(row["var"])) < 1e-9


def test_per_year_var_missing_out_year_is_zero_and_flagged():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    # ZiPS base present, 2027 present, 2028 missing.
    zips_by_year = {
        2026: anchor,
        2027: {**anchor, "hr": anchor["hr"] * 0.9},
        2028: None,
    }
    pyv, flags = kv.per_year_var(
        anchor, list(row["positions"]), row["player_type"], zips_by_year, scale
    )
    assert set(pyv) == {2026, 2027, 2028}
    assert pyv[2028] == 0.0
    assert "no_zips_2028" in flags
    assert abs(pyv[2026] - float(row["var"])) < 1e-9  # base year == board var


def test_per_year_var_low_pt_base_falls_back_to_approach_a():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    # ZiPS base line has AB below the 100 default -> out-years use approach A.
    tiny_base = {**anchor, "ab": 40}
    zips_2027 = {**anchor, "hr": 20}
    zips_by_year = {2026: tiny_base, 2027: zips_2027, 2028: zips_2027}
    pyv, flags = kv.per_year_var(
        anchor, list(row["positions"]), row["player_type"], zips_by_year, scale
    )
    assert "fallback_A" in flags
    # Approach A: out-year V equals scoring the raw ZiPS 2027 line directly.
    expected = kv._value_of_line(zips_2027, list(row["positions"]), row["player_type"], scale)
    assert abs(pyv[2027] - expected) < 1e-9


def test_discounted_total_weights_by_year():
    pyv = {2026: 10.0, 2027: 10.0, 2028: 10.0}
    assert abs(kv.discounted_total(pyv, 2026, 0.8, 3) - (10.0 + 8.0 + 6.4)) < 1e-9


def test_out_year_share_scales_with_discount_and_guards():
    pyv = {2026: 10.0, 2027: 10.0, 2028: 10.0}
    t_low = kv.discounted_total(pyv, 2026, 0.5, 3)
    t_high = kv.discounted_total(pyv, 2026, 0.9, 3)
    s_low = kv.out_year_share(pyv, 2026, t_low)
    s_high = kv.out_year_share(pyv, 2026, t_high)
    assert s_low is not None and s_high is not None
    assert s_high > s_low  # out-years weigh more at a shallower discount
    # guard: a total at/below eps_share -> None, never a blown-up ratio
    assert kv.out_year_share(pyv, 2026, t_low, eps_share=1e9) is None


def test_keeper_value_horizon_1_equals_board_var():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    res = kv.keeper_value(
        row["player_id"],
        row["name"],
        anchor,
        list(row["positions"]),
        row["player_type"],
        {2026: anchor},
        scale,
        horizon=1,
    )
    assert abs(res.total - float(row["var"])) < 1e-9  # currency parity


def test_youth_premium_emerges_and_widens_as_discount_shallows():
    """Two players, identical 2026 VAR, different ZiPS decline curves.
    The flatter (younger) curve ranks higher, and the gap widens as discount rises."""
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    pt, positions = row["player_type"], list(row["positions"])

    young = {2026: anchor, 2027: anchor, 2028: anchor}
    decayed_27 = {
        **anchor,
        "r": anchor["r"] * 0.85,
        "hr": anchor["hr"] * 0.85,
        "rbi": anchor["rbi"] * 0.85,
        "sb": anchor["sb"] * 0.85,
    }
    decayed_28 = {
        **anchor,
        "r": anchor["r"] * 0.70,
        "hr": anchor["hr"] * 0.70,
        "rbi": anchor["rbi"] * 0.70,
        "sb": anchor["sb"] * 0.70,
    }
    old = {2026: anchor, 2027: decayed_27, 2028: decayed_28}

    def total(zbys, discount):
        return kv.keeper_value(
            "y", "y", anchor, positions, pt, zbys, scale, discount=discount
        ).total

    gap_steep = total(young, 0.60) - total(old, 0.60)
    gap_shallow = total(young, 0.90) - total(old, 0.90)
    assert gap_steep > 0  # young always wins
    assert gap_shallow > gap_steep  # advantage grows as out-years count more


def test_keeper_value_none_share_when_total_below_eps():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Meh Bat"].iloc[0]  # low/near-replacement value
    anchor = row.to_dict()
    res = kv.keeper_value(
        row["player_id"],
        row["name"],
        anchor,
        list(row["positions"]),
        row["player_type"],
        {2026: anchor, 2027: anchor, 2028: anchor},
        scale,
        eps_share=1e9,  # force the guard
    )
    assert res.pct_from_out_years is None


def test_keeper_value_zero_year_is_kept_not_dropped():
    """A year whose V is exactly 0.0 is a real value: it stays in per_year_var
    and participates in the discounted sum (numeric-default guard)."""
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    anchor = row.to_dict()
    res = kv.keeper_value(
        row["player_id"],
        row["name"],
        anchor,
        list(row["positions"]),
        row["player_type"],
        {2026: anchor, 2027: anchor, 2028: None},
        scale,
        discount=1.0,  # 2028 missing -> 0.0
    )
    assert res.per_year_var[2028] == 0.0
    assert 2028 in res.per_year_var  # not dropped
    assert abs(res.total - (res.per_year_var[2026] + res.per_year_var[2027] + 0.0)) < 1e-9


def test_pct_from_saves_zero_for_hitter():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Star Bat"].iloc[0]
    assert kv.pct_from_saves(row.to_dict(), "hitter", scale) == 0.0


def test_pct_from_saves_positive_for_closer_and_beats_starter():
    # Robust relational assertion (avoids a magic threshold that depends on the
    # tiny fixture's pool-derived denominators): a closer's saves share is
    # positive and strictly greater than a save-less starter's (which is 0.0).
    board, scale = _tiny_scale_and_board()
    closer = board[board["name"] == "Closer Guy"].iloc[0]
    ace = board[board["name"] == "Ace Arm"].iloc[0]
    closer_share = kv.pct_from_saves(closer.to_dict(), "pitcher", scale)
    ace_share = kv.pct_from_saves(ace.to_dict(), "pitcher", scale)
    assert ace_share == 0.0  # no saves -> 0 share
    assert closer_share is not None and closer_share > 0.0
    assert closer_share > ace_share


def test_pct_from_saves_none_when_sgp_below_eps():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Closer Guy"].iloc[0]
    assert kv.pct_from_saves(row.to_dict(), "pitcher", scale, eps_share=1e9) is None


def test_keeper_value_populates_pct_from_saves():
    board, scale = _tiny_scale_and_board()
    row = board[board["name"] == "Closer Guy"].iloc[0]
    anchor = row.to_dict()
    res = kv.keeper_value(
        row["player_id"],
        row["name"],
        anchor,
        list(row["positions"]),
        "pitcher",
        {2026: anchor, 2027: anchor, 2028: anchor},
        scale,
    )
    assert res.pct_from_saves is not None and res.pct_from_saves > 0.0
