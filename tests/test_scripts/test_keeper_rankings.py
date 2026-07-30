"""Unit tests for the pure pricing helpers in the keeper-rankings script.

`_slots_for` exists because a pitcher was being priced against a hitter floor.
"""

import pandas as pd
import pytest

from fantasy_baseball.models.player import PlayerType
from scripts import keeper_rankings as module
from scripts.keeper_rankings import FALLBACK_POS, _dedupe_two_way, _slots_for

POSITIONS = {
    "shohei ohtani": ["UTIL"],
    "tarik skubal": ["P"],
    "ivan herrera": ["C", "UTIL"],
    "otto lopez": ["2B", "SS", "IF", "UTIL"],
    "someone hurt": ["OF", "IL"],
}


def test_a_util_only_pitcher_is_still_priced_as_a_pitcher():
    """Yahoo lists Ohtani as UTIL. In the pitcher pool that sent his pitching
    projection to the UTIL floor via calculate_var's hitter branch."""
    assert _slots_for(POSITIONS, "Shohei Ohtani", PlayerType.PITCHER) == FALLBACK_POS["pitcher"]


def test_a_pitcher_only_player_is_not_priced_as_a_hitter():
    assert _slots_for(POSITIONS, "Tarik Skubal", PlayerType.HITTER) == FALLBACK_POS["hitter"]


def test_a_hitter_keeps_only_his_batting_slots():
    assert _slots_for(POSITIONS, "Otto Lopez", PlayerType.HITTER) == ["2B", "SS", "IF", "UTIL"]


def test_a_bench_or_il_token_is_not_a_position_to_price_against():
    """An allowlist, not a denylist: IL is not in HITTER_ELIGIBLE."""
    assert _slots_for(POSITIONS, "Someone Hurt", PlayerType.HITTER) == ["OF"]


def test_an_unknown_player_falls_back_to_the_deepest_floor():
    assert _slots_for(POSITIONS, "Nobody At All", PlayerType.HITTER) == FALLBACK_POS["hitter"]


def _board(ids, names, proj_var):
    return pd.DataFrame({"name": names, "proj_var": proj_var}, index=pd.Index(ids, name="mlbam_id"))


def test_a_two_way_player_collapses_to_his_better_side():
    board = _board([660271, 660271, 592450], ["Ohtani", "Ohtani", "Judge"], [11.8, 7.8, 10.0])
    out = _dedupe_two_way(board)
    assert len(out) == 2
    assert out.loc[out["name"] == "Ohtani", "proj_var"].tolist() == [11.8]


def test_two_different_players_sharing_a_name_both_survive():
    """2022 had two Will Smiths, two Diego Castillos and two Luis Garcias. A
    name-keyed drop deletes a real rival, and probability_top_n then spreads the
    same slot mass over fewer people -- inflating everyone while the sum-to-slots
    check still passes."""
    board = _board([519293, 669257], ["Will Smith", "Will Smith"], [9.0, 8.0])
    assert len(_dedupe_two_way(board)) == 2


def test_dedupe_leaves_a_board_with_no_duplicates_alone():
    board = _board([1, 2, 3], ["A", "B", "C"], [3.0, 2.0, 1.0])
    assert _dedupe_two_way(board)["name"].tolist() == ["A", "B", "C"]


def test_dedupe_refuses_a_frame_that_lost_its_id_index():
    """The dedupe's correctness rests on the index being mlbam_id, and a synthetic
    index passes the tests above just as well -- so a reset_index upstream would
    silently restore the double-count with everything green."""
    board = pd.DataFrame({"name": ["A", "A"], "proj_var": [2.0, 1.0]})
    with pytest.raises(ValueError, match="mlbam_id"):
        _dedupe_two_way(board)


def _stub_pool(monkeypatch, *, kind, raw, feat):
    """Drive `_positional_residuals` off synthetic frames.

    It is the only regenerator of the positional and per-role evidence
    `projection.scarcity_floors` argues from, and it is otherwise reachable only
    by running `--study` by hand -- so a renamed BBRef column would break the
    evidence silently while the pricing it justifies kept working.
    """
    monkeypatch.setattr(module, "_raw", lambda year, table: raw)
    monkeypatch.setattr(module, "_transition", lambda year, k, denoms: feat)
    monkeypatch.setattr(module, "composite_pct", lambda frame, k, weights=None: frame["c"])
    monkeypatch.setattr(module, "ALL_TRANSITION_YEARS", (2022,))
    return module._positional_residuals(kind, None, ({"cj abrams": ["SS"]}, {"SS": -1.0}))


def test_pitchers_are_grouped_by_role_not_by_the_single_p_slot(monkeypatch):
    """Every pitcher shares one slot, so grouping on it would print one row and
    regenerate nothing. The SP/RP split is what the merged floor is an argument
    about, so it is what has to come out."""
    raw = pd.DataFrame(
        {"mlbID": ["a", "b"], "Name": ["A Starter", "A Closer"], "G": [30, 60], "GS": [30, 0]}
    )
    feat = pd.DataFrame({"c": [0.9, 0.8], "target_sgp": [10.0, 4.0]}, index=["a", "b"])
    out = _stub_pool(monkeypatch, kind=PlayerType.PITCHER, raw=raw, feat=feat)
    assert list(out["slot"]) == ["SP", "RP"]


def test_a_swingman_at_exactly_half_starts_counts_as_a_starter(monkeypatch):
    """Pins the same >= 0.5 boundary `_role_equivalent_ip` uses; the two must not
    drift apart or the diagnostic stops describing the pricing."""
    raw = pd.DataFrame({"mlbID": ["a"], "Name": ["A Swingman"], "G": [20], "GS": [10]})
    feat = pd.DataFrame({"c": [0.5], "target_sgp": [5.0]}, index=["a"])
    assert list(_stub_pool(monkeypatch, kind=PlayerType.PITCHER, raw=raw, feat=feat)["slot"]) == [
        "SP"
    ]


def test_a_pitcher_who_never_appeared_is_a_reliever_not_a_nan_group(monkeypatch):
    """G == 0 divides by zero. Left as NaN the comparison is False anyway, but
    only by accident -- fillna makes the RP routing the stated behavior."""
    raw = pd.DataFrame({"mlbID": ["a"], "Name": ["A Ghost"], "G": [0], "GS": [0]})
    feat = pd.DataFrame({"c": [0.5], "target_sgp": [0.0]}, index=["a"])
    assert list(_stub_pool(monkeypatch, kind=PlayerType.PITCHER, raw=raw, feat=feat)["slot"]) == [
        "RP"
    ]


def test_hitters_are_grouped_by_slot_and_credited_from_the_floor(monkeypatch):
    """The hitter branch is the one whose spread is left in place, so its credit
    column is what the `resid ~ credit` slope regresses against."""
    raw = pd.DataFrame({"mlbID": ["a"], "Name": ["CJ Abrams"], "G": [150], "GS": [150]})
    feat = pd.DataFrame({"c": [0.9], "target_sgp": [12.0]}, index=["a"])
    out = _stub_pool(monkeypatch, kind=PlayerType.HITTER, raw=raw, feat=feat)
    assert list(out["slot"]) == ["SS"]
    assert out["credit"].iloc[0] == 1.0
    assert bool(out["mapped"].iloc[0])
