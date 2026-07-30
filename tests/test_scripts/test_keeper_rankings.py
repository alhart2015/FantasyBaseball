"""Unit tests for the pure pricing helpers in the keeper-rankings script.

`_slots_for` exists because a pitcher was being priced against a hitter floor.
"""

from fantasy_baseball.models.player import PlayerType
from scripts.keeper_rankings import FALLBACK_POS, _slots_for

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
