"""Unit tests for the pure pricing helpers in the keeper-rankings script.

`_slots_for` and `_role_equivalent_ip` are the two functions with real branch
logic, and both existed to fix a bug that shipped: a pitcher priced against a
hitter floor, and 120 of 198 pitchers priced against the wrong pitcher floor.
"""

import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import STARTER_IP_THRESHOLD
from scripts.keeper_rankings import FALLBACK_POS, _role_equivalent_ip, _slots_for

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


def test_a_starter_routes_above_the_threshold_and_a_reliever_below():
    """Role comes from start share, so a partial season cannot flip it. The values
    only have to straddle STARTER_IP_THRESHOLD."""
    frame = pd.DataFrame({"start_share": [1.0, 0.6, 0.5, 0.4, 0.0]})
    roles = _role_equivalent_ip(frame, PlayerType.PITCHER)
    assert [r >= STARTER_IP_THRESHOLD for r in roles] == [True, True, True, False, False]


def test_a_starter_with_few_innings_so_far_still_routes_as_a_starter():
    """The bug this replaced: 90 innings in July read as a reliever."""
    frame = pd.DataFrame({"start_share": [1.0], "pt": [90.0]})
    assert _role_equivalent_ip(frame, PlayerType.PITCHER)[0] >= STARTER_IP_THRESHOLD


def test_hitters_get_no_role_at_all():
    frame = pd.DataFrame({"pt": [600.0, 550.0]})
    assert _role_equivalent_ip(frame, PlayerType.HITTER) == [None, None]


def test_a_pitcher_frame_without_start_share_degrades_to_no_role():
    """Rather than raising: calculate_var then routes off the player's own ip."""
    frame = pd.DataFrame({"pt": [120.0]})
    assert _role_equivalent_ip(frame, PlayerType.PITCHER) == [None]
