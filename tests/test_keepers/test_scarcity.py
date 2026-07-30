"""Positional scarcity measured on the projection's own scale.

The floors this module produces decide who the top keeper in the league is -- the
catcher credit alone is worth over 1.5 SGP -- so the fill order and the flex-slot
rules are load-bearing, not bookkeeping.
"""

import pandas as pd
import pytest

from fantasy_baseball.keepers.scarcity import (
    NATIVE_CREDITS,
    centred_credits,
    credit_levels,
    marginal_starter_floors,
    slot_capacities,
)

# One team's worth, scaled up by `slot_capacities`.
SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "OF": 4, "UTIL": 2, "BN": 2, "IL": 2}


def test_bench_and_il_are_not_slots_a_keeper_competes_for():
    """The marginal player is the last one who cracks a LINEUP. Counting bench
    spots would push replacement deeper and inflate every credit."""
    caps = slot_capacities(SLOTS, 10)
    assert "BN" not in caps and "IL" not in caps
    assert caps["OF"] == 40
    assert caps["C"] == 10


def test_a_zero_count_slot_is_dropped_not_kept_as_zero():
    assert "C" not in slot_capacities({"C": 0, "OF": 3}, 10)


def _floors(values, eligibility, caps):
    return marginal_starter_floors(pd.Series(values), eligibility, caps)


def test_the_floor_is_the_best_player_who_starts_nowhere():
    """Two 1B slots and three 1B: the floor is the third, not the second."""
    floors = _floors(
        {0: 10.0, 1: 9.0, 2: 8.0},
        {0: {"1B"}, 1: {"1B"}, 2: {"1B"}},
        {"1B": 2},
    )
    assert floors["1B"] == pytest.approx(8.0)


def test_a_scarce_position_gets_a_lower_floor_than_a_deep_one():
    """The whole point: one catcher slot with two catchers, one OF slot with four
    outfielders, so the replacement OF is better than the replacement catcher."""
    values = {0: 10.0, 1: 4.0, 2: 10.0, 3: 9.0, 4: 8.5, 5: 8.0}
    elig = {0: {"C"}, 1: {"C"}, 2: {"OF"}, 3: {"OF"}, 4: {"OF"}, 5: {"OF"}}
    floors = _floors(values, elig, {"C": 1, "OF": 1})
    assert floors["C"] == pytest.approx(4.0)
    assert floors["OF"] == pytest.approx(9.0)
    assert centred_credits(floors)["C"] > centred_credits(floors)["OF"]


def test_util_does_not_swallow_a_catcher_before_his_own_slot_is_filled():
    """Fill order is the subtle part. If UTIL ran first it would take the best
    catcher, leaving the C slot to the scrub and making catcher look DEEP -- the
    exact inversion of the effect being measured."""
    values = {0: 10.0, 1: 3.0, 2: 9.9}
    elig = {0: {"C"}, 1: {"C"}, 2: {"1B"}}
    floors = _floors(values, elig, {"C": 1, "UTIL": 1})
    # the 10.0 catcher takes C, the 9.9 first baseman takes UTIL, so the 3.0
    # catcher is the one left over
    assert floors["C"] == pytest.approx(3.0)


def test_the_infield_flex_slot_accepts_any_infielder():
    values = {0: 10.0, 1: 9.0, 2: 8.0}
    elig = {0: {"SS"}, 1: {"2B"}, 2: {"SS"}}
    floors = _floors(values, elig, {"SS": 1, "IF": 1})
    # SS takes the 10.0, IF absorbs the 9.0 second baseman, leaving the 8.0 SS
    assert floors["SS"] == pytest.approx(8.0)


def test_util_is_priced_at_the_deepest_hitter_floor():
    """It is a flex slot with no marginal starter of its own, but every DH-only bat
    is priced against it, so it inherits the easiest position to replace at."""
    # Deep enough that a leftover survives at BOTH positions after UTIL takes one.
    values = {0: 10.0, 1: 4.0, 2: 10.0, 3: 9.0, 4: 8.0}
    elig = {0: {"C"}, 1: {"C"}, 2: {"OF"}, 3: {"OF"}, 4: {"OF"}}
    floors = _floors(values, elig, {"C": 1, "OF": 1, "UTIL": 0})
    assert "UTIL" not in floors  # zero capacity is not a slot
    floors = _floors(values, elig, {"C": 1, "OF": 1, "UTIL": 1})
    assert floors["C"] == pytest.approx(4.0)
    assert floors["OF"] == pytest.approx(8.0)
    # the deepest hitter floor is OF's, and UTIL inherits it rather than catcher's
    assert floors["UTIL"] == pytest.approx(8.0)


def test_a_position_with_nobody_left_over_is_omitted_not_guessed():
    floors = _floors({0: 10.0}, {0: {"C"}}, {"C": 1})
    assert "C" not in floors


def test_the_pitcher_pool_needs_no_role_split():
    """One starting slot type means one floor by construction -- the reason this
    table cannot repeat the SP/RP double-count that the old draft-time floors did."""
    assert "SP" not in NATIVE_CREDITS
    assert "RP" not in NATIVE_CREDITS
    assert "P" in NATIVE_CREDITS


def test_credits_are_mean_centred_so_they_add_to_nothing_overall():
    floors = {"C": 7.0, "OF": 9.0, "SS": 8.0}
    assert sum(centred_credits(floors).values()) == pytest.approx(0.0)


def test_centring_leaves_every_gap_untouched():
    """It is a display offset. Any two players' difference must be unchanged, or
    the ranking and every P(top-N) would move."""
    floors = {"C": 7.0, "OF": 9.0, "SS": 8.0}
    credits = centred_credits(floors)
    assert credits["C"] - credits["OF"] == pytest.approx(floors["OF"] - floors["C"])
    assert credits["SS"] - credits["OF"] == pytest.approx(floors["OF"] - floors["SS"])


def test_centring_an_empty_table_is_empty():
    assert centred_credits({}) == {}


def test_the_levels_handed_to_calculate_var_are_the_negated_credits():
    """`calculate_var` SUBTRACTS what it is given, so a scarce position must arrive
    as a NEGATIVE level to come out as a positive credit. Getting this backwards
    would invert the entire positional adjustment silently."""
    levels = credit_levels()
    assert levels["C"] == pytest.approx(-NATIVE_CREDITS["C"])
    assert levels["C"] < 0 < levels["OF"]


def test_the_shipped_table_prices_catcher_as_the_scarcest_hitter_slot():
    """A regression guard on the conclusion of the measurement, not on its digits."""
    hitters = {s: c for s, c in NATIVE_CREDITS.items() if s != "P"}
    assert max(hitters, key=lambda s: hitters[s]) == "C"
    assert hitters["C"] > 0 > max(c for s, c in hitters.items() if s != "C")
