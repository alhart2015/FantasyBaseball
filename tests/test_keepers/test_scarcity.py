"""Positional scarcity measured on the projection's own scale.

The floors this module produces decide who the top keeper in the league is -- the
catcher credit alone is worth over 1.5 SGP -- so the fill order and the flex-slot
rules are load-bearing, not bookkeeping.
"""

from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.config import load_config
from fantasy_baseball.keepers.scarcity import (
    MEASURED_UNDER,
    NATIVE_CREDITS,
    centred_credits,
    credit_levels,
    marginal_starter_floors,
    slot_capacities,
)

# One team's worth, scaled up by `slot_capacities`.
SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "OF": 4, "UTIL": 2, "BN": 2, "IL": 2}


def test_measured_under_matches_the_live_config_it_documents():
    """`NATIVE_CREDITS` were measured under `MEASURED_UNDER`'s league shape, and unlike
    `SGP_FIT` (append-only history, can only go stale) they depend on MUTABLE config:
    change `roster_slots` or `num_teams` and the credits are not stale but WRONG, with no
    other signal. `MEASURED_UNDER`'s docstring claims this test enforces that -- so it
    must actually trip when the live config drifts from the shape the credits assume."""
    # Anchored to the repo root via __file__, not cwd: the tripwire must bind to the
    # config, not to where pytest happens to be invoked from.
    config_path = Path(__file__).resolve().parents[2] / "config" / "league.yaml"
    cfg = load_config(config_path)
    assert slot_capacities(cfg.roster_slots, cfg.num_teams) == MEASURED_UNDER


def test_bench_and_il_are_not_slots_a_keeper_competes_for():
    """The marginal player is the last one who cracks a LINEUP. Counting bench
    spots would push replacement deeper and inflate every credit -- and
    `can_fill_slot` returns True for every bench variant, so one slipping through
    would absorb players from every position at once."""
    caps = slot_capacities({**SLOTS, "IL+": 1, "DL": 1}, 10)
    for benched in ("BN", "IL", "IL+", "DL"):
        assert benched not in caps
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


def test_a_yahoo_mixed_case_slot_still_matches():
    """Yahoo writes "Util", not "UTIL". A bare string compare matches nothing, so
    the mixed-case player is never seated and turns up as the floor instead --
    understating it. `can_fill_slot` parses the spelling."""
    values = {0: 10.0, 1: 9.0, 2: 4.0}
    elig = {0: {"1B"}, 1: {"Util", "1B"}, 2: {"1B"}}
    floors = _floors(values, elig, {"1B": 1, "UTIL": 1})
    # 10.0 takes 1B, the mixed-case 9.0 is absorbed by UTIL, so 4.0 is the leftover
    assert floors["1B"] == pytest.approx(4.0)


def test_a_pitcher_is_not_absorbed_by_a_hitter_flex_slot():
    """UTIL takes anyone with HITTER eligibility, not anyone at all. If it swallowed
    the ace, the outfield floor would be measured against a depleted pool."""
    values = {0: 10.0, 1: 9.0, 2: 8.0}
    elig = {0: {"P"}, 1: {"OF"}, 2: {"OF"}}
    floors = _floors(values, elig, {"OF": 1, "UTIL": 1})
    # OF takes 9.0; UTIL must refuse the 10.0 pitcher and take the 8.0 outfielder,
    # leaving no outfielder over -- so OF has no floor and the pitcher is untouched
    assert "OF" not in floors


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


def test_util_floor_can_be_set_by_a_util_only_leftover():
    """A UTIL-only bat (a DH) who is the best leftover must SET the UTIL floor. The
    old code maxed over dedicated floors only, so a player eligible for no dedicated
    slot could never set it, understating UTIL."""
    values = {0: 10.0, 1: 4.0, 2: 10.0, 3: 9.5, 4: 9.0, 5: 3.0}
    elig = {0: {"C"}, 1: {"C"}, 2: {"OF"}, 3: {"DH"}, 4: {"DH"}, 5: {"OF"}}
    floors = _floors(values, elig, {"C": 1, "OF": 1, "UTIL": 1})
    # Fill: C<-10@0, OF<-10@2, UTIL(flex)<-best remaining hitter = 9.5 DH@3.
    # Leftovers: 4.0 C@1, 9.0 DH@4, 3.0 OF@5.
    #   old (max dedicated floors): max(C 4.0, OF 3.0) = 4.0  -- WRONG
    #   new (best UTIL-eligible leftover): the 9.0 DH@4        -- RIGHT
    assert floors["UTIL"] == pytest.approx(9.0)


def test_util_floor_omitted_when_no_leftover():
    """Like any slot, UTIL is omitted when nobody is left over to price it."""
    values = {0: 10.0, 1: 4.0}
    elig = {0: {"C"}, 1: {"C"}}
    floors = _floors(values, elig, {"C": 1, "UTIL": 1})
    # C<-10@0, UTIL(flex)<-4@1. No player is left over, so UTIL has no floor.
    assert "UTIL" not in floors


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
