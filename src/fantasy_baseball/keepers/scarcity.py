"""Positional scarcity measured on the projection's OWN scale.

`sgp.replacement`'s floors are draft-time waiver lines: what a freely available
player at each position produces over a full season. They are the right answer for
a draft board, where an ace projects ~20 SGP and the floors sit near 8-10.

They are the wrong answer for `keepers.projection`, whose output is a REGRESSED
conditional mean topping out near 13 for hitters with a median around 7. Against
that scale the waiver floors sit above the median player, so more than half the
qualified pool prices below replacement -- a statement about the two scales, not
about the players.

Nor is the gap uniform across positions, and no mechanism is asserted for that
here: `REPLACEMENT_BY_POSITION` is itself calibrated from this league's own
un-rostered players, so it is NOT blind to how many of each position get started.
The two tables simply disagree by position, most of all at outfield, and
`--scarcity` prints the disagreement rather than explaining it.

This module measures replacement the other way, natively and on whatever scale it
is handed: fill every starting slot in the league with the best eligible player,
then a position's floor is the best player who did NOT win a job anywhere. That is
the player you would actually be forced to field, which is what the adjustment is
supposed to price.

Measured this way the draft-time table it replaces was right in SHAPE -- catcher is
genuinely the scarcest slot -- and about 1.4x too wide, almost entirely at outfield.

A warning for anyone tempted to validate these against realized value: you cannot.
Regressing next-season SGP on the credit returns a slope near zero, and that is the
CORRECT result, not a refutation -- a catcher who earns 10 SGP earns 10 SGP, and
scarcity is about the alternative you would have been forced to field instead, which
never appears in his own line. An earlier version of this model cited exactly that
null slope as evidence the spread was unsupported. It is not evidence either way.
The measurement has to be structural, which is what `--scarcity` does.
"""

from __future__ import annotations

import pandas as pd

from fantasy_baseball.models.positions import BENCH_SLOTS, HITTER_ELIGIBLE, Position
from fantasy_baseball.utils.constants import compute_starters_per_position
from fantasy_baseball.utils.positions import can_fill_slot

# Mean-centred credit per slot, measured on `keepers.projection`'s scale over the
# four COMPLETE seasons 2022-2025, each against its OWN >= 10-game position
# eligibility (`keepers.appearances`), not the current Yahoo map. Regenerate with
# `keeper_rankings.py --scarcity`, which prints the per-season centred credit these
# average -- that is where the year-to-year spread is read; it is deliberately not
# restated here, because nothing in this file would regenerate a number written in it.
#
# One "P" entry by construction: the pool has a single starting slot type, so there
# is no role split to get wrong here, which is why the table needs no SP/RP merge
# and why `build` passes no `role_ip`. If the league ever splits P into SP/RP
# slots, regenerating this table will produce SP and RP keys and `calculate_var`
# will start routing by role -- at which point `build` MUST pass a full-season
# equivalent IP, not the to-date total, or every under-pace starter is priced as
# a reliever mid-season. See `sgp.var.calculate_var`.
# The league shape the table above was measured under. Unlike `SGP_FIT`, which
# depends only on append-only history and can merely go stale, these credits
# depend on MUTABLE config: change `roster_slots` or `num_teams` and they are not
# stale but wrong, with no signal. The largest credit ("P") is also the most
# exposed, being measured against 9 x 10 = 90 starting pitcher slots.
# `test_scarcity.py` fails if config drifts from this.
MEASURED_UNDER: dict[str, int] = {
    "C": 10,
    "1B": 10,
    "2B": 10,
    "3B": 10,
    "SS": 10,
    "IF": 10,
    "OF": 40,
    "UTIL": 20,
    "P": 90,
}

NATIVE_CREDITS: dict[str, float] = {
    "P": 1.440,
    "C": 0.298,
    "1B": -0.169,
    "SS": -0.208,
    "OF": -0.285,
    "2B": -0.327,
    "3B": -0.345,
    "UTIL": -0.404,
}

# `calculate_var` SUBTRACTS what it is handed, so the levels it wants are the
# negated credits. Kept as a function rather than a second literal so the two can
# never disagree.


def credit_levels() -> dict[str, float]:
    """`NATIVE_CREDITS` in the sign `calculate_var` expects."""
    return {slot: -credit for slot, credit in NATIVE_CREDITS.items()}


# Slots that absorb players from several positions. Filled AFTER the dedicated
# slots so a catcher is never burned on a UTIL spot he was not needed in -- that
# would understate catcher scarcity, the thing being measured. What each slot
# ACCEPTS is not defined here: `can_fill_slot` owns that, and it already handles
# Yahoo's mixed-case spellings ("Util") which a bare string comparison would miss.
FLEX_SLOTS = ("IF", "UTIL")


def slot_capacities(roster_slots: dict[str, int], teams: int) -> dict[str, int]:
    """League-wide starting slots, every bench and IL variant excluded.

    A bench spot is not a slot a keeper decision competes for: the question is who
    STARTS, so the marginal player is the last one who cracks a lineup. Filters on
    `BENCH_SLOTS` rather than a local ("BN", "IL") tuple, which misses IL+, DL and
    DL+ -- and `can_fill_slot` returns True for all of those, so one slipping
    through would silently absorb players and lift every floor.
    """
    benched = {slot.value for slot in BENCH_SLOTS}
    return {
        slot: count
        for slot, count in compute_starters_per_position(roster_slots, teams).items()
        if slot not in benched and count > 0
    }


def _fill_order(capacities: dict[str, int]) -> list[str]:
    """Dedicated slots before flex ones, scarcest dedicated slot first.

    Order matters: filling UTIL early would let it swallow a scarce catcher and
    flatten the very difference this measures.
    """
    dedicated = sorted((s for s in capacities if s not in FLEX_SLOTS), key=lambda s: capacities[s])
    # A multi-position slot that is not declared flex would sort in here by
    # capacity and run BEFORE catcher, swallowing the scarce catchers and
    # inverting the effect this measures. Refuse rather than mis-order.
    for slot in dedicated:
        if slot != Position.P.value and _accepts_many(slot):
            raise ValueError(
                f"{slot!r} accepts more than one position but is not in FLEX_SLOTS; "
                "add it there or the fill order will understate catcher scarcity"
            )
    flex = [s for s in FLEX_SLOTS if s in capacities]
    return [*dedicated, *flex]


def _accepts_many(slot: str) -> bool:
    """Whether `slot` takes players from more than one position."""
    return sum(1 for position in HITTER_ELIGIBLE if can_fill_slot([position], slot)) > 1


def marginal_starter_floors(
    values: pd.Series,
    eligibility: dict[object, set[str]],
    capacities: dict[str, int],
) -> dict[str, float]:
    """Each position's floor: the best player at it who starts NOWHERE.

    `values` is any per-player quantity the floors should be denominated in --
    pass `proj_sgp` to get floors on the projection's scale. `eligibility` maps the
    same index to that player's slots. Positions with no unrostered eligible player
    are omitted rather than guessed at.

    Greedy by descending value. A greedy fill is not the optimal assignment, but it
    only errs by leaving a slot to a slightly worse player, which moves a floor by
    less than the year-to-year noise in the floors themselves.
    """
    ranked = values.dropna().sort_values(ascending=False)
    remaining = dict(capacities)
    started: set[object] = set()
    for slot in _fill_order(capacities):
        if remaining.get(slot, 0) <= 0:
            continue
        for player in ranked.index:
            if remaining[slot] <= 0:
                break
            if player in started:
                continue
            if can_fill_slot(eligibility.get(player, ()), slot):
                started.add(player)
                remaining[slot] -= 1

    floors: dict[str, float] = {}
    for slot in capacities:
        if slot in FLEX_SLOTS:
            continue
        for player in ranked.index:
            if player not in started and can_fill_slot(eligibility.get(player, ()), slot):
                floors[slot] = float(ranked.loc[player])
                break
    # UTIL is a flex slot with no dedicated marginal starter, but every DH-only bat
    # is priced against it, so its floor is the best leftover who can fill UTIL --
    # found the same way as every dedicated slot. A genuinely UTIL-only player (a real
    # DH) is eligible for no per-position floor, so the old `max(dedicated)` could
    # never let him set UTIL even when he was the best hitter starting nowhere. Like
    # any slot, UTIL is omitted when nobody is left over to price it -- and since every
    # dedicated-slot leftover is itself UTIL-eligible, that only happens when no hitter
    # is left over at all.
    if capacities.get("UTIL", 0) > 0:
        util_floor = next(
            (
                float(ranked.loc[player])
                for player in ranked.index
                if player not in started and can_fill_slot(eligibility.get(player, ()), "UTIL")
            ),
            None,
        )
        if util_floor is not None:
            floors["UTIL"] = util_floor
    return floors


def centred_credits(floors: dict[str, float]) -> dict[str, float]:
    """Floors turned into the additive credit `build` applies, mean-centred.

    A display offset, not a model decision: subtracting a constant shared by every
    slot leaves every gap, the ordering and every P(top-N) untouched, and only makes
    the column read positive for the scarce slots. Which slots enter the mean is
    therefore arbitrary and cannot change a conclusion.
    """
    if not floors:
        return {}
    average = sum(floors.values()) / len(floors)
    return {slot: average - level for slot, level in floors.items()}
