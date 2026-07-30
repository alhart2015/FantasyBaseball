"""Positional scarcity measured on the projection's OWN scale.

`sgp.replacement`'s floors are draft-time waiver lines: what a freely available
player at each position produces over a full season. They are the right answer for
a draft board, where an ace projects ~20 SGP and the floors sit near 8-10.

They are the wrong answer for `keepers.projection`, whose output is a REGRESSED
conditional mean topping out near 13 for hitters with a median around 7. Against
that scale the waiver floors sit above the median player, so more than half the
qualified pool prices below replacement -- a statement about the two scales, not
about the players. The gap is not uniform either: the waiver concept prices a
position by what a replacement AT that position hits, ignoring how many of them a
league has to start, which penalizes outfielders hardest because ten teams start
forty of them.

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

# Mean-centred credit per slot, measured on `keepers.projection`'s scale over the
# four COMPLETE seasons 2022-2025 (2026 is partial; including it moves nothing by
# more than 0.08, since a credit is a difference and the mid-season level shift is
# common to every slot). Regenerate with `keeper_rankings.py --scarcity`, which also
# prints the per-season table these average -- the catcher figure ranges 0.50 to
# 2.18 year to year, so the average is the number to use and no single season is.
#
# One "P" entry by construction: the pool has a single starting slot type, so there
# is no role split to get wrong here. This is why the table needs no SP/RP merge.
NATIVE_CREDITS: dict[str, float] = {
    "P": 1.446,
    "C": 1.176,
    "1B": -0.347,
    "SS": -0.405,
    "3B": -0.455,
    "2B": -0.465,
    "OF": -0.465,
    "UTIL": -0.485,
}

# `calculate_var` SUBTRACTS what it is handed, so the levels it wants are the
# negated credits. Kept as a function rather than a second literal so the two can
# never disagree.


def credit_levels() -> dict[str, float]:
    """`NATIVE_CREDITS` in the sign `calculate_var` expects."""
    return {slot: -credit for slot, credit in NATIVE_CREDITS.items()}


# Slots that absorb players from several positions, and what each accepts. Filled
# AFTER the dedicated slots so a catcher is never burned on a UTIL spot he was not
# needed in -- that would understate catcher scarcity, the thing being measured.
INFIELD = ("1B", "2B", "3B", "SS")
FLEX_ACCEPTS: dict[str, tuple[str, ...]] = {
    "IF": (*INFIELD, "IF"),
    "UTIL": (),  # empty means "anyone in this pool"
}


def slot_capacities(roster_slots: dict[str, int], teams: int) -> dict[str, int]:
    """League-wide starting slots, bench and IL excluded.

    A bench spot is not a slot a keeper decision competes for: the question is who
    STARTS, so the marginal player is the last one who cracks a lineup.
    """
    return {
        slot: count * teams
        for slot, count in roster_slots.items()
        if slot not in ("BN", "IL") and count > 0
    }


def _fill_order(capacities: dict[str, int]) -> list[str]:
    """Dedicated slots before flex ones, scarcest dedicated slot first.

    Order matters: filling UTIL early would let it swallow a scarce catcher and
    flatten the very difference this measures.
    """
    dedicated = sorted(
        (s for s in capacities if s not in FLEX_ACCEPTS), key=lambda s: capacities[s]
    )
    flex = [s for s in ("IF", "UTIL") if s in capacities]
    return [*dedicated, *flex]


def _accepts(slot: str, eligible: set[str]) -> bool:
    if slot not in FLEX_ACCEPTS:
        return slot in eligible
    allowed = FLEX_ACCEPTS[slot]
    return True if not allowed else bool(eligible.intersection(allowed))


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
            if _accepts(slot, eligibility.get(player, set())):
                started.add(player)
                remaining[slot] -= 1

    floors: dict[str, float] = {}
    for slot in capacities:
        if slot in FLEX_ACCEPTS:
            continue
        for player in ranked.index:
            if player not in started and _accepts(slot, eligibility.get(player, set())):
                floors[slot] = float(ranked.loc[player])
                break
    # UTIL is a flex slot so it has no marginal starter of its own, but it IS the
    # fallback every DH-only bat is priced against. Same rule `sgp.replacement`
    # uses: the deepest hitter floor, i.e. the position it is easiest to replace at.
    hitters = {slot: level for slot, level in floors.items() if slot != "P"}
    if hitters and capacities.get("UTIL", 0) > 0:
        floors["UTIL"] = max(hitters.values())
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
