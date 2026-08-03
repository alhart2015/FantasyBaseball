"""Restate a trajectory as VALUE ABOVE REPLACEMENT instead of raw SGP.

Raw SGP answers "how much will he produce". VAR answers "how much more than the man who
would replace him", and those rank differently because the waiver floor is not one
number: it spans 2.54 SGP, from RP 7.42 and C 7.70 up to OF/UTIL 9.96.

That gap is not cosmetic. A closer projected for 9.96 SGP and a starter projected for
9.87 look interchangeable on the raw scale and are 2.54 versus 0.58 on the VAR scale --
the closer is worth four times as much, because replacing a reliever is easy and
replacing a starter is not. Ranking raw SGP against a VAR board silently penalises every
catcher and reliever in the table.

Floors come from `sgp/replacement.py`, the same source the draft board and the keeper
board use, so a trajectory, a draft pick and a keeper all net against ONE definition of
replacement.

**Eligibility is the league's own rule: 10 games at a position in the season**, via
`keepers.appearances.season_eligibility` over the MLB Stats fielding leaderboard. That
is ingest and normalization, not keeper scoring -- the layer the 2026-08-01 teardown
kept precisely because it is real plumbing -- and it reads the same MLB Stats cache the
trajectory panel is built from, so no new source is introduced.

The alternative sources are both wrong here, and measurably. MLBAM's `primaryPosition`
is a single career-long label: it calls Ivan Herrera a DH, so he would net against the
UTIL floor at 9.96 rather than the catcher floor at 7.70 -- 2.26 SGP a year of pure
scale error on a real keeper candidate. Yahoo's live map has only the current season and
so cannot price a historical one.

`sgp.var.calculate_var` is deliberately NOT reused. Its job is picking the best floor
among eligible slots, which is one line here, and it also wants an `ip` this model never
projects -- so calling it would hide a subtraction behind a helper whose other half does
not apply.
"""

from __future__ import annotations

from dataclasses import replace

from .comps import Trajectory

#: A starter's floor sits 1.87 SGP above a reliever's, so misrouting a pitcher hands him
#: value he never earned. Role comes from the share of appearances that were starts, not
#: from an innings threshold -- see #253 on why total IP is the wrong shape for this.
STARTER_SHARE = 0.5

#: Nothing eligible, so he fills a UTIL slot -- the HIGHEST floor. A missing lookup can
#: therefore only understate a player, never invent value for him.
DEFAULT_SLOT = "UTIL"


def best_floor(slots: set[str], replacement_levels: dict[str, float]) -> tuple[str, float]:
    """The cheapest floor among `slots` -- the slot where the player is worth most.

    Same rule `calculate_var` applies: a multi-eligible player is priced at his scarcest
    position, because that is the roster hole he actually fills.
    """
    usable = {s: replacement_levels[s] for s in slots if s in replacement_levels}
    if not usable:
        return DEFAULT_SLOT, replacement_levels.get(DEFAULT_SLOT, 0.0)
    slot = min(usable, key=lambda s: usable[s])
    return slot, usable[slot]


def resolve_slots(
    eligible: set[str] | None, kind: str, starts: float = 0.0, games: float = 0.0
) -> set[str]:
    """Eligible slots for ONE pool's trajectory.

    Scoped to the pool, because this league scores a two-way player as two separate
    assets: his bat nets against a hitter's floor and his arm against a pitcher's, and
    neither should borrow the other's. Letting the fielding leaderboard's "P" reach the
    hitter side put Ohtani's BAT on the reliever floor -- 2.54 SGP a year of value his
    hitting never earned.

    For pitchers the leaderboard only says he pitched in 10+ games; SP versus RP is a
    question about ROLE, and the panel's own starts and games answer it.
    """
    if kind == "pitcher":
        return {"SP" if games > 0 and starts / games >= STARTER_SHARE else "RP"}
    return {slot for slot in (eligible or ()) if slot != "P"}


def to_var(
    trajectory: Trajectory, replacement_levels: dict[str, float], slots: set[str]
) -> tuple[Trajectory, str, float]:
    """Net every horizon of the cheapest eligible floor.

    Returns the shifted trajectory, the slot chosen, and the floor applied. Only the
    LEVEL moves: `se` and `spread` are widths of a distribution and a constant shift
    leaves them alone, `median` slides with the mean, and survival and support are
    untouched. A horizon with no support stays untouched -- there is no estimate to net.
    """
    slot, floor = best_floor(slots, replacement_levels)
    shifted = tuple(
        point
        if point.n == 0
        else replace(point, mean=point.mean - floor, median=point.median - floor)
        for point in trajectory.path
    )
    return replace(trajectory, path=shifted), slot, floor
