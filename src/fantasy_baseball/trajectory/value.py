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
board use, so a trajectory, a draft pick and a keeper net against one definition of
replacement -- for HITTERS exactly, and for pitchers with the SP/RP routing caveat on
`STARTER_SHARE` below.

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

#: A starter's floor sits 1.87 SGP above a reliever's, so misrouting a pitcher hands him
#: value he never earned.
#:
#: This DIVERGES from `utils.constants.role_from_ip` (ip >= 100), which the draft board
#: and keeper board route on via `sgp.var._pitcher_floor_key`. The two disagree on 814
#: of 4793 pitcher-seasons (17%) in the 2021-2026 panel -- a 90-IP all-starts rookie is
#: a starter here and a reliever there. Kept deliberately: #253 records that a total-IP
#: threshold is the wrong shape for role, and this model has the starts the shared
#: classifier lacks. It is a real inconsistency between boards, not a hidden one; the
#: resolution is to move starts-based routing into the shared classifier under #253,
#: not to fork it further.
STARTER_SHARE = 0.5

#: Which slots each pool can possibly net against. A hitter cannot be priced on the
#: reliever floor and a pitcher cannot be priced at catcher, but argparse offers one flat
#: list, so `--pool hitter --position RP` was accepted and printed "RP floor 7.42" over a
#: hitter -- 2.54 SGP a year, stated as fact. Validated in one place rather than at each
#: call site, because the two-way guard was added at one site and this was missed.
POOL_SLOTS: dict[str, frozenset[str]] = {
    "hitter": frozenset({"C", "1B", "2B", "3B", "SS", "OF", "UTIL"}),
    "pitcher": frozenset({"SP", "RP"}),
}

#: Appearances before an in-progress season is trusted to describe a pitcher's ROLE.
#: The same 10-game threshold the league uses for eligibility: a starter back from the
#: IL with two September relief outings is not a reliever, but `starts / games` on that
#: fragment says he is. The anchor that gives his SGP a full-season line deliberately
#: leaves appearances alone -- a projection must not pick a replacement level (#348) --
#: so the fragment is still what this has to see through.
ROLE_MIN_GAMES = 10

#: Nothing eligible, so he fills a UTIL slot -- the HIGHEST floor. A missing lookup can
#: therefore only understate a player, never invent value for him.
DEFAULT_SLOT = "UTIL"


def best_floor(slots: set[str], replacement_levels: dict[str, float]) -> tuple[str, float]:
    """The cheapest floor among `slots` -- the slot where the player is worth most.

    Same rule `calculate_var` applies: a multi-eligible player is priced at his scarcest
    position, because that is the roster hole he actually fills.
    """
    if isinstance(slots, str):
        # A str is iterable, so `{s for s in "SP"}` silently becomes {"S", "P"}, matches
        # no floor, and falls through to UTIL -- the whole feature quietly disabled for
        # every position but the one-character "C". Refuse rather than accept it.
        raise TypeError(f"slots must be a set of slot names, not the string {slots!r}")
    usable = {s: replacement_levels[s] for s in slots if s in replacement_levels}
    if not usable:
        return DEFAULT_SLOT, replacement_levels.get(DEFAULT_SLOT, 0.0)
    slot = min(usable, key=lambda s: usable[s])
    return slot, usable[slot]


def check_position(position: str, pool: str) -> str | None:
    """None if `position` can price `pool`, else why not."""
    if position in POOL_SLOTS[pool]:
        return None
    other = "pitcher" if pool == "hitter" else "hitter"
    return (
        f"--position {position} is a {other} slot and cannot price a {pool}; "
        f"{pool} slots are {', '.join(sorted(POOL_SLOTS[pool]))}"
    )


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


def replacement_for(slots: set[str], replacement_levels: dict[str, float]) -> tuple[str, float]:
    """The slot and floor a trajectory should be scored against.

    Thin on purpose. Flooring happens inside the estimators, which see the individual
    comp outcomes; a post-hoc shift of the finished mean charged the floor a second time
    against every comp who had left the league, and moved `mean` while leaving `median`,
    `spread` and `mean_if_survived` on the raw scale.
    """
    return best_floor(slots, replacement_levels)
