"""The roster-to-board join, shared by the CLI and the web board.

`(normalized_name, player_type)` is the only key available: roster blobs carry no
`mlbam_id` (#284), so this join is NOT unique -- the live board has two hitters
called Max Muncy. That is surfaced through `ambiguous` rather than resolved here.

WHY THIS LIVES UNDER `src/`. Nothing under `src/` can import from `scripts/`, and
`scripts/trajectory_board.py` owned the only implementation. A web team filter
would have needed its own copy -- two spellings of a fragile join, free to drift
on exactly the ambiguity handling that matters. Same reason `trajectory/sweep.py`
was extracted, recorded in its module docstring.

WHY IT RETURNS A LOOKUP AND MUTATES NOTHING. The predecessor stamped
`row["team"]` in place, which is safe for rows the CLI just built. The web's rows
come from `trajectory_view._ranked_rows`, a cache shared across requests whose
comment states rows are "never mutated after `add_ranks`". One helper that
mutated would be correct in one caller and a cross-request data race in the
other, so the hazard is made unreachable instead of merely avoided.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.utils.name_utils import normalize_name


@dataclass(frozen=True)
class RosterIndex:
    """Everything a board needs to know about who owns which row."""

    #: (normalized name, pool) -> EVERY team rostering that key, usually one.
    #:
    #: OWNERSHIP IS MEMBERSHIP, NOT A WINNER, and conflating the two was a real bug.
    #: "Which team does this row belong to" needs a single answer and so needs a
    #: winner; "do I own a player under this key" is a set test and needs none.
    #: Deriving the second from the first meant that when an opponent's spot sorted
    #: first, my own rostered player rendered as neither mine nor ambiguous -- worse
    #: than the arbitrary-attribution bug the winner was introduced to fix, because
    #: the reader saw no marking at all rather than a wrong one.
    #:
    #: A row whose key is rostered by two teams therefore shows under BOTH, flagged.
    #: Hiding the one you own is worse than showing one you do not, clearly marked --
    #: the same rule the board-side collision already followed.
    owners_of: dict[tuple[str, str], frozenset[str]]
    #: (normalized name, pool) -> the spot that won the key, for the places that
    #: genuinely need ONE: the CLI's `[IL10]` suffix and its CSV `team` column.
    #: Deterministic (see `index_rosters`), never used to decide ownership.
    spot_of: dict[tuple[str, str], RosterSpot]
    #: Keys the join cannot resolve, from EITHER side: more than one board row under
    #: the key, or more than one roster spot under it. Both mean a consumer
    #: attributing such a row to a team is guessing, and must say so on screen.
    ambiguous: set[tuple[str, str]]
    #: Roster spots with NO board row under their key, per team. ONE cause: the
    #: model could not price the player. It briefly carried a second -- "another
    #: spot won the key" -- back when ownership was winner-derived, and that forced
    #: every consumer's copy to hedge about which cause applied. Membership
    #: ownership removes the second cause outright: a spot whose key has a row is
    #: shown under its team, so it is not missing and has no business here.
    #:
    #: Spots rather than names so a consumer can filter by pool -- the list renders
    #: under a table that may be showing hitters only, and naming a pitcher there
    #: reads as a hole in the hitter list.
    unscored_spots: dict[str, list[RosterSpot]]
    #: Teams owning at least one spot that WON a scored row. Falls out of the same
    #: pass that builds `unscored` -- a spot not pushed there is a spot that matched
    #: -- so a caller asking "did this team join anything" gets a set lookup instead
    #: of rescanning every board row and re-normalizing every name to find out.
    matched_teams: frozenset[str]
    #: Dropdown order: my team first, then the rest alphabetically. Plain
    #: alphabetical when `my_team` is None or names no team on any roster.
    teams: tuple[str, ...] = ()

    def owners_for(self, name: str, pool: str) -> frozenset[str]:
        """Every team rostering this key. The ownership question."""
        return self.owners_of.get((normalize_name(name), pool), frozenset())

    def team_for(self, name: str, pool: str) -> str | None:
        """The single team to print in a one-team-per-row context. NOT ownership --
        use `owners_for` for "is this mine" or "does this team hold him"."""
        spot = self.spot_of.get((normalize_name(name), pool))
        return spot.team if spot else None

    def status_for(self, name: str, pool: str) -> str:
        spot = self.spot_of.get((normalize_name(name), pool))
        return spot.status if spot else ""

    def is_ambiguous(self, name: str, pool: str) -> bool:
        return (normalize_name(name), pool) in self.ambiguous

    def unscored_for(self, team: str, pool: str = "both") -> list[str]:
        """Names this team rosters that the board has no row for, sorted.

        `pool` filters to match the table the list renders under; "both" keeps all.
        """
        return sorted(
            s.name
            for s in self.unscored_spots.get(team, ())
            if pool == "both" or s.player_type == pool
        )


def index_rosters(
    rows: Sequence[dict],
    spots: Sequence[RosterSpot],
    my_team: str | None,
) -> RosterIndex:
    """Join `spots` against `rows` without touching either.

    `rows` need only carry "name" and "pool".
    """
    # THE KEY COLLIDES ON THE ROSTER SIDE TOO, and that is the worse direction.
    # Two board rows under one key at least leave the row on screen; two SPOTS
    # under one key -- two different humans on two different teams -- mean one of
    # them silently loses the key, so the row renders under a team that does not
    # own him and the true owner's page is short a player. A dict comprehension
    # over `spots` resolved that by roster iteration order, which is Yahoo's, and
    # so attributed the same row to a different team from one read to the next.
    # Sorting makes the winner a fixed function of the data instead.
    ordered = sorted(spots, key=lambda s: (s.normalized, s.player_type, s.team, s.name, s.yahoo_id))
    by_key: dict[tuple[str, str], RosterSpot] = {}
    spot_counts: dict[tuple[str, str], int] = {}
    for spot in ordered:
        key = (spot.normalized, spot.player_type)
        spot_counts[key] = spot_counts.get(key, 0) + 1
        by_key.setdefault(key, spot)

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (normalize_name(row["name"]), row["pool"])
        counts[key] = counts.get(key, 0) + 1

    owners: dict[tuple[str, str], set[str]] = {}
    unscored_spots: dict[str, list[RosterSpot]] = {}
    matched: set[str] = set()
    for spot in spots:
        key = (spot.normalized, spot.player_type)
        owners.setdefault(key, set()).add(spot.team)
        # Purely "did the board price anyone under this key". It used to also send
        # the loser of a key contest here, which put a player on the missing list
        # while his row sat on screen under the winning team.
        if key in counts:
            matched.add(spot.team)
        else:
            unscored_spots.setdefault(spot.team, []).append(spot)

    rostered = {s.team for s in spots}
    # `my_team in rostered` rather than `is not None`: a renamed or mistyped team
    # must not be promoted into a dropdown it cannot filter to.
    promoted = (my_team,) if my_team in rostered else ()
    teams = promoted + tuple(sorted(rostered - set(promoted)))

    return RosterIndex(
        owners_of={k: frozenset(v) for k, v in owners.items()},
        spot_of=by_key,
        ambiguous={k for k, c in counts.items() if c > 1}
        | {k for k, c in spot_counts.items() if c > 1},
        unscored_spots=unscored_spots,
        matched_teams=frozenset(matched),
        teams=teams,
    )
