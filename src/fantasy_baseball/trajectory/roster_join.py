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

    #: (normalized name, pool) -> the spot that won the key. ONE dict rather than a
    #: parallel `team_of` / `status_of` pair: both were built from this same mapping
    #: and each held a single attribute off it, so every further field a consumer
    #: wanted meant a third parallel dict keyed identically.
    spot_of: dict[tuple[str, str], RosterSpot]
    #: Keys the join cannot resolve, from EITHER side: more than one board row under
    #: the key, or more than one roster spot under it. Both mean a consumer
    #: attributing such a row to a team is guessing, and must say so on screen.
    ambiguous: set[tuple[str, str]]
    #: Roster players this board cannot attribute a row to, per team. Two causes,
    #: deliberately in one list: nothing on the board matched the name at all, and
    #: the name matched but another team's spot won the key. The reader is asking
    #: the same question in both cases -- "where is my guy?" -- and the answer is
    #: the same, that this page is not showing him a row for that player. Rendering
    #: it is what keeps "the model could not price him" from reading as "he ranked
    #: last", and what keeps a collision loser from vanishing off his owner's page
    #: with nothing on screen to say he was ever there.
    unscored: dict[str, list[str]]
    #: Teams owning at least one spot that WON a scored row. Falls out of the same
    #: pass that builds `unscored` -- a spot not pushed there is a spot that matched
    #: -- so a caller asking "did this team join anything" gets a set lookup instead
    #: of rescanning every board row and re-normalizing every name to find out.
    matched_teams: frozenset[str]
    #: Dropdown order: my team first, then the rest alphabetically. Plain
    #: alphabetical when `my_team` is None or names no team on any roster.
    teams: tuple[str, ...] = ()

    def team_for(self, name: str, pool: str) -> str | None:
        spot = self.spot_of.get((normalize_name(name), pool))
        return spot.team if spot else None

    def status_for(self, name: str, pool: str) -> str:
        spot = self.spot_of.get((normalize_name(name), pool))
        return spot.status if spot else ""

    def is_ambiguous(self, name: str, pool: str) -> bool:
        return (normalize_name(name), pool) in self.ambiguous


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

    unscored: dict[str, list[str]] = {}
    matched: set[str] = set()
    for spot in spots:
        key = (spot.normalized, spot.player_type)
        # `is not` and not `!=`: two spots for the same player on the same team
        # are distinct roster entries even though the dataclass compares equal.
        if key not in counts or by_key[key] is not spot:
            unscored.setdefault(spot.team, []).append(spot.name)
        else:
            matched.add(spot.team)

    rostered = {s.team for s in spots}
    # `my_team in rostered` rather than `is not None`: a renamed or mistyped team
    # must not be promoted into a dropdown it cannot filter to.
    promoted = (my_team,) if my_team in rostered else ()
    teams = promoted + tuple(sorted(rostered - set(promoted)))

    return RosterIndex(
        spot_of=by_key,
        ambiguous={k for k, c in counts.items() if c > 1}
        | {k for k, c in spot_counts.items() if c > 1},
        unscored=unscored,
        matched_teams=frozenset(matched),
        teams=teams,
    )
