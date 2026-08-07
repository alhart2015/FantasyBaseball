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
from dataclasses import dataclass, field

from fantasy_baseball.data.rosters import RosterSpot
from fantasy_baseball.utils.name_utils import normalize_name


@dataclass(frozen=True)
class RosterIndex:
    """Everything a board needs to know about who owns which row."""

    #: (normalized name, pool) -> owning team.
    team_of: dict[tuple[str, str], str]
    #: Same key -> "" / "IL10" / "DTD". The CLI renders this; the web does not.
    status_of: dict[tuple[str, str], str]
    #: Keys matching MORE THAN ONE board row. A consumer attributing such a row to
    #: a team is guessing, and must say so on screen.
    ambiguous: set[tuple[str, str]]
    #: Roster players with no scored row at all, per team. Rendering this is what
    #: keeps "the model could not price him" from reading as "he ranked last".
    unscored: dict[str, list[str]]
    #: Dropdown order: my team first, then the rest alphabetically. Plain
    #: alphabetical when `my_team` is None or names no team on any roster.
    teams: tuple[str, ...] = field(default=())

    def team_for(self, name: str, pool: str) -> str | None:
        return self.team_of.get((normalize_name(name), pool))

    def status_for(self, name: str, pool: str) -> str:
        return self.status_of.get((normalize_name(name), pool), "")

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
    by_key: dict[tuple[str, str], RosterSpot] = {(s.normalized, s.player_type): s for s in spots}

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (normalize_name(row["name"]), row["pool"])
        counts[key] = counts.get(key, 0) + 1

    unscored: dict[str, list[str]] = {}
    for spot in spots:
        if (spot.normalized, spot.player_type) not in counts:
            unscored.setdefault(spot.team, []).append(spot.name)

    rostered = {s.team for s in spots}
    # `my_team in rostered` rather than `is not None`: a renamed or mistyped team
    # must not be promoted into a dropdown it cannot filter to.
    promoted = (my_team,) if my_team in rostered else ()
    teams = promoted + tuple(sorted(rostered - set(promoted)))

    return RosterIndex(
        team_of={k: s.team for k, s in by_key.items()},
        status_of={k: s.status for k, s in by_key.items()},
        ambiguous={k for k, c in counts.items() if c > 1},
        unscored=unscored,
        teams=teams,
    )
