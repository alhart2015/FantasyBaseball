"""Roster entry + snapshot dataclasses.

A :class:`Roster` is one weekly lock snapshot — it represents the set of
players on a team as of ``effective_date`` (the Tuesday the lineup
becomes active). A :class:`RosterEntry` is the lightweight per-player
payload: name, positions, selected_position, status, and yahoo_id. This
is the same information stored in the ``weekly_rosters`` SQLite table
after the Step 0 schema migration.

Callers that need projection-backed fields (ros/preseason stats, wSGP,
pace, rank) hydrate entries into ``Player`` objects via
``hydrate_roster_entries`` (added later in the migration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterator

from fantasy_baseball.models.positions import Position


@dataclass
class RosterEntry:
    """One player's slot-level assignment on a team."""
    name: str
    positions: list[Position]
    selected_position: Position
    status: str = ""          # "", "IL15", "IL60", "DTD", "NA", ...
    yahoo_id: str = ""


@dataclass
class Roster:
    """A team's roster at a specific effective_date.

    ``effective_date`` is the date the lineup becomes active — typically
    the Tuesday of a scoring week. A roster with
    ``effective_date=2026-04-14`` is in effect from 2026-04-14 until the
    next snapshot supersedes it.
    """
    effective_date: date
    entries: list[RosterEntry] = field(default_factory=list)

    def __iter__(self) -> Iterator[RosterEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> set[str]:
        """Return the set of player names on the roster."""
        return {e.name for e in self.entries}

    def by_slot(self) -> dict[Position, list[RosterEntry]]:
        """Group entries by ``selected_position``.

        Slots with multiple players (OF, UTIL, P) return the list in
        the order the entries appear in ``self.entries``. Slots with no
        players are not present in the returned dict.
        """
        out: dict[Position, list[RosterEntry]] = {}
        for entry in self.entries:
            out.setdefault(entry.selected_position, []).append(entry)
        return out
