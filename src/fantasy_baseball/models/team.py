"""Team dataclass — a name/team_key + sorted roster history.

The roster list is sorted on access (not in ``__init__``) so Team is
trivially constructable from DB row groups that may or may not be in
effective_date order. Methods that care about ordering iterate over a
freshly-sorted view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from fantasy_baseball.models.roster import Roster, RosterEntry


@dataclass
class Team:
    name: str
    team_key: str
    rosters: list[Roster] = field(default_factory=list)

    def _sorted(self) -> list[Roster]:
        return sorted(self.rosters, key=lambda r: r.effective_date)

    def latest_roster(self) -> Roster:
        """Return the snapshot with the greatest effective_date.

        In-season, this is typically the future-dated Tuesday lock
        written by the refresh pipeline — i.e. the roster state that
        will be active when the next lineup locks.

        Raises:
            ValueError: if the team has no rosters.
        """
        if not self.rosters:
            raise ValueError(f"team {self.name!r} has no rosters")
        return max(self.rosters, key=lambda r: r.effective_date)

    def roster_as_of(self, d: date) -> Roster | None:
        """Return the most recent roster with ``effective_date <= d``.

        Returns ``None`` if ``d`` is earlier than every known snapshot.
        """
        candidates = [r for r in self.rosters if r.effective_date <= d]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.effective_date)

    def ownership_periods(
        self,
        season_start: date,
        season_end: date,
        today: date,
    ) -> list[tuple[RosterEntry, date, date]]:
        """Yield (entry, period_start, period_end) for every player across
        all snapshots, clipped to the season window and to today.

        Each snapshot ``i`` contributes a period
        ``[rosters[i].effective_date, rosters[i+1].effective_date)``
        for every RosterEntry in it (or up to ``today`` / ``season_end``
        for the last snapshot).

        The period is clipped so that:
        - ``period_start = max(snap.effective_date, season_start)``
        - ``period_end   = min(next_snap.effective_date or today, today, season_end)``

        Snapshots whose effective_date is today or later contribute no
        days — they're scheduled state, not yet owned state. A roster
        that locks today has not played any games under that lineup yet.

        Used by SPoE to scale preseason projections by days-owned.
        """
        sorted_rosters = self._sorted()
        if not sorted_rosters:
            return []

        out: list[tuple[RosterEntry, date, date]] = []
        for i, snap in enumerate(sorted_rosters):
            if snap.effective_date >= today:
                continue  # future-dated, no contribution yet

            if i + 1 < len(sorted_rosters):
                raw_end = sorted_rosters[i + 1].effective_date
            else:
                raw_end = today

            start = max(snap.effective_date, season_start)
            end = min(raw_end, today, season_end)

            if end <= start:
                continue  # zero-length or negative period after clipping

            for entry in snap.entries:
                out.append((entry, start, end))

        return out
