"""League dataclass — the root of the in-season data model.

Owns a list of :class:`Team` and a list of :class:`StandingsSnapshot`.
Per the Step-scoping decision in the design doc, League does NOT own
free agents, projections, leverage, or schedule data — those remain
separate inputs to analysis functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from fantasy_baseball.models.standings import StandingsSnapshot
from fantasy_baseball.models.team import Team


@dataclass
class League:
    season_year: int
    teams: list[Team] = field(default_factory=list)
    standings: list[StandingsSnapshot] = field(default_factory=list)

    # -- Team lookups --

    def team_by_name(self, name: str) -> Team:
        for t in self.teams:
            if t.name == name:
                return t
        raise KeyError(f"Unknown team name: {name!r}")

    def team_by_key(self, team_key: str) -> Team:
        for t in self.teams:
            if t.team_key == team_key:
                return t
        raise KeyError(f"Unknown team_key: {team_key!r}")

    # -- Standings lookups --

    def latest_standings(self) -> StandingsSnapshot:
        """Return the snapshot with the greatest effective_date.

        Raises:
            ValueError: if the league has no standings snapshots.
        """
        if not self.standings:
            raise ValueError("league has no standings snapshots")
        return max(self.standings, key=lambda s: s.effective_date)

    def standings_as_of(self, d: date) -> StandingsSnapshot | None:
        """Return the most recent standings with ``effective_date <= d``.

        Returns ``None`` if ``d`` is earlier than every known snapshot.
        """
        candidates = [s for s in self.standings if s.effective_date <= d]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.effective_date)
