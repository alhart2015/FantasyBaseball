"""League dataclass — the root of the in-season data model.

Owns a list of :class:`Team` and a list of :class:`StandingsSnapshot`.
Per the Step-scoping decision in the design doc, League does NOT own
free agents, projections, leverage, or schedule data — those remain
separate inputs to analysis functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.roster import Roster, RosterEntry
from fantasy_baseball.models.standings import (
    CategoryStats,
    StandingsEntry,
    StandingsSnapshot,
)
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

    @classmethod
    def from_db(cls, conn, season_year: int) -> "League":
        """Load complete league state for a season from SQLite.

        Reads two tables:

        - ``weekly_rosters`` (all rows where ``snapshot_date`` starts
          with ``"{season_year}-"``) — produces the sorted
          ``Team.rosters`` lists.
        - ``standings`` (all rows where ``year = season_year``) —
          produces the :class:`StandingsSnapshot` list.

        Team identity is joined by team name. ``Team.team_key`` is
        taken from the most recent standings row for that team, or
        ``""`` if the team appears only in ``weekly_rosters``.

        Raises:
            ValueError: if any stored position token is unknown.
        """
        # ---- Load weekly_rosters rows, group by team then snapshot ----
        roster_rows = conn.execute(
            "SELECT snapshot_date, team, slot, player_name, positions, "
            "       status, yahoo_id "
            "FROM weekly_rosters "
            "WHERE snapshot_date LIKE ? "
            "ORDER BY snapshot_date, team",
            (f"{season_year}-%",),
        ).fetchall()

        # {team_name: {snapshot_date_str: [RosterEntry, ...]}}
        by_team_snap: dict[str, dict[str, list[RosterEntry]]] = {}
        for row in roster_rows:
            team_name = row["team"]
            snap_key = row["snapshot_date"]
            entry = RosterEntry(
                name=row["player_name"],
                positions=Position.parse_list(row["positions"]),
                selected_position=Position.parse(row["slot"]),
                status=row["status"] or "",
                yahoo_id=row["yahoo_id"] or "",
            )
            by_team_snap.setdefault(team_name, {}).setdefault(
                snap_key, []
            ).append(entry)

        # ---- Load standings rows, group by snapshot ----
        standings_rows = conn.execute(
            "SELECT snapshot_date, team, team_key, rank, "
            "       r, hr, rbi, sb, avg, w, k, sv, era, whip "
            "FROM standings "
            "WHERE year = ? "
            "ORDER BY snapshot_date, team",
            (season_year,),
        ).fetchall()

        snapshots_by_date: dict[str, list[StandingsEntry]] = {}
        team_key_by_name: dict[str, str] = {}
        for row in standings_rows:
            stats = CategoryStats(
                r=row["r"] or 0.0,
                hr=row["hr"] or 0.0,
                rbi=row["rbi"] or 0.0,
                sb=row["sb"] or 0.0,
                avg=row["avg"] or 0.0,
                w=row["w"] or 0.0,
                k=row["k"] or 0.0,
                sv=row["sv"] or 0.0,
                era=row["era"] if row["era"] is not None else 99.0,
                whip=row["whip"] if row["whip"] is not None else 99.0,
            )
            standings_entry = StandingsEntry(
                team_name=row["team"],
                team_key=row["team_key"] or "",
                rank=int(row["rank"] or 0),
                stats=stats,
            )
            snapshots_by_date.setdefault(
                row["snapshot_date"], []
            ).append(standings_entry)
            # Keep updating so the most-recent snapshot wins (ORDER BY
            # snapshot_date ASC means the last overwrite is the latest).
            if standings_entry.team_key:
                team_key_by_name[row["team"]] = standings_entry.team_key

        return cls._assemble(
            season_year, by_team_snap, snapshots_by_date, team_key_by_name,
        )

    @classmethod
    def from_redis(cls, season_year: int) -> "League":
        """Load complete league state for a season from Redis.

        Reads two hashes:

        - ``weekly_rosters_history`` — all snapshot dates that start
          with ``"{season_year}-"``; builds ``Team.rosters``.
        - ``standings_history`` — same date filter; builds
          :class:`StandingsSnapshot` list.

        Team identity is joined by team name. ``Team.team_key`` is
        taken from any standings entry that carries one, or ``""`` if
        the team appears only in rosters.

        Raises:
            ValueError: if any stored position token is unknown.
        """
        from fantasy_baseball.data.redis_store import (
            get_default_client,
            get_standings_history,
            get_weekly_roster_history,
        )

        client = get_default_client()
        all_rosters = get_weekly_roster_history(client)
        all_standings = get_standings_history(client)

        prefix = f"{season_year}-"

        # {team_name: {snapshot_date_str: [RosterEntry, ...]}}
        by_team_snap: dict[str, dict[str, list[RosterEntry]]] = {}
        for snap_date, entries in all_rosters.items():
            if not snap_date.startswith(prefix):
                continue
            for e in entries:
                entry = RosterEntry(
                    name=e["player_name"],
                    positions=Position.parse_list(e["positions"]),
                    selected_position=Position.parse(e["slot"]),
                    status=e.get("status") or "",
                    yahoo_id=e.get("yahoo_id") or "",
                )
                by_team_snap.setdefault(
                    e["team"], {}
                ).setdefault(snap_date, []).append(entry)

        snapshots_by_date: dict[str, list[StandingsEntry]] = {}
        team_key_by_name: dict[str, str] = {}
        for snap_date in sorted(all_standings.keys()):
            if not snap_date.startswith(prefix):
                continue
            payload = all_standings[snap_date]
            entries_list: list[StandingsEntry] = []
            for row in payload.get("teams", []):
                stats = CategoryStats(
                    r=row.get("r") or 0.0,
                    hr=row.get("hr") or 0.0,
                    rbi=row.get("rbi") or 0.0,
                    sb=row.get("sb") or 0.0,
                    avg=row.get("avg") or 0.0,
                    w=row.get("w") or 0.0,
                    k=row.get("k") or 0.0,
                    sv=row.get("sv") or 0.0,
                    era=row["era"] if row.get("era") is not None else 99.0,
                    whip=row["whip"] if row.get("whip") is not None else 99.0,
                )
                standings_entry = StandingsEntry(
                    team_name=row["team"],
                    team_key=row.get("team_key") or "",
                    rank=int(row.get("rank") or 0),
                    stats=stats,
                )
                entries_list.append(standings_entry)
                if standings_entry.team_key:
                    team_key_by_name[row["team"]] = standings_entry.team_key
            snapshots_by_date[snap_date] = entries_list

        return cls._assemble(
            season_year, by_team_snap, snapshots_by_date, team_key_by_name,
        )

    @classmethod
    def _assemble(
        cls,
        season_year: int,
        by_team_snap: dict[str, dict[str, list[RosterEntry]]],
        snapshots_by_date: dict[str, list[StandingsEntry]],
        team_key_by_name: dict[str, str],
    ) -> "League":
        """Shared stitching step for ``from_db`` and ``from_redis``.

        Takes already-grouped intermediate structures and builds the
        final ``League`` object: ``Team`` list (sorted by name, each
        with sorted ``Roster`` list) and ``StandingsSnapshot`` list
        (sorted by effective_date).
        """
        standings_snapshots = [
            StandingsSnapshot(
                effective_date=date.fromisoformat(snap_key),
                entries=entries,
            )
            for snap_key, entries in sorted(snapshots_by_date.items())
        ]

        # Build Team list (union of team names across both sources)
        all_team_names = set(by_team_snap.keys()) | set(team_key_by_name.keys())

        teams: list[Team] = []
        for team_name in sorted(all_team_names):
            rosters: list[Roster] = []
            for snap_key, entries in by_team_snap.get(team_name, {}).items():
                rosters.append(Roster(
                    effective_date=date.fromisoformat(snap_key),
                    entries=entries,
                ))
            rosters.sort(key=lambda r: r.effective_date)
            teams.append(Team(
                name=team_name,
                team_key=team_key_by_name.get(team_name, ""),
                rosters=rosters,
            ))

        return cls(
            season_year=season_year,
            teams=teams,
            standings=standings_snapshots,
        )
