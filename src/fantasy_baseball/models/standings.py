"""League-level roto category statistics.

Snapshot-layer dataclasses used by League:
:class:`CategoryStats` (the ten roto totals), :class:`StandingsEntry`
(one team's stats + rank), and :class:`Standings` (all teams at an
effective_date). :class:`ProjectedStandings` is the projected twin of
``Standings`` built from rosters.

``CategoryStats`` is keyed exclusively by :class:`Category` enum. Bare
string access raises ``TypeError``. ``StandingsEntry.extras`` is
keyed by :class:`OpportunityStat` on the same contract — non-roto
volume stats (PA, IP) from Yahoo standings that ride alongside the
ten roto categories.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category, OpportunityStat

# Private: single source of truth for Category <-> attribute mapping.
_CAT_TO_FIELD: dict[Category, str] = {
    Category.R: "r",
    Category.HR: "hr",
    Category.RBI: "rbi",
    Category.SB: "sb",
    Category.AVG: "avg",
    Category.W: "w",
    Category.K: "k",
    Category.SV: "sv",
    Category.ERA: "era",
    Category.WHIP: "whip",
}


@dataclass
class CategoryStats:
    r: float = 0.0
    hr: float = 0.0
    rbi: float = 0.0
    sb: float = 0.0
    avg: float = 0.0
    w: float = 0.0
    k: float = 0.0
    sv: float = 0.0
    era: float = 99.0
    whip: float = 99.0

    def __getitem__(self, cat: Category) -> float:
        if not isinstance(cat, Category):
            raise TypeError(
                f"CategoryStats indexing requires a Category enum, got {type(cat).__name__}"
            )
        return float(getattr(self, _CAT_TO_FIELD[cat]))

    def items(self) -> Iterator[tuple[Category, float]]:
        for cat in ALL_CATEGORIES:
            yield cat, float(getattr(self, _CAT_TO_FIELD[cat]))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CategoryStats:
        """Build from an UPPERCASE-string-keyed dict (I/O boundary only).

        Missing keys fall back to dataclass defaults (0 for counting
        stats, 99 for ERA/WHIP).
        """
        kwargs: dict[str, Any] = {}
        for cat in ALL_CATEGORIES:
            if cat.value in d:
                kwargs[_CAT_TO_FIELD[cat]] = float(d[cat.value])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, float]:
        """Produce an UPPERCASE-string-keyed dict (I/O boundary only)."""
        return {cat.value: float(getattr(self, _CAT_TO_FIELD[cat])) for cat in ALL_CATEGORIES}


@dataclass
class CategoryPoints:
    """Per-category roto points plus total, for one team.

    Replaces the ``{"R_pts": ..., "HR_pts": ..., "total": ...}`` dict
    returned by the old ``score_roto``. ``values`` is the per-category
    map; ``total`` is the sum of ``values`` by default, but ``score_roto``
    may override it with ``yahoo_points_for`` when the display layer
    needs an exact match with Yahoo's official standings page.
    """

    values: dict[Category, float]
    total: float

    def __getitem__(self, cat: Category) -> float:
        if not isinstance(cat, Category):
            raise TypeError(
                f"CategoryPoints indexing requires a Category enum, got {type(cat).__name__}"
            )
        return self.values[cat]


@dataclass
class StandingsEntry:
    """One team's standings row at a point in time.

    ``yahoo_points_for`` is Yahoo's authoritative roto total, computed
    internally from full-precision stats. It's set only for snapshots
    built from live Yahoo standings (not for projected snapshots). When
    present, the display layer prefers it over ``score_roto``'s total so
    our UI exactly matches Yahoo's standings page — otherwise display
    ties in rounded rate stats (AVG, ERA, WHIP) make our averaged-rank
    scoring differ by up to ±0.5 points per tie from Yahoo's real total.

    ``extras`` holds non-roto volume stats (PA, IP) that Yahoo ships
    alongside the scoring categories in ``team_stats``. Keyed by
    :class:`OpportunityStat` enum — bare-string access is deliberately
    not supported (same contract as :class:`CategoryStats`).
    """

    team_name: str
    team_key: str
    rank: int
    stats: CategoryStats
    yahoo_points_for: float | None = None
    extras: dict[OpportunityStat, float] = field(default_factory=dict)


@dataclass
class Standings:
    """All teams' live standings at a single effective_date.

    ``entries`` carry real Yahoo ``team_key`` and ``rank`` (non-optional)
    plus ``yahoo_points_for`` when Yahoo has scored the week.
    """

    effective_date: date
    entries: list[StandingsEntry]

    def by_team(self) -> dict[str, StandingsEntry]:
        out: dict[str, StandingsEntry] = {}
        for entry in self.entries:
            if entry.team_name in out:
                raise ValueError(f"duplicate team in standings: {entry.team_name!r}")
            out[entry.team_name] = entry
        return out

    def sorted_by_rank(self) -> list[StandingsEntry]:
        return sorted(self.entries, key=lambda e: e.rank)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Standings:
        """Canonical shape only: {'effective_date', 'teams': [{'name', ...}]}.

        Raises ``ValueError`` on legacy shapes ('team' instead of
        'name', lowercase stat keys, missing wrapper date).
        """
        if not isinstance(d, Mapping) or "teams" not in d or "effective_date" not in d:
            got = sorted(d) if isinstance(d, Mapping) else type(d).__name__
            raise ValueError(
                f"Standings.from_json: legacy or unknown payload shape — "
                f"missing 'effective_date' or 'teams' wrapper (got keys: {got})"
            )
        eff = date.fromisoformat(d["effective_date"])
        entries: list[StandingsEntry] = []
        for row in d["teams"]:
            if "team" in row and "name" not in row:
                raise ValueError(
                    "Standings.from_json: legacy row shape detected ('team' field "
                    "instead of 'name') — run scripts/migrate_standings_history.py"
                )
            if "stats" not in row:
                raise ValueError(
                    f"Standings.from_json: row missing 'stats' wrapper "
                    f"(likely legacy flat-lowercase shape): {row.get('name')!r}"
                )
            raw_extras = row.get("extras") or {}
            extras: dict[OpportunityStat, float] = {}
            for k, v in raw_extras.items():
                try:
                    extras[OpportunityStat(k)] = float(v)
                except ValueError:
                    # Unknown key — ignore to keep legacy/in-flight
                    # writers from breaking the read path.
                    continue
            entries.append(
                StandingsEntry(
                    team_name=row["name"],
                    team_key=row["team_key"],
                    rank=int(row["rank"]),
                    stats=CategoryStats.from_dict(row["stats"]),
                    yahoo_points_for=row.get("yahoo_points_for"),
                    extras=extras,
                )
            )
        return cls(effective_date=eff, entries=entries)

    def to_json(self) -> dict[str, Any]:
        return {
            "effective_date": self.effective_date.isoformat(),
            "teams": [
                {
                    "name": e.team_name,
                    "team_key": e.team_key,
                    "rank": e.rank,
                    "yahoo_points_for": e.yahoo_points_for,
                    "stats": e.stats.to_dict(),
                    "extras": {k.value: float(v) for k, v in e.extras.items()},
                }
                for e in self.entries
            ],
        }


@dataclass
class ProjectedStandingsEntry:
    team_name: str
    stats: CategoryStats


@dataclass
class ProjectedStandings:
    effective_date: date
    entries: list[ProjectedStandingsEntry]

    def by_team(self) -> dict[str, ProjectedStandingsEntry]:
        out: dict[str, ProjectedStandingsEntry] = {}
        for entry in self.entries:
            if entry.team_name in out:
                raise ValueError(f"duplicate team in projected standings: {entry.team_name!r}")
            out[entry.team_name] = entry
        return out

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> ProjectedStandings:
        if not isinstance(d, Mapping) or "teams" not in d or "effective_date" not in d:
            raise ValueError("ProjectedStandings.from_json: missing 'effective_date' or 'teams'")
        return cls(
            effective_date=date.fromisoformat(d["effective_date"]),
            entries=[
                ProjectedStandingsEntry(
                    team_name=row["name"],
                    stats=CategoryStats.from_dict(row["stats"]),
                )
                for row in d["teams"]
            ],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "effective_date": self.effective_date.isoformat(),
            "teams": [{"name": e.team_name, "stats": e.stats.to_dict()} for e in self.entries],
        }

    @classmethod
    def from_rosters(
        cls,
        team_rosters: Mapping[str, Any],
        effective_date: date,
    ) -> ProjectedStandings:
        """Build from {team_name: roster_list} using project_team_stats.

        ``ProjectedStandings`` projects end-of-season totals — read
        ``full_season_projection`` (= ROS + YTD per player). The optimizer
        and other forward-looking decision paths use the default
        ``rest_of_season`` (ROS-only) instead so a hot-YTD player's locked
        accumulated value doesn't bias start/sit and trade decisions.

        TODO: Replace this approximation with current_standings + ROS
        contribution when team-level YTD AB ingest lands. Yahoo standings
        only surface AVG, not H/AB components, so the rate-stat
        recombination can't be implemented correctly today. See
        ``docs/feature_specs/ros_only_decision_projections.md`` Phase 3
        scope reduction (deferred to follow-up).
        """
        from fantasy_baseball.scoring import project_team_stats

        return cls(
            effective_date=effective_date,
            entries=[
                ProjectedStandingsEntry(
                    team_name=tname,
                    stats=project_team_stats(
                        roster,
                        displacement=True,
                        projection_source="full_season_projection",
                    ),
                )
                for tname, roster in team_rosters.items()
            ],
        )
