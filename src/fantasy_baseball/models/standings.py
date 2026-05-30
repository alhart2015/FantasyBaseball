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

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from fantasy_baseball.utils.constants import AB_PER_PA, ALL_CATEGORIES, Category, OpportunityStat

log = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class TeamYtdComponents:
    """Rate-stat ingredients for a team's YTD totals.

    Derived from ``StandingsEntry.stats`` (rates) + ``extras`` (volumes).
    Designed to be summed with the analogous ROS components by a
    ``team_end_of_season`` helper that recomputes AVG/ERA/WHIP from the
    combined ingredients so the displayed standings reflect true YTD +
    ROS arithmetic rather than averaging rates.

    BB and H_allowed are combined because Yahoo only exposes their sum
    via WHIP * IP; the projection math only needs the sum.
    """

    # Counting stats (passed through)
    r: float = 0.0
    hr: float = 0.0
    rbi: float = 0.0
    sb: float = 0.0
    w: float = 0.0
    k: float = 0.0
    sv: float = 0.0

    # Rate-stat components
    h: float = 0.0
    ab: float = 0.0
    ip: float = 0.0
    er: float = 0.0
    bb_plus_h_allowed: float = 0.0


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

    def ytd_components(self, *, fallback_ab: float = 0.0) -> TeamYtdComponents:
        """Decompose this entry's stats + extras into rate-stat ingredients.

        AB sourcing tier (first hit wins):
        1. ``extras[OpportunityStat.AB]`` -- Yahoo's direct AB stat.
        2. ``extras[OpportunityStat.PA] * AB_PER_PA`` -- derive from PA.
        3. ``fallback_ab`` -- caller-provided hint (e.g., per-roster sum of
           ``full_season.ab - ros.ab``, or games-played * league-avg AB/game).
           Last resort before giving up on AVG recombination.
        4. ``0`` -- no AB recoverable; callers see h=0/ab=0 and must skip AVG.

        ER/IP/BB+H_allowed come from IP + rate stats. When IP is zero
        (pre-season), they zero out cleanly (not NaN).
        """
        ip = float(self.extras.get(OpportunityStat.IP, 0.0))
        ab = float(self.extras.get(OpportunityStat.AB, 0.0))
        if ab <= 0.0:
            pa = float(self.extras.get(OpportunityStat.PA, 0.0))
            if pa > 0.0:
                ab = pa * AB_PER_PA
            elif fallback_ab > 0.0:
                ab = fallback_ab
        h = self.stats.avg * ab if ab > 0.0 else 0.0
        er = self.stats.era * ip / 9.0 if ip > 0.0 else 0.0
        bbha = self.stats.whip * ip if ip > 0.0 else 0.0
        return TeamYtdComponents(
            r=self.stats.r,
            hr=self.stats.hr,
            rbi=self.stats.rbi,
            sb=self.stats.sb,
            w=self.stats.w,
            k=self.stats.k,
            sv=self.stats.sv,
            h=h,
            ab=ab,
            ip=ip,
            er=er,
            bb_plus_h_allowed=bbha,
        )


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
    """One team's projected end-of-season totals.

    ``stats`` is the user-facing CategoryStats (rates already recombined
    from YTD + ROS components). ``total_ab`` and ``total_ip`` are the
    end-of-season volume denominators used to recombine those rates, so
    downstream callers (``apply_swap_delta``) can back out current
    hits/ER/BH using the same denominators that produced the rates.
    Pre-PR-110 entries (loaded from older persisted JSON, or hand-built
    in tests) leave them at 0.0; ``apply_swap_delta`` then falls back to
    the legacy ``_TEAM_AB`` / ``_TEAM_IP`` heuristics.
    """

    team_name: str
    stats: CategoryStats
    total_ab: float = 0.0
    total_ip: float = 0.0


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

    def field_stats(self, exclude: str) -> dict[str, CategoryStats]:
        """Stats for every team except ``exclude`` (the user's team).

        The fixed field for deltaRoto confidence-band sampling.
        """
        return {e.team_name: e.stats for e in self.entries if e.team_name != exclude}

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
                    # Older persisted payloads predate the AB/IP carry --
                    # missing keys decay to 0.0, which apply_swap_delta
                    # treats as the legacy-constant fallback.
                    total_ab=float(row.get("total_ab", 0.0)),
                    total_ip=float(row.get("total_ip", 0.0)),
                )
                for row in d["teams"]
            ],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "effective_date": self.effective_date.isoformat(),
            "teams": [
                {
                    "name": e.team_name,
                    "stats": e.stats.to_dict(),
                    "total_ab": float(e.total_ab),
                    "total_ip": float(e.total_ip),
                }
                for e in self.entries
            ],
        }

    @classmethod
    def from_rosters(
        cls,
        team_rosters: Mapping[str, Any],
        effective_date: date,
        *,
        actual_standings: Standings | None = None,
        fraction_remaining: float = 1.0,
    ) -> ProjectedStandings:
        """Build end-of-season projection from rosters and team YTD.

        ``actual_standings`` is the league's Yahoo standings snapshot at the
        same effective_date. Per team, the projected end-of-season totals
        are computed as ``team_YTD + project_ros_components(roster)``, then
        AVG/ERA/WHIP are recombined from summed components by
        :func:`team_end_of_season`.

        When ``actual_standings`` is ``None`` (pre-season), each team's YTD
        components are empty and the projection collapses to ROS-only,
        matching the pre-season behavior.

        Two-pass displacement:
        1. Pass 1 -- SGP-based displacement on each team's ROS, combined
           with the team's YTD via :func:`team_end_of_season` to produce
           the end-of-season baseline ``{team: CategoryStats}`` consumed by
           the DeltaRoto picker via ``LeagueContext.baseline_other_team_stats``
           in Pass 2. Pass-1 must include YTD because per-team YTD shifts
           are not uniform across the picker's argmax -- they materially
           change which alternative wins category swaps.
        2. Pass 2 -- each team re-displaces against the frozen Pass-1
           baseline of the OTHER teams; the resulting ROS components are
           summed with that team's YTD via ``team_end_of_season``.

        Other callers of ``project_team_stats`` (optimizer, draft, trade
        evaluator) keep their ROS-only forward-looking math -- they don't
        need YTD because their question is "what does going forward look
        like?", not "what's the end-of-season total?".
        """
        from fantasy_baseball.scoring import (
            LeagueContext,
            build_team_sds,
            project_ros_components,
            team_end_of_season,
        )

        # Pass-1 needs the YTD-by-team map first, so build it before the
        # baseline loop instead of after.
        ytd_by_team: dict[str, TeamYtdComponents] = {}
        if actual_standings is not None:
            for entry in actual_standings.entries:
                ytd_by_team[entry.team_name] = entry.ytd_components()

        warned_names: set[str] = set()

        def _ytd_for(tname: str) -> TeamYtdComponents:
            ytd = ytd_by_team.get(tname)
            if ytd is None:
                if actual_standings is not None and tname not in warned_names:
                    log.warning(
                        "Team YTD lookup miss for %r in from_rosters; falling back "
                        "to zero YTD components (team_rosters key not found in "
                        "actual_standings.entries)",
                        tname,
                    )
                    warned_names.add(tname)
                return TeamYtdComponents()
            return ytd

        baseline_stats: dict[str, CategoryStats] = {}
        for tname, roster in team_rosters.items():
            ytd = _ytd_for(tname)
            ros = project_ros_components(list(roster), displacement=True)
            baseline_stats[tname] = team_end_of_season(ytd, ros)

        team_sds = build_team_sds(
            {tname: list(roster) for tname, roster in team_rosters.items()},
            sd_scale=fraction_remaining**0.5,
        )

        entries: list[ProjectedStandingsEntry] = []
        for tname, roster in team_rosters.items():
            ros = project_ros_components(
                list(roster),
                displacement=True,
                league_context=LeagueContext(
                    baseline_other_team_stats={
                        t: s for t, s in baseline_stats.items() if t != tname
                    },
                    team_sds=team_sds,
                    team_name=tname,
                ),
            )
            ytd = _ytd_for(tname)
            entries.append(
                ProjectedStandingsEntry(
                    team_name=tname,
                    stats=team_end_of_season(ytd, ros),
                    total_ab=ytd.ab + ros.ab,
                    total_ip=ytd.ip + ros.ip,
                )
            )
        return cls(effective_date=effective_date, entries=entries)
