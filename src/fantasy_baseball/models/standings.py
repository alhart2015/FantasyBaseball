"""League-level roto category statistics.

Three snapshot-layer dataclasses used by League:
:class:`CategoryStats` (the ten roto totals), :class:`StandingsEntry`
(one team's stats + rank), and :class:`StandingsSnapshot` (all teams
at an effective_date).

``CategoryStats`` is keyed exclusively by :class:`Category` enum. Bare
string access raises ``TypeError``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category

# Private: single source of truth for Category <-> attribute mapping.
_CAT_TO_FIELD: dict[Category, str] = {
    Category.R:    "r",
    Category.HR:   "hr",
    Category.RBI:  "rbi",
    Category.SB:   "sb",
    Category.AVG:  "avg",
    Category.W:    "w",
    Category.K:    "k",
    Category.SV:   "sv",
    Category.ERA:  "era",
    Category.WHIP: "whip",
}


@dataclass
class CategoryStats:
    r:    float = 0.0
    hr:   float = 0.0
    rbi:  float = 0.0
    sb:   float = 0.0
    avg:  float = 0.0
    w:    float = 0.0
    k:    float = 0.0
    sv:   float = 0.0
    era:  float = 99.0
    whip: float = 99.0

    def __getitem__(self, cat: Category) -> float:
        if not isinstance(cat, Category):
            raise TypeError(
                f"CategoryStats indexing requires a Category enum, got "
                f"{type(cat).__name__}"
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
        return {
            cat.value: float(getattr(self, _CAT_TO_FIELD[cat]))
            for cat in ALL_CATEGORIES
        }


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
    """
    team_name: str
    team_key: str
    rank: int
    stats: CategoryStats
    yahoo_points_for: float | None = None


@dataclass
class StandingsSnapshot:
    """All teams' standings at a single effective_date.

    ``effective_date`` is the lineup-lock date the snapshot represents
    (typically the Tuesday the scoring week begins). Entries are not
    required to be in rank order — call :meth:`by_team` for lookup.
    """
    effective_date: date
    entries: list[StandingsEntry]

    def by_team(self) -> dict[str, StandingsEntry]:
        """Return a {team_name: entry} lookup.

        Raises:
            ValueError: if any team name is duplicated in ``entries``.
        """
        out: dict[str, StandingsEntry] = {}
        for entry in self.entries:
            if entry.team_name in out:
                raise ValueError(
                    f"duplicate team in snapshot: {entry.team_name!r}"
                )
            out[entry.team_name] = entry
        return out
