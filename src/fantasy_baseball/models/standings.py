"""League-level roto category statistics.

Defines the three snapshot-layer dataclasses used by League:
:class:`CategoryStats` (the ten roto totals), :class:`StandingsEntry`
(one team's stats + rank), and :class:`StandingsSnapshot` (all teams
at an effective_date).

``CategoryStats`` has a small dict-compat surface (``__getitem__``,
``get``, ``items``) so call sites that currently do
``stats["R"]``-style access keep working during the migration. These
compat methods get deleted in Step 9 of the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterator

from fantasy_baseball.utils.constants import Category

# Canonical category order — used for iteration and display
CATEGORY_ORDER: tuple[str, ...] = (
    "R", "HR", "RBI", "SB", "AVG",
    "W", "K", "SV", "ERA", "WHIP",
)

# Map between uppercase category key and dataclass field name
_KEY_TO_FIELD: dict[str, str] = {
    "R": "r", "HR": "hr", "RBI": "rbi", "SB": "sb", "AVG": "avg",
    "W": "w", "K": "k", "SV": "sv", "ERA": "era", "WHIP": "whip",
}


def _normalize_key(key: Any) -> str | None:
    """Return the uppercase string form of a category key, or None if unknown.

    Accepts either a ``Category`` enum member or a bare uppercase string
    (``"R"``, ``"HR"``, …) — during the StrEnum→Enum migration, the
    dict-compat surface needs to keep working for callers that still
    pass strings while also accepting the new enum-typed keys.
    """
    if isinstance(key, Category):
        return key.value
    if isinstance(key, str):
        return key
    return None


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

    # -- Dict-compat surface (deleted in Step 9 of the migration) --

    def __getitem__(self, key: str | Category) -> float:
        normalized = _normalize_key(key)
        field_name = _KEY_TO_FIELD.get(normalized) if normalized is not None else None
        if field_name is None:
            raise KeyError(key)
        return float(getattr(self, field_name))

    def get(self, key: str | Category, default: float = 0.0) -> float:
        normalized = _normalize_key(key)
        field_name = _KEY_TO_FIELD.get(normalized) if normalized is not None else None
        if field_name is None:
            return default
        return float(getattr(self, field_name))

    def keys(self) -> list[str]:
        """Return the category keys in canonical order.

        Present so ``dict(cs)`` and ``**cs`` unpacking work in legacy
        call sites (e.g. ``scripts/simulate_draft.py``) that treat the
        return of ``project_team_stats`` as a plain mapping. Step 9
        cleanup removes this along with the rest of the dict-compat
        surface.
        """
        return list(CATEGORY_ORDER)

    def __iter__(self) -> Iterator[str]:
        """Iterate keys in canonical order (mapping protocol)."""
        return iter(CATEGORY_ORDER)

    def items(self) -> Iterator[tuple[str, float]]:
        for key in CATEGORY_ORDER:
            yield key, getattr(self, _KEY_TO_FIELD[key])

    # -- Constructors --

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CategoryStats":
        """Build from a ``{"R": val, "HR": val, ...}`` dict.

        Missing keys fall back to the dataclass defaults (0 for counting
        stats, 99 for ERA/WHIP which are inverse and shouldn't default
        to 0 — that would make a missing team rank first).
        """
        kwargs: dict[str, Any] = {}
        for key, field_name in _KEY_TO_FIELD.items():
            if key in d:
                kwargs[field_name] = float(d[key])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, float]:
        """Return the dict form used by cache JSON."""
        return {key: getattr(self, field_name)
                for key, field_name in _KEY_TO_FIELD.items()}


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
