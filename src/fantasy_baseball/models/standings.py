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
from typing import Any, Iterator


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

    def __getitem__(self, key: str) -> float:
        field_name = _KEY_TO_FIELD.get(key)
        if field_name is None:
            raise KeyError(key)
        return getattr(self, field_name)

    def get(self, key: str, default: Any = 0.0) -> Any:
        field_name = _KEY_TO_FIELD.get(key)
        if field_name is None:
            return default
        return getattr(self, field_name)

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
