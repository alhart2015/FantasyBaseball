"""Typed payload for the daily summary email.

One frozen dataclass per section so ``assemble`` and ``render`` never pass raw
dicts around. A section with no data is an empty list / sentinel; ``render``
omits it (except the first-run standings baseline, which renders a message).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class PlayerLine:
    """One rostered player's box-score line from last night."""

    name: str
    group: str  # "hitting" or "pitching"
    stats: dict[str, float]  # hitting: h/hr/r/rbi/sb/ab; pitching: ip/k/er/bb/w/sv/h_allowed


@dataclass(frozen=True)
class StreakItem:
    name: str
    category: str  # e.g. "hr", "sb", "avg"
    label: str  # "hot" or "cold"
    probability: float


@dataclass(frozen=True)
class TeamDelta:
    name: str
    rank_prev: int
    rank_now: int
    points_prev: float
    points_now: float
    category_points_delta: dict[str, float]


@dataclass(frozen=True)
class StandingsDelta:
    is_first_run: bool
    user_team_name: str
    teams: list[TeamDelta] = field(default_factory=list)


@dataclass(frozen=True)
class LineupMove:
    player: str
    action: str  # "start", "sit", "swap"
    from_slot: str
    to_slot: str
    roto_delta: float


@dataclass(frozen=True)
class InjuryItem:
    name: str
    status: str  # IL15 / IL60 / DTD / ...
    note: str


@dataclass(frozen=True)
class ProbableMatchup:
    pitcher: str
    starts: int
    days: str
    opponents: str
    quality: str  # "Great" / "Fair" / "Tough"


@dataclass(frozen=True)
class DailySummary:
    as_of: date
    last_night: list[PlayerLine]
    unmatched: list[str]
    streaks: list[StreakItem]
    standings_delta: StandingsDelta
    lineup_moves: list[LineupMove]
    injuries: list[InjuryItem]
    probables: list[ProbableMatchup]
    section_errors: list[str]
