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
class CategoryEroto:
    """One category's expected-roto (Gaussian) points for the user's team on the
    PROJECTED end-of-season standings, with the prior-snapshot value for the
    overnight delta (``prev`` is None on the first run)."""

    category: str  # "R", "HR", ... "ERA", "WHIP"
    now: float
    prev: float | None


@dataclass(frozen=True)
class ProjectionDelta:
    """Projected-finish movement for the user's team: per-category expected roto
    (eRoto) and Monte Carlo championship odds, each with its overnight change.

    All fields default to an empty/first-run state so a summary with no cached
    projections is still constructible; ``render`` omits the panel via
    ``has_content``."""

    is_first_run: bool = True
    eroto: list[CategoryEroto] = field(default_factory=list)
    eroto_total_now: float = 0.0
    eroto_total_prev: float | None = None
    champ_pct_now: float | None = None  # MC first_pct (championship odds)
    champ_pct_prev: float | None = None

    @property
    def has_content(self) -> bool:
        return bool(self.eroto) or self.champ_pct_now is not None


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
    projections: ProjectionDelta = field(default_factory=ProjectionDelta)
