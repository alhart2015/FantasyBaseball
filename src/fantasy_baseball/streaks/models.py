"""Dataclass models for the streaks DuckDB tables.

Each dataclass corresponds 1:1 to a DuckDB table in `streaks/data/schema.py`.
Field declaration order is the table's column order; the loaders derive their
SQL column tuples from `dataclasses.fields(...)` so this file is the single
source of truth for column names and order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class QualifiedHitter:
    """One entry from the MLB Stats API ≥min_pa leaderboard.

    Not persisted directly — used as the producer→orchestrator handoff for
    "which players should we fetch game logs for this season."
    """

    player_id: int
    name: str
    team: str | None
    pa: int


@dataclass(frozen=True, slots=True)
class HitterGame:
    """One game of hitter counting stats. Maps to `hitter_games` row.

    PK is (player_id, date).
    """

    player_id: int
    name: str
    team: str | None
    season: int
    date: date
    pa: int
    ab: int
    h: int
    hr: int
    r: int
    rbi: int
    sb: int
    bb: int
    k: int


@dataclass(frozen=True, slots=True)
class HitterStatcastPA:
    """One terminal-PA row from Baseball Savant. Maps to `hitter_statcast_pa` row.

    PK is (player_id, date, pa_index). pa_index is sort-derived
    (groupby+cumcount within `pitches_to_pa_rows`) and is not currently
    chronologically stable across re-fetches; suitable for counting and
    aggregation, not for ordered event walks.
    """

    player_id: int
    date: date
    pa_index: int
    event: str | None
    launch_speed: float | None
    launch_angle: float | None
    estimated_woba_using_speedangle: float | None
    barrel: bool | None


@dataclass(frozen=True, slots=True)
class HitterWindow:
    """Rolling-window aggregate. Maps to `hitter_windows` row.

    PK is (player_id, window_end, window_days). Populated in Phase 2.
    """

    player_id: int
    window_end: date
    window_days: int
    pa: int
    hr: int
    r: int
    rbi: int
    sb: int
    avg: float | None
    babip: float | None
    k_pct: float | None
    bb_pct: float | None
    iso: float | None
    ev_avg: float | None
    barrel_pct: float | None
    xwoba_avg: float | None
    pt_bucket: str


@dataclass(frozen=True, slots=True)
class Threshold:
    """One calibrated percentile threshold. Maps to `thresholds` row.

    PK is (season_set, category, window_days, pt_bucket). Populated in Phase 2.
    """

    season_set: str
    category: str
    window_days: int
    pt_bucket: str
    p10: float
    p90: float


@dataclass(frozen=True, slots=True)
class HitterStreakLabel:
    """One hot/cold label for a (player, window, category). Maps to `hitter_streak_labels` row.

    PK is (player_id, window_end, window_days, category). Populated in Phase 2.
    """

    player_id: int
    window_end: date
    window_days: int
    category: str
    label: str
