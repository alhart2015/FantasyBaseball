"""Streak-report orchestrator: Yahoo hitters -> the :class:`Report` the /streaks
page shows (via :mod:`streaks.dashboard`). Pure inference lives in
:mod:`streaks.inference` and is invoked from :func:`build_report`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from fantasy_baseball.streaks.data.projections import discover_projection_files
from fantasy_baseball.streaks.inference import (
    REPORT_CATEGORIES,
    STALE_TOLERANCE_DAYS,
    Driver,
    FittedModel,
    PlayerCategoryScore,
    ScoreSkip,
    score_player_windows,
)
from fantasy_baseball.streaks.models import StreakCategory, StreakDirection, StreakLabel
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter

logger = logging.getLogger(__name__)


# Default season year used when scoring 2026 windows -- the report itself
# is parameterized but the scoring season comes from league_config.
_DEFAULT_SCORING_SEASON = 2026


@dataclass(frozen=True)
class YahooHitter:
    """One hitter pulled from Yahoo (roster or FA), normalized for matching."""

    name: str
    positions: tuple[str, ...]
    yahoo_id: str
    status: str


@dataclass(frozen=True)
class ReportRow:
    """One player's row in the report.

    ``scores`` is keyed by category; a category with no entry has no score.

    ``days_since_last_game`` is set (an int) only when the player's
    most-recent window is stale (> STALE_TOLERANCE_DAYS before the run
    date) -- i.e. he is inactive; ``None`` means active/recent.
    """

    name: str
    positions: tuple[str, ...]
    player_id: int
    composite: int
    scores: dict[StreakCategory, PlayerCategoryScore]
    max_probability: float
    days_since_last_game: int | None = None

    @property
    def sort_key(self) -> tuple[float, float, str]:
        """Roster ordering: composite desc, max_probability desc, name asc."""
        return (-self.composite, -self.max_probability, self.name.lower())

    @property
    def fa_sort_key(self) -> tuple[float, float, str]:
        """FA ordering: |composite| desc, max_probability desc, name asc."""
        return (-abs(self.composite), -self.max_probability, self.name.lower())


@dataclass(frozen=True)
class DriverLine:
    """One driver-attribution line under the tables."""

    player_name: str
    category: StreakCategory
    label: StreakLabel
    probability: float
    drivers: tuple[Driver, ...]


@dataclass(frozen=True)
class Report:
    """Full streak report payload, cached for the /streaks page."""

    report_date: date
    window_end: date | None
    team_name: str
    league_id: int
    season_set_train: str
    roster_rows: tuple[ReportRow, ...]
    fa_rows: tuple[ReportRow, ...]
    driver_lines: tuple[DriverLine, ...]
    skipped: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------- #
#                       Yahoo player -> mlbam_id                          #
# --------------------------------------------------------------------- #


def build_name_to_mlbam_map(projections_root: Path, *, season: int) -> dict[str, int]:
    """Build a ``{normalized_name: mlbam_id}`` lookup from the season's CSVs.

    Loads every per-system hitter CSV under
    ``{projections_root}/{season}/`` and keys by ``normalize_name(Name)``.
    Conflicts (same normalized name pointing at different MLBAMIDs across
    systems) are resolved by first-write-wins -- they're rare in practice
    and the report can tolerate one such miss; logged at WARNING.

    Returns an empty dict when no per-system CSVs are found -- the caller
    should treat that as a hard error.
    """
    out: dict[str, int] = {}
    files = discover_projection_files(projections_root, season=season)
    if not files:
        logger.warning("No hitter projection CSVs found under %s/%d/", projections_root, season)
        return out
    for path in files:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=["Name", "MLBAMID"])
        except (ValueError, FileNotFoundError):
            logger.warning("Skipping %s: missing Name or MLBAMID column", path)
            continue
        df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
        df = df.dropna(subset=["MLBAMID"])
        for name, mlbam in zip(df["Name"], df["MLBAMID"].astype(int), strict=True):
            normalized = normalize_name(str(name))
            if normalized in out and out[normalized] != int(mlbam):
                logger.debug(
                    "name->mlbam conflict for %r: %d vs %d (keeping first)",
                    normalized,
                    out[normalized],
                    int(mlbam),
                )
                continue
            out.setdefault(normalized, int(mlbam))
    return out


def resolve_hitters(
    yahoo_hitters: Sequence[YahooHitter],
    name_to_mlbam: dict[str, int],
) -> tuple[list[tuple[YahooHitter, int]], list[str]]:
    """Pair Yahoo hitters with mlbam_ids; collect names we could not resolve.

    Returns ``(resolved, unresolved_names)``. Pitcher positions (SP, RP)
    are filtered out before lookup -- pitcher streaks are out of scope.
    """
    resolved: list[tuple[YahooHitter, int]] = []
    unresolved: list[str] = []
    for h in yahoo_hitters:
        # Drop pitchers -- Phase 5 is hitters-only.
        if not is_hitter(h.positions):
            continue
        mlbam = name_to_mlbam.get(normalize_name(h.name))
        if mlbam is None:
            unresolved.append(h.name)
            continue
        resolved.append((h, mlbam))
    return resolved, unresolved


# --------------------------------------------------------------------- #
#                       Report construction                              #
# --------------------------------------------------------------------- #


def _composite(scores: Iterable[PlayerCategoryScore]) -> int:
    """``#hot - #cold`` over the player's category scores.

    Counts the label regardless of whether a probability was computed --
    a sparse-cat cold without a model still represents a cold streak.
    """
    composite = 0
    for s in scores:
        if s.label == "hot":
            composite += 1
        elif s.label == "cold":
            composite -= 1
    return composite


def _max_probability(scores: Iterable[PlayerCategoryScore]) -> float:
    """Max continuation probability across cats; 0.0 if none scored.

    Used as the tiebreaker in roster + FA ordering.
    """
    best = 0.0
    for s in scores:
        if s.probability is not None and s.probability > best:
            best = s.probability
    return best


def _row_from_scores(
    *,
    name: str,
    positions: tuple[str, ...],
    player_id: int,
    scores: list[PlayerCategoryScore],
    today: date,
) -> ReportRow:
    by_cat = {s.category: s for s in scores}
    # window_end is identical across a player's category scores (stamped once
    # per player in score_player_windows), so any score carries it.
    window_end = scores[0].window_end if scores else None
    days_since_last_game: int | None = None
    if window_end is not None:
        days = (today - window_end).days
        if days > STALE_TOLERANCE_DAYS:
            days_since_last_game = days
    return ReportRow(
        name=name,
        positions=positions,
        player_id=player_id,
        composite=_composite(scores),
        scores=by_cat,
        max_probability=_max_probability(scores),
        days_since_last_game=days_since_last_game,
    )


def _driver_lines_from_rows(rows: Iterable[ReportRow]) -> list[DriverLine]:
    """Surface every (player, cat) prediction that has both a probability
    and at least one driver attribution. Composite-sorted by row, then
    by category order -- same as the rendered tables for visual continuity.
    """
    lines: list[DriverLine] = []
    for row in rows:
        for cat in REPORT_CATEGORIES:
            score = row.scores.get(cat)
            if score is None or score.probability is None or not score.drivers:
                continue
            lines.append(
                DriverLine(
                    player_name=row.name,
                    category=cat,
                    label=score.label,
                    probability=score.probability,
                    drivers=score.drivers,
                )
            )
    return lines


def build_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    league_config_team_name: str,
    league_config_league_id: int,
    models: dict[tuple[StreakCategory, StreakDirection], FittedModel],
    roster_hitters: Sequence[YahooHitter],
    fa_hitters: Sequence[YahooHitter],
    name_to_mlbam: dict[str, int],
    today: date,
    season_set_train: str = "2023-2025",
    scoring_season: int = _DEFAULT_SCORING_SEASON,
    window_days: int = 14,
    top_n_fas: int = 10,
) -> Report:
    """Build a :class:`Report` end-to-end from already-fetched Yahoo data.

    Doesn't fetch from Yahoo -- streaks.pipeline does that, so this
    function stays testable without Yahoo mocks. Doesn't refit models --
    same reasoning; refitting is streaks.pipeline's job.
    """
    resolved_roster, roster_unresolved = resolve_hitters(roster_hitters, name_to_mlbam)
    resolved_fas, _fa_unresolved = resolve_hitters(fa_hitters, name_to_mlbam)

    roster_player_ids = [mlbam for _, mlbam in resolved_roster]
    fa_player_ids = [mlbam for _, mlbam in resolved_fas]

    all_scores, all_skips = score_player_windows(
        conn,
        models=models,
        player_ids=roster_player_ids + fa_player_ids,
        window_end_on_or_before=today,
        window_days=window_days,
        scoring_season=scoring_season,
        season_set=season_set_train,
    )
    scores_by_pid: dict[int, list[PlayerCategoryScore]] = {}
    for s in all_scores:
        scores_by_pid.setdefault(s.player_id, []).append(s)

    skipped_pids = {sk.player_id for sk in all_skips}

    # Build roster rows for every resolved roster hitter who has any scores.
    roster_rows: list[ReportRow] = []
    skipped_names: list[str] = list(roster_unresolved)
    for hitter, mlbam in resolved_roster:
        scores = scores_by_pid.get(mlbam)
        if scores is None or mlbam in skipped_pids:
            skipped_names.append(hitter.name)
            continue
        roster_rows.append(
            _row_from_scores(
                name=hitter.name,
                positions=tuple(hitter.positions),
                player_id=mlbam,
                scores=scores,
                today=today,
            )
        )
    roster_rows.sort(key=lambda r: r.sort_key)

    # FA rows. Composite=0 rows aren't useful in the FA section so they
    # are dropped before sorting -- that lets the "top 10 by |composite|"
    # logic do the right thing when more than 10 FAs share composite=0.
    fa_rows: list[ReportRow] = []
    for hitter, mlbam in resolved_fas:
        scores = scores_by_pid.get(mlbam)
        if scores is None or mlbam in skipped_pids:
            skipped_names.append(hitter.name)
            continue
        row = _row_from_scores(
            name=hitter.name,
            positions=tuple(hitter.positions),
            player_id=mlbam,
            scores=scores,
            today=today,
        )
        if row.composite == 0:
            continue
        fa_rows.append(row)
    fa_rows.sort(key=lambda r: r.fa_sort_key)
    fa_rows = fa_rows[:top_n_fas]

    driver_lines = _driver_lines_from_rows(roster_rows + fa_rows)

    # Determine the most recent window_end across all scored rows.
    window_end: date | None = None
    for s in all_scores:
        if s.window_end is not None and (window_end is None or s.window_end > window_end):
            window_end = s.window_end

    skipped_names.extend(_skip_reason_text(sk) for sk in all_skips)

    return Report(
        report_date=today,
        window_end=window_end,
        team_name=league_config_team_name,
        league_id=league_config_league_id,
        season_set_train=season_set_train,
        roster_rows=tuple(roster_rows),
        fa_rows=tuple(fa_rows),
        driver_lines=tuple(driver_lines),
        skipped=tuple(skipped_names),
    )


def _skip_reason_text(skip: ScoreSkip) -> str:
    """One-line summary of an inference-skip for the footer."""
    return f"player_id={skip.player_id} ({skip.reason})"
