"""Sunday-report orchestrator + formatter.

Knows about Yahoo + report layout. Pure inference lives in
:mod:`streaks.inference` and is invoked from :func:`build_report`.

The orchestrator is split from the renderers so unit tests can hand-build
a :class:`Report` and snapshot the markdown without touching Yahoo or
sklearn.
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


# Display labels for the five categories — the column headers shown in
# the rendered tables.
_CATEGORY_HEADERS: dict[StreakCategory, str] = {
    "hr": "HR",
    "r": "R",
    "rbi": "RBI",
    "sb": "SB",
    "avg": "AVG",
}

# Sparse cats have no cold model in Phase 4 (per design). When the label
# is cold for these cats we render ``cold —`` (label, no probability).
_SPARSE_CATEGORIES: frozenset[StreakCategory] = frozenset({"hr", "sb"})

# Default season year used when scoring 2026 windows — the report itself
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
    """One player's row in the rendered report.

    ``scores`` is keyed by category for fast cell lookup in the renderer;
    missing entries render as ``—``.
    """

    name: str
    positions: tuple[str, ...]
    player_id: int
    composite: int
    scores: dict[StreakCategory, PlayerCategoryScore]
    max_probability: float

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
    """Full Sunday report payload — input to both renderers."""

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
#                       Yahoo player → mlbam_id                          #
# --------------------------------------------------------------------- #


def build_name_to_mlbam_map(projections_root: Path, *, season: int) -> dict[str, int]:
    """Build a ``{normalized_name: mlbam_id}`` lookup from the season's CSVs.

    Loads every per-system hitter CSV under
    ``{projections_root}/{season}/`` and keys by ``normalize_name(Name)``.
    Conflicts (same normalized name pointing at different MLBAMIDs across
    systems) are resolved by first-write-wins — they're rare in practice
    and the report can tolerate one such miss; logged at WARNING.

    Returns an empty dict when no per-system CSVs are found — the caller
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
    are filtered out before lookup — pitcher streaks are out of scope.
    """
    resolved: list[tuple[YahooHitter, int]] = []
    unresolved: list[str] = []
    for h in yahoo_hitters:
        # Drop pitchers — Phase 5 is hitters-only.
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

    Counts the label regardless of whether a probability was computed —
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
) -> ReportRow:
    by_cat = {s.category: s for s in scores}
    return ReportRow(
        name=name,
        positions=positions,
        player_id=player_id,
        composite=_composite(scores),
        scores=by_cat,
        max_probability=_max_probability(scores),
    )


def _driver_lines_from_rows(rows: Iterable[ReportRow]) -> list[DriverLine]:
    """Surface every (player, cat) prediction that has both a probability
    and at least one driver attribution. Composite-sorted by row, then
    by category order — same as the rendered tables for visual continuity.
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

    Doesn't fetch from Yahoo — the CLI is responsible for that, so this
    function stays testable without Yahoo mocks. Doesn't refit models —
    same reasoning; refitting is the CLI's job.
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
            )
        )
    roster_rows.sort(key=lambda r: r.sort_key)

    # FA rows. Composite=0 rows aren't useful in the FA section so they
    # are dropped before sorting — that lets the "top 10 by |composite|"
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


# --------------------------------------------------------------------- #
#                         Markdown rendering                             #
# --------------------------------------------------------------------- #


def _format_cell(score: PlayerCategoryScore | None, *, sparse: bool) -> str:
    """Render one category cell.

    - Neutral / missing score: ``—``.
    - Hot/cold with probability: ``"hot 0.71"``, ``"cold 0.78"``.
    - Hot/cold without probability: ``"hot —"`` / ``"cold —"``. (Sparse
      cold has no model; missing projection rate also lands here.)
    """
    if score is None or score.label == "neutral":
        return "—"
    if score.probability is None:
        if sparse and score.label == "cold":
            return "cold —"
        return f"{score.label} —"
    return f"{score.label} {score.probability:.2f}"


def _signed(n: int) -> str:
    """Signed composite-score string: ``+3``, ``0``, ``−2`` (true minus)."""
    if n > 0:
        return f"+{n}"
    if n < 0:
        return f"−{abs(n)}"
    return "0"


def _format_positions(positions: tuple[str, ...]) -> str:
    return "/".join(positions) if positions else ""


def _markdown_separator_row(n_cols: int) -> str:
    """Three dashes per column — minimal valid markdown separator."""
    return "|" + "|".join([" --- "] * n_cols) + "|"


def _roster_table_markdown(rows: Sequence[ReportRow]) -> list[str]:
    """Render the roster table — uniform 5-cat grid."""
    headers = ["Player", "Pos", "Comp", *(_CATEGORY_HEADERS[c] for c in REPORT_CATEGORIES)]
    lines = [
        "| " + " | ".join(headers) + " |",
        _markdown_separator_row(len(headers)),
    ]
    for row in rows:
        cells = [
            row.name,
            _format_positions(row.positions),
            _signed(row.composite),
        ]
        for cat in REPORT_CATEGORIES:
            cells.append(_format_cell(row.scores.get(cat), sparse=cat in _SPARSE_CATEGORIES))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _fa_table_markdown(rows: Sequence[ReportRow]) -> list[str]:
    """Render the free-agent table — non-neutral cats concatenated."""
    headers = ["Player", "Pos", "Comp", "Active Streaks"]
    lines = [
        "| " + " | ".join(headers) + " |",
        _markdown_separator_row(len(headers)),
    ]
    for row in rows:
        active: list[str] = []
        for cat in REPORT_CATEGORIES:
            score = row.scores.get(cat)
            if score is None or score.label == "neutral":
                continue
            cell = _format_cell(score, sparse=cat in _SPARSE_CATEGORIES)
            active.append(f"{cell} {_CATEGORY_HEADERS[cat]}")
        cells = [
            row.name,
            _format_positions(row.positions),
            _signed(row.composite),
            ", ".join(active) if active else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _driver_block_markdown(lines: Sequence[DriverLine]) -> list[str]:
    out: list[str] = []
    for line in lines:
        drivers = ", ".join(_format_driver(d) for d in line.drivers)
        cat_header = _CATEGORY_HEADERS[line.category]
        out.append(
            f"**{line.player_name} — {cat_header} {line.label} "
            f"{line.probability:.2f}**  →  {drivers}"
        )
    return out


def _format_driver(driver: Driver) -> str:
    """``xwoba_avg +1.8σ`` — sigma symbol matches the spec."""
    sign = "+" if driver.z_score >= 0 else "−"
    return f"{driver.feature} {sign}{abs(driver.z_score):.1f}σ"


def render_markdown(report: Report) -> str:
    """Render the full report as a markdown string.

    Matches the skeleton in the design spec; intended to be written to
    ``data/streaks/reports/YYYY-MM-DD.md``.
    """
    parts: list[str] = []
    parts.append(f"# Streaks — Sunday Report — {report.report_date.isoformat()}")
    window_text = (
        report.window_end.isoformat() if report.window_end is not None else "(no recent windows)"
    )
    parts.append(f"*Models refit on {report.season_set_train}; data through {window_text}*")
    parts.append("")
    parts.append(f"## Your Roster — {report.team_name} (League {report.league_id})")
    parts.append("Sorted by composite (#hot − #cold), tiebreak max continuation probability.")
    parts.append("")
    if report.roster_rows:
        parts.extend(_roster_table_markdown(report.roster_rows))
    else:
        parts.append("_No roster hitters resolved._")
    parts.append("")
    parts.append("## Top 10 Free Agent Signals")
    parts.append("Top 10 available hitters by |composite|, non-neutral cats only.")
    parts.append("")
    if report.fa_rows:
        parts.extend(_fa_table_markdown(report.fa_rows))
    else:
        parts.append("_No free-agent signals._")
    parts.append("")
    if report.driver_lines:
        parts.append("## Drivers (top peripheral signal per active prediction)")
        parts.append("")
        parts.extend(_driver_block_markdown(report.driver_lines))
        parts.append("")
    if report.skipped:
        parts.append(f"_Skipped {len(report.skipped)} player(s): {', '.join(report.skipped)}._")
        parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------- #
#                       Terminal rendering                               #
# --------------------------------------------------------------------- #

# ANSI codes for hot/cold cells. Disabled when ``no_color=True`` so the
# output remains readable when piped to a file or non-TTY.
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"


def _paint_cell(score: PlayerCategoryScore | None, *, sparse: bool, no_color: bool) -> str:
    """Same content as ``_format_cell``, plus ANSI color for hot/cold."""
    text = _format_cell(score, sparse=sparse)
    if no_color or score is None or score.label == "neutral":
        return text
    color = _ANSI_GREEN if score.label == "hot" else _ANSI_RED
    return f"{color}{text}{_ANSI_RESET}"


def _column_widths(rows: Sequence[Sequence[str]]) -> list[int]:
    """Compute the visible width of each column, stripping ANSI."""
    widths: list[int] = []
    for r_idx, row in enumerate(rows):
        for c_idx, cell in enumerate(row):
            w = _visible_width(cell)
            if c_idx >= len(widths):
                widths.append(w)
            elif w > widths[c_idx]:
                widths[c_idx] = w
        if r_idx == 0:
            continue
    return widths


def _visible_width(s: str) -> int:
    """Width of ``s`` ignoring ANSI escape sequences (e.g. ``\\033[32m``)."""
    visible = []
    i = 0
    while i < len(s):
        if s[i] == "\033":
            # Skip to the ``m`` that closes the escape.
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
            continue
        visible.append(s[i])
        i += 1
    return len("".join(visible))


def _pad(cell: str, width: int) -> str:
    return cell + " " * (width - _visible_width(cell))


def _render_text_table(rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a fixed-width text table with simple ASCII separators."""
    widths = _column_widths(rows)
    lines: list[str] = []
    for r_idx, row in enumerate(rows):
        padded = [_pad(cell, widths[c]) for c, cell in enumerate(row)]
        lines.append("  ".join(padded).rstrip())
        if r_idx == 0:
            lines.append("  ".join("-" * widths[c] for c in range(len(widths))))
    return lines


def render_terminal(report: Report, *, no_color: bool = False) -> str:
    """Render the report as a single string suitable for ``print``.

    Returns a string rather than printing so callers can choose where the
    output goes (stdout, a log, a test capture). The CLI prints it.
    """
    parts: list[str] = []
    parts.append(f"Streaks — Sunday Report — {report.report_date.isoformat()}")
    window_text = (
        report.window_end.isoformat() if report.window_end is not None else "(no recent windows)"
    )
    parts.append(f"Models refit on {report.season_set_train}; data through {window_text}")
    parts.append("")
    parts.append(f"Your Roster — {report.team_name} (League {report.league_id})")
    if report.roster_rows:
        roster_rows = [
            ["Player", "Pos", "Comp", *(_CATEGORY_HEADERS[c] for c in REPORT_CATEGORIES)]
        ]
        for r in report.roster_rows:
            row = [
                r.name,
                _format_positions(r.positions),
                _signed(r.composite),
            ]
            for cat in REPORT_CATEGORIES:
                row.append(
                    _paint_cell(
                        r.scores.get(cat), sparse=cat in _SPARSE_CATEGORIES, no_color=no_color
                    )
                )
            roster_rows.append(row)
        parts.extend(_render_text_table(roster_rows))
    else:
        parts.append("(no roster hitters resolved)")
    parts.append("")
    parts.append("Top 10 Free Agent Signals")
    if report.fa_rows:
        fa_rows = [["Player", "Pos", "Comp", "Active Streaks"]]
        for r in report.fa_rows:
            active_chunks: list[str] = []
            for cat in REPORT_CATEGORIES:
                score = r.scores.get(cat)
                if score is None or score.label == "neutral":
                    continue
                cell = _paint_cell(score, sparse=cat in _SPARSE_CATEGORIES, no_color=no_color)
                active_chunks.append(f"{cell} {_CATEGORY_HEADERS[cat]}")
            fa_rows.append(
                [
                    r.name,
                    _format_positions(r.positions),
                    _signed(r.composite),
                    ", ".join(active_chunks) if active_chunks else "—",
                ]
            )
        parts.extend(_render_text_table(fa_rows))
    else:
        parts.append("(no free-agent signals)")
    parts.append("")
    if report.driver_lines:
        parts.append("Drivers (top peripheral signal per active prediction)")
        for line in report.driver_lines:
            drivers = ", ".join(_format_driver(d) for d in line.drivers)
            cat_header = _CATEGORY_HEADERS[line.category]
            parts.append(
                f"  {line.player_name} — {cat_header} {line.label} "
                f"{line.probability:.2f}  →  {drivers}"
            )
        parts.append("")
    if report.skipped:
        parts.append(f"Skipped {len(report.skipped)}: {', '.join(report.skipped)}")
    return "\n".join(parts)
