"""ASCII terminal report for the Yahoo-free (manual) roster audit.

Presentation only. This module formats :class:`~fantasy_baseball.lineup.
roster_audit.AuditEntry` objects that ``audit_roster`` already produced; it
computes nothing. No SGP, no deltaRoto, no confidence bands, no re-ranking --
every number printed here is read straight off an entry (or off the top
candidate dict hanging on that entry), and the row order is the order
``audit_roster`` returned, which is already biggest-deltaRoto-first.

If a number you want is not already on an ``AuditEntry``, it does not belong
in this file: it belongs upstream, in the audit.

Strictly ASCII, per the repo rule -- this renders on a Windows cp1252 console
where a single non-ASCII glyph raises ``UnicodeEncodeError`` and kills the
run. Arrows are ``->``, the rule characters are ``=``, ``-`` and ``|``, and
nothing here is padded to a fixed width that could truncate a real player
name (some are long, and the accented ones arrive from the data unchanged --
the entry-point script is responsible for
``sys.stdout.reconfigure(encoding="utf-8", errors="replace")``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from fantasy_baseball.lineup.roster_audit import AuditEntry
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.utils.ansi import column_widths, pad
from fantasy_baseball.utils.constants import ALL_CATEGORIES

#: Rendered in a numeric cell whose source value is missing (never 0.0 -- a
#: real zero and an absent number must not look the same).
MISSING = "--"

#: Rendered in the "best free agent" column of a roster spot the audit found
#: no upgrade for. Spelled out rather than left blank so an empty cell can
#: never be misread as a rendering failure.
NO_UPGRADE = "no upgrade"

_MIN_RULE_WIDTH = 64

_CAVEAT_LINES = (
    "the free-agent pool is projection-derived, NOT a live Yahoo availability",
    "list. Ownership and injury status are UNKNOWN -- confirm each add is",
    "really a free agent, and really healthy, before you make the move.",
)


# --------------------------------------------------------------------------
# Reads. Every function below pulls an already-computed value off an entry.
# --------------------------------------------------------------------------


def _top_candidate(entry: AuditEntry) -> dict[str, Any] | None:
    """The candidate backing ``entry.best_fa``, or None when there is no upgrade.

    ``audit_roster`` sorts ``candidates`` by deltaRoto and only promotes
    ``candidates[0]`` to ``best_fa`` when its deltaRoto is positive, so
    "``best_fa`` is set" and "``candidates[0]`` is the recommendation" are
    the same condition.
    """
    if entry.best_fa is None or not entry.candidates:
        return None
    top = entry.candidates[0]
    return top if isinstance(top, dict) else None


def _delta_roto(entry: AuditEntry) -> Mapping[str, Any] | None:
    """The top candidate's ``DeltaRotoResult.to_dict()`` payload, if present."""
    top = _top_candidate(entry)
    if top is None:
        return None
    dr = top.get("delta_roto")
    return dr if isinstance(dr, Mapping) else None


def _delta_roto_total(entry: AuditEntry) -> float | None:
    """Projected roto-point gain of the recommended swap. Read, not computed."""
    dr = _delta_roto(entry)
    if dr is None:
        return None
    total = dr.get("total")
    return float(total) if total is not None else None


def _category_deltas(entry: AuditEntry) -> dict[str, float]:
    """``{category: roto_delta}`` for the recommended swap, as the audit stored it."""
    dr = _delta_roto(entry)
    if dr is None:
        return {}
    cats = dr.get("categories")
    if not isinstance(cats, Mapping):
        return {}
    out: dict[str, float] = {}
    for name, payload in cats.items():
        if isinstance(payload, Mapping):
            value = payload.get("roto_delta")
        else:
            value = payload
        if value is not None:
            out[str(name)] = float(value)
    return out


def _band(entry: AuditEntry) -> Mapping[str, Any] | None:
    """The top candidate's ``DeltaRotoBand.to_dict()`` payload, if present."""
    top = _top_candidate(entry)
    if top is None:
        return None
    band = top.get("band")
    return band if isinstance(band, Mapping) else None


def _p_helps(entry: AuditEntry) -> float | None:
    band = _band(entry)
    if band is None:
        return None
    p = band.get("p_positive")
    return float(p) if p is not None else None


def _verdict(entry: AuditEntry) -> str:
    band = _band(entry)
    if band is None:
        return MISSING
    verdict = band.get("verdict")
    return MISSING if verdict is None else str(verdict)


def _before_total(entries: Sequence[AuditEntry]) -> float | None:
    """The team's projected roto total as currently constructed.

    Every candidate's ``delta_roto`` carries the same pre-swap team total, so
    the first one that has it answers for the whole roster. Nothing is
    derived -- this is a lookup with a fallback, not a calculation.
    """
    for entry in entries:
        dr = _delta_roto(entry)
        if dr is None:
            continue
        before = dr.get("before_total")
        if before is not None:
            return float(before)
    return None


def is_upgrade(entry: AuditEntry) -> bool:
    """True when the audit found a free agent worth adding for this player."""
    return entry.best_fa is not None


def is_injured(entry: AuditEntry) -> bool:
    """True for the IL rows ``audit_roster`` appends after the scored entries."""
    return entry.slot == "IL"


# --------------------------------------------------------------------------
# Formatting primitives.
# --------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    """A finite real number, or None for anything else.

    Every number in this report arrives out of a cached JSON blob, and a blob is not
    a type. What gets rejected is chosen so that a value nobody computed can never
    print as one that looks computed:

    - a bool, because ``True`` IS a float in Python and would render "+1.00", a
      perfectly plausible roto delta;
    - a string, even a numeric one. ``float("0.46")`` succeeds, so accepting strings
      would let a writer that stringified its numbers render a full report that looks
      right while the type mismatch -- the actual defect -- stays invisible. "--" is
      the honest answer: this build did not produce that number;
    - NaN and the infinities, which format as "+nan" and "inf%" inside a table a
      reader scans for magnitudes.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _num(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    return MISSING if number is None else f"{number:.{digits}f}"


def _signed(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    return MISSING if number is None else f"{number:+.{digits}f}"


def _pct(value: Any, digits: int = 0) -> str:
    number = _as_float(value)
    return MISSING if number is None else f"{number * 100:.{digits}f}%"


def _positions(positions: Sequence[str] | None) -> str:
    if not positions:
        return MISSING
    return "/".join(str(p) for p in positions)


def _cell_delta(value: float | None) -> str:
    """A per-category roto delta, with a rounds-to-nothing value shown as '.'."""
    if value is None:
        return MISSING
    text = f"{value:+.2f}"
    return "." if text in ("+0.00", "-0.00") else text


def _format_entry(entry: AuditEntry) -> str:
    """One-line description of an entry's recommended move.

    ``"Yoan Moncada (3B, BN) -> Jose Ramirez (3B)"`` for an upgrade,
    ``"Yoan Moncada (3B, BN) -> no upgrade"`` when the audit found none.
    """
    left = f"{entry.player} ({_positions(entry.positions)}, {entry.slot})"
    if entry.best_fa is None:
        return f"{left} -> {NO_UPGRADE}"
    return f"{left} -> {entry.best_fa} ({_positions(entry.best_fa_positions)})"


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    right_align: Sequence[int] = (),
) -> list[str]:
    """A header + separator + body table, every column sized from its content.

    No column has a fixed cap, so a long or accented name is never truncated.
    """
    grid = [list(headers), *[list(r) for r in rows]]
    widths = column_widths(grid)
    right = set(right_align)
    out: list[str] = []
    for index, row in enumerate(grid):
        cells = [pad(cell, widths[col], right=col in right) for col, cell in enumerate(row)]
        out.append(" | ".join(cells).rstrip())
        if index == 0:
            out.append("-+-".join("-" * w for w in widths))
    return out


# --------------------------------------------------------------------------
# Sections.
# --------------------------------------------------------------------------


def _provenance_lines(
    *,
    team_name: str,
    effective_date: date,
    fraction_remaining: float | None,
    ros_snapshot_date: str,
    kv_path: str | None,
    dropped_rows: int = 0,
) -> list[str]:
    fields = [
        ("team", team_name),
        ("pipeline", "manual (Yahoo-free; no Yahoo API calls were made)"),
        ("kv store", MISSING if kv_path is None else str(kv_path)),
        ("roster snapshot", effective_date.isoformat()),
        ("ros projections", ros_snapshot_date),
        # `_pct1`, not a 0.0 default upstream: an absent number rendered as
        # "0.0%" is a specific and false claim that the season is over, on the
        # PROVENANCE block -- the one part of the report a reader trusts to say
        # what the rest was built from. MISSING is why it exists.
        ("season remaining", _pct(fraction_remaining, digits=1)),
    ]
    if dropped_rows:
        # IN THE REPORT, not only on the terminal. This file is the artifact -- what a
        # later session reads and what the user comes back to -- and a partial audit
        # presented as a complete one is the failure this whole block exists to
        # prevent. A stdout NOTE scrolls away; this does not.
        fields.append(
            (
                "INCOMPLETE",
                f"{dropped_rows} audit row(s) were unreadable and are missing from "
                "this report -- re-run the pipeline to rebuild them",
            )
        )
    label_width = max(len(label) for label, _ in fields)
    lines = [f"{pad(label, label_width)} : {value}" for label, value in fields]
    lines.append("")
    lines.append("CAVEAT:")
    lines.extend(f"  {line}" for line in _CAVEAT_LINES)
    return lines


def _standings_lines(
    entries: Sequence[AuditEntry],
    *,
    team_name: str,
    projected_standings: ProjectedStandings | None,
    roto_standings: Sequence[tuple[str, float]] | None,
) -> list[str]:
    lines: list[str] = []

    before = _before_total(entries)
    if before is not None:
        lines.append(f"projected roto points as currently rostered : {_num(before)}")
        lines.append("")

    if roto_standings:
        rows = []
        for index, (name, points) in enumerate(roto_standings, start=1):
            marker = "*" if name == team_name else ""
            rows.append([str(index), f"{name}{marker}", _num(points)])
        lines.extend(_render_table(["#", "TEAM", "ROTO PTS"], rows, right_align=(0, 2)))
        lines.append("")
        lines.append("(* your team; order and points come from the pipeline as-scored)")
    elif projected_standings is not None:
        entry_by_team = projected_standings.by_team()
        mine = entry_by_team.get(team_name)
        if mine is None:
            lines.append(f"{team_name} is not in the projected standings payload.")
        else:
            stats = mine.stats.to_dict()
            headers = [c.value for c in ALL_CATEGORIES]
            row = [
                MISSING if stats.get(c.value) is None else f"{float(stats[c.value]):g}"
                for c in ALL_CATEGORIES
            ]
            lines.append("projected end-of-season totals:")
            lines.extend(_render_table(headers, [row], right_align=tuple(range(len(headers)))))
            lines.append("")
            lines.append("(per-team roto points were not supplied; category totals only)")
    else:
        lines.append("no standings payload supplied.")

    return lines


def _upgrade_lines(upgrades: Sequence[AuditEntry], *, shown: int, total: int) -> list[str]:
    if not upgrades:
        return [
            "no upgrades found -- the audit could not beat any roster spot with",
            "the available free agents. Hold the roster as-is.",
        ]

    headers = [
        "#",
        "DROP",
        "POS",
        "SLOT",
        "SGP",
        "ADD",
        "POS",
        "SGP",
        "ROTO GAIN",
        "P(HELPS)",
        "VERDICT",
        "SGP GAP",
    ]
    rows: list[list[str]] = []
    for index, entry in enumerate(upgrades, start=1):
        rows.append(
            [
                str(index),
                entry.player,
                _positions(entry.positions),
                entry.slot,
                _num(entry.player_sgp),
                MISSING if entry.best_fa is None else entry.best_fa,
                _positions(entry.best_fa_positions),
                _num(entry.best_fa_sgp),
                _signed(_delta_roto_total(entry)),
                _pct(_p_helps(entry)),
                _verdict(entry),
                _signed(entry.gap),
            ]
        )
    lines = _render_table(headers, rows, right_align=(0, 4, 7, 8, 9, 11))
    lines.append("")
    if shown < total:
        lines.append(f"showing the top {shown} of {total} upgrades.")
    lines.append("ROTO GAIN is the projected roto-point change of the swap; SGP GAP is the")
    lines.append("raw SGP difference between the two players. Rows are ranked by ROTO GAIN.")
    return lines


def _category_lines(upgrades: Sequence[AuditEntry]) -> list[str]:
    if not upgrades:
        return ["no upgrades found -- nothing to break out by category."]

    seen: list[str] = []
    per_move: list[dict[str, float]] = []
    for entry in upgrades:
        deltas = _category_deltas(entry)
        per_move.append(deltas)
        for name in deltas:
            if name not in seen:
                seen.append(name)

    if not seen:
        return ["the audit stored no per-category breakdown for these moves."]

    # Canonical roto order first, then anything the audit stored that isn't a
    # known Category (so a new category can never silently vanish from view).
    ordered: list[str] = [c.value for c in ALL_CATEGORIES if c.value in seen]
    ordered.extend(name for name in seen if name not in ordered)

    headers = ["#", "MOVE", *ordered]
    rows: list[list[str]] = []
    for index, (entry, deltas) in enumerate(zip(upgrades, per_move, strict=True), start=1):
        rows.append(
            [str(index), _format_entry(entry), *[_cell_delta(deltas.get(c)) for c in ordered]]
        )
    lines = _render_table(headers, rows, right_align=(0, *range(2, len(headers))))
    lines.append("")
    lines.append("Cells are roto points gained or lost in that category; '.' is no change.")
    return lines


def _hold_lines(holds: Sequence[AuditEntry]) -> list[str]:
    if not holds:
        return ["every active roster spot has an upgrade available."]
    headers = ["PLAYER", "POS", "SLOT", "SGP", "BEST FA", "CANDIDATES SEEN"]
    rows = [
        [
            entry.player,
            _positions(entry.positions),
            entry.slot,
            _num(entry.player_sgp),
            NO_UPGRADE,
            str(len(entry.candidates)),
        ]
        for entry in holds
    ]
    return _render_table(headers, rows, right_align=(3, 5))


def _il_lines(injured: Sequence[AuditEntry]) -> list[str]:
    if not injured:
        return ["no players on the IL."]
    headers = ["PLAYER", "POS", "TYPE"]
    rows = [[entry.player, _positions(entry.positions), entry.player_type] for entry in injured]
    lines = _render_table(headers, rows)
    lines.append("")
    lines.append("IL players are carried, not scored -- the audit does not evaluate them.")
    return lines


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------

REPORT_TITLE = "MANUAL ROSTER AUDIT -- YAHOO-FREE PIPELINE"

_STANDINGS_TITLE = "WHERE YOU STAND"
_LINEUP_TITLE = "LINEUP MOVES -- FREE, NO TRANSACTION NEEDED"
_MOVES_TITLE = "RECOMMENDED MOVES -- BEST FIRST"
_CATEGORY_TITLE = "PER-CATEGORY IMPACT OF EACH MOVE"
_HOLD_TITLE = "HOLD -- NO UPGRADE AVAILABLE"
_IL_TITLE = "INJURED LIST"
_EMPTY_TITLE = "NO AUDIT ENTRIES"


def _lineup_lines(moves: Mapping[str, Any] | None) -> list[str]:
    """Start/bench swaps the optimizer wants, from ``cache:lineup_optimal``.

    These are ranked FIRST in the report because they cost nothing: no waiver
    claim, no roster spot, nobody can be beaten to them. An add/drop can be
    outbid; a lineup change cannot. The add/drop deltas elsewhere in this
    report are computed against the OPTIMAL lineup, so they stack on top of
    these rather than competing with them.

    Presentation only -- every number here was computed by
    ``optimize_pitcher_lineup`` / ``optimize_hitter_lineup``.
    """
    if not moves:
        return ["(no lineup data available)"]
    swaps = moves.get("swaps") or []
    unpaired_starts = moves.get("unpaired_starts") or []
    unpaired_benches = moves.get("unpaired_benches") or []
    if not swaps and not unpaired_starts and not unpaired_benches:
        return ["Your active lineup is already optimal -- no start/bench change helps."]

    headers = ("START", "BENCH", "ROTO GAIN", "P(HELPS)", "VERDICT")
    rows: list[Sequence[str]] = []
    for sw in swaps:
        start = sw.get("start") or {}
        bench = sw.get("bench") or {}
        band = start.get("band") or {}
        rows.append(
            (
                "{} ({} -> {})".format(
                    start.get("name") or start.get("player") or "?",
                    start.get("from", "?"),
                    start.get("to", "?"),
                ),
                bench.get("name") or bench.get("player") or "?",
                _signed(start.get("roto_delta")),
                _pct(band.get("p_positive")),
                str(band.get("verdict") or ""),
            )
        )
    lines = _render_table(headers, rows, right_align=(2, 3))
    for u in unpaired_starts:
        lines.append("  also start: {}".format(u.get("name") or u.get("player") or "?"))
    for u in unpaired_benches:
        lines.append("  also bench: {}".format(u.get("name") or u.get("player") or "?"))
    lines.append("")
    lines.append("These cost nothing and cannot be sniped. Make them before any add/drop --")
    lines.append("the add/drop gains below are measured on top of this lineup, not instead of it.")
    return lines


def _assemble(header: Sequence[str], sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    """Frame the report, sizing every rule to the widest line actually rendered.

    Rules are sized last, from the content, so a wide category table or a long
    player name widens the frame instead of being trimmed to fit it.
    """
    measured: list[str] = [REPORT_TITLE, *header]
    for title, body in sections:
        measured.append(title)
        measured.extend(body)
    width = max(_MIN_RULE_WIDTH, max(len(line) for line in measured))

    out: list[str] = ["=" * width, REPORT_TITLE, "=" * width, *header]
    for title, body in sections:
        out.append("")
        out.append(title)
        out.append("-" * width)
        out.extend(body)
    out.append("=" * width)
    return "\n".join(out)


def render_audit_report(
    entries: Sequence[AuditEntry],
    *,
    team_name: str,
    effective_date: date,
    fraction_remaining: float | None,
    ros_snapshot_date: str,
    kv_path: str | None = None,
    top_n: int | None = None,
    projected_standings: ProjectedStandings | None = None,
    roto_standings: Sequence[tuple[str, float]] | None = None,
    lineup_moves: Mapping[str, Any] | None = None,
    dropped_rows: int = 0,
) -> str:
    """Render a roster audit as an ASCII terminal report.

    Args:
        entries: ``audit_roster`` output, in the order it returned them
            (biggest projected roto gain first, then the no-upgrade rows,
            then the IL rows). The renderer never reorders them.
        team_name: the user's team, used in the banner and to mark their row
            in the standings block.
        effective_date: the roster snapshot's date.
        fraction_remaining: fraction of the season still to play, rendered as
            a percentage. ``None`` when the pipeline did not record one, and
            printed as ``"--"`` rather than as a fabricated 0.0%.
        ros_snapshot_date: the rest-of-season projection snapshot date, as a
            display string (the manual pipeline stores it as text).
        kv_path: resolved KV store path, printed for provenance so a reader
            can tell the isolated manual store from the Yahoo baseline.
        dropped_rows: audit rows the caller could not read back out of the cache.
            Named in the provenance block when non-zero, because the saved file is
            the artifact and a partial audit that does not say so reads as complete.
        top_n: cap on how many upgrades to list. ``None`` lists all of them.
            The hold and IL sections are never capped.
        projected_standings: optional. Only the stored category totals for
            ``team_name`` are printed; nothing is scored or ranked from it.
        lineup_moves: optional ``cache:lineup_optimal['moves']`` payload --
            the free start/bench swaps. Rendered ABOVE the add/drops because
            they cost nothing and the add/drop deltas are measured against
            the optimal lineup. ``None`` prints a 'no lineup data' note.
        roto_standings: optional ``(team, roto_points)`` pairs **already
            scored and already ordered by the caller**. The renderer prints
            them in the given order and numbers them; it does not sort, score
            or rank.

    Returns:
        The report as a single ASCII string, with no trailing newline.
    """
    header = _provenance_lines(
        team_name=team_name,
        effective_date=effective_date,
        fraction_remaining=fraction_remaining,
        dropped_rows=dropped_rows,
        ros_snapshot_date=ros_snapshot_date,
        kv_path=kv_path,
    )

    if not entries:
        return _assemble(
            header,
            [
                (
                    _EMPTY_TITLE,
                    [
                        "The audit returned nothing. That means the roster came back",
                        "empty -- NOT that the roster is optimal. Check the roster",
                        "transcription and re-run.",
                    ],
                )
            ],
        )

    upgrades = [e for e in entries if is_upgrade(e) and not is_injured(e)]
    holds = [e for e in entries if not is_upgrade(e) and not is_injured(e)]
    injured = [e for e in entries if is_injured(e)]
    shown = upgrades if top_n is None else upgrades[:top_n]

    sections: list[tuple[str, Sequence[str]]] = [
        (
            _STANDINGS_TITLE,
            _standings_lines(
                entries,
                team_name=team_name,
                projected_standings=projected_standings,
                roto_standings=roto_standings,
            ),
        ),
        (_LINEUP_TITLE, _lineup_lines(lineup_moves)),
        (_MOVES_TITLE, _upgrade_lines(shown, shown=len(shown), total=len(upgrades))),
        (_CATEGORY_TITLE, _category_lines(shown)),
        (_HOLD_TITLE, _hold_lines(holds)),
        (_IL_TITLE, _il_lines(injured)),
    ]
    return _assemble(header, sections)
