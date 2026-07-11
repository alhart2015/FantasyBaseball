"""Render a DailySummary into an HTML email body and a plain-text fallback."""

from __future__ import annotations

import html as html_lib

from fantasy_baseball.summary.models import DailySummary


def subject_line(summary: DailySummary) -> str:
    return f"Fantasy daily summary - {summary.as_of.isoformat()}"


def _hitter_line(stats: dict[str, float]) -> str:
    return (
        f"{int(stats.get('h', 0))}-{int(stats.get('ab', 0))}, "
        f"{int(stats.get('hr', 0))} HR, {int(stats.get('r', 0))} R, "
        f"{int(stats.get('rbi', 0))} RBI, {int(stats.get('sb', 0))} SB"
    )


def _pitcher_line(stats: dict[str, float]) -> str:
    return (
        f"{stats.get('ip', 0)} IP, {int(stats.get('k', 0))} K, "
        f"{int(stats.get('er', 0))} ER, {int(stats.get('w', 0))} W, {int(stats.get('sv', 0))} SV"
    )


def _sections(summary: DailySummary) -> list[tuple[str, list[str]]]:
    """Return (heading, lines) for each NON-empty section, in email order."""
    out: list[tuple[str, list[str]]] = []

    if summary.last_night:
        lines = [
            f"{p.name}: {_hitter_line(p.stats) if p.group == 'hitting' else _pitcher_line(p.stats)}"
            for p in summary.last_night
        ]
        out.append(("Last night", lines))

    if summary.streaks:
        out.append(
            (
                "Hot / cold (hitters)",
                [
                    f"{s.name} - {s.category} {s.label} ({s.probability:.2f})"
                    for s in summary.streaks
                ],
            )
        )

    sd = summary.standings_delta
    if sd.is_first_run:
        out.append(("Standings", ["Baseline established - deltas start next run."]))
    elif sd.teams:
        mine = next((t for t in sd.teams if t.name == sd.user_team_name), None)
        if mine is not None:
            change = mine.points_now - mine.points_prev
            out.append(
                (
                    "Standings",
                    [
                        f"{mine.name}: rank {mine.rank_prev} -> {mine.rank_now}, "
                        f"roto {mine.points_prev:.1f} -> {mine.points_now:.1f} ({change:+.1f})",
                        "(AVG/ERA/WHIP movement approximate - averaged-rank recompute.)",
                    ],
                )
            )

    if summary.lineup_moves:
        out.append(
            (
                "Lineup moves",
                [
                    f"{m.action.upper()} {m.player}: {m.from_slot} -> {m.to_slot}"
                    for m in summary.lineup_moves
                ],
            )
        )

    if summary.injuries:
        out.append(
            (
                "Injuries",
                [f"{i.name} ({i.status}): {i.note}".rstrip(": ") for i in summary.injuries],
            )
        )

    if summary.probables:
        out.append(
            (
                "Probable starts",
                [
                    f"{p.pitcher}: {p.starts} start(s) - {p.opponents} [{p.quality}]"
                    for p in summary.probables
                ],
            )
        )

    notes: list[str] = []
    if summary.unmatched:
        notes.append(
            f"{len(summary.unmatched)} roster player(s) unmatched: {', '.join(summary.unmatched)}"
        )
    if summary.section_errors:
        notes.append(f"Could not build: {', '.join(summary.section_errors)}")
    if notes:
        out.append(("Notes", notes))

    return out


def render_text(summary: DailySummary) -> str:
    parts = [f"Fantasy daily summary - {summary.as_of.isoformat()}", ""]
    for heading, lines in _sections(summary):
        parts.append(heading)
        parts.extend(f"  {line}" for line in lines)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_html(summary: DailySummary) -> str:
    blocks: list[str] = []
    for heading, lines in _sections(summary):
        items = "".join(f"<li>{html_lib.escape(line)}</li>" for line in lines)
        blocks.append(
            f'<h2 style="font-size:16px;margin:16px 0 4px">{html_lib.escape(heading)}</h2>'
            f'<ul style="margin:0;padding-left:20px">{items}</ul>'
        )
    body = "".join(blocks)
    title = html_lib.escape(subject_line(summary))
    return (
        '<html><head><meta charset="utf-8"></head>'
        '<body style="font-family:Arial,Helvetica,sans-serif;color:#111">'
        f'<h1 style="font-size:20px">{title}</h1>{body}</body></html>'
    )
