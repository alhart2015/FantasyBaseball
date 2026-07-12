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

    pd = summary.projections
    if pd.has_content:
        lines = []
        if pd.champ_pct_now is not None:
            cp = f"Championship {pd.champ_pct_now:.1f}%"
            if pd.champ_pct_prev is not None:
                cp += f" ({pd.champ_pct_now - pd.champ_pct_prev:+.1f})"
            lines.append(cp)
        for e in pd.eroto:
            s = f"{e.category} eRoto {e.now:.1f}"
            if e.prev is not None:
                s += f" ({e.now - e.prev:+.1f})"
            lines.append(s)
        if pd.eroto:
            t = f"Total eRoto {pd.eroto_total_now:.1f}"
            if pd.eroto_total_prev is not None:
                t += f" ({pd.eroto_total_now - pd.eroto_total_prev:+.1f})"
            lines.append(t)
        out.append(("Projected finish (end of season)", lines))

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


# --- HTML email design system ---------------------------------------------
# A ballpark-scoreboard treatment: a deep-ink header band with a brass rule,
# box-score stat tables in tabular figures, and hot/cold rendered in
# baseball-stitch red vs. cool blue. Styles are INLINE (mail clients strip
# <style>/classes), layout is table-based (Outlook), fonts are web-safe, and
# every glyph is an HTML entity so the Python source stays ASCII-only.
_INK = "#14273d"  # scoreboard navy
_TEXT = "#1d2833"
_MUTED = "#748089"
_HAIR = "#e6e2d8"
_ROW = "#faf8f3"  # zebra tint
_HOT = "#b23a2e"  # baseball-stitch red
_HOT_BG = "#f7e9e6"
_COLD = "#2f5e8c"  # cool blue
_COLD_BG = "#e8eef6"
_GAIN = "#2f7d57"
_AMBER = "#a9791f"
_AMBER_BG = "#f6efdd"
_CREAM = "#f4f1ea"
_GOLD = "#c9a24b"  # brass scoreboard rule
_SERIF = "Georgia,'Times New Roman',serif"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif"
_NUM = f"'SF Mono',Menlo,Consolas,{_SANS}"  # tabular figures for stat lines


def _esc(text: str) -> str:
    return html_lib.escape(str(text))


def _eyebrow(label: str) -> str:
    return (
        f'<div style="font:600 11px/1 {_SANS};letter-spacing:.15em;'
        f'text-transform:uppercase;color:{_MUTED}">{_esc(label)}</div>'
    )


def _section(eyebrow: str, inner: str) -> str:
    return (
        f'<tr><td style="padding:22px 28px 4px">{_eyebrow(eyebrow)}'
        f'<div style="height:1px;background:{_HAIR};margin:9px 0 13px"></div>'
        f"{inner}</td></tr>"
    )


def _pill(text: str, fg: str, bg: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
        f"background:{bg};color:{fg};font:700 11px/1.5 {_SANS};"
        f'letter-spacing:.03em">{_esc(text)}</span>'
    )


def _boxscore(lines: list) -> str:  # list[PlayerLine]
    rows = []
    for i, p in enumerate(lines):
        stat = _hitter_line(p.stats) if p.group == "hitting" else _pitcher_line(p.stats)
        bg = _ROW if i % 2 else "#ffffff"
        rows.append(
            f'<tr><td style="padding:7px 10px;background:{bg};font:600 14px/1.3 {_SANS};'
            f'color:{_TEXT}">{_esc(p.name)}</td>'
            f'<td align="right" style="padding:7px 10px;background:{bg};font:13px/1.3 {_NUM};'
            f'color:{_MUTED};white-space:nowrap">{_esc(stat)}</td></tr>'
        )
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="border-collapse:collapse;border:1px solid {_HAIR};border-radius:6px">'
        f"{''.join(rows)}</table>"
    )


def _streak_chips(streaks: list) -> str:  # list[StreakItem]
    chips = []
    for s in streaks:
        hot = s.label == "hot"
        fg, bg = (_HOT, _HOT_BG) if hot else (_COLD, _COLD_BG)
        arrow = "&uarr;" if hot else "&darr;"
        pct = f"{s.probability * 100:.0f}%"
        chips.append(
            f'<span style="display:inline-block;margin:0 8px 8px 0;padding:6px 11px;'
            f'border-radius:8px;background:{bg};font:13px/1.2 {_SANS};color:{fg}">'
            f"<b>{_esc(s.name)}</b> &middot; {_esc(s.category.upper())} "
            f'{arrow} <span style="font-family:{_NUM}">{pct}</span></span>'
        )
    return "".join(chips)


def _moves(moves: list) -> str:  # list[LineupMove]
    rows = []
    for m in moves:
        start = m.action == "start"
        tag = _pill("START", "#ffffff", _GAIN) if start else _pill("SIT", "#ffffff", _MUTED)
        delta = (
            f'&nbsp;<span style="color:{_GAIN};font:700 13px {_NUM}">+{m.roto_delta:.1f}</span>'
            if start and m.roto_delta
            else ""
        )
        rows.append(
            f'<div style="padding:6px 0;font:14px/1.4 {_SANS};color:{_TEXT}">{tag}&nbsp;'
            f"<b>{_esc(m.player)}</b>&nbsp;"
            f'<span style="color:{_MUTED}">{_esc(m.from_slot)} &rarr; {_esc(m.to_slot)}</span>'
            f"{delta}</div>"
        )
    return "".join(rows)


def _injuries(injuries: list) -> str:  # list[InjuryItem]
    rows = []
    for i in injuries:
        il = i.status.upper().startswith("IL")
        tag = _pill(i.status, _HOT if il else _AMBER, _HOT_BG if il else _AMBER_BG)
        note = (
            f'<div style="margin-top:3px;font:13px/1.5 {_SANS};color:{_MUTED}">{_esc(i.note)}</div>'
            if i.note
            else ""
        )
        rows.append(
            f'<div style="padding:8px 0;border-bottom:1px solid {_HAIR}">'
            f'<span style="font:600 14px {_SANS};color:{_TEXT}">{_esc(i.name)}</span>&nbsp;'
            f"{tag}{note}</div>"
        )
    return "".join(rows)


def _probables(probables: list) -> str:  # list[ProbableMatchup]
    dot_of = {"Great": _GAIN, "Fair": _AMBER, "Tough": _HOT}
    rows = []
    for p in probables:
        color = dot_of.get(p.quality, _MUTED)
        dot = (
            f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;'
            f'background:{color};margin-right:8px"></span>'
        )
        rows.append(
            f'<div style="padding:6px 0;font:14px/1.4 {_SANS};color:{_TEXT}">{dot}'
            f"<b>{_esc(p.pitcher)}</b> "
            f'<span style="color:{_MUTED}">&middot; {p.starts} start(s) &middot; '
            f"{_esc(p.opponents)}</span> "
            f'<span style="color:{color};font-weight:600">{_esc(p.quality)}</span></div>'
        )
    return "".join(rows)


def _standings_block(sd) -> str:  # StandingsDelta -> a full <tr> (hero) or ""
    if sd.is_first_run:
        return _section(
            "Standings",
            f'<div style="font:14px/1.5 {_SANS};color:{_MUTED}">'
            "Baseline established &mdash; overnight deltas start with tomorrow's report.</div>",
        )
    mine = next((t for t in sd.teams if t.name == sd.user_team_name), None)
    if mine is None:
        return ""
    change = mine.points_now - mine.points_prev
    rank_up = mine.rank_now < mine.rank_prev
    rank_arrow = (
        "&uarr;" if rank_up else ("&darr;" if mine.rank_now > mine.rank_prev else "&middot;")
    )
    rank_color = _GAIN if rank_up else (_HOT if mine.rank_now > mine.rank_prev else _MUTED)
    chg_color = _GAIN if change > 0 else (_HOT if change < 0 else _MUTED)
    return (
        f'<tr><td style="padding:22px 28px 4px">{_eyebrow("Overnight standings")}'
        f'<div style="height:1px;background:{_HAIR};margin:9px 0 13px"></div>'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="border-collapse:separate;background:{_INK};border-radius:8px">'
        f'<tr><td style="padding:16px 20px">'
        f'<div style="font:600 12px/1 {_SANS};letter-spacing:.04em;color:{_GOLD}">'
        f"{_esc(mine.name)}</div>"
        f'<table role="presentation" cellpadding="0" cellspacing="0" style="margin-top:10px">'
        f"<tr>"
        f'<td style="padding-right:26px">'
        f'<div style="font:600 10px/1 {_SANS};letter-spacing:.12em;text-transform:uppercase;'
        f'color:#93a2b3">Rank</div>'
        f'<div style="margin-top:5px;font:22px/1 {_NUM};color:{_CREAM}">'
        f'{mine.rank_prev} <span style="color:{rank_color}">&rarr; {mine.rank_now} '
        f"{rank_arrow}</span></div></td>"
        f'<td style="border-left:1px solid #2b415a;padding-left:26px">'
        f'<div style="font:600 10px/1 {_SANS};letter-spacing:.12em;text-transform:uppercase;'
        f'color:#93a2b3">Roto points</div>'
        f'<div style="margin-top:5px;font:22px/1 {_NUM};color:{_CREAM}">'
        f"{mine.points_prev:.1f} &rarr; {mine.points_now:.1f} "
        f'<span style="color:{chg_color}">({change:+.1f})</span></div></td>'
        f"</tr></table></td></tr></table>"
        f'<div style="margin-top:8px;font:12px/1.5 {_SANS};color:{_MUTED}">'
        "AVG/ERA/WHIP movement is approximate (averaged-rank recompute).</div></td></tr>"
    )


def _signed(delta: float, decimals: int, up_good: bool = True) -> str:
    thresh = 0.5 * 10 ** (-decimals)
    if abs(delta) < thresh:
        return f'<span style="color:{_MUTED}">&middot;</span>'
    color = _GAIN if (delta > 0) == up_good else _HOT
    return f'<span style="color:{color}">{delta:+.{decimals}f}</span>'


def _delta_cell(now: float, prev: float | None, decimals: int) -> str:
    if prev is None:
        return f'<span style="color:{_MUTED}">&mdash;</span>'
    return _signed(now - prev, decimals)


def _projection_block(pd) -> str:  # ProjectionDelta -> a full <tr> or ""
    if not pd.has_content:
        return ""
    parts = []
    if pd.champ_pct_now is not None:
        delta = (
            f' <span style="font:600 13px {_NUM}">('
            f"{_signed(pd.champ_pct_now - pd.champ_pct_prev, 1)})</span>"
            if pd.champ_pct_prev is not None
            else ""
        )
        parts.append(
            f'<div style="margin-bottom:12px;font:14px/1.5 {_SANS};color:{_TEXT}">'
            f'<span style="color:{_MUTED}">Championship odds</span>&nbsp;'
            f'<b style="font:700 16px {_NUM}">{pd.champ_pct_now:.1f}%</b>{delta}</div>'
        )
    if pd.eroto:
        head = (
            f'<tr><td style="padding:0 10px 5px;font:600 10px/1 {_SANS};letter-spacing:.1em;'
            f'text-transform:uppercase;color:{_MUTED}">Cat</td>'
            f'<td align="right" style="padding:0 10px 5px;font:600 10px/1 {_SANS};'
            f'letter-spacing:.1em;text-transform:uppercase;color:{_MUTED}">eRoto</td>'
            f'<td align="right" style="padding:0 10px 5px;font:600 10px/1 {_SANS};'
            f'letter-spacing:.1em;text-transform:uppercase;color:{_MUTED}">O/N</td></tr>'
        )
        rows = []
        for i, e in enumerate(pd.eroto):
            bg = _ROW if i % 2 else "#ffffff"
            rows.append(
                f'<tr><td style="padding:5px 10px;background:{bg};font:600 12px/1.3 {_SANS};'
                f'color:{_TEXT}">{_esc(e.category)}</td>'
                f'<td align="right" style="padding:5px 10px;background:{bg};font:13px/1.3 {_NUM};'
                f'color:{_TEXT}">{e.now:.1f}</td>'
                f'<td align="right" style="padding:5px 10px;background:{bg};font:13px/1.3 {_NUM};'
                f'white-space:nowrap">{_delta_cell(e.now, e.prev, 1)}</td></tr>'
            )
        rows.append(
            f'<tr><td style="padding:7px 10px;border-top:2px solid {_HAIR};'
            f'font:700 12px {_SANS};color:{_TEXT}">Total</td>'
            f'<td align="right" style="padding:7px 10px;border-top:2px solid {_HAIR};'
            f'font:700 13px {_NUM};color:{_TEXT}">{pd.eroto_total_now:.1f}</td>'
            f'<td align="right" style="padding:7px 10px;border-top:2px solid {_HAIR};'
            f'font:700 13px {_NUM};white-space:nowrap">'
            f"{_delta_cell(pd.eroto_total_now, pd.eroto_total_prev, 1)}</td></tr>"
        )
        parts.append(
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            f'style="border-collapse:collapse;border:1px solid {_HAIR};border-radius:6px">'
            f"{head}{''.join(rows)}</table>"
        )
    return _section("Projected finish &middot; end of season", "".join(parts))


def _notes(summary: DailySummary) -> str:
    items = []
    if summary.unmatched:
        items.append(
            f"{len(summary.unmatched)} roster player(s) unmatched: {', '.join(summary.unmatched)}"
        )
    if summary.section_errors:
        items.append(f"Could not build: {', '.join(summary.section_errors)}")
    if not items:
        return ""
    lis = "".join(
        f'<div style="padding:3px 0;font:13px/1.5 {_SANS};color:{_MUTED}">{_esc(x)}</div>'
        for x in items
    )
    return _section("Notes", lis)


def render_html(summary: DailySummary) -> str:
    sd = summary.standings_delta
    team = sd.user_team_name or "Your team"
    d = summary.as_of
    date_str = f"{d.strftime('%A, %B')} {d.day}, {d.year}"  # portable (no %-d)
    body_rows = [
        # Header band -- the scoreboard.
        f'<tr><td style="background:{_INK};padding:24px 28px 20px">'
        f'<div style="font:600 11px/1 {_SANS};letter-spacing:.18em;text-transform:uppercase;'
        f'color:{_GOLD}">Morning Report</div>'
        f'<h1 style="margin:8px 0 2px;font:400 26px/1.15 {_SERIF};color:{_CREAM}">'
        f"{_esc(team)}</h1>"
        f'<div style="font:13px/1.4 {_SANS};color:#93a2b3">Overnight of '
        f"{_esc(date_str)}</div>"
        f'<div style="height:3px;width:52px;background:{_GOLD};margin-top:14px"></div></td></tr>',
        _standings_block(sd),
        _projection_block(summary.projections),
    ]
    if summary.last_night:
        body_rows.append(_section("Last night", _boxscore(summary.last_night)))
    if summary.streaks:
        body_rows.append(_section("Hot &amp; cold hitters", _streak_chips(summary.streaks)))
    if summary.lineup_moves:
        body_rows.append(_section("Today's lineup", _moves(summary.lineup_moves)))
    if summary.injuries:
        body_rows.append(_section("Injury report", _injuries(summary.injuries)))
    if summary.probables:
        body_rows.append(_section("On the mound", _probables(summary.probables)))
    body_rows.append(_notes(summary))
    body_rows.append(
        f'<tr><td style="padding:20px 28px 26px;border-top:1px solid {_HAIR}">'
        f'<div style="font:12px/1.5 {_SANS};color:{_MUTED}">'
        "Generated automatically from your Yahoo roster and the projection model.</div></td></tr>"
    )
    inner = "".join(r for r in body_rows if r)
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(subject_line(summary))}</title></head>"
        f'<body style="margin:0;padding:0;background:#eceae3">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:#eceae3;padding:24px 0">'
        '<tr><td align="center">'
        f'<table role="presentation" width="600" cellpadding="0" cellspacing="0" '
        f'style="width:600px;max-width:600px;background:#ffffff;border-radius:12px;'
        f'overflow:hidden;box-shadow:0 1px 3px rgba(20,39,61,.12)">'
        f"{inner}</table></td></tr></table></body></html>"
    )
