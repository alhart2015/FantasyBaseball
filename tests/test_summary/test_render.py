from datetime import date

from fantasy_baseball.summary.models import (
    DailySummary,
    InjuryItem,
    PlayerLine,
    StandingsDelta,
    StreakItem,
)
from fantasy_baseball.summary.render import render_html, render_text, subject_line


def _summary(**overrides):
    base = dict(
        as_of=date(2026, 7, 10),
        last_night=[
            PlayerLine(
                name="Aaron Judge",
                group="hitting",
                stats={"h": 2, "hr": 1, "r": 2, "rbi": 3, "sb": 0, "ab": 4, "pa": 4},
            )
        ],
        unmatched=[],
        streaks=[StreakItem(name="Judge", category="hr", label="hot", probability=0.71)],
        standings_delta=StandingsDelta(is_first_run=False, user_team_name="My Team"),
        lineup_moves=[],
        injuries=[InjuryItem(name="Hurt Guy", status="IL15", note="hamstring")],
        probables=[],
        section_errors=[],
    )
    base.update(overrides)
    return DailySummary(**base)


def test_render_projections_panel_html_and_text():
    from fantasy_baseball.summary.models import CategoryEroto, ProjectionDelta

    pd = ProjectionDelta(
        is_first_run=False,
        eroto=[CategoryEroto("HR", 8.2, 7.9), CategoryEroto("SB", 6.0, 6.5)],
        eroto_total_now=72.3,
        eroto_total_prev=70.1,
        champ_pct_now=18.4,
        champ_pct_prev=17.2,
    )
    summary = _summary(projections=pd)
    html = render_html(summary)
    assert "18.4%" in html
    assert "Projected finish" in html
    assert "HR" in html
    text = render_text(summary)
    assert "Championship 18.4%" in text
    assert "HR eRoto 8.2" in text


def test_render_omits_projections_when_empty():
    # Default ProjectionDelta() has no content -> no panel.
    html = render_html(_summary())
    assert "Projected finish" not in html


def test_render_html_includes_populated_sections():
    html = render_html(_summary())
    assert "Aaron Judge" in html
    assert "Hurt Guy" in html
    assert "hamstring" in html
    assert "<html" in html.lower()


def test_render_omits_empty_and_notes_errors_and_firstrun():
    html = render_html(
        _summary(
            last_night=[],
            injuries=[],
            streaks=[],
            standings_delta=StandingsDelta(is_first_run=True, user_team_name="My Team"),
            section_errors=["build_streaks"],
            unmatched=["Ghost"],
        )
    )
    assert "baseline established" in html.lower()
    assert "build_streaks" in html
    assert "unmatched" in html.lower()


def test_render_text_is_plain_and_nonempty():
    text = render_text(_summary())
    assert "Aaron Judge" in text
    assert "<" not in text  # no HTML tags in the text part


def test_subject_line_mentions_date():
    assert "2026-07-10" in subject_line(_summary())
