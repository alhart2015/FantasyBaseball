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
