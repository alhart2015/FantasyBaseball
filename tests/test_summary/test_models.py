from datetime import date

from fantasy_baseball.summary.models import (
    DailySummary,
    InjuryItem,
    LineupMove,
    PlayerLine,
    ProbableMatchup,
    StandingsDelta,
    StreakItem,
    TeamDelta,
)


def test_daily_summary_is_constructible_and_frozen():
    summary = DailySummary(
        as_of=date(2026, 7, 10),
        last_night=[PlayerLine(name="Aaron Judge", group="hitting", stats={"h": 2, "hr": 1})],
        unmatched=["Nobody Matched"],
        streaks=[StreakItem(name="Judge", category="hr", label="hot", probability=0.71)],
        standings_delta=StandingsDelta(
            is_first_run=False,
            user_team_name="My Team",
            teams=[
                TeamDelta(
                    name="My Team",
                    rank_prev=3,
                    rank_now=2,
                    points_prev=52.0,
                    points_now=54.5,
                    category_points_delta={"HR": 1.0, "SB": -1.0},
                )
            ],
        ),
        lineup_moves=[
            LineupMove(player="X", action="start", from_slot="BN", to_slot="OF", roto_delta=0.3)
        ],
        injuries=[InjuryItem(name="Y", status="IL15", note="hamstring")],
        probables=[
            ProbableMatchup(
                pitcher="Z", starts=2, days="Mon, Sat", opponents="@ BAL, vs TOR", quality="Great"
            )
        ],
        section_errors=["build_streaks"],
    )
    assert summary.as_of == date(2026, 7, 10)
    assert summary.last_night[0].stats["hr"] == 1
    try:
        summary.as_of = date(2026, 1, 1)  # type: ignore[misc]
        raise AssertionError("expected frozen dataclass")
    except AttributeError:
        pass
