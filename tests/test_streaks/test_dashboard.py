"""Tests for streaks/dashboard.py — serialization and indicator."""

from __future__ import annotations

from datetime import date

from fantasy_baseball.streaks.dashboard import (
    deserialize_report,
    serialize_report,
)
from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine,
    Report,
    ReportRow,
)


def _example_report() -> Report:
    score_hr = PlayerCategoryScore(
        player_id=665742,
        category="hr",
        label="hot",
        probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=1.8),),
        window_end=date(2026, 5, 10),
    )
    score_avg = PlayerCategoryScore(
        player_id=665742,
        category="avg",
        label="neutral",
        probability=None,
        drivers=(),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Juan Soto",
        positions=("OF",),
        player_id=665742,
        composite=1,
        scores={"hr": score_hr, "avg": score_avg},
        max_probability=0.62,
    )
    driver_line = DriverLine(
        player_name="Juan Soto",
        category="hr",
        label="hot",
        probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=1.8),),
    )
    return Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="Hart of the Order",
        league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(driver_line,),
        skipped=("Foo — no_window",),
    )


def test_serialize_report_round_trips() -> None:
    original = _example_report()
    payload = serialize_report(original)
    rebuilt = deserialize_report(payload)
    assert rebuilt == original
