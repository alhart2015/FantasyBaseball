"""Tests for streaks/dashboard.py — serialization and indicator."""

from __future__ import annotations

from datetime import date

from fantasy_baseball.streaks.dashboard import (
    Indicator,
    build_indicator,
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


def test_build_indicator_hot_picks_top_hot_cat() -> None:
    payload = serialize_report(_example_report())
    ind = build_indicator("Juan Soto", payload)
    assert ind is not None
    assert ind.tone == "hot"
    assert ind.label == "HOT · HR"


def test_build_indicator_cold_picks_top_cold_cat() -> None:
    # Modify the example: flip HR to COLD, AVG stays NEUTRAL.
    # Resulting composite = -1, top cat = HR with prob 0.62.
    original = _example_report()
    row = original.roster_rows[0]
    cold_score = PlayerCategoryScore(
        player_id=row.player_id,
        category="hr",
        label="cold",
        probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=-1.8),),
        window_end=date(2026, 5, 10),
    )
    flipped_row = ReportRow(
        name=row.name,
        positions=row.positions,
        player_id=row.player_id,
        composite=-1,
        scores={"hr": cold_score, "avg": row.scores["avg"]},
        max_probability=0.62,
    )
    flipped = Report(
        report_date=original.report_date,
        window_end=original.window_end,
        team_name=original.team_name,
        league_id=original.league_id,
        season_set_train=original.season_set_train,
        roster_rows=(flipped_row,),
        fa_rows=(),
        driver_lines=(),
        skipped=(),
    )
    payload = serialize_report(flipped)
    ind = build_indicator("Juan Soto", payload)
    assert ind is not None
    assert ind.tone == "cold"
    assert ind.label == "COLD · HR"


def test_build_indicator_neutral_when_composite_zero() -> None:
    neutral_score = PlayerCategoryScore(
        player_id=1,
        category="hr",
        label="neutral",
        probability=None,
        drivers=(),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Neutral Guy",
        positions=("OF",),
        player_id=1,
        composite=0,
        scores={"hr": neutral_score},
        max_probability=0.0,
    )
    rpt = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="t",
        league_id=1,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(),
        skipped=(),
    )
    payload = serialize_report(rpt)
    ind = build_indicator("Neutral Guy", payload)
    assert ind is not None
    assert ind.tone == "neutral"
    assert ind.label == "—"


def test_build_indicator_unresolved_player() -> None:
    payload = serialize_report(_example_report())
    ind = build_indicator("Unknown Hitter", payload)
    assert ind is not None
    assert ind.tone == "neutral"
    assert ind.label == "—"
    assert "No streak data" in ind.tooltip


def test_build_indicator_returns_none_when_cache_missing() -> None:
    assert build_indicator("Juan Soto", None) is None


def test_build_indicator_tiebreak_alphabetical() -> None:
    score_hr = PlayerCategoryScore(
        player_id=1,
        category="hr",
        label="hot",
        probability=0.6,
        drivers=(),
        window_end=date(2026, 5, 10),
    )
    score_r = PlayerCategoryScore(
        player_id=1,
        category="r",
        label="hot",
        probability=0.6,
        drivers=(),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Tied Guy",
        positions=("OF",),
        player_id=1,
        composite=2,
        scores={"hr": score_hr, "r": score_r},
        max_probability=0.6,
    )
    rpt = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="t",
        league_id=1,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(),
        skipped=(),
    )
    payload = serialize_report(rpt)
    ind = build_indicator("Tied Guy", payload)
    assert ind is not None
    assert ind.label == "HOT · HR"  # HR alphabetically before R


def test_indicator_is_frozen() -> None:
    ind = Indicator(tone="hot", label="HOT · HR", tooltip="x")
    try:
        ind.tone = "cold"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Indicator should be frozen")
