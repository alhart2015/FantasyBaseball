"""Dashboard glue for the hot-streaks pipeline.

Serialization helpers translate the in-memory :class:`Report` dataclass to
JSON-safe dicts (and back) for transport through the Redis/SQLite cache.
These helpers are refresh-side only — they import :mod:`streaks.inference`
and :mod:`streaks.reports.sunday`, both of which ``import duckdb`` at
module load. The Render dashboard cannot load this module.

The Lineup-page chip lives in :mod:`streaks.indicator` precisely so the
Render-side ``/lineup`` import path stays duckdb-free; :class:`Indicator`
and :func:`build_indicator` are re-exported here so existing tests and
docs that import them from this module continue to work.

The schema mirrors the dataclass fields 1:1 — round-trip equality holds
because every dataclass involved is ``frozen=True`` (default-generated
``__eq__`` compares fields).
"""

from __future__ import annotations

from datetime import date
from typing import Any, cast

from fantasy_baseball.streaks.indicator import Indicator, build_indicator
from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.models import StreakCategory, StreakLabel
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine,
    Report,
    ReportRow,
)

__all__ = [
    "Indicator",
    "build_indicator",
    "deserialize_report",
    "serialize_report",
]


def _serialize_driver(d: Driver) -> dict[str, Any]:
    return {"feature": d.feature, "z_score": d.z_score}


def _deserialize_driver(p: dict[str, Any]) -> Driver:
    return Driver(feature=str(p["feature"]), z_score=float(p["z_score"]))


def _serialize_score(s: PlayerCategoryScore) -> dict[str, Any]:
    return {
        "player_id": s.player_id,
        "category": s.category,  # string Literal
        "label": s.label,  # string Literal
        "probability": s.probability,
        "drivers": [_serialize_driver(d) for d in s.drivers],
        "window_end": s.window_end.isoformat() if s.window_end else None,
    }


def _deserialize_score(p: dict[str, Any]) -> PlayerCategoryScore:
    probability = p["probability"]
    return PlayerCategoryScore(
        player_id=int(p["player_id"]),
        category=cast(StreakCategory, p["category"]),
        label=cast(StreakLabel, p["label"]),
        probability=float(probability) if probability is not None else None,
        drivers=tuple(_deserialize_driver(d) for d in p["drivers"]),
        window_end=date.fromisoformat(p["window_end"]) if p["window_end"] else None,
    )


def _serialize_row(r: ReportRow) -> dict[str, Any]:
    return {
        "name": r.name,
        "positions": list(r.positions),
        "player_id": r.player_id,
        "composite": r.composite,
        "max_probability": r.max_probability,
        "scores": {cat: _serialize_score(score) for cat, score in r.scores.items()},
    }


def _deserialize_row(p: dict[str, Any]) -> ReportRow:
    return ReportRow(
        name=str(p["name"]),
        positions=tuple(p["positions"]),
        player_id=int(p["player_id"]),
        composite=int(p["composite"]),
        max_probability=float(p["max_probability"]),
        scores={
            cast(StreakCategory, cat): _deserialize_score(score)
            for cat, score in p["scores"].items()
        },
    )


def _serialize_driver_line(dl: DriverLine) -> dict[str, Any]:
    return {
        "player_name": dl.player_name,
        "category": dl.category,
        "label": dl.label,
        "probability": dl.probability,
        "drivers": [_serialize_driver(d) for d in dl.drivers],
    }


def _deserialize_driver_line(p: dict[str, Any]) -> DriverLine:
    return DriverLine(
        player_name=str(p["player_name"]),
        category=cast(StreakCategory, p["category"]),
        label=cast(StreakLabel, p["label"]),
        probability=float(p["probability"]),
        drivers=tuple(_deserialize_driver(d) for d in p["drivers"]),
    )


def serialize_report(report: Report) -> dict[str, Any]:
    """Convert a :class:`Report` into a JSON-safe dict.

    All dates are encoded as ISO-8601 strings. ``None`` is preserved as
    JSON ``null`` for the nullable fields (``Report.window_end`` and
    ``PlayerCategoryScore.probability`` / ``window_end``).
    """
    return {
        "report_date": report.report_date.isoformat(),
        "window_end": report.window_end.isoformat() if report.window_end else None,
        "team_name": report.team_name,
        "league_id": report.league_id,
        "season_set_train": report.season_set_train,
        "roster_rows": [_serialize_row(r) for r in report.roster_rows],
        "fa_rows": [_serialize_row(r) for r in report.fa_rows],
        "driver_lines": [_serialize_driver_line(dl) for dl in report.driver_lines],
        "skipped": list(report.skipped),
    }


def deserialize_report(payload: dict[str, Any]) -> Report:
    """Reconstruct a :class:`Report` from the dict produced by :func:`serialize_report`.

    Round-trip equality holds (the dataclasses are frozen and generate
    structural ``__eq__``); this is exercised by
    ``tests/test_streaks/test_dashboard.py``.
    """
    return Report(
        report_date=date.fromisoformat(payload["report_date"]),
        window_end=(date.fromisoformat(payload["window_end"]) if payload["window_end"] else None),
        team_name=str(payload["team_name"]),
        league_id=int(payload["league_id"]),
        season_set_train=str(payload["season_set_train"]),
        roster_rows=tuple(_deserialize_row(r) for r in payload["roster_rows"]),
        fa_rows=tuple(_deserialize_row(r) for r in payload["fa_rows"]),
        driver_lines=tuple(_deserialize_driver_line(dl) for dl in payload["driver_lines"]),
        skipped=tuple(payload["skipped"]),
    )
