"""Dashboard glue for the hot-streaks pipeline.

Serialization helpers translate the in-memory :class:`Report` dataclass to
JSON-safe dicts (and back) for transport through the Redis/SQLite cache.
``build_indicator`` (added in Task 7) is the Lineup-page hook.

The schema mirrors the dataclass fields 1:1 — round-trip equality holds
because every dataclass involved is ``frozen=True`` (default-generated
``__eq__`` compares fields).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.models import StreakCategory, StreakLabel
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine,
    Report,
    ReportRow,
)
from fantasy_baseball.utils.name_utils import normalize_name


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


@dataclass(frozen=True)
class Indicator:
    """One Lineup-page chip: tone + label + tooltip."""

    tone: Literal["hot", "cold", "neutral"]
    label: str
    tooltip: str


def _row_lookup_by_normalized_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build ``{normalize_name(row.name): row_dict}`` from roster + FAs.

    Roster wins ties with FAs (a player can theoretically appear in both
    if the cache was written mid-roster-move). Already-normalized name
    comparison is the contract.
    """
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("fa_rows", []):
        out[normalize_name(row["name"])] = row
    for row in payload.get("roster_rows", []):
        out[normalize_name(row["name"])] = row
    return out


def _top_cat_label(row: dict[str, Any], tone: Literal["hot", "cold"]) -> str:
    """Find the cat with the highest probability matching the tone.

    Alphabetical tiebreak on the category enum value for determinism.
    The displayed label uppercases both the tone and the category code.
    """
    target = tone  # labels in the cache are lowercase ("hot"/"cold")
    candidates: list[tuple[float, str]] = []
    for cat_value, score in row["scores"].items():
        if score["label"] != target:
            continue
        prob = score["probability"] or 0.0
        candidates.append((prob, cat_value))
    if not candidates:
        return "—"
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top_cat = candidates[0][1]
    return f"{target.upper()} · {top_cat.upper()}"


def build_indicator(name: str, payload: dict[str, Any] | None) -> Indicator | None:
    """Build the Lineup-page chip for one hitter name.

    Returns ``None`` when the cache is missing (so the route can decide
    to render a default placeholder). Returns ``Indicator(tone='neutral',
    label='—', tooltip='No streak data')`` when the name doesn't resolve
    against either the roster or the FA list in the cached report.
    """
    if payload is None:
        return None

    lookup = _row_lookup_by_normalized_name(payload)
    row = lookup.get(normalize_name(name))
    if row is None:
        return Indicator(tone="neutral", label="—", tooltip="No streak data")

    composite = row["composite"]
    tone: Literal["hot", "cold", "neutral"]
    if composite > 0:
        tone = "hot"
    elif composite < 0:
        tone = "cold"
    else:
        return Indicator(
            tone="neutral",
            label="—",
            tooltip="composite=0 (no active streaks)",
        )

    label = _top_cat_label(row, tone)
    target = tone  # lowercase label key in the cache
    bits: list[str] = []
    for cat_value, score in row["scores"].items():
        if score["label"] == target and score["probability"] is not None:
            bits.append(f"{cat_value.upper()} ({round(score['probability'] * 100)}%)")
    bits.sort()
    sign = "+" if composite > 0 else ""
    tooltip = f"composite={sign}{composite} · top: " + ", ".join(bits)
    return Indicator(tone=tone, label=label, tooltip=tooltip)
