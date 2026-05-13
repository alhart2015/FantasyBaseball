"""Lineup-page streak chip: pure dict-payload consumer.

Intentionally lives outside :mod:`streaks.dashboard` so the request-time
``/lineup`` import path on Render does not transitively pull in
:mod:`streaks.inference` / :mod:`streaks.reports.sunday`, which both
``import duckdb`` at module load. DuckDB is dev-only — Render never
runs the streaks pipeline; it just reads :data:`CacheKey.STREAK_SCORES`
written by a refresh on a developer machine.

If you add a new helper here, keep it duckdb-free. Any function that
needs DuckDB belongs in ``streaks/dashboard.py`` (refresh-side) instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fantasy_baseball.utils.name_utils import normalize_name


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
        prob = score["probability"] if score["probability"] is not None else 0.0
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
