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
    The displayed label uppercases both the tone and the category code, then
    appends the top cat's P(continuation) as a trailing percent (e.g. a "62%"
    suffix) so a weak streak ("55%") reads differently from a strong one
    ("80%") on the chip itself, not just in the tooltip. The percent is dropped
    only when the model has no probability estimate for that cat (``top_prob``
    is None); a genuinely computed low value still renders (even "0%"), because
    that number is exactly the signal the chip exists to surface.
    """
    target = tone  # labels in the cache are lowercase ("hot"/"cold")
    candidates: list[tuple[str, float | None]] = [
        (cat_value, score["probability"])
        for cat_value, score in row["scores"].items()
        if score["label"] == target
    ]
    if not candidates:
        return "—"
    # Highest P(continuation) first (None sorts as 0.0), alphabetical cat tiebreak.
    candidates.sort(key=lambda c: (-(c[1] if c[1] is not None else 0.0), c[0]))
    top_cat, top_prob = candidates[0]
    base = f"{target.upper()} · {top_cat.upper()}"
    if top_prob is None:
        return base
    return f"{base} {round(top_prob * 100)}%"


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
        days = row.get("days_since_last_game")
        if days is not None:
            tooltip = f"Inactive - {days} days"
        else:
            tooltip = "composite=0 (no active streaks)"
        return Indicator(tone="neutral", label="—", tooltip=tooltip)

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
