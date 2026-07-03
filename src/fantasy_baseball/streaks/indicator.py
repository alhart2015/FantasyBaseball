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


def _lift_pts(score: dict[str, Any]) -> int | None:
    """Signed percentage-point lift of P(continuation) over the stratum base rate.

    Raw probabilities are not comparable across categories/directions —
    base rates range ~0.15 (sparse hot) to ~0.81 (dense cold), so a raw
    "78%" can carry less signal than a raw "46%". The lift is the part
    the streak actually adds. ``None`` when either number is missing
    (old cached payloads predate ``probability_baserate``).
    """
    prob = score.get("probability")
    base = score.get("probability_baserate")
    if prob is None or base is None:
        return None
    return round((float(prob) - float(base)) * 100)


def _top_cat_label(row: dict[str, Any], tone: Literal["hot", "cold"]) -> str:
    """Find the strongest tone-matching cat and format the chip label.

    Ranking: highest lift first (see :func:`_lift_pts`); cats without a
    base rate rank below lifted ones by raw probability (old payloads);
    unscored cats last. Alphabetical cat tiebreak for determinism.

    Display: ``HOT · RBI +34`` — the lift in percentage points, signed,
    so a weak streak ("+4") reads differently from a strong one ("+34")
    and a big-but-empty raw number can't masquerade as signal. Falls back
    to the raw percent ("78%") when the payload has no base rate, and to
    the bare cat when there's no probability at all.
    """
    target = tone  # labels in the cache are lowercase ("hot"/"cold")
    candidates: list[tuple[str, dict[str, Any]]] = [
        (cat_value, score) for cat_value, score in row["scores"].items() if score["label"] == target
    ]
    if not candidates:
        return "—"

    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, str]:
        cat_value, score = item
        lift = _lift_pts(score)
        if lift is not None:
            return (0, -lift, cat_value)
        prob = score.get("probability")
        if prob is not None:
            return (1, -prob, cat_value)
        return (2, 0.0, cat_value)

    candidates.sort(key=sort_key)
    top_cat, top_score = candidates[0]
    base = f"{target.upper()} · {top_cat.upper()}"
    lift = _lift_pts(top_score)
    if lift is not None:
        return f"{base} {lift:+d}"
    top_prob = top_score.get("probability")
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
        if score["label"] != target:
            continue
        lift = _lift_pts(score)
        prob = score.get("probability")
        if lift is not None:
            base_pct = round(score["probability_baserate"] * 100)
            bits.append(
                f"{cat_value.upper()} ({lift:+d}: {round(prob * 100)}% vs {base_pct}% base)"
            )
        elif prob is not None:
            bits.append(f"{cat_value.upper()} ({round(prob * 100)}%)")
        else:
            # Labeled but unscoreable (e.g. sparse cold has no model): list the
            # cat bare rather than dropping it, so the tooltip never renders a
            # dangling "top: " with nothing after it.
            bits.append(cat_value.upper())
    bits.sort()
    sign = "+" if composite > 0 else ""
    tooltip = f"composite={sign}{composite} · top: " + ", ".join(bits)
    return Indicator(tone=tone, label=label, tooltip=tooltip)
