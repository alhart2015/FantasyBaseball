"""Lineup-page streak chip, from the cached streak report.

DuckDB-free, so the request-time ``/lineup`` path on Render can import it: it reads
:class:`~fantasy_baseball.streaks.models.Report` (the payload type, which lives in
the DuckDB-free :mod:`streaks.models`; load it with
:func:`streaks.payload.load_report`), never the DuckDB code that builds the report
(:mod:`streaks.inference`, :mod:`streaks.reports.sunday`). DuckDB is dev-only --
Render never runs the streaks pipeline; it just reads :data:`CacheKey.STREAK_SCORES`
written by a refresh on a developer machine.

If you add a new helper here, keep it duckdb-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fantasy_baseball.streaks.models import PlayerCategoryScore, Report, ReportRow
from fantasy_baseball.utils.name_utils import normalize_name


@dataclass(frozen=True)
class Indicator:
    """One Lineup-page chip: tone + label + tooltip."""

    tone: Literal["hot", "cold", "neutral"]
    label: str
    tooltip: str


def _row_lookup_by_normalized_name(report: Report) -> dict[str, ReportRow]:
    """Build ``{normalize_name(row.name): row}`` from roster + FAs.

    Roster wins ties with FAs (a player can theoretically appear in both
    if the cache was written mid-roster-move). Already-normalized name
    comparison is the contract.
    """
    out: dict[str, ReportRow] = {}
    for row in report.fa_rows:
        out[normalize_name(row.name)] = row
    for row in report.roster_rows:
        out[normalize_name(row.name)] = row
    return out


def _lift_pts(score: PlayerCategoryScore) -> int | None:
    """Signed percentage-point lift of P(continuation) over the stratum base rate.

    Raw probabilities are not comparable across categories/directions —
    base rates range ~0.15 (sparse hot) to ~0.81 (dense cold), so a raw
    "78%" can carry less signal than a raw "46%". The lift is the part
    the streak actually adds. ``None`` when either number is missing
    (old cached payloads predate ``probability_baserate``).
    """
    if score.probability is None or score.probability_baserate is None:
        return None
    return round((score.probability - score.probability_baserate) * 100)


def _top_cat_label(row: ReportRow, tone: Literal["hot", "cold"]) -> str:
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
    candidates: list[tuple[str, PlayerCategoryScore]] = [
        (str(cat), score) for cat, score in row.scores.items() if score.label == tone
    ]
    if not candidates:
        return "—"

    def sort_key(item: tuple[str, PlayerCategoryScore]) -> tuple[int, float, str]:
        cat_value, score = item
        lift = _lift_pts(score)
        if lift is not None:
            return (0, -lift, cat_value)
        if score.probability is not None:
            return (1, -score.probability, cat_value)
        return (2, 0.0, cat_value)

    candidates.sort(key=sort_key)
    top_cat, top_score = candidates[0]
    base = f"{tone.upper()} · {top_cat.upper()}"
    lift = _lift_pts(top_score)
    if lift is not None:
        return f"{base} {lift:+d}"
    if top_score.probability is None:
        return base
    return f"{base} {round(top_score.probability * 100)}%"


def build_indicator(name: str, report: Report | None) -> Indicator | None:
    """Build the Lineup-page chip for one hitter name.

    Returns ``None`` when there is no report (so the route can decide
    to render a default placeholder). Returns ``Indicator(tone='neutral',
    label='—', tooltip='No streak data')`` when the name doesn't resolve
    against either the roster or the FA list in the report.
    """
    if report is None:
        return None

    row = _row_lookup_by_normalized_name(report).get(normalize_name(name))
    if row is None:
        return Indicator(tone="neutral", label="—", tooltip="No streak data")

    composite = row.composite
    tone: Literal["hot", "cold", "neutral"]
    if composite > 0:
        tone = "hot"
    elif composite < 0:
        tone = "cold"
    else:
        days = row.days_since_last_game
        if days is not None:
            tooltip = f"Inactive - {days} days"
        else:
            tooltip = "composite=0 (no active streaks)"
        return Indicator(tone="neutral", label="—", tooltip=tooltip)

    label = _top_cat_label(row, tone)
    bits: list[str] = []
    for cat, score in row.scores.items():
        if score.label != tone:
            continue
        cat_value = str(cat)
        lift = _lift_pts(score)
        prob = score.probability
        if lift is not None and prob is not None and score.probability_baserate is not None:
            base_pct = round(score.probability_baserate * 100)
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
