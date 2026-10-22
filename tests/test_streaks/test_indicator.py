"""Tests for streaks/indicator.py — the duckdb-free Lineup chip module.

The first test is a regression lock for the Render deployment bug where
``/lineup`` 500'd because importing the old ``streaks.dashboard`` transitively
pulled in ``streaks.inference`` and ``streaks.reports.sunday``, both of
which ``import duckdb`` at module load. The indicator, the payload types
(``streaks.models``) and the payload loader (``streaks.payload``) must be
importable in environments that do not have duckdb installed.

The rest cover the chip's behavior on typed reports (#175): the chip reads
``Report`` / ``ReportRow`` / ``PlayerCategoryScore``, the one definition of the
cached payload's shape, rather than fingering raw dicts.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from typing import cast

from fantasy_baseball.streaks.indicator import Indicator, build_indicator
from fantasy_baseball.streaks.models import (
    PlayerCategoryScore,
    Report,
    ReportRow,
    StreakCategory,
    StreakLabel,
)
from fantasy_baseball.streaks.payload import load_report, serialize_report
from tests.test_streaks.conftest import example_report, example_report_inactive

# One scored category: (label, probability, probability_baserate).
Scored = tuple[str, float | None, float | None]


def _row(
    name: str, composite: int, scores: dict[str, Scored], days: int | None = None
) -> ReportRow:
    return ReportRow(
        name=name,
        positions=("OF",),
        player_id=abs(hash(name)) % 100_000,
        composite=composite,
        scores={
            cast(StreakCategory, cat): PlayerCategoryScore(
                player_id=0,
                category=cast(StreakCategory, cat),
                label=cast(StreakLabel, label),
                probability=prob,
                drivers=(),
                window_end=date(2026, 5, 10),
                probability_baserate=base,
            )
            for cat, (label, prob, base) in scores.items()
        },
        max_probability=max((p for _, p, _ in scores.values() if p is not None), default=0.0),
        days_since_last_game=days,
    )


def _report(roster: tuple[ReportRow, ...] = (), fa: tuple[ReportRow, ...] = ()) -> Report:
    return Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="t",
        league_id=1,
        season_set_train="2023-2025",
        roster_rows=roster,
        fa_rows=fa,
        driver_lines=(),
    )


def test_indicator_module_imports_without_duckdb() -> None:
    """Regression: the chip's whole path must NOT transitively import duckdb.

    Render does not install duckdb (it is a [dev] extra), and the chip is
    served from a cached payload — there is no DuckDB on the request path.
    We run the import in a subprocess with sys.modules['duckdb'] pre-set
    to ``None`` so that any transitive ``import duckdb`` raises ImportError
    and fails the subprocess.
    """
    src = (
        "import sys\n"
        "sys.modules['duckdb'] = None\n"  # Force ImportError on `import duckdb`
        "import fantasy_baseball.streaks.indicator  # noqa: F401\n"
        "from fantasy_baseball.streaks.indicator import build_indicator, Indicator\n"
        "from fantasy_baseball.streaks.payload import load_report\n"
        "assert build_indicator('Nobody', None) is None\n"
        "rpt = load_report({'report_date': '2026-05-11', 'window_end': None, 'team_name': 't',\n"
        "    'league_id': 1, 'season_set_train': 's', 'roster_rows': [], 'fa_rows': [],\n"
        "    'driver_lines': [], 'skipped': []})\n"
        "ind = build_indicator('Nobody', rpt)\n"
        "assert isinstance(ind, Indicator) and ind.tone == 'neutral'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"the streak chip path pulled in duckdb-tainted modules.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_build_indicator_none_payload_returns_none() -> None:
    assert build_indicator("Anyone", None) is None


def test_build_indicator_missing_player_returns_neutral_placeholder() -> None:
    ind = build_indicator("Ghost Player", _report())
    assert ind == Indicator(tone="neutral", label="—", tooltip="No streak data")


def test_build_indicator_hot_from_report() -> None:
    rpt = _report(
        roster=(_row("Juan Soto", 1, {"hr": ("hot", 0.62, None), "avg": ("neutral", None, None)}),)
    )
    ind = build_indicator("Juan Soto", rpt)
    assert ind is not None
    assert ind.tone == "hot"
    assert ind.label == "HOT · HR 62%"
    assert "HR (62%)" in ind.tooltip


def test_build_indicator_label_distinguishes_continuation_probability() -> None:
    """The chip label carries P(continuation) so weak and strong streaks differ.

    A cold streak only 55% likely to continue must read differently on the
    chip itself from one 80% likely to continue -- the whole point of the
    'surface continuation probability on the chip' work.
    """

    def _cold(prob: float) -> Report:
        return _report(roster=(_row("Streaky Guy", -1, {"sb": ("cold", prob, None)}),))

    weak = build_indicator("Streaky Guy", _cold(0.55))
    strong = build_indicator("Streaky Guy", _cold(0.80))
    assert weak is not None and strong is not None
    assert weak.label == "COLD · SB 55%"
    assert strong.label == "COLD · SB 80%"
    assert weak.label != strong.label


def test_build_indicator_label_omits_percent_when_probability_none() -> None:
    """A hot/cold cat with no model probability keeps the bare label (no '0%')."""
    rpt = _report(roster=(_row("No Model", 1, {"hr": ("hot", None, None)}),))
    ind = build_indicator("No Model", rpt)
    assert ind is not None
    assert ind.label == "HOT · HR"


def test_build_indicator_roster_beats_fa_on_name_collision() -> None:
    """If the same player appears in both roster_rows and fa_rows, roster wins."""
    rpt = _report(
        roster=(_row("Juan Soto", 1, {"hr": ("hot", 0.7, None)}),),
        fa=(_row("Juan Soto", -1, {"hr": ("cold", 0.5, None)}),),
    )
    ind = build_indicator("Juan Soto", rpt)
    assert ind is not None
    assert ind.tone == "hot"


def test_build_indicator_label_shows_lift_when_baserate_present() -> None:
    """With a base rate in the payload, the chip shows signed lift, not raw %.

    Raw probabilities are incomparable across categories (base rates span
    ~0.15-0.81); the lift is the part the streak adds.
    """
    rpt = _report(roster=(_row("Junior Caminero", 1, {"rbi": ("hot", 0.78, 0.44)}),))
    ind = build_indicator("Junior Caminero", rpt)
    assert ind is not None
    assert ind.label == "HOT · RBI +34"
    assert "RBI (+34: 78% vs 44% base)" in ind.tooltip


def test_build_indicator_negative_lift_renders_signed() -> None:
    """A streak *less* likely to continue than baseline shows a minus lift."""
    rpt = _report(roster=(_row("Nothing Burger", -1, {"r": ("cold", 0.63, 0.67)}),))
    ind = build_indicator("Nothing Burger", rpt)
    assert ind is not None
    assert ind.label == "COLD · R -4"


def test_build_indicator_top_cat_ranked_by_lift_not_raw_probability() -> None:
    """A high-raw/low-lift dense-cold cat must not outrank a real signal.

    r: 78% raw but base 81% (lift -3). rbi: 71% raw, base 56% (lift +15).
    Raw ranking picks R; lift ranking must pick RBI.
    """
    rpt = _report(
        roster=(_row("Cold Guy", -2, {"r": ("cold", 0.78, 0.81), "rbi": ("cold", 0.71, 0.56)}),)
    )
    ind = build_indicator("Cold Guy", rpt)
    assert ind is not None
    assert ind.label == "COLD · RBI +15"


def test_build_indicator_tooltip_never_dangles_when_no_probabilities() -> None:
    """A tone with only unscoreable cats lists them bare -- no dangling 'top: '."""
    # sparse cold has no model by design: label without probability
    rpt = _report(roster=(_row("Sparse Cold", -1, {"sb": ("cold", None, None)}),))
    ind = build_indicator("Sparse Cold", rpt)
    assert ind is not None
    assert ind.tooltip.endswith("top: SB")


# Formerly in test_dashboard.py: the chip read back through the cache's shape
# (serialize_report -> load_report), as the dashboard reads it.


def test_build_indicator_hot_picks_top_hot_cat() -> None:
    ind = build_indicator("Juan Soto", load_report(serialize_report(example_report())))
    assert ind is not None
    assert ind.tone == "hot"
    assert ind.label == "HOT · HR 62%"


def test_build_indicator_cold_picks_top_cold_cat() -> None:
    # Flip HR to COLD, AVG stays NEUTRAL: composite = -1, top cat = HR at 0.62.
    original = example_report()
    row = original.roster_rows[0]
    cold_score = PlayerCategoryScore(
        player_id=row.player_id,
        category="hr",
        label="cold",
        probability=0.62,
        drivers=row.scores["hr"].drivers,
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
    flipped = _report(roster=(flipped_row,))
    ind = build_indicator("Juan Soto", load_report(serialize_report(flipped)))
    assert ind is not None
    assert ind.tone == "cold"
    assert ind.label == "COLD · HR 62%"


def test_build_indicator_neutral_when_composite_zero() -> None:
    rpt = _report(roster=(_row("Neutral Guy", 0, {"hr": ("neutral", None, None)}),))
    ind = build_indicator("Neutral Guy", load_report(serialize_report(rpt)))
    assert ind is not None
    assert ind.tone == "neutral"
    assert ind.label == "—"


def test_build_indicator_inactive_row_shows_days_tooltip() -> None:
    ind = build_indicator("Injured Guy", load_report(serialize_report(example_report_inactive())))
    assert ind is not None
    assert ind.tone == "neutral"
    assert ind.tooltip == "Inactive - 30 days"


def test_build_indicator_unresolved_player() -> None:
    ind = build_indicator("Unknown Hitter", load_report(serialize_report(example_report())))
    assert ind is not None
    assert ind.tone == "neutral"
    assert ind.label == "—"
    assert "No streak data" in ind.tooltip


def test_build_indicator_tiebreak_alphabetical() -> None:
    rpt = _report(
        roster=(_row("Tied Guy", 2, {"hr": ("hot", 0.6, None), "r": ("hot", 0.6, None)}),)
    )
    ind = build_indicator("Tied Guy", load_report(serialize_report(rpt)))
    assert ind is not None
    assert ind.label == "HOT · HR 60%"  # HR alphabetically before R


def test_indicator_is_frozen() -> None:
    ind = Indicator(tone="hot", label="HOT · HR", tooltip="x")
    try:
        ind.tone = "cold"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Indicator should be frozen")
