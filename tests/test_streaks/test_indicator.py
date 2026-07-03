"""Tests for streaks/indicator.py — the duckdb-free Lineup chip module.

The first test is a regression lock for the Render deployment bug where
``/lineup`` 500'd because importing ``streaks.dashboard`` transitively
pulled in ``streaks.inference`` and ``streaks.reports.sunday``, both of
which ``import duckdb`` at module load. The indicator module is supposed
to be importable in environments that do not have duckdb installed.

The remaining tests cover the chip's behavior on plain-dict payloads.
"""

from __future__ import annotations

import subprocess
import sys


def test_indicator_module_imports_without_duckdb() -> None:
    """Regression: streaks.indicator must NOT transitively import duckdb.

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
        "assert build_indicator('Nobody', None) is None\n"
        "ind = build_indicator('Nobody', {'roster_rows': [], 'fa_rows': []})\n"
        "assert isinstance(ind, Indicator) and ind.tone == 'neutral'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"streaks.indicator pulled in duckdb-tainted modules.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_build_indicator_none_payload_returns_none() -> None:
    from fantasy_baseball.streaks.indicator import build_indicator

    assert build_indicator("Anyone", None) is None


def test_build_indicator_missing_player_returns_neutral_placeholder() -> None:
    from fantasy_baseball.streaks.indicator import Indicator, build_indicator

    ind = build_indicator("Ghost Player", {"roster_rows": [], "fa_rows": []})
    assert ind == Indicator(tone="neutral", label="—", tooltip="No streak data")


def test_build_indicator_hot_from_dict_payload() -> None:
    """Indicator works from a plain dict — no Report/PlayerCategoryScore needed."""
    from fantasy_baseball.streaks.indicator import build_indicator

    payload = {
        "roster_rows": [
            {
                "name": "Juan Soto",
                "composite": 1,
                "scores": {
                    "hr": {"label": "hot", "probability": 0.62},
                    "avg": {"label": "neutral", "probability": None},
                },
            }
        ],
        "fa_rows": [],
    }
    ind = build_indicator("Juan Soto", payload)
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
    from fantasy_baseball.streaks.indicator import build_indicator

    def _cold_payload(prob: float) -> dict[str, object]:
        return {
            "roster_rows": [
                {
                    "name": "Streaky Guy",
                    "composite": -1,
                    "scores": {"sb": {"label": "cold", "probability": prob}},
                }
            ],
            "fa_rows": [],
        }

    weak = build_indicator("Streaky Guy", _cold_payload(0.55))
    strong = build_indicator("Streaky Guy", _cold_payload(0.80))
    assert weak is not None and strong is not None
    assert weak.label == "COLD · SB 55%"
    assert strong.label == "COLD · SB 80%"
    assert weak.label != strong.label


def test_build_indicator_label_omits_percent_when_probability_none() -> None:
    """A hot/cold cat with no model probability keeps the bare label (no '0%')."""
    from fantasy_baseball.streaks.indicator import build_indicator

    payload = {
        "roster_rows": [
            {
                "name": "No Model",
                "composite": 1,
                "scores": {"hr": {"label": "hot", "probability": None}},
            }
        ],
        "fa_rows": [],
    }
    ind = build_indicator("No Model", payload)
    assert ind is not None
    assert ind.label == "HOT · HR"


def test_build_indicator_roster_beats_fa_on_name_collision() -> None:
    """If the same player appears in both roster_rows and fa_rows, roster wins."""
    from fantasy_baseball.streaks.indicator import build_indicator

    payload = {
        "fa_rows": [
            {
                "name": "Juan Soto",
                "composite": -1,
                "scores": {"hr": {"label": "cold", "probability": 0.5}},
            }
        ],
        "roster_rows": [
            {
                "name": "Juan Soto",
                "composite": 1,
                "scores": {"hr": {"label": "hot", "probability": 0.7}},
            }
        ],
    }
    ind = build_indicator("Juan Soto", payload)
    assert ind is not None
    assert ind.tone == "hot"


def test_build_indicator_label_shows_lift_when_baserate_present() -> None:
    """With a base rate in the payload, the chip shows signed lift, not raw %.

    Raw probabilities are incomparable across categories (base rates span
    ~0.15-0.81); the lift is the part the streak adds.
    """
    from fantasy_baseball.streaks.indicator import build_indicator

    payload = {
        "roster_rows": [
            {
                "name": "Junior Caminero",
                "composite": 1,
                "scores": {
                    "rbi": {
                        "label": "hot",
                        "probability": 0.78,
                        "probability_baserate": 0.44,
                    },
                },
            }
        ],
        "fa_rows": [],
    }
    ind = build_indicator("Junior Caminero", payload)
    assert ind is not None
    assert ind.label == "HOT · RBI +34"
    assert "RBI (+34: 78% vs 44% base)" in ind.tooltip


def test_build_indicator_negative_lift_renders_signed() -> None:
    """A streak *less* likely to continue than baseline shows a minus lift."""
    from fantasy_baseball.streaks.indicator import build_indicator

    payload = {
        "roster_rows": [
            {
                "name": "Nothing Burger",
                "composite": -1,
                "scores": {
                    "r": {
                        "label": "cold",
                        "probability": 0.63,
                        "probability_baserate": 0.67,
                    },
                },
            }
        ],
        "fa_rows": [],
    }
    ind = build_indicator("Nothing Burger", payload)
    assert ind is not None
    assert ind.label == "COLD · R -4"


def test_build_indicator_top_cat_ranked_by_lift_not_raw_probability() -> None:
    """A high-raw/low-lift dense-cold cat must not outrank a real signal.

    r: 78% raw but base 81% (lift -3). rbi: 71% raw, base 56% (lift +15).
    Raw ranking picks R; lift ranking must pick RBI.
    """
    from fantasy_baseball.streaks.indicator import build_indicator

    payload = {
        "roster_rows": [
            {
                "name": "Cold Guy",
                "composite": -2,
                "scores": {
                    "r": {
                        "label": "cold",
                        "probability": 0.78,
                        "probability_baserate": 0.81,
                    },
                    "rbi": {
                        "label": "cold",
                        "probability": 0.71,
                        "probability_baserate": 0.56,
                    },
                },
            }
        ],
        "fa_rows": [],
    }
    ind = build_indicator("Cold Guy", payload)
    assert ind is not None
    assert ind.label == "COLD · RBI +15"


def test_build_indicator_tooltip_never_dangles_when_no_probabilities() -> None:
    """A tone with only unscoreable cats lists them bare -- no dangling 'top: '."""
    from fantasy_baseball.streaks.indicator import build_indicator

    payload = {
        "roster_rows": [
            {
                "name": "Sparse Cold",
                "composite": -1,
                # sparse cold has no model by design: label without probability
                "scores": {"sb": {"label": "cold", "probability": None}},
            }
        ],
        "fa_rows": [],
    }
    ind = build_indicator("Sparse Cold", payload)
    assert ind is not None
    assert ind.tooltip.endswith("top: SB")
