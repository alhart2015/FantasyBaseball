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
    assert ind.label == "HOT · HR"
    assert "HR (62%)" in ind.tooltip


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
