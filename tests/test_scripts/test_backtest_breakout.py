"""Unit tests for scripts/backtest_breakout.py:build_corpus and its two MLB Stats
API / ZiPS readers. No network: skill_luck's season loaders and the module's own
ZiPS-by-MLBAM reader are monkeypatched to tiny in-memory fake data keyed by MLBAM.

Two silent-failure modes are pinned here:
  (a) CorpusEntry is a positional 5-tuple with TWO same-typed (Line) slots --
      surface (year-Y counting line) and actual_next (year-Y+1 RATE line). A
      swap would evade mypy + every other test and invert the backtest.
  (b) _zips_hitters_by_mlbam must join on MLBAMID and skip present-but-NaN cells
      (treat like absent -> 0.0) rather than silently propagating NaN into the
      corpus.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import backtest_breakout as script

from fantasy_baseball.analysis.breakout import SkillLuckRow, line_rates

MLBAM = 100

# Derived MLB Stats API hitter lines (post skill_luck.load_mlb_hitters shape), one
# per fake year -- HR differs year to year so year-Y vs year-Y+1 mixups are visible
# in assertions. No "ab" key: _ROTO_KEYS (pa, hr, r, rbi, sb, avg) doesn't carry it.
_MLB_LINES = {
    2020: {
        "mlbam": MLBAM,
        "pa": 600.0,
        "ab": 550.0,
        "h": 143.0,
        "hr": 20.0,
        "r": 70.0,
        "rbi": 70.0,
        "sb": 10.0,
        "avg": 0.260,
        "k_pct": 0.20,
        "bb_pct": 0.09,
        "babip": 0.300,
    },
    2021: {
        "mlbam": MLBAM,
        "pa": 610.0,
        "ab": 560.0,
        "h": 157.0,
        "hr": 30.0,
        "r": 80.0,
        "rbi": 85.0,
        "sb": 8.0,
        "avg": 0.280,
        "k_pct": 0.19,
        "bb_pct": 0.10,
        "babip": 0.310,
    },
    2022: {
        "mlbam": MLBAM,
        "pa": 590.0,
        "ab": 540.0,
        "h": 135.0,
        "hr": 15.0,
        "r": 60.0,
        "rbi": 55.0,
        "sb": 12.0,
        "avg": 0.250,
        "k_pct": 0.22,
        "bb_pct": 0.08,
        "babip": 0.290,
    },
    2023: {
        "mlbam": MLBAM,
        "pa": 600.0,
        "ab": 550.0,
        "h": 148.0,
        "hr": 25.0,
        "r": 75.0,
        "rbi": 75.0,
        "sb": 9.0,
        "avg": 0.270,
        "k_pct": 0.20,
        "bb_pct": 0.09,
        "babip": 0.300,
    },
}

# ZiPS forecast lines keyed by the FORECAST year (i.e. the year they were published
# for), distinct per year so an off-by-one in the y+1 lookup is visible. Keyed by
# MLBAM (mirrors _zips_hitters_by_mlbam's MLBAMID join).
_ZIPS_FORECASTS = {
    2021: {
        "mlbam": MLBAM,
        "pa": 595.0,
        "hr": 24.0,
        "r": 74.0,
        "rbi": 74.0,
        "sb": 9.0,
        "avg": 0.265,
    },
    2022: {
        "mlbam": MLBAM,
        "pa": 585.0,
        "hr": 18.0,
        "r": 65.0,
        "rbi": 60.0,
        "sb": 10.0,
        "avg": 0.255,
    },
    2023: {
        "mlbam": MLBAM,
        "pa": 600.0,
        "hr": 22.0,
        "r": 70.0,
        "rbi": 70.0,
        "sb": 9.0,
        "avg": 0.260,
    },
}


def _fake_skill_luck_row(year: int) -> SkillLuckRow:
    avg = _MLB_LINES[year]["avg"]
    return SkillLuckRow(
        mlbam=MLBAM,
        player_type="hitter",
        pa=_MLB_LINES[year]["pa"],
        ip=0.0,
        age=27.0,
        barrel_pct=0.08,
        xslg=0.45,
        slg=0.45,
        xba=avg,
        ba=avg,
        babip=0.300,
        xwoba=0.340,
        woba=0.340,
        k_pct=0.20,
        bb_pct=0.09,
    )


def _expected_rate_line(year: int) -> dict[str, float]:
    raw = _MLB_LINES[year]
    return line_rates(
        {
            "pa": raw["pa"],
            "hr": raw["hr"],
            "r": raw["r"],
            "rbi": raw["rbi"],
            "sb": raw["sb"],
            "avg": raw["avg"],
        },
        "hitter",
    )


def _patch_no_network_loaders(monkeypatch, skill_luck_years) -> None:
    """Monkeypatch skill_luck's MLB Stats API loaders + the module's own ZiPS-by-
    MLBAM reader so build_corpus never touches the MLB Stats API/Savant/disk CSVs."""

    def fake_load_mlb_hitters(cache_dir, year, **kwargs):
        if year not in _MLB_LINES:
            return pd.DataFrame()
        return pd.DataFrame([_MLB_LINES[year]])

    def fake_build_hitter_skill_luck(cache_dir, year, **kwargs):
        if year not in skill_luck_years:
            return {}, None
        return {MLBAM: _fake_skill_luck_row(year)}, None

    def fake_zips_hitters_by_mlbam(projections_root, year):
        if year not in _ZIPS_FORECASTS:
            return None
        row = _ZIPS_FORECASTS[year]
        return {row["mlbam"]: {k: v for k, v in row.items() if k != "mlbam"}}

    monkeypatch.setattr(script.skill_luck, "load_mlb_hitters", fake_load_mlb_hitters)
    monkeypatch.setattr(script.skill_luck, "build_hitter_skill_luck", fake_build_hitter_skill_luck)
    monkeypatch.setattr(script, "_zips_hitters_by_mlbam", fake_zips_hitters_by_mlbam)


def _build_fake_corpus(tmp_path, monkeypatch):
    cache_dir = tmp_path / "skill_luck"
    _patch_no_network_loaders(monkeypatch, skill_luck_years=set(_MLB_LINES))
    years = [2020, 2021, 2022]
    corpus = script.build_corpus(cache_dir, tmp_path / "projections", years)
    return corpus


def test_build_corpus_no_network_and_report_year_has_history_and_actual_next(tmp_path, monkeypatch):
    """Sanity check on the fixture itself: with 3 consecutive fake years, report
    year 2021 gets a real actual_next (from 2022) AND a real (non-empty) marcel
    history (from 2020) -- exercising both lookback paths in build_corpus."""
    corpus = _build_fake_corpus(tmp_path, monkeypatch)
    _surface, sl, actual_next, hist, _zips_line = corpus[2021][MLBAM]
    assert isinstance(sl, SkillLuckRow)
    assert actual_next  # real actual_next -- mlbam played the following year
    assert hist  # real (non-empty) marcel history from 2020


def test_build_corpus_entry_tuple_is_not_transposed(tmp_path, monkeypatch):
    """Anti-swap guard: surface (slot 0) must be year-Y's RAW roto line and
    actual_next (slot 2) must be year-(Y+1)'s RATE line. Both slots are typed
    `Line = dict[str, float]`, so a transpose evades mypy -- this test only
    catches it because counting and rate values live at very different
    magnitudes (raw HR count ~15-30 vs HR rate ~0.02-0.05)."""
    corpus = _build_fake_corpus(tmp_path, monkeypatch)
    surface, _sl, actual_next, _hist, _zips_line = corpus[2021][MLBAM]

    # surface: year 2021's raw roto line (_ROTO_KEYS only), unchanged from the fake.
    expected_surface = {
        "pa": 610.0,
        "hr": 30.0,
        "r": 80.0,
        "rbi": 85.0,
        "sb": 8.0,
        "avg": 0.280,
    }
    assert surface == expected_surface
    assert surface["hr"] == 30.0  # a raw count, not a per-PA rate

    # actual_next: year 2022's RATE line (line_rates of the 2022 raw roto line).
    expected_actual_next = _expected_rate_line(2022)
    assert actual_next == pytest.approx(expected_actual_next)
    assert actual_next["hr"] == pytest.approx(15.0 / 590.0)  # a rate, not a raw count

    # Cross-checks that would fail if surface and actual_next were swapped:
    assert surface["hr"] != pytest.approx(actual_next["hr"])
    assert actual_next["hr"] < 1.0 < surface["hr"]  # rate vs count magnitude
    # and that would fail if either slot leaked the OTHER year's data:
    assert surface["hr"] != 15.0  # not year 2022's raw HR count
    assert actual_next["hr"] != pytest.approx(30.0 / 610.0)  # not year 2021's own rate


@pytest.mark.parametrize("report_year", [2020, 2021, 2022])
def test_build_corpus_zips_line_and_actual_next_come_from_year_plus_one(
    tmp_path, monkeypatch, report_year
):
    """Y -> Y+1 alignment: for report year Y, zips_line must be the ZiPS forecast
    published for Y+1 (not Y, not Y+2), and actual_next must be Y+1's actuals."""
    corpus = _build_fake_corpus(tmp_path, monkeypatch)
    _surface, _sl, actual_next, _hist, zips_line = corpus[report_year][MLBAM]
    forecast_year = report_year + 1

    expected_zips = {k: v for k, v in _ZIPS_FORECASTS[forecast_year].items() if k != "mlbam"}
    assert zips_line == expected_zips
    # off-by-one guards: not the report year's own forecast, not two years out
    if report_year in _ZIPS_FORECASTS:
        assert zips_line != {k: v for k, v in _ZIPS_FORECASTS[report_year].items() if k != "mlbam"}

    expected_actual_next = _expected_rate_line(forecast_year)
    assert actual_next == pytest.approx(expected_actual_next)


def test_zips_hitters_by_mlbam_skips_nan_cells_instead_of_propagating_nan(tmp_path):
    """A present-but-NaN ZiPS cell must be treated like an absent one (0.0), not
    silently propagate as float('nan') into the corpus. Exercises the REAL
    (unpatched) _zips_hitters_by_mlbam against a CSV written to disk -- no
    network, but real MLBAMID-column parsing."""
    proj_dir = tmp_path / "projections" / "2021"
    proj_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "MLBAMID": MLBAM,
                "PA": float("nan"),
                "HR": 20.0,
                "R": float("nan"),
                "RBI": 60.0,
                "SB": 5.0,
                "AVG": 0.270,
            }
        ]
    ).to_csv(proj_dir / "zips-hitters.csv", index=False)

    idx = script._zips_hitters_by_mlbam(tmp_path / "projections", 2021)
    assert idx is not None
    line = idx[MLBAM]
    assert line["pa"] == 0.0
    assert not math.isnan(line["pa"])
    assert line["r"] == 0.0
    assert not math.isnan(line["r"])
    assert line["hr"] == 20.0  # untouched, non-NaN cells pass through normally
    assert line["rbi"] == 60.0


def test_zips_hitters_by_mlbam_returns_none_when_no_archive(tmp_path):
    """No zips-hitters*.csv on disk for that year -- treated as ZiPS-uncovered,
    not a crash."""
    assert script._zips_hitters_by_mlbam(tmp_path / "projections", 1999) is None
