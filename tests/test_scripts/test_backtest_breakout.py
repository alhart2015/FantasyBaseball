"""Unit tests for scripts/backtest_breakout.py:build_corpus and its two
FanGraphs/ZiPS readers. No network: skill_luck's season loaders and
keeper_value's load_zips_year are monkeypatched to tiny in-memory fake data.

Two silent-failure modes are pinned here:
  (a) CorpusEntry is a positional 5-tuple with TWO same-typed (Line) slots --
      surface (year-Y counting line) and actual_next (year-Y+1 RATE line). A
      swap would evade mypy + every other test and invert the backtest.
  (b) _raw_fg_hitter_lines relies on fetch_or_cache caching the PRE-rename raw
      pybaseball frame (so HR/R/RBI/SB/AVG survive). A renamed-only frame
      must raise, not silently default those to 0.0.
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

FGID = 100

# Raw FanGraphs counting lines (pre-rename column names), one per fake year --
# HR differs year to year so year-Y vs year-Y+1 mixups are visible in assertions.
_RAW_COUNTING = {
    2020: {
        "IDfg": FGID,
        "PA": 600,
        "AB": 550,
        "HR": 20,
        "R": 70,
        "RBI": 70,
        "SB": 10,
        "AVG": 0.260,
    },
    2021: {"IDfg": FGID, "PA": 610, "AB": 560, "HR": 30, "R": 80, "RBI": 85, "SB": 8, "AVG": 0.280},
    2022: {
        "IDfg": FGID,
        "PA": 590,
        "AB": 540,
        "HR": 15,
        "R": 60,
        "RBI": 55,
        "SB": 12,
        "AVG": 0.250,
    },
    2023: {"IDfg": FGID, "PA": 600, "AB": 550, "HR": 25, "R": 75, "RBI": 75, "SB": 9, "AVG": 0.270},
}

# ZiPS forecast lines keyed by the FORECAST year (i.e. the year they were published
# for), distinct per year so an off-by-one in the y+1 lookup is visible.
_ZIPS_FORECASTS = {
    2021: {"fg_id": FGID, "pa": 595.0, "hr": 24.0, "r": 74.0, "rbi": 74.0, "sb": 9.0, "avg": 0.265},
    2022: {
        "fg_id": FGID,
        "pa": 585.0,
        "hr": 18.0,
        "r": 65.0,
        "rbi": 60.0,
        "sb": 10.0,
        "avg": 0.255,
    },
    2023: {"fg_id": FGID, "pa": 600.0, "hr": 22.0, "r": 70.0, "rbi": 70.0, "sb": 9.0, "avg": 0.260},
}


def _fake_skill_luck_row(year: int) -> SkillLuckRow:
    avg = _RAW_COUNTING[year]["AVG"]
    return SkillLuckRow(
        mlbam=FGID,
        player_type="hitter",
        pa=float(_RAW_COUNTING[year]["PA"]),
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
    raw = _RAW_COUNTING[year]
    return line_rates(
        {
            "pa": raw["PA"],
            "ab": raw["AB"],
            "hr": raw["HR"],
            "r": raw["R"],
            "rbi": raw["RBI"],
            "sb": raw["SB"],
            "avg": raw["AVG"],
        },
        "hitter",
    )


def _write_fake_raw_csvs(cache_dir: Path, years) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for y in years:
        pd.DataFrame([_RAW_COUNTING[y]]).to_csv(cache_dir / f"fg_h_{y}.csv", index=False)


def _patch_no_network_loaders(monkeypatch, skill_luck_years) -> None:
    """Monkeypatch the skill_luck season loaders + the ZiPS loader so build_corpus
    never touches pybaseball/FanGraphs/Savant."""

    def fake_load_fg_hitters(cache_dir, year, **kwargs):
        # Real fetch_or_cache side effect (ensuring the CSV cache file exists) is
        # replaced by the test pre-writing the CSVs directly -- this is a no-op.
        return None

    def fake_build_hitter_skill_luck(cache_dir, year, **kwargs):
        if year not in skill_luck_years:
            return {}, None
        return {FGID: _fake_skill_luck_row(year)}, None

    def fake_load_zips_year(projections_root, year):
        if year not in _ZIPS_FORECASTS:
            raise FileNotFoundError(f"no fake ZiPS archive for {year}")
        hitters = pd.DataFrame([_ZIPS_FORECASTS[year]])
        pitchers = pd.DataFrame({"fg_id": [999]})  # unused by _zips_hitters_index
        return hitters, pitchers

    monkeypatch.setattr(script.skill_luck, "load_fg_hitters", fake_load_fg_hitters)
    monkeypatch.setattr(script.skill_luck, "build_hitter_skill_luck", fake_build_hitter_skill_luck)
    monkeypatch.setattr(script.kv_script, "load_zips_year", fake_load_zips_year)


def _build_fake_corpus(tmp_path, monkeypatch):
    cache_dir = tmp_path / "skill_luck"
    _write_fake_raw_csvs(cache_dir, _RAW_COUNTING)
    _patch_no_network_loaders(monkeypatch, skill_luck_years=set(_RAW_COUNTING))
    years = [2020, 2021, 2022]
    corpus = script.build_corpus(cache_dir, tmp_path / "projections", years)
    return corpus


def test_build_corpus_no_network_and_report_year_has_history_and_actual_next(tmp_path, monkeypatch):
    """Sanity check on the fixture itself: with 3 consecutive fake years, report
    year 2021 gets a real actual_next (from 2022) AND a real (non-empty) marcel
    history (from 2020) -- exercising both lookback paths in build_corpus."""
    corpus = _build_fake_corpus(tmp_path, monkeypatch)
    _surface, sl, actual_next, hist, _zips_line = corpus[2021][FGID]
    assert isinstance(sl, SkillLuckRow)
    assert actual_next  # real actual_next -- fgid played the following year
    assert hist  # real (non-empty) marcel history from 2020


def test_build_corpus_entry_tuple_is_not_transposed(tmp_path, monkeypatch):
    """Anti-swap guard: surface (slot 0) must be year-Y's RAW COUNTING line and
    actual_next (slot 2) must be year-(Y+1)'s RATE line. Both slots are typed
    `Line = dict[str, float]`, so a transpose evades mypy -- this test only
    catches it because counting and rate values live at very different
    magnitudes (raw HR count ~20-30 vs HR rate ~0.02-0.05)."""
    corpus = _build_fake_corpus(tmp_path, monkeypatch)
    surface, _sl, actual_next, _hist, _zips_line = corpus[2021][FGID]

    # surface: year 2021's raw counting line, unchanged from the fake CSV row.
    expected_surface = {
        "pa": 610.0,
        "ab": 560.0,
        "hr": 30.0,
        "r": 80.0,
        "rbi": 85.0,
        "sb": 8.0,
        "avg": 0.280,
    }
    assert surface == expected_surface
    assert surface["hr"] == 30.0  # a raw count, not a per-PA rate

    # actual_next: year 2022's RATE line (line_rates of the 2022 raw counting line).
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
    _surface, _sl, actual_next, _hist, zips_line = corpus[report_year][FGID]
    forecast_year = report_year + 1

    expected_zips = {k: v for k, v in _ZIPS_FORECASTS[forecast_year].items() if k != "fg_id"}
    assert zips_line == expected_zips
    # off-by-one guards: not the report year's own forecast, not two years out
    if report_year in _ZIPS_FORECASTS:
        assert zips_line != {k: v for k, v in _ZIPS_FORECASTS[report_year].items() if k != "fg_id"}

    expected_actual_next = _expected_rate_line(forecast_year)
    assert actual_next == pytest.approx(expected_actual_next)


def test_raw_fg_hitter_lines_raises_on_renamed_only_frame(tmp_path):
    """A cached fg_h_{year}.csv that lacks the raw counting columns (HR/R/RBI/SB/
    AVG) -- e.g. if a future refactor accidentally cached the post-rename frame
    instead of pybaseball's raw one -- must raise loud, not silently emit a
    counting line of zeros that would score as a plausible-but-garbage verdict."""
    cache_dir = tmp_path / "skill_luck"
    cache_dir.mkdir()
    # Renamed-only-shaped frame: keeps IDfg + PA (so the id/PT columns "look" fine)
    # but is missing AB/HR/R/RBI/SB/AVG entirely.
    pd.DataFrame({"IDfg": [FGID], "PA": [600]}).to_csv(cache_dir / "fg_h_2020.csv", index=False)

    with pytest.raises(ValueError, match="counting columns"):
        script._raw_fg_hitter_lines(cache_dir, 2020)


def test_raw_fg_hitter_lines_ok_with_full_raw_frame(tmp_path):
    """Control case: a full raw frame (all _COUNTING_COLS present) parses fine and
    is not caught by the new guard."""
    cache_dir = tmp_path / "skill_luck"
    _write_fake_raw_csvs(cache_dir, [2020])
    out = script._raw_fg_hitter_lines(cache_dir, 2020)
    assert out[FGID]["hr"] == 20.0
    assert out[FGID]["pa"] == 600.0


def test_zips_hitters_index_skips_nan_cells_instead_of_propagating_nan(tmp_path, monkeypatch):
    """Gap 2: a present-but-NaN ZiPS cell must be treated like an absent one (0.0),
    matching the pd.notna pattern _raw_fg_hitter_lines already uses -- not silently
    propagate as float('nan') into the corpus (float(row.get('pa', 0.0)) returns nan
    for a present NaN cell; the dict-level default never fires)."""

    def fake_load_zips_year(projections_root, year):
        row = {
            "fg_id": FGID,
            "pa": float("nan"),
            "hr": 20.0,
            "r": float("nan"),
            "rbi": 60.0,
            "sb": 5.0,
            "avg": 0.270,
        }
        return pd.DataFrame([row]), pd.DataFrame({"fg_id": [999]})

    monkeypatch.setattr(script.kv_script, "load_zips_year", fake_load_zips_year)

    idx = script._zips_hitters_index(tmp_path / "projections", 2021)
    assert idx is not None
    line = idx[FGID]
    assert line["pa"] == 0.0
    assert not math.isnan(line["pa"])
    assert line["r"] == 0.0
    assert not math.isnan(line["r"])
    assert line["hr"] == 20.0  # untouched, non-NaN cells pass through normally
    assert line["rbi"] == 60.0
