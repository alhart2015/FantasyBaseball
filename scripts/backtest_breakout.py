"""Year-over-year backtest for the breakout/mirage keeper diagnostic (Task 10,
Phase 4 -- the go/no-go gate for any future automation of the report in
scripts/run_breakout_report.py).

Builds a per-year corpus from cached data/skill_luck/ hitter frames spanning
2015-2024 -- surface = that year's actual FanGraphs counting line, SkillLuckRow
= that year's Statcast xStats/FanGraphs-rate underlying, actual_next = the
FOLLOWING year's actual rates, history = up to 3 prior years' actual rate
lines (for the Marcel prior), zips_line = the archived ZiPS forecast published
for the following season (via scripts/keeper_value.py:load_zips_year; None
when that archive isn't cached). Calls
fantasy_baseball.analysis.breakout_backtest.run_backtest to compare three
estimators -- surface (last year's line, unadjusted), skill_adjusted (this
diagnostic's regression), and pure_zips (a professional projection system's
own forecast) -- against realized next-year performance, tuning the
w-mapping on 2015-2022 only and holding out 2023-2024 to report on. Writes
data/stats/breakout_backtest_results.csv.

v1 is HITTERS-ONLY (pitcher expected-stats coverage on Savant is thinner and
starts later); pitcher backtest is a named follow-up.

Historical FanGraphs/Statcast fetches (10 years x FanGraphs hitters + 2
Statcast endpoints + the Chadwick id-map register) proxy through pybaseball,
which is subject to the same FanGraphs Cloudflare 403 blocking already
documented for the live ROS fetch path -- a first run with an empty
data/skill_luck/ cache is expected to need retries or a manual CSV mirror,
the same as scripts/refresh_ros.py's known failure mode.

See docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import keeper_value as kv_script  # scripts/keeper_value.py: load_zips_year

from fantasy_baseball.analysis.breakout import line_rates
from fantasy_baseball.analysis.breakout_backtest import Corpus, CorpusEntry, Line, run_backtest
from fantasy_baseball.data import skill_luck

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_LUCK_CACHE_DIR = REPO_ROOT / "data" / "skill_luck"
OUT_PATH = REPO_ROOT / "data" / "stats" / "breakout_backtest_results.csv"

FIT_YEARS = range(2015, 2023)  # 2015..2022: tuning only, never scored
REPORT_YEARS = [2023, 2024]  # held-out: the years run_backtest reports on
ALL_YEARS = list(range(2015, 2025))  # 2015..2024: the full corpus span

_COUNTING_COLS = ("PA", "AB", "HR", "R", "RBI", "SB", "AVG")
_COUNTING_KEYS = {
    "PA": "pa",
    "AB": "ab",
    "HR": "hr",
    "R": "r",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
}

V1_HITTERS_ONLY_NOTE = (
    "v1 is HITTERS-ONLY: pitcher expected-stats coverage on Savant is thinner and "
    "starts later. Pitcher backtest is a named follow-up."
)


def _raw_fg_hitter_lines(cache_dir: Path, year: int) -> dict[int, Line]:
    """Full FanGraphs counting line (PA, AB, HR, R, RBI, SB, AVG) keyed by fg_id,
    read from the RAW cached fg_h_{year}.csv. skill_luck.load_fg_hitters strips
    its in-memory return value to a narrow rename map for SkillLuckRow (Age,
    K%, BB%, BABIP, HR/FB, Contact%, PA) -- the counting stats the backtest's
    surface/actual_next/history lines need live outside that map, but
    fetch_or_cache still writes the FULL raw pybaseball frame to this path, so
    reading it directly recovers them without a second fetch.
    """
    path = cache_dir / f"fg_h_{year}.csv"
    if not path.exists():
        return {}
    raw = pd.read_csv(path)
    out: dict[int, Line] = {}
    for _, r in raw.iterrows():
        fg = r.get("IDfg")
        if pd.isna(fg):
            continue
        line: Line = {}
        for col in _COUNTING_COLS:
            v = r.get(col)
            line[_COUNTING_KEYS[col]] = float(v) if v is not None and pd.notna(v) else 0.0
        out[int(fg)] = line
    return out


def _ensure_raw_fg_hitters(cache_dir: Path, year: int) -> dict[int, Line]:
    """Populate/reuse the fg_h_{year}.csv cache via skill_luck's fetch-or-cache
    path (same default pybaseball fetcher, same file), then read it back for
    the full counting line."""
    skill_luck.load_fg_hitters(cache_dir, year)  # side effect: ensures the cache file exists
    return _raw_fg_hitter_lines(cache_dir, year)


def _zips_hitters_index(projections_root: Path, year: int) -> dict[int, Line] | None:
    """fg_id -> ZiPS counting line for the archived `year` hitters export, or
    None when that year's ZiPS archive isn't cached (load_zips_year raises
    FileNotFoundError with a download link; the backtest treats this player-
    year as ZiPS-uncovered rather than failing the whole run)."""
    try:
        hitters, _pitchers = kv_script.load_zips_year(projections_root, year)
    except FileNotFoundError:
        return None
    idx: dict[int, Line] = {}
    for _, row in hitters.iterrows():
        fg = row.get("fg_id")
        if fg is None or pd.isna(fg):
            continue
        try:
            fgid = int(fg)
        except (TypeError, ValueError):
            continue
        idx[fgid] = {
            "pa": float(row.get("pa", 0.0)),
            "hr": float(row.get("hr", 0.0)),
            "r": float(row.get("r", 0.0)),
            "rbi": float(row.get("rbi", 0.0)),
            "sb": float(row.get("sb", 0.0)),
            "avg": float(row.get("avg", 0.0)),
        }
    return idx


def build_corpus(cache_dir: Path, projections_root: Path, years: list[int]) -> Corpus:
    """Assemble the year -> {fg_id -> (surface, SkillLuckRow, actual_next_rates,
    history, zips_line)} corpus run_backtest consumes (hitters only). `years`
    must be requested with enough lead years already cached/fetchable for the
    Marcel history lookback -- this function fetches `years` plus one trailing
    year (for actual_next) but does NOT reach further back for history than
    `years` itself provides, so the earliest requested years will have a
    thinner-than-3-year history (gracefully handled by marcel_prior).
    """
    span = sorted(set(years) | {y + 1 for y in years})
    raw_lines = {y: _ensure_raw_fg_hitters(cache_dir, y) for y in span}
    skill_luck_by_year = {y: skill_luck.build_hitter_skill_luck(cache_dir, y)[0] for y in span}
    zips_by_forecast_year = {y + 1: _zips_hitters_index(projections_root, y + 1) for y in years}

    corpus: Corpus = {}
    for year in years:
        year_data: dict[int, CorpusEntry] = {}
        next_lines = raw_lines.get(year + 1, {})
        zidx = zips_by_forecast_year.get(year + 1)
        for fgid, surface in raw_lines[year].items():
            sl = skill_luck_by_year[year].get(fgid)
            if sl is None:
                continue  # no Statcast/FanGraphs-rate underlying this year -- can't score
            next_line = next_lines.get(fgid)
            if next_line is None:
                continue  # didn't play (enough) the following year -- no outcome to grade against
            actual_next = line_rates(next_line, "hitter")
            hist = [
                (y2, line_rates(raw_lines[y2][fgid], "hitter"))
                for y2 in range(year - 1, year - 4, -1)
                if y2 in raw_lines and fgid in raw_lines[y2]
            ]
            zips_line = zidx.get(fgid) if zidx is not None else None
            year_data[fgid] = (surface, sl, actual_next, hist, zips_line)
        corpus[year] = year_data
    return corpus


def _zips_covered_count(corpus: Corpus, years: list[int]) -> int:
    """Report-year population (pre-candidate-filter) with a ZiPS line attached --
    the subset run_backtest's ci_skill_vs_zips is actually computed over is a
    further (smaller) restriction to the candidate/deviator population; this is
    the raw coverage size, for the printed summary."""
    return sum(1 for y in years for entry in corpus[y].values() if entry[4] is not None)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = build_corpus(SKILL_LUCK_CACHE_DIR, kv_script.PROJECTIONS_ROOT, ALL_YEARS)
    results = run_backtest(corpus, fit_years=FIT_YEARS, report_years=REPORT_YEARS)
    n_zips_covered = _zips_covered_count(corpus, REPORT_YEARS)

    print("Breakout/mirage diagnostic backtest")
    print(f"  fit years: {list(FIT_YEARS)}  report years: {REPORT_YEARS}")
    print(f"  {V1_HITTERS_ONLY_NOTE}")
    print(f"  candidates scored (report years): {results['n']}")
    print(f"  ZiPS-covered report-year population (pre-candidate-filter): {n_zips_covered}")
    print("  spearman vs realized next-year ruler-SGP:")
    for k, v in results["spearman"].items():
        print(f"    {k:15s} {v:+.3f}")
    lo, hi = results["ci_skill_vs_surface"]
    print(f"  95% CI skill_adjusted - surface: [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = results["ci_skill_vs_zips"]
    print(f"  95% CI skill_adjusted - pure_zips: [{lo:+.3f}, {hi:+.3f}]")
    print("  rate MAE:")
    for k, v in results["rate_mae"].items():
        print(f"    {k:15s} {v:.4f}")
    print(f"  label lift (top vs bottom believed-deviation tercile): {results['label_lift']:+.3f}")
    print(f"  verdict: {results['verdict']}")
    if results["verdict"] != "clears gate":
        print("  -> automation stays deferred")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"metric": "spearman_surface", "value": results["spearman"]["surface"]},
        {"metric": "spearman_skill_adjusted", "value": results["spearman"]["skill_adjusted"]},
        {"metric": "spearman_pure_zips", "value": results["spearman"]["pure_zips"]},
        {"metric": "ci_skill_vs_surface_low", "value": results["ci_skill_vs_surface"][0]},
        {"metric": "ci_skill_vs_surface_high", "value": results["ci_skill_vs_surface"][1]},
        {"metric": "ci_skill_vs_zips_low", "value": results["ci_skill_vs_zips"][0]},
        {"metric": "ci_skill_vs_zips_high", "value": results["ci_skill_vs_zips"][1]},
        {"metric": "rate_mae_surface", "value": results["rate_mae"]["surface"]},
        {"metric": "rate_mae_skill_adjusted", "value": results["rate_mae"]["skill_adjusted"]},
        {"metric": "label_lift", "value": results["label_lift"]},
        {"metric": "verdict", "value": results["verdict"]},
        {"metric": "n_candidates", "value": results["n"]},
        {"metric": "n_zips_covered", "value": n_zips_covered},
    ]
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
