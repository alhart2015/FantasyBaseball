"""Year-over-year backtest for the breakout/mirage keeper diagnostic (Task 10,
Phase 4 -- the go/no-go gate for any future automation of the report in
scripts/run_breakout_report.py).

Builds a per-year corpus (hitters only), keyed by MLBAM, from:
- MLB Stats API season lines (surface / actual_next / history counting lines +
  derived K%/BB%/BABIP) via fantasy_baseball.data.skill_luck, and
- Baseball Savant expected stats (the SkillLuckRow underlying), and
- the archived ZiPS forecast published for the following season (data/
  projections/{year+1}/zips-hitters*.csv, joined on MLBAMID; None when that
  archive isn't cached).

surface = that year's actual line, actual_next = the FOLLOWING year's actual
rates, history = up to 3 prior years' actual rate lines (Marcel prior),
zips_line = ZiPS's own forecast for the following season. Calls
fantasy_baseball.analysis.breakout_backtest.run_backtest to compare three
estimators -- surface (unadjusted), skill_adjusted (this diagnostic), and
pure_zips -- against realized next-year performance, tuning the w-mapping on
2015-2022 only and holding out 2023-2024 to report on. Writes
data/stats/breakout_backtest_results.csv.

v1 is HITTERS-ONLY (pitcher expected-stats coverage on Savant is thinner and
starts later); pitcher backtest is a named follow-up.

Data sources are the MLB Stats API (public, no auth) + Baseball Savant -- no
FanGraphs, so no Cloudflare 403. pure_zips is limited to the report years whose
following-season ZiPS archive is on disk (2022-2025).

See docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fantasy_baseball.analysis.breakout import line_rates
from fantasy_baseball.analysis.breakout_backtest import Corpus, CorpusEntry, Line, run_backtest
from fantasy_baseball.data import skill_luck

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_LUCK_CACHE_DIR = REPO_ROOT / "data" / "skill_luck"
PROJECTIONS_ROOT = REPO_ROOT / "data" / "projections"
OUT_PATH = REPO_ROOT / "data" / "stats" / "breakout_backtest_results.csv"

FIT_YEARS = range(2015, 2023)  # 2015..2022: tuning only, never scored
REPORT_YEARS = [2023, 2024]  # held-out: the years run_backtest reports on
ALL_YEARS = list(range(2015, 2025))  # 2015..2024: the full corpus span

_ROTO_KEYS = ("pa", "hr", "r", "rbi", "sb", "avg")

V1_HITTERS_ONLY_NOTE = (
    "v1 is HITTERS-ONLY: pitcher expected-stats coverage on Savant is thinner and "
    "starts later. Pitcher backtest is a named follow-up."
)


def _mlb_hitter_roto(cache_dir: Path, year: int) -> dict[int, Line]:
    """MLB Stats API counting line (pa/hr/r/rbi/sb/avg) keyed by MLBAM."""
    df = skill_luck.load_mlb_hitters(cache_dir, year)
    out: dict[int, Line] = {}
    for r in df.itertuples(index=False):
        out[int(r.mlbam)] = {k: float(getattr(r, k)) for k in _ROTO_KEYS}
    return out


def _zips_hitters_by_mlbam(projections_root: Path, year: int) -> dict[int, Line] | None:
    """MLBAM -> ZiPS counting line for the archived `year` hitters export, or None
    when that year's ZiPS archive isn't on disk (treated as ZiPS-uncovered rather
    than failing the whole run)."""
    matches = glob.glob(str(projections_root / str(year) / "zips-hitters*.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    if "MLBAMID" not in df.columns:
        return None
    src = {"pa": "PA", "hr": "HR", "r": "R", "rbi": "RBI", "sb": "SB", "avg": "AVG"}
    idx: dict[int, Line] = {}
    for _, row in df.iterrows():
        m = row.get("MLBAMID")
        if m is None or pd.isna(m):
            continue
        idx[int(m)] = {
            k: (float(v) if (v := row.get(col)) is not None and pd.notna(v) else 0.0)
            for k, col in src.items()
        }
    return idx


def build_corpus(cache_dir: Path, projections_root: Path, years: list[int]) -> Corpus:
    """Assemble the year -> {mlbam -> (surface, SkillLuckRow, actual_next_rates,
    history, zips_line)} corpus run_backtest consumes (hitters only). Fetches
    `years` plus one trailing year (for actual_next); history reaches back only
    as far as `years` provides, so the earliest requested years get a thinner
    Marcel history (gracefully handled by marcel_prior)."""
    span = sorted(set(years) | {y + 1 for y in years})
    roto = {y: _mlb_hitter_roto(cache_dir, y) for y in span}
    # Savant only exists from 2015; a year with no xStats yields an empty dict.
    skill_by_year = {
        y: (skill_luck.build_hitter_skill_luck(cache_dir, y)[0] if y >= 2015 else {}) for y in span
    }
    zips_by_forecast_year = {y + 1: _zips_hitters_by_mlbam(projections_root, y + 1) for y in years}

    corpus: Corpus = {}
    for year in years:
        year_data: dict[int, CorpusEntry] = {}
        next_lines = roto.get(year + 1, {})
        zidx = zips_by_forecast_year.get(year + 1)
        for mlbam, surface in roto[year].items():
            sl = skill_by_year[year].get(mlbam)
            if sl is None:
                continue  # no Statcast underlying this year -- can't score
            next_line = next_lines.get(mlbam)
            if next_line is None:
                continue  # didn't play the following year -- no outcome to grade against
            actual_next = line_rates(next_line, "hitter")
            hist = [
                (y2, line_rates(roto[y2][mlbam], "hitter"))
                for y2 in range(year - 1, year - 4, -1)
                if y2 in roto and mlbam in roto[y2]
            ]
            zips_line = zidx.get(mlbam) if zidx is not None else None
            year_data[mlbam] = (surface, sl, actual_next, hist, zips_line)
        corpus[year] = year_data
    return corpus


def _zips_covered_count(corpus: Corpus, years: list[int]) -> int:
    """Report-year population (pre-candidate-filter) with a ZiPS line attached."""
    return sum(1 for y in years for entry in corpus[y].values() if entry[4] is not None)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = build_corpus(SKILL_LUCK_CACHE_DIR, PROJECTIONS_ROOT, ALL_YEARS)
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
