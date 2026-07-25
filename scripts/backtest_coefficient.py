"""Coefficient backtest: does the PER-PLAYER-STAT regression coefficient (1-w,
i.e. the skill-adjusted blend) beat the current FLAT global
out_year_regression=0.6 at predicting the future?

Same 2015-2024 corpus + held-out w-tuning as scripts/backtest_breakout.py, but
scores three estimators of a player's forward rate line against realized
next-year rates (hitters only). For each stat, with prior = the reconstructed
Marcel projection (stale re: the current year) and surface = the current-year
line:

    surface    = all current           (coefficient 0.0 on the prior)
    flat_0.6   = prior + 0.4*(surface-prior)   (today's global: 60% to prior)
    skill_adj  = prior +   w*(surface-prior)   (per-stat coefficient = 1-w)

If skill_adj does NOT beat flat_0.6, the per-player coefficient is not worth the
complexity -- a flat regression does the job. This is the test matched to the
real goal (a current-informed forward line), and it settles the 1-w form.

See docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backtest_breakout as bt  # build_corpus + year ranges + cache/projection paths

from fantasy_baseball.analysis.breakout_backtest import (
    DEFAULT_RULER,
    _bootstrap_diff,
    _records,
    _spearman,
    rate_mae,
    sgp_on_ruler,
    tune_wmap,
)

FLAT_REGRESSION = 0.6  # today's global out_year_regression: weight on the prior side


def _flat_rates(rec: dict) -> dict[str, float]:
    """The flat-0.6-regression forward line: prior + 0.4*(surface - prior) per stat."""
    return {
        s: FLAT_REGRESSION * rec["prior_rates"].get(s, sr) + (1.0 - FLAT_REGRESSION) * sr
        for s, sr in rec["surface_rates"].items()
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = bt.build_corpus(bt.SKILL_LUCK_CACHE_DIR, bt.PROJECTIONS_ROOT, bt.ALL_YEARS)
    params = tune_wmap(corpus, list(bt.FIT_YEARS))
    recs = _records(corpus, bt.REPORT_YEARS, params, DEFAULT_RULER)
    actual = [r["actual"] for r in recs]
    flat = [sgp_on_ruler(_flat_rates(r), DEFAULT_RULER) for r in recs]

    spearman = {
        "surface (all current)": _spearman([r["surface"] for r in recs], actual),
        "flat_0.6 (global today)": _spearman(flat, actual),
        "skill_adj (per-stat 1-w)": _spearman([r["skill"] for r in recs], actual),
    }
    mae = {
        "surface": fmean([rate_mae(r["surface_rates"], r["actual_rates"]) for r in recs]),
        "flat_0.6": fmean([rate_mae(_flat_rates(r), r["actual_rates"]) for r in recs]),
        "skill_adj": fmean([rate_mae(r["adjusted_rates"], r["actual_rates"]) for r in recs]),
    }
    ci_skill_vs_flat = _bootstrap_diff([r["skill"] for r in recs], flat, actual)

    print("Coefficient backtest -- per-stat (1-w) vs flat 0.6, predicting next-year")
    print(
        f"  fit years {list(bt.FIT_YEARS)}  report years {bt.REPORT_YEARS}  candidates {len(recs)}"
    )
    print("  Spearman vs realized next-year SGP (higher = ranks the future better):")
    for k, v in spearman.items():
        print(f"    {k:26s} {v:+.3f}")
    print("  rate MAE vs realized next-year rates (LOWER = more accurate forward line):")
    for k, v in mae.items():
        print(f"    {k:26s} {v:.4f}")
    lo, hi = ci_skill_vs_flat
    print(f"  95% CI  Spearman(skill_adj) - Spearman(flat_0.6): [{lo:+.3f}, {hi:+.3f}]")
    verdict = (
        "per-stat coefficient BEATS flat 0.6 (CI excludes 0)"
        if lo > 0
        else "per-stat coefficient does NOT clearly beat flat 0.6 (CI includes 0)"
    )
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
