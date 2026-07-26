"""Backtest: does barrel rate or park-adjusted xHR confirm next-year HR better than
the SLG-vs-xSLG proxy in w_for_stat? Hitters, common support 2016-2024 (xHR starts
2016). Go/no-go = challenger's bootstrap CI vs xslg excludes 0, with a level-control
veto. Wires nothing in -- that is a verdict-gated follow-up.

See docs/superpowers/specs/2026-07-26-hr-confirmation-backtest-design.md (#262).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

import backtest_breakout as bt  # build_corpus + cache/projection paths
from fantasy_baseball.analysis import hr_confirm

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "stats" / "hr_confirm_backtest_results.csv"

SOURCE_YEARS = list(range(2016, 2025))  # 2016..2024 predicting 2017..2025 (common support)
FIT_YEARS = list(range(2016, 2021))  # 2016..2020: tuning/calibration only, never scored
REPORT_YEARS = list(range(2021, 2025))  # 2021..2024: held-out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = bt.build_corpus(bt.SKILL_LUCK_CACHE_DIR, bt.PROJECTIONS_ROOT, SOURCE_YEARS)
    res = hr_confirm.run(corpus, fit_years=FIT_YEARS, report_years=REPORT_YEARS)

    print("HR-confirmation backtest -- barrels/xHR vs the SLG-vs-xSLG proxy")
    print(f"  fit {FIT_YEARS}  report {REPORT_YEARS}")
    print(f"  n fit {res['n_fit']}  n report {res['n_report']}")
    slope, intercept = res["barrel_calib"]
    print(f"  barrel calib: HR/PA = {slope:+.4f}*brl_pa {intercept:+.4f}")
    print("  tuned scales:", {k: round(v, 3) for k, v in res["scales"].items()})
    print("  Spearman(forward HR/PA, next-year HR/PA)  (higher = ranks the future better):")
    for k, v in res["spearman"].items():
        print(f"    {k:14s} {v:+.3f}")
    print("  rate MAE (lower = better forward line):")
    for k, v in res["mae"].items():
        print(f"    {k:14s} {v:.4f}")
    for cand, d in res["verdicts"].items():
        lo, hi = d["ci_vs_xslg"]
        tiers = ", ".join(f"{t:+.2f}" for t in d["tier_spearman"])
        print(
            f"  {cand.upper()} vs xslg: CI [{lo:+.3f}, {hi:+.3f}]  "
            f"level-control(low->high tiers) [{tiers}]  -> {d['verdict']}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": "n_report", "value": res["n_report"]}]
    for k, v in res["spearman"].items():
        rows.append({"metric": f"spearman_{k}", "value": v})
    for k, v in res["mae"].items():
        rows.append({"metric": f"mae_{k}", "value": v})
    for cand, d in res["verdicts"].items():
        rows.append({"metric": f"{cand}_ci_low", "value": d["ci_vs_xslg"][0]})
        rows.append({"metric": f"{cand}_ci_high", "value": d["ci_vs_xslg"][1]})
        rows.append({"metric": f"{cand}_verdict", "value": d["verdict"]})
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
