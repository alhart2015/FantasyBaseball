"""Gate backtest for barrel-anchored HR (issue #262 follow-on): does regressing
surface HR toward a calibrated barrel-expected anchor beat the shipped surface->Marcel
HR line on held-out years? Hitters, common support 2016-2024 (fit 2016-20 / report
2021-24). Prints the go/no-go and the all-seasons production constants for freezing.

See docs/superpowers/specs/2026-07-26-barrel-anchored-hr-design.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fantasy_baseball.analysis import hr_confirm as H
from fantasy_baseball.analysis.breakout_backtest import _bootstrap_diff, _spearman

# NOTE: `backtest_breakout` lives in scripts/ and is imported lazily inside main()
# only. Importing it at module scope would break the importlib-based integration test
# (scripts/ is on sys.path only when this file is RUN directly, not when imported).

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "stats" / "hr_level_backtest_results.csv"
SOURCE_YEARS = list(range(2016, 2025))
FIT_YEARS = list(range(2016, 2021))
REPORT_YEARS = list(range(2021, 2025))
REL_CW_GRID = [0.1 * i for i in range(1, 11)]  # reliability-scaled cw, 0.1 .. 1.0


def _by_year(corpus, years, top_n=None, hr_move_min=H.HR_MOVE_MIN):
    out = []
    for y in years:
        r = H.build_hr_records(corpus, [y], hr_move_min=hr_move_min)
        r.sort(key=lambda x: x["surface_hr"] * x["pa"], reverse=True)
        out.extend(r[:top_n] if top_n else r)
    return out


def _direct_level(report, calib):
    actual = [r["actual_hr"] for r in report]
    surf = [r["surface_hr"] for r in report]
    xhr = [r["xhr_rate"] for r in report]
    brl = [H.expected_hr_rate("barrel", r, calib) for r in report]
    sp = {"surface": _spearman(surf, actual), "xhr": _spearman(xhr, actual),
          "barrel": _spearman(brl, actual)}
    ci = {"barrel_minus_surface": _bootstrap_diff(brl, surf, actual, seed=H.SEED),
          "xhr_minus_surface": _bootstrap_diff(xhr, surf, actual, seed=H.SEED)}
    return sp, ci


def _rel_forward(rec, calib, cw):
    """Reliability-scaled variant: w_s = reliability*cw, reliability = pa/(pa+120)."""
    barrel = H.expected_hr_rate("barrel", rec, calib)
    rel = H._reliability(rec["pa"], H.HR_STABILIZE)
    return barrel + rel * cw * (rec["surface_hr"] - barrel)


def _tune_rel_cw(fit_records, calib):
    actual = [r["actual_hr"] for r in fit_records]
    best_cw, best_rho = REL_CW_GRID[0], -2.0
    for cw in REL_CW_GRID:
        rho = _spearman([_rel_forward(r, calib, cw) for r in fit_records], actual)
        if rho > best_rho:
            best_cw, best_rho = cw, rho
    return best_cw


def run_level_gate(corpus, *, fit_years, report_years):
    calib = H.fit_barrel_calibration(H.build_hr_records(corpus, fit_years, hr_move_min=0.0))
    fit = _by_year(corpus, fit_years)
    report = _by_year(corpus, report_years)
    actual = [r["actual_hr"] for r in report]
    shipped_fwd = H._forwards(report, "xslg", calib, H.SHIPPED_XSLG_SCALE)

    # flat weight (primary) vs reliability-scaled weight (reported variant)
    w_s = H.tune_level_weight(fit, calib)
    flat_fwd = [H.level_blend_forward(r, calib, w_s) for r in report]
    flat_ci = _bootstrap_diff(flat_fwd, shipped_fwd, actual, seed=H.SEED)
    cw = _tune_rel_cw(fit, calib)
    rel_fwd = [_rel_forward(r, calib, cw) for r in report]
    rel_ci = _bootstrap_diff(rel_fwd, shipped_fwd, actual, seed=H.SEED)

    # default flat; the scaled form is chosen only if its CI lower bound is strictly
    # higher (spec: "default to flat unless the scaled one clearly wins").
    use_rel = rel_ci[0] > flat_ci[0]
    weight_form = "rel" if use_rel else "flat"
    gate_ci = rel_ci if use_rel else flat_ci
    sp_dl, ci_dl = _direct_level(report, calib)

    # production constants: refit calib + tune both weights on ALL source years (no holdout)
    all_years = sorted(set(fit_years) | set(report_years))
    prod_calib = H.fit_barrel_calibration(H.build_hr_records(corpus, all_years, hr_move_min=0.0))
    prod_fit = _by_year(corpus, all_years)
    prod_w = H.tune_level_weight(prod_fit, prod_calib)
    prod_cw = _tune_rel_cw(prod_fit, prod_calib)

    return {
        "n_fit": len(fit), "n_report": len(report),
        "calib": calib, "w_s": w_s, "cw": cw,
        "weight_form": weight_form,
        "barrel_spearman": _spearman(flat_fwd, actual),
        "rel_spearman": _spearman(rel_fwd, actual),
        "shipped_spearman": _spearman(shipped_fwd, actual),
        "barrel_mae": fmean([abs(a - b) for a, b in zip(flat_fwd, actual)]),
        "shipped_mae": fmean([abs(a - b) for a, b in zip(shipped_fwd, actual)]),
        "gate_ci": gate_ci,             # the CHOSEN form's CI vs shipped
        "flat_ci": flat_ci, "rel_ci": rel_ci,
        "gate_clears": gate_ci[0] > 0,
        "direct_level_spearman": sp_dl,
        "direct_level_ci": ci_dl,
        "prod_constants": {"slope": prod_calib[0], "intercept": prod_calib[1],
                           "w_s": prod_w, "cw": prod_cw},
    }


def main():
    import backtest_breakout as bt  # lazy: scripts/ is only on sys.path on direct run

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = bt.build_corpus(bt.SKILL_LUCK_CACHE_DIR, bt.PROJECTIONS_ROOT, SOURCE_YEARS)
    res = run_level_gate(corpus, fit_years=FIT_YEARS, report_years=REPORT_YEARS)
    print("Barrel-anchored HR level gate -- vs the shipped surface->Marcel line")
    print(f"  fit {FIT_YEARS} report {REPORT_YEARS}  n_fit {res['n_fit']} n_report {res['n_report']}")
    print(f"  Spearman: flat(w_s={res['w_s']:.2f}) {res['barrel_spearman']:+.3f}  "
          f"rel(cw={res['cw']:.2f}) {res['rel_spearman']:+.3f}  shipped {res['shipped_spearman']:+.3f}")
    print(f"  MAE: barrel(flat) {res['barrel_mae']:.4f}  shipped {res['shipped_mae']:.4f}")
    print(f"  flat CI vs shipped {tuple(round(x,3) for x in res['flat_ci'])}  "
          f"rel CI vs shipped {tuple(round(x,3) for x in res['rel_ci'])}  "
          f"-> chosen form: {res['weight_form']}")
    lo, hi = res["gate_ci"]
    print(f"  GATE ({res['weight_form']}) CI barrel-anchored - shipped [{lo:+.3f}, {hi:+.3f}] -> "
          f"{'GATE CLEARS' if res['gate_clears'] else 'GATE DOES NOT CLEAR'}")
    print("  direct-level Spearman:", {k: round(v, 3) for k, v in res["direct_level_spearman"].items()})
    print("  direct-level CI:", {k: (round(v[0], 3), round(v[1], 3))
                                  for k, v in res["direct_level_ci"].items()})
    # robustness: top-100/50 + unfiltered, on the chosen weight form
    for label, kw in [("TOP-100", {"top_n": 100}), ("TOP-50", {"top_n": 50}),
                      ("UNFILTERED", {"hr_move_min": 0.0})]:
        rep = _by_year(corpus, REPORT_YEARS, **kw)
        act = [r["actual_hr"] for r in rep]
        if res["weight_form"] == "rel":
            b = [_rel_forward(r, res["calib"], res["cw"]) for r in rep]
        else:
            b = [H.level_blend_forward(r, res["calib"], res["w_s"]) for r in rep]
        s = H._forwards(rep, "xslg", res["calib"], H.SHIPPED_XSLG_SCALE)
        ci = _bootstrap_diff(b, s, act, seed=H.SEED)
        print(f"  robustness {label}: n={len(rep)} barrel {_spearman(b,act):+.3f} "
              f"shipped {_spearman(s,act):+.3f} CI [{ci[0]:+.3f},{ci[1]:+.3f}]")
    pc = res["prod_constants"]
    wtd = pc["cw"] if res["weight_form"] == "rel" else pc["w_s"]
    print(f"  PRODUCTION CONSTANTS (refit all {SOURCE_YEARS[0]}-{SOURCE_YEARS[-1]}, "
          f"form={res['weight_form']}): HR_BARREL_SLOPE={pc['slope']:.5f} "
          f"HR_BARREL_INTERCEPT={pc['intercept']:.5f} HR_BARREL_WEIGHT={wtd:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": k, "value": v} for k, v in {
        "n_report": res["n_report"], "weight_form": res["weight_form"],
        "w_s": res["w_s"], "cw": res["cw"],
        "barrel_spearman": res["barrel_spearman"], "rel_spearman": res["rel_spearman"],
        "shipped_spearman": res["shipped_spearman"],
        "gate_ci_low": res["gate_ci"][0], "gate_ci_high": res["gate_ci"][1],
        "gate_clears": res["gate_clears"],
        "prod_slope": pc["slope"], "prod_intercept": pc["intercept"],
        "prod_w_s": pc["w_s"], "prod_cw": pc["cw"],
    }.items()]
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
