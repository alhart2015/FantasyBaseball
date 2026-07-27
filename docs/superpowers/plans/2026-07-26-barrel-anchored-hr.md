# Barrel-anchored HR Level Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-26-barrel-anchored-hr-design.md`

**Goal:** For HR in the breakout diagnostic, regress surface HR toward a calibrated barrel-expected HR anchor (weighted average), gated on beating the shipped surface->Marcel HR line on held-out years; wire into `breakout.py` only if the gate clears.

**Architecture:** Phase 1 = a committed gate backtest (`scripts/backtest_hr_level.py` reusing `hr_confirm.py`), which decides go/no-go and prints the production constants. Phase 2 (conditional) = freeze those constants in `breakout.py` and swap HR's regression anchor from the Marcel prior to `barrel_expected` (weight from `w_for_stat`, anchor from `adjust_line`), with a Marcel fallback when barrels are absent.

**Tech Stack:** Python 3.12, pandas, stdlib. Reuses `hr_confirm.py` (`build_hr_records`, `fit_barrel_calibration`, `expected_hr_rate`, `_forwards`, `SHIPPED_XSLG_SCALE`), `breakout_backtest.py` (`_spearman`, `_bootstrap_diff`, `rate_mae`), and `backtest_breakout.build_corpus`.

## Global Constraints

- **ASCII-only** in source, logs, reports. Entry-point scripts may `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`.
- **No `x or default` for numeric defaults.** The barrel fallback keys on `brl_pa is not None` (a real `0.0` barrel rate is valid, not missing).
- **No circular imports:** `barrel_expected_rate` lives in `breakout.py`; `hr_confirm.py` imports it from `breakout` (never the reverse).
- **Isolation:** only HR changes. Non-HR stats and non-barrel (`brl_pa is None`) players must be byte-for-byte unaffected -- guarded by a characterization test.
- **Determinism:** all bootstrap CIs pass an explicit `seed=hr_confirm.SEED`.
- **Gate before wire-in:** Phase 2 runs ONLY if Task 2's gate CI excludes 0. If it does not, stop after Phase 1 and report the negative result.
- **End-of-effort checks:** `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` (the `analysis/` package is covered -- `breakout.py`/`hr_confirm.py` must pass). Show outputs. Pre-existing `resend` ModuleNotFoundError in `summary/send` is unrelated and may remain.

---

## Phase 1 -- the gate backtest (always runs)

### Task 1: Shared barrel-level functions

**Files:**
- Modify: `src/fantasy_baseball/analysis/breakout.py` (add `barrel_expected_rate` near the other pure helpers, ~after `_confirm_gap`)
- Modify: `src/fantasy_baseball/analysis/hr_confirm.py` (add `level_blend_forward`, `tune_level_weight`, `LEVEL_WEIGHT_GRID`; refactor `expected_hr_rate` to use `barrel_expected_rate`)
- Test: `tests/test_analysis/test_hr_confirm.py` (extend); `tests/test_analysis/test_breakout.py` (extend)

**Interfaces:**
- Produces:
  - `breakout.barrel_expected_rate(brl_pa: float, slope: float, intercept: float) -> float` = `max(0.0, intercept + slope*brl_pa)`.
  - `hr_confirm.level_blend_forward(rec: HrRecord, calib: BarrelCalib, w_s: float) -> float` = `barrel_expected + w_s*(surface_hr - barrel_expected)`.
  - `hr_confirm.tune_level_weight(fit_records, calib) -> float` (grid over `LEVEL_WEIGHT_GRID`, max fit-year Spearman).
  - `hr_confirm.LEVEL_WEIGHT_GRID = [0.00, 0.05, ..., 1.00]` (weight on SURFACE, i.e. `w_s`; `w_b = 1 - w_s`).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_analysis/test_breakout.py`:

```python
def test_barrel_expected_rate_is_line_clamped_at_zero():
    # intercept + slope*brl_pa, clamped >= 0
    assert breakout.barrel_expected_rate(0.08, 0.5, 0.01) == 0.05
    assert breakout.barrel_expected_rate(0.0, 0.5, 0.01) == 0.01
    assert breakout.barrel_expected_rate(0.0, 1.0, -0.5) == 0.0  # clamp
```

Add to `tests/test_analysis/test_hr_confirm.py`:

```python
def test_level_blend_forward_is_weighted_average():
    calib = (0.5, 0.01)  # slope, intercept -> barrel_expected(0.08) = 0.05
    rec = _rec(brl_pa=0.08, surface_hr=0.07)
    f = hr_confirm.level_blend_forward(rec, calib, w_s=0.25)
    # barrel + w_s*(surface - barrel) == w_b*barrel + (1-w_b)*surface
    assert math.isclose(f, 0.05 + 0.25 * (0.07 - 0.05))
    assert math.isclose(f, 0.75 * 0.05 + 0.25 * 0.07)


def test_tune_level_weight_picks_grid_argmax():
    recs = [_rec(mlbam=i, brl_pa=0.02 + 0.002 * i,
                 surface_hr=0.03 + 0.001 * i, actual_hr=0.03 + 0.001 * i) for i in range(30)]
    w = hr_confirm.tune_level_weight(recs, (0.5, 0.01))
    assert w in hr_confirm.LEVEL_WEIGHT_GRID
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_breakout.py::test_barrel_expected_rate_is_line_clamped_at_zero tests/test_analysis/test_hr_confirm.py::test_level_blend_forward_is_weighted_average tests/test_analysis/test_hr_confirm.py::test_tune_level_weight_picks_grid_argmax -v`
Expected: FAIL -- `AttributeError: module ... has no attribute 'barrel_expected_rate'` / `level_blend_forward`.

- [ ] **Step 3: Add `barrel_expected_rate` to `breakout.py`**

Insert after `_confirm_gap` (~line 145):

```python
def barrel_expected_rate(brl_pa: float, slope: float, intercept: float) -> float:
    """Barrel-implied expected HR/PA: the calibrated line HR/PA ~ brl_pa, clamped to
    >= 0 (a degenerate/extreme brl_pa must not yield a negative expected HR rate).
    Shared by the gate backtest (fitted calib) and the live diagnostic (frozen
    constants) so both agree."""
    return max(0.0, intercept + slope * brl_pa)
```

- [ ] **Step 4: Add level-blend helpers to `hr_confirm.py`, refactor `expected_hr_rate`**

In `hr_confirm.py`, import the shared helper and add the grid + functions:

```python
from fantasy_baseball.analysis.breakout import (
    _confirm_gap,
    _reliability,
    barrel_expected_rate,   # add to the existing breakout import block
    line_rates,
)
```

Add near the other constants:

```python
LEVEL_WEIGHT_GRID = [0.05 * i for i in range(21)]  # w_s (weight on surface), 0.00 .. 1.00
```

Refactor `expected_hr_rate`'s barrel branch to use the shared clamped helper:

```python
    if candidate == "barrel":
        slope, intercept = calib
        return barrel_expected_rate(rec["brl_pa"], slope, intercept)
```

Add:

```python
def level_blend_forward(rec: HrRecord, calib: BarrelCalib, w_s: float) -> float:
    """Barrel-anchored HR forward: barrel_expected + w_s*(surface - barrel_expected),
    i.e. w_b*barrel_expected + (1-w_b)*surface with w_b = 1 - w_s."""
    barrel = expected_hr_rate("barrel", rec, calib)
    return barrel + w_s * (rec["surface_hr"] - barrel)


def tune_level_weight(fit_records: list[HrRecord], calib: BarrelCalib) -> float:
    """Grid-search the surface weight w_s on FIT records, max fit-year Spearman."""
    actual = [r["actual_hr"] for r in fit_records]
    best_w, best_rho = LEVEL_WEIGHT_GRID[0], -2.0
    for w_s in LEVEL_WEIGHT_GRID:
        rho = _spearman([level_blend_forward(r, calib, w_s) for r in fit_records], actual)
        if rho > best_rho:
            best_w, best_rho = w_s, rho
    return best_w
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_breakout.py tests/test_analysis/test_hr_confirm.py -q`
Expected: PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout.py src/fantasy_baseball/analysis/hr_confirm.py tests/test_analysis/test_breakout.py tests/test_analysis/test_hr_confirm.py
git commit -m "feat(breakout): shared barrel_expected_rate + level-blend helpers (#262)"
```

---

### Task 2: The gate backtest script + run

**Files:**
- Create: `scripts/backtest_hr_level.py`
- Test: `tests/test_scripts/test_backtest_hr_level.py`

**Interfaces:**
- Consumes: `hr_confirm` (`build_hr_records`, `fit_barrel_calibration`, `expected_hr_rate`, `level_blend_forward`, `tune_level_weight`, `_forwards`, `SHIPPED_XSLG_SCALE`, `SEED`, `CANDIDATES`), `breakout_backtest` (`_spearman`, `_bootstrap_diff`, `rate_mae`), `backtest_breakout.build_corpus`.
- Produces: a `run_level_gate(corpus, *, fit_years, report_years) -> dict` in the script (importable by the test) + a `main()` that prints the direct-level table, the gate CI, top-N/unfiltered robustness, the all-seasons production constants, and a clear `GATE CLEARS`/`GATE DOES NOT CLEAR`; writes `data/stats/hr_level_backtest_results.csv`.

- [ ] **Step 1: Write a failing integration test on a synthetic corpus**

Create `tests/test_scripts/test_backtest_hr_level.py`. Reuse the mover-by-construction synthetic corpus shape from `test_backtest_hr_confirm.py`:

```python
import importlib.util
from pathlib import Path

from fantasy_baseball.analysis import breakout

_SPEC = importlib.util.spec_from_file_location(
    "backtest_hr_level",
    Path(__file__).resolve().parents[2] / "scripts" / "backtest_hr_level.py",
)
bhl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bhl)


def _row(mlbam, pa, slg, xslg, brl_pa, xhr):
    return breakout.SkillLuckRow(
        mlbam=mlbam, player_type="hitter", pa=pa, ip=0.0, age=27.0,
        barrel_pct=None, xslg=xslg, slg=slg, xba=None, ba=None, babip=None,
        xwoba=None, woba=None, k_pct=None, bb_pct=None, brl_pa=brl_pa, xhr=xhr,
    )


def _entry(hr, next_hr, brl_pa, mlbam):
    surface = {"pa": 600.0, "hr": hr, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}
    actual_next = breakout.line_rates(
        {"pa": 600.0, "hr": next_hr, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}, "hitter")
    prior = {"pa": 600.0, "hr": 8, "r": 60, "rbi": 60, "sb": 5, "avg": 0.250}
    hist = [(2018, breakout.line_rates(prior, "hitter")),
            (2019, breakout.line_rates(prior, "hitter"))]
    return (surface, _row(mlbam, 600.0, 0.520, 0.470, brl_pa, hr - 4), actual_next, hist, None)


def test_run_level_gate_wellformed():
    def year():
        return {1000 + i: _entry(26 + i, 26 + i - 4, 0.04 + 0.004 * i, 1000 + i)
                for i in range(12)}
    corpus = {2020: year(), 2021: year()}
    res = bhl.run_level_gate(corpus, fit_years=[2020], report_years=[2021])
    assert res["n_report"] == 12
    assert "gate_ci" in res and len(res["gate_ci"]) == 2
    assert res["gate_clears"] in (True, False)
    assert res["weight_form"] in ("flat", "rel")
    assert set(res["direct_level_spearman"]) >= {"surface", "barrel", "xhr"}
    assert 0.0 <= res["prod_constants"]["w_s"] <= 1.0
    assert 0.0 <= res["prod_constants"]["cw"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_backtest_hr_level.py -q`
Expected: FAIL -- file/attribute not found.

- [ ] **Step 3: Write `scripts/backtest_hr_level.py`**

```python
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


REL_CW_GRID = [0.1 * i for i in range(1, 11)]  # reliability-scaled cw, 0.1 .. 1.0


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
```

- [ ] **Step 4: Run the integration test**

Run: `pytest tests/test_scripts/test_backtest_hr_level.py -q`
Expected: PASS.

- [ ] **Step 5: Run the real gate**

Run: `python scripts/backtest_hr_level.py`
Expected: prints the gate verdict, robustness, and production constants; writes the CSV. Uses the already-cached `sc_hr_h_*` / barrel / MLB data (no new fetch).

- [ ] **Step 6: Commit**

```bash
git add scripts/backtest_hr_level.py tests/test_scripts/test_backtest_hr_level.py data/stats/hr_level_backtest_results.csv
git commit -m "feat(breakout): barrel-anchored HR gate backtest + result (#262)"
```

- [ ] **Step 7: DECISION GATE**

Read the printed `GATE CLEARS` / `GATE DOES NOT CLEAR` and the `chosen form`
(`flat` or `rel`).
- **CLEARS** -> proceed to Phase 2 with the printed `PRODUCTION CONSTANTS` and the
  chosen `form` (flat -> `HR_BARREL_WEIGHT = w_s` used directly; rel ->
  `HR_BARREL_WEIGHT = cw` used as `reliability*cw`).
- **DOES NOT CLEAR** -> STOP. Do not wire in. The deliverable is the committed
  direct-level + gate backtest and the negative result; report it (numbers + why) and
  run the Phase-1 end-of-effort checks (Task 5, minus the wire-in tests).

---

## Phase 2 -- freeze + wire-in (ONLY if the gate clears)

### Task 3: Freeze constants + wire barrel-anchored HR into `breakout.py`

**Files:**
- Modify: `src/fantasy_baseball/analysis/breakout.py` (constants + `w_for_stat` HR branch + `adjust_line` anchor override)
- Test: `tests/test_analysis/test_breakout.py`

**Interfaces:**
- Consumes: `barrel_expected_rate` (Task 1), the `PRODUCTION CONSTANTS` from Task 2's run.
- Produces: `HR_BARREL_SLOPE`, `HR_BARREL_INTERCEPT`, `HR_BARREL_WEIGHT` (the surface weight `w_s`) module constants; barrel-anchored HR in `adjust_line`/`w_for_stat` with Marcel fallback.

- [ ] **Step 1: Write failing tests (barrel-backed holds, thin regresses, fallback unchanged)**

Add to `tests/test_analysis/test_breakout.py`:

```python
def test_hr_barrel_backed_surge_holds_more_than_thin_surge():
    # Same big HR surface; backed (high brl_pa -> high barrel_expected) should keep
    # more of the surge than thin (low brl_pa -> low barrel_expected).
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}
    surf = {"pa": 600, "ab": 540, "hr": 40, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}
    backed = _row(brl_pa=0.14, xslg=0.560, slg=0.560)
    thin = _row(brl_pa=0.04, xslg=0.560, slg=0.560)
    hr_backed = breakout.adjust_line(surf, proj, backed, "hitter").adjusted_line["hr"]
    hr_thin = breakout.adjust_line(surf, proj, thin, "hitter").adjusted_line["hr"]
    assert hr_backed > hr_thin  # barrels back the surge -> higher adjusted HR


def test_hr_fallback_unchanged_when_no_barrels():
    # brl_pa is None -> exact pre-change surface->Marcel behavior (characterization).
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}
    surf = {"pa": 600, "ab": 540, "hr": 40, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}
    row = _row(brl_pa=None, xslg=0.470, slg=0.560)  # no barrels
    res = breakout.adjust_line(surf, proj, row, "hitter")
    # w_for_stat falls back to the xslg confirm blend; recompute it independently
    w = breakout.w_for_stat("hr", row, "hitter", breakout.DEFAULT_WMAP)
    p_hr, s_hr = 20 / 600, 40 / 600
    assert abs(res.adjusted_line["hr"] - (p_hr + w * (s_hr - p_hr)) * 600) < 1e-9
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_analysis/test_breakout.py -k "hr_barrel_backed or hr_fallback" -v`
Expected: FAIL (`hr_backed == hr_thin` today, since HR ignores barrels).

- [ ] **Step 3: Add the frozen constants** (values from Task 2's `PRODUCTION CONSTANTS` -- substitute the actual printed numbers):

Insert near `DEFAULT_WMAP` in `breakout.py`:

```python
# Barrel-anchored HR level (issue #262). For HR, the regression target is a
# barrel-expected HR/PA instead of the Marcel prior, and the weight is the frozen
# surface weight below. Fit by scripts/backtest_hr_level.py on 2016-2024 (gate
# validated on held-out 2021-2024); refresh if Statcast redefines barrels.
HR_BARREL_SLOPE = <PROD_SLOPE>       # B in barrel_expected = A + B*brl_pa
HR_BARREL_INTERCEPT = <PROD_INTERCEPT>  # A
HR_BARREL_WEIGHT = <PROD_W_S>        # w_s: weight on surface (w_b = 1 - w_s on barrels)
```

- [ ] **Step 4: Weight in `w_for_stat` (early return when barrels present)**

In `w_for_stat`, right after `reliability = _reliability(sample, stabilize)`:

```python
    if stat == "hr" and player_type == "hitter" and row.brl_pa is not None:
        return HR_BARREL_WEIGHT  # barrel-anchored HR: fixed surface weight w_s
```

(If Task 2 selected the reliability-scaled variant instead of the flat one, return
`reliability * HR_BARREL_WEIGHT` here and name the constant `HR_BARREL_CW`; the flat
form is the primary per the spec.)

- [ ] **Step 5: Anchor override in `adjust_line`**

In `adjust_line`, immediately after `p_rates = line_rates(projection_line, player_type)`:

```python
    if player_type == "hitter" and row.brl_pa is not None:
        # HR regresses toward the barrel-expected level, not the Marcel prior, so
        # adj_hr, the believed/surface deviation, and the label's HR term all use it.
        p_rates = {
            **p_rates,
            "hr": barrel_expected_rate(row.brl_pa, HR_BARREL_SLOPE, HR_BARREL_INTERCEPT),
        }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_breakout.py -q`
Expected: PASS (new + all existing breakout tests).

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout.py tests/test_analysis/test_breakout.py
git commit -m "feat(breakout): wire barrel-anchored HR level into adjust_line (#262)"
```

---

### Task 4: Report barrel_expected in the breakout report

**Files:**
- Modify: `scripts/run_breakout_report.py` (surface `barrel_expected` for HR)

- [ ] **Step 1: Read the report script's HR row rendering (authoritative)**

Run: `grep -n "barrel_pct\|hr\|xslg\|w_by_stat\|SkillLuckRow\|print(" scripts/run_breakout_report.py`
and read the surrounding function. Identify the EXACT line that renders per-player HR
context (surface HR / xSLG). This line's format string is the idiom Step 2 extends --
do not invent a new column layout.

- [ ] **Step 2: Add `barrel_expected` next to surface HR**

On the exact render line from Step 1, for a player with `row.brl_pa is not None`, add an
ASCII field `barrelHR={val:.1f}` where
`val = breakout.barrel_expected_rate(row.brl_pa, breakout.HR_BARREL_SLOPE, breakout.HR_BARREL_INTERCEPT) * pa`
(HR count; use the same PA the row uses for other counting stats), matched to that
line's existing spacing/format. If the row has no `brl_pa`, render nothing extra
(the fallback path is unchanged).

- [ ] **Step 3: Smoke-run the report**

Run: `python scripts/run_breakout_report.py 2>&1 | head -30`
Expected: runs; HR rows show `barrel_expected` for barrel-covered players.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_breakout_report.py
git commit -m "feat(breakout): show barrel_expected HR in the breakout report (#262)"
```

---

### Task 5: End-of-effort checks + report

- [ ] **Step 1: Full checks** (run all; fix every failure in touched files; show output):

```bash
pytest -n auto -q
ruff check src/fantasy_baseball/analysis/breakout.py src/fantasy_baseball/analysis/hr_confirm.py scripts/backtest_hr_level.py scripts/run_breakout_report.py tests/test_analysis/test_breakout.py tests/test_analysis/test_hr_confirm.py tests/test_scripts/test_backtest_hr_level.py
ruff format --check <same files>
vulture src/fantasy_baseball/analysis/hr_confirm.py scripts/backtest_hr_level.py
mypy src/fantasy_baseball/analysis/breakout.py src/fantasy_baseball/analysis/hr_confirm.py
```

The `resend` ModuleNotFoundError in `tests/test_summary` / `tests/test_scripts/test_send_daily_summary.py` is pre-existing (fails identically on main) and unrelated.

- [ ] **Step 2: Report the outcome** -- the gate verdict, the tuned weight and Spearman/MAE deltas, the direct-level table with CIs, the frozen constants, and (if wired) the adjust_line behavior change. State plainly whether barrels shipped.

## Self-Review

**Spec coverage:**
- Weighted-average estimate + calibration + >=0 clamp -> Task 1 (`barrel_expected_rate`, `level_blend_forward`). [covered]
- Weight grids (flat primary + reliability-scaled variant, report comparison, ship
  winner default-flat) -> Task 1 `LEVEL_WEIGHT_GRID`; Task 2 `run_level_gate` computes
  BOTH forms, their CIs, and `weight_form`; Task 3 Step 4 wires the chosen form. [covered]
- Fallback on `brl_pa is None` -> Task 3 Steps 4-5 (both key on `is not None`); characterization test Task 3 Step 1. [covered]
- Remove xSLG-HR confirm for barrel players -> Task 3 Step 4 early return (fallback keeps it). [covered]
- Gate backtest (direct-level + level-blend vs shipped, seeded CI, top-N + unfiltered, method-vs-production-fit) -> Task 2. [covered]
- Gate rule (CI excludes 0) + conditional Phase 2 -> Task 2 Step 7 decision gate. [covered]
- Constants refit on all seasons -> Task 2 `run_level_gate` prod_constants; frozen in Task 3 Step 3. [covered]
- adjust_line anchor-only override; weight solely in w_for_stat -> Task 3 Steps 4-5. [covered]
- Report barrel_expected -> Task 4. [covered]
- Testing: weighted-avg identity, tuning argmax, gate rule (`gate_clears`), determinism (seeded CIs), fallback characterization, existing suites -> Tasks 1-3 + Task 5. [covered]

**Placeholder scan:** The only intentional placeholders are `<PROD_SLOPE>`/`<PROD_INTERCEPT>`/`<PROD_W_S>` in Task 3 Step 3 -- these are filled from Task 2's runtime output (a real, named dependency), not vague TODOs. Everything else is concrete.

**Type/name consistency:** `barrel_expected_rate(brl_pa, slope, intercept)`, `level_blend_forward(rec, calib, w_s)`, `tune_level_weight`, `LEVEL_WEIGHT_GRID`, `HR_BARREL_SLOPE/INTERCEPT/WEIGHT`, and `run_level_gate(...)['gate_clears'|'prod_constants']` are used identically across Tasks 1-3 and the tests. The gate keys on `w_s` (weight on surface) throughout; `w_b = 1 - w_s` is only display.
