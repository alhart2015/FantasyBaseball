# Games-based MC -- Phase 6 (SD-calibration validation gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Validate that the ROS-direct engine's SAMPLED per-team category SDs reproduce the calibrated analytic SDs (`project_team_sds`), which `scripts/backtest_sd_calibration.py` already validated against realized 2022-2025 variance. This is the GATE on (a) the `pt_mean_fraction` horizon split -- did keeping `fraction_remaining` for the SD/dispersion term preserve the remaining-season variance, or flatten it? -- and (b) the `f^2` bench-fill variance approximation. If the MC SDs match the analytic SDs (ratio ~ 1), the engine inherits ERoto's realized-calibration and ships. If they are systematically too tight/wide, the `_ROS_DIRECT_HITTERS`/`_ROS_DIRECT_PITCHERS` flags are the one-line fallback.

**Why this comparison is apples-to-apples (verified):** `project_team_sds` -> `player_category_variance` reads `mu` via `_stat(player, stat_key)`, whose default source is `"rest_of_season"` (scoring.py:82) -- so the analytic SD is the ROS-horizon variance `sum_i [negbin_perf_variance(stat, mu_ROS_i) + mu_ROS_i^2 * cv_pt_i^2]`. The MC samples ROS-direct, so `np.std(batch[t][c])` (batch = YTD + ROS_sampled; the constant YTD does not affect SD) is also the ROS-horizon SD. Both use the SAME `cv_pt` curve lookup at FULL-SEASON volume (`_full_season_volume` in the analytic; `_projected_volume` in `_apply_variance_batch`) by design (scoring.py:1201 "cv_pt band matches the calibration and the MC"). `project_team_sds(displacement=True)` applies the SAME displacement the new engine does, on the active+IL set, EXCLUDING bench -- so it matches the new engine's active set. The new engine ADDS bench-fill (with its `f^2` variance) that the analytic does NOT model, so any MC_SD EXCESS over the analytic is exactly the fill-variance contribution this gate measures.

**Architecture:** Extend the existing gated selection-attribution diagnostic (`mc_selection.py`, run via `FB_SELECTION_ATTRIBUTION` in `_run_ros_monte_carlo`) to ALSO emit a per-category SD-calibration table for the `new_engine` arm: `mc_sd = std(batch[t][c])` vs `analytic_sd = team_sds[t][c]`, and the ratio. Then run ONE auth'd local refresh and adjudicate the gate. No production-path code changes; this is diagnostic + analysis. The means were already validated by the Phase 4/5 before/after evidence; Phase 6 is the SD half.

**Tech Stack:** Python, numpy. Touched: `src/fantasy_baseball/mc_selection.py`; test `tests/test_mc_selection.py`.

## Global Constraints

- ASCII-only; numeric defaults via `is not None`; imports at top.
- Spec: `docs/superpowers/specs/2026-06-26-games-based-availability-mc-design.md` Component 6 + the "Variance note (acknowledged)" (the `f^2` approximation) + the `fraction_remaining` horizon-split GATE language (Component 4).
- `mc_selection.py` is under `[tool.mypy].files` (added in Phase 4c); run mypy on it.
- Do NOT change the means arms or the production MC. Diagnostic-only.
- Calibration threshold (matches the existing backtest, scripts/backtest_sd_calibration.py:155): ratio in [0.8, 1.25] = calibrated; > 1.25 = too wide; < 0.8 = too tight.

---

### Task 1: SD-calibration computation + table in the diagnostic

**Files:** Modify `src/fantasy_baseball/mc_selection.py`; test `tests/test_mc_selection.py`.

**Interfaces:**
- Consumes: the `new_engine` arm's per-team batch distributions (already computed inside `run_selection_attribution` -- the `batch[t][c]` arrays it currently medians at mc_selection.py:177); the analytic `team_sds: Mapping[str, Mapping[Category, float]]` (already a param of `run_selection_attribution` since Phase 4c).
- Produces: `compute_sd_calibration(new_engine_batch: dict[str, dict[str, np.ndarray]], team_sds) -> dict[str, dict[str, tuple[float, float, float]]]` returning per-team per-category `(mc_sd, analytic_sd, ratio)`; `format_sd_calibration_table(calib) -> str`. `run_selection_attribution` returns the new_engine batch (or the calibration) alongside its existing medians so the hook can render the table.

- [ ] **Step 1: Write failing tests** in `tests/test_mc_selection.py`:

```python
def test_sd_calibration_ratio_computed():
    import numpy as np
    from fantasy_baseball.models.category import Category
    from fantasy_baseball.mc_selection import compute_sd_calibration
    # One team, R: a batch with known std vs an analytic sd -> ratio = mc/analytic.
    rng = np.random.default_rng(0)
    batch = {"TeamA": {"R": rng.normal(800, 30, 5000)}}
    team_sds = {"TeamA": {Category.R: 30.0}}
    calib = compute_sd_calibration(batch, team_sds)
    mc_sd, analytic_sd, ratio = calib["TeamA"]["R"]
    assert abs(analytic_sd - 30.0) < 1e-9
    assert abs(mc_sd - 30.0) < 3.0          # sample std ~= 30
    assert abs(ratio - mc_sd / analytic_sd) < 1e-9

def test_sd_calibration_handles_zero_analytic():
    # analytic_sd == 0 -> ratio is NaN (not a div-by-zero crash), skipped in pooling.
    import numpy as np
    from fantasy_baseball.models.category import Category
    from fantasy_baseball.mc_selection import compute_sd_calibration
    calib = compute_sd_calibration({"T": {"SV": np.zeros(10)}}, {"T": {Category.SV: 0.0}})
    assert calib["T"]["SV"][2] != calib["T"]["SV"][2]  # NaN

def test_format_sd_calibration_table_has_ratio_column():
    from fantasy_baseball.mc_selection import format_sd_calibration_table
    calib = {"TeamA": {"R": (31.0, 30.0, 31.0/30.0)}}
    txt = format_sd_calibration_table(calib)
    assert "ratio" in txt.lower() and "TeamA" in txt and "R" in txt
```

- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement** `compute_sd_calibration` + `format_sd_calibration_table` in `mc_selection.py`. The Category-vs-string key mapping must match how `_CATS` / the batch keys are keyed (the batch uses string cat keys like `"R"`; `team_sds` uses `Category` enum keys -- map via `Category(<str>)` or the existing `_CATS` list; READ how mc_selection.py:158/177 keys the batch and how team_sds is keyed before writing, and handle the AVG/ERA/WHIP rate cats which `team_sds` may key differently -- if a cat is absent from team_sds, emit NaN and skip it, do not crash). `compute_sd_calibration` uses `float(np.std(arr))` for mc_sd and guards `analytic_sd > 0` (else ratio = `float("nan")`, per the falsy-zero rule use `is not None`/`> 0`, never `x or default`). `format_sd_calibration_table`: per team, rows = cats, cols = `mc_sd`, `analytic_sd`, `ratio`, plus a POOLED line (median ratio across all team-cats with a finite ratio) and a verdict (`calibrated` if 0.8 <= pooled <= 1.25, else `too tight`/`too wide`).
- [ ] **Step 4: Have `run_selection_attribution` retain the new_engine batch** (it currently discards it after medianing at :177). Return it (or the computed calibration) in the result dict under a `"_new_engine_batch"` / `"_sd_calibration"` key so the hook can render the table. Keep the existing arm medians unchanged.
- [ ] **Step 5: Run, confirm PASS:** `pytest tests/test_mc_selection.py -v`.
- [ ] **Step 6: Wire into the hook** (`web/refresh_pipeline.py` `_run_ros_monte_carlo`, the `FB_SELECTION_ATTRIBUTION` block): after the means table, compute + write the SD-calibration table (it has `self.team_sds` and the attribution result). Append to the same `phase0_attribution.txt`-style output the means table writes to. (READ the existing hook block first; mirror exactly how it writes the means table.)
- [ ] **Step 7: No regression:** `pytest tests/test_mc_selection.py tests/test_mc_integration.py -q`; ruff check + format --check + `mypy src/fantasy_baseball/mc_selection.py src/fantasy_baseball/web/refresh_pipeline.py` (clean; ignore pre-existing category_odds.py).
- [ ] **Step 8: Commit:** `feat(mc): SD-calibration table in selection-attribution diagnostic (Phase 6)`.

---

### Task 2: Run the gate + adjudicate (controller, not a subagent)

This is the controller's analysis step, not an implementer task.

- [ ] Run the auth'd local refresh with `FB_SELECTION_ATTRIBUTION=1` (same invocation as Phases 4c/5), producing the SD-calibration table on the live rosters.
- [ ] Read the POOLED ratio + per-category ratios. ADJUDICATE:
  - **Pooled ratio in [0.8, 1.25] across counting cats** -> GATE PASSES. The horizon split preserved the dispersion and the `f^2` fill variance is bounded. Document and proceed to PR.
  - **Systematically too TIGHT (< 0.8)** -> the horizon split flattened the dispersion (likely `fraction_remaining` not reaching the SD/dispersion term as intended). Investigate `_apply_variance_batch`'s SD path under `pt_mean_fraction`; if unresolved, set `_ROS_DIRECT_HITTERS`/`_ROS_DIRECT_PITCHERS = False` (the documented fallback: keep full-season sampling, source only games/displacement/fill from ROS) and re-validate.
  - **Systematically too WIDE (> 1.25)** -> the bench-fill `f^2` variance (or double-counted PT variance) is inflating; localize to the cats with bench-fill (hitters) vs pitchers (no fill -- pitchers should track the analytic TIGHTLY since they are pure active-set sampling).
- [ ] Write the evidence note `docs/superpowers/games-mc-phase6-sd-evidence-2026-06-26.md` (+ raw table) with the verdict, and update the ledger.

---

## Self-Review

**Spec coverage:** Component 6 (validation backtest, means AND SDs) -- means done in Phase 4/5 evidence; this adds the SD half as the GATE on the `pt_mean_fraction` horizon split and the `f^2` fill-variance approximation (spec's "Variance note (acknowledged)" + Component 4 GATE language). Reuses the existing analytic SD (`project_team_sds`, validated vs realized by `scripts/backtest_sd_calibration.py`) rather than re-deriving realized variance.

**Placeholder scan:** Concrete functions + tests + threshold (0.8-1.25, matching the existing backtest). Task 2 is explicitly a controller adjudication, not placeholder code.

**Type consistency:** `compute_sd_calibration(batch, team_sds) -> per-(team,cat) (mc_sd, analytic_sd, ratio)`; the Category-enum (team_sds) vs string (batch) key mismatch is called out as a must-handle, with NaN for absent/zero-analytic cats (no div-by-zero, `is not None`/`>0` guards).
