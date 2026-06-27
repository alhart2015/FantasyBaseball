# Games-based MC -- Phase 6 (cv_pt-volume fix + SD-calibration validation gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** (1) FIX a variance bug the Phase-6 plan-review surfaced: the ROS-direct path looks up the playing-time curve at ROS volume, not full-season volume, inflating per-team SDs ~1.5x. (2) Then VALIDATE that the corrected engine's SAMPLED counting-category SDs reproduce the calibrated analytic SDs (`project_team_sds`, validated vs realized 2022-2025 by `scripts/backtest_sd_calibration.py`) -- the GATE on the `pt_mean_fraction` horizon split and the `f^2` bench-fill variance.

**The bug (Task 1).** `_apply_variance_batch` looks up `playing_time_params`/`playing_time_shape` at `vol = _projected_volume(p)` (simulation.py:735), which reads `pa`/`ip` from the player's flat dict. On the ROS-direct path the flat dict is `to_flat_dict()` = `rest_of_season`, so `vol` is ROS PA (~305 mid-season) instead of full-season PA (~620). The playing-time curve is calibrated and INDEXED by full-season volume; at ROS volume a full-timer is misclassified as a part-timer -- `cv_pt` 0.20 -> 0.42 (2.07x), `mean_scale` 0.94 -> 0.75 (measured). The original full-season MC path reads full-season PA (correct); ROS-direct (Phase 4/5) regressed it. Means-only evidence missed it: pitchers use `pt_mean_fraction=0` (eff_mean=1, volume-independent mean) so their means matched ERoto; hitter means were masked by the bench-fill compensating for the over-aggressive 0.75 haircut. SDs were never checked until now.

**Why the gate is apples-to-apples AFTER the fix (the precise rationale).** For a counting cat, both sides reduce to the same closed form:
- Analytic (`player_category_variance`, scoring.py:1264): `negbin_perf_variance(stat, mu_ROS) + mu_ROS^2 * cv_pt(FULL_vol)^2`, then `build_team_sds` multiplies the team SD by `sd_scale = sqrt(fraction_remaining)` (scoring.py:1385; refresh_pipeline.py:901/952). `mu_ROS` because `_stat` defaults to `rest_of_season` (scoring.py:82); `cv_pt(FULL_vol)` via `_full_season_volume` (scoring.py:1256/1206).
- MC (ROS-direct, AFTER Task 1): NegBin mean `mu_ROS` (ROS flat dict), perf term `_negbin_copula_counts` sets `var_target = fraction_remaining * var_full` (simulation.py:559), PT term `eff_sd = cv_pt(FULL_vol) * sqrt(fraction_remaining)` (playing_time.py:104) -- `cv_pt(FULL_vol)` ONLY because Task 1 passes full-season volume.

So `mc_var ~= fraction_remaining * [negbin_perf_variance(stat, mu_ROS) + mu_ROS^2 * cv_pt(FULL_vol)^2]` and `analytic_sd^2 = fraction_remaining * [same]`. The `sqrt(fraction_remaining)` appears on BOTH sides (analytic via `sd_scale`, MC via `var_target` + `eff_sd`) and CANCELS in the ratio. **Sanity check for Task 3: a pooled ratio near `sqrt(fraction_remaining)` (~0.7 mid-season), not near 1.0, means one side dropped the damping -- investigate before adjudicating.** Without Task 1, the MC's `cv_pt(ROS_vol)` is ~2x the analytic's `cv_pt(FULL_vol)` and the gate reads ~1.5x "too wide" -- the bug, not the `f^2` term.

**Scope note (counting cats only).** The SD gate covers the 7 COUNTING cats (R/HR/RBI/SB/W/K/SV), where `np.std(team_total) == np.std(ROS)` because team_total = YTD + ROS adds a CONSTANT YTD. The 3 RATE cats (AVG/ERA/WHIP) are EXCLUDED from the SD gate: their `team_total` is a ratio `(YTD+ROS)/(YTD_vol+ROS_vol)`, so `np.std(EOS_rate)` is structurally TIGHTER than the ROS-rate analytic SD (YTD denominator dilution) -- comparing them is apples-to-oranges and would falsely read "too tight." Rate-cat SD calibration is deferred/noted, not gated here.

**Architecture:** Task 1 fixes the sampler + both ROS-direct helpers (additive `pt_volumes` param; default None = byte-identical legacy). Task 2 extends the gated diagnostic (`mc_selection.py`) to emit a counting-cat SD-calibration table via a SEPARATELY-TYPED return channel. Task 3 (controller) runs ONE auth'd refresh and adjudicates means (re-validate; the fix shrinks the hitter premium) AND SDs.

**Tech Stack:** Python, numpy. Touched: `src/fantasy_baseball/simulation.py`, `src/fantasy_baseball/mc_selection.py`; tests `tests/test_mc_integration.py`, `tests/test_mc_selection.py`.

## Global Constraints

- ASCII-only; numeric defaults via `is not None`/`> 0`, never `x or default`; imports at top.
- Spec: Component 4 (the `fraction_remaining` horizon-split GATE), Component 6 (validation), the "Variance note (acknowledged)" (`f^2`).
- `simulation.py` and `mc_selection.py` are under `[tool.mypy].files`; run mypy.
- Task 1 default path (`pt_volumes=None`) MUST stay byte-identical: the `effective_rosters=None` byte-anchor (`test_whole_context_fallback_to_topk`) and the scalar path must not move.
- SD-ratio tolerance is an ENGINEERING band [0.8, 1.25] (the same numeric band the realized backtest uses at backtest_sd_calibration.py:155). It is NOT inherited realized-calibration for THIS comparison -- it is a near-1 tolerance for `mc_sd/analytic_sd`; the realized calibration is established only transitively (analytic-vs-realized by the existing backtest + MC-vs-analytic by this gate). Do NOT copy the backtest's verdict expression verbatim: its `SD(z)>1.25 -> TOO TIGHT` measures the INVERSE quantity; here `mc/analytic > 1.25 -> MC too WIDE`, `< 0.8 -> MC too TIGHT`.

---

### Task 1: cv_pt-volume fix (full-season volume for the ROS-direct PT curve lookup)

**Files:** Modify `src/fantasy_baseball/simulation.py`; test `tests/test_mc_integration.py`.

**Interfaces:**
- Produces: `_apply_variance_batch(..., *, pt_mean_fraction=None, suppress_repl=False, pt_volumes: np.ndarray | None = None)`. When `pt_volumes` is given, the per-player curve lookup uses `pt_volumes[j]` for BOTH `playing_time_params` and `playing_time_shape`, instead of `_projected_volume(p)`. The two ROS-direct helpers compute and pass full-season volumes.
- A module helper `_full_season_pt_volume(player, is_hitter) -> float`: reads `full_season_projection.pa` (hitters) / `.ip` (pitchers); hitter fallback `ab / AB_PER_PA`; if `full_season_projection` is None or volume <= 0, fall back to the ROS volume (`_projected_volume` on the ROS flat dict) so preseason/missing-FS players keep working (the falsy-zero rule: guard `is not None` and `> 0`).

- [ ] **Step 1: Write failing tests** in `tests/test_mc_integration.py`:

```python
def test_ros_direct_uses_full_season_volume_for_cv_pt():
    # A full-timer (full-season PA 620, ROS PA 305) must be sampled with the
    # FULL-SEASON cv_pt band (~0.20), NOT the ROS-volume band (~0.42). Assert the
    # sampled ROS-R SD is close to the full-vol analytic PT-SD, not ~2x it.
    import numpy as np
    from fantasy_baseball.simulation import _simulate_team_hitters_ros_direct
    eff = _one_full_timer_hitter(full_pa=620.0, ros_pa=305.0, ros_r=45.0)
    out = _simulate_team_hitters_ros_direct(eff, 0.49, np.random.default_rng(0), 8000)
    # Full-vol PT-SD of R ~= 45 * cv_pt(620) * sqrt(0.49); ROS-vol would be ~2x.
    # Assert the realized SD is within ~25% of the full-vol expectation (NOT ~2x).
    from fantasy_baseball.utils.playing_time import playing_time_params
    from fantasy_baseball.models.player import PlayerType
    cv_full = playing_time_params(PlayerType.HITTER, 620.0)[1]
    cv_ros = playing_time_params(PlayerType.HITTER, 305.0)[1]
    pt_sd_full = 45.0 * cv_full * (0.49 ** 0.5)
    sd = out["R"].std()
    # Must be far below the ROS-vol band; near (perf+PT) full-vol scale.
    assert sd < 45.0 * cv_ros * (0.49 ** 0.5) * 0.9   # well under the 2x-wide ROS band
```

```python
def test_apply_variance_batch_pt_volumes_default_is_byte_identical():
    # pt_volumes=None reproduces the legacy per-player vol == _projected_volume.
    # Seed-pinned: counts identical with pt_volumes=None vs omitted.
    ...  # mirror the existing 4a byte-equality test pattern
```

- [ ] **Step 2: Run, confirm FAIL** (the helper still uses ROS volume).
- [ ] **Step 3: Implement.** Add `pt_volumes` kwarg to `_apply_variance_batch`; in the per-player loop (simulation.py:734-742) use `vol = float(pt_volumes[j]) if pt_volumes is not None else _projected_volume(p, is_hitter)` for BOTH `playing_time_params(player_type, vol)` and `playing_time_shape(player_type, vol)`. Add `_full_season_pt_volume`. In `_simulate_team_hitters_ros_direct` and `_simulate_team_pitchers_ros_direct`, build `pt_volumes = np.array([_full_season_pt_volume(b.player, is_hitter) for b in <active bodies>])` and pass it to the `_apply_variance_batch` call. (No change to the `effective_rosters=None` flat-dict path -> byte-anchor safe.)
- [ ] **Step 4: Run, confirm PASS.**
- [ ] **Step 5: No regression:** `pytest tests/test_mc_integration.py tests/test_simulation.py tests/test_mc_selection.py tests/test_web/ -q`. The byte-anchor + scalar path MUST pass. The Phase-4/5 mechanism tests are magnitude-tolerant so should pass; if `test_pitcher_mean_matches_projection_no_haircut` shifts, note that pitcher MEAN is volume-INDEPENDENT (pt_mean_fraction=0) so it must NOT change -- if it does, the fix wrongly touched the mean path.
- [ ] **Step 6: ruff + format --check + `mypy src/fantasy_baseball/simulation.py`** (clean; ignore pre-existing category_odds.py).
- [ ] **Step 7: Commit:** `fix(sim): ROS-direct PT curve lookup at full-season volume, not ROS (Phase 6 Task 1)`.

---

### Task 2: counting-cat SD-calibration table in the diagnostic

**Files:** Modify `src/fantasy_baseball/mc_selection.py`; test `tests/test_mc_selection.py`.

**Interfaces:**
- Produces: `compute_sd_calibration(new_engine_batch: dict[str, dict[str, np.ndarray]], team_sds: Mapping[str, Mapping[Category, float]]) -> dict[str, dict[str, tuple[float, float, float]]]` -- per team, per COUNTING cat, `(mc_sd, analytic_sd, ratio)`. `format_sd_calibration_table(calib) -> str`. To avoid breaking the arm-keyed return dict and its mypy type, `run_selection_attribution` returns a 2-tuple `(arm_medians, sd_calibration | None)` (sd_calibration computed INSIDE the function where the raw new_engine `batch` is live, BEFORE it is discarded at mc_selection.py:177) -- NOT a sentinel key inside the arm dict. Update the one call site (the hook) to unpack the tuple.

- [ ] **Step 1: Write failing tests** in `tests/test_mc_selection.py`:

```python
def test_sd_calibration_counting_cats_only_with_enum_keys():
    import numpy as np
    from fantasy_baseball.models.category import Category
    from fantasy_baseball.mc_selection import compute_sd_calibration
    rng = np.random.default_rng(0)
    batch = {"T": {"R": rng.normal(800, 30, 5000), "AVG": rng.normal(0.27, 0.01, 5000)}}
    team_sds = {"T": {Category.R: 30.0}}     # rate cats absent / keyed differently
    calib = compute_sd_calibration(batch, team_sds)
    assert "R" in calib["T"] and "AVG" not in calib["T"]   # counting only; rate excluded
    mc_sd, analytic_sd, ratio = calib["T"]["R"]
    assert abs(analytic_sd - 30.0) < 1e-9 and abs(ratio - mc_sd / 30.0) < 1e-9

def test_sd_calibration_string_to_enum_roundtrip():
    # The batch keys are bare strings ("R"); team_sds keys are Category enums.
    # Pin that Category("R") round-trips to the same enum used in team_sds.
    from fantasy_baseball.models.category import Category
    assert Category("R") == Category.R   # if this is false, the join silently NaNs

def test_sd_calibration_zero_or_missing_analytic_is_nan():
    import numpy as np
    from fantasy_baseball.models.category import Category
    from fantasy_baseball.mc_selection import compute_sd_calibration
    calib = compute_sd_calibration({"T": {"SV": np.ones(10)}}, {"T": {Category.SV: 0.0}})
    assert calib["T"]["SV"][2] != calib["T"]["SV"][2]   # ratio NaN, no div-by-zero

def test_format_sd_calibration_table_has_ratio_and_pooled():
    from fantasy_baseball.mc_selection import format_sd_calibration_table
    txt = format_sd_calibration_table({"T": {"R": (31.0, 30.0, 31.0 / 30.0)}})
    assert "ratio" in txt.lower() and "POOLED" in txt and "T" in txt
```

- [ ] **Step 2: Run, confirm FAIL.**
- [ ] **Step 3: Implement.** `compute_sd_calibration`: iterate the 7 COUNTING cats only (define `_COUNTING_CATS = ["R","HR","RBI","SB","W","K","SV"]`; READ how `_CATS` and the batch are keyed at mc_selection.py:23/177 first). For each team-cat: `mc_sd = float(np.std(batch[t][cat_str]))`; `analytic_sd = team_sds[t].get(Category(cat_str))`; if `analytic_sd is None or analytic_sd <= 0`: `ratio = float("nan")` else `ratio = mc_sd / analytic_sd`. Skip a cat absent from the batch. `format_sd_calibration_table`: per-team rows (cat, mc_sd, analytic_sd, ratio), a POOLED line (median of finite ratios across all team-cats), and a verdict (`calibrated` if 0.8 <= pooled <= 1.25 else `MC too tight`/`MC too wide`). Then make `run_selection_attribution` compute the calibration inside itself (it already has `team_sds`) and return `(out, calib)`; update the hook call site to unpack.
- [ ] **Step 4: Run, confirm PASS:** `pytest tests/test_mc_selection.py -v`.
- [ ] **Step 5: Wire into the hook** (`web/refresh_pipeline.py` `_run_ros_monte_carlo`, the `FB_SELECTION_ATTRIBUTION` block): unpack the new 2-tuple from `run_selection_attribution`; after the means table, append the SD-calibration table to the same output file. (READ the existing hook block; mirror how it writes + the tuple unpack at the call site.)
- [ ] **Step 6: No regression:** `pytest tests/test_mc_selection.py tests/test_mc_integration.py -q`; ruff + format --check + `mypy src/fantasy_baseball/mc_selection.py src/fantasy_baseball/web/refresh_pipeline.py`.
- [ ] **Step 7: Commit:** `feat(mc): counting-cat SD-calibration table in selection diagnostic (Phase 6 Task 2)`.

---

### Task 3: Run the gate + adjudicate (controller, not a subagent)

- [ ] Run the auth'd local refresh with `FB_SELECTION_ATTRIBUTION=1` (as Phases 4c/5), producing BOTH the means table (re-validate -- the Task-1 fix raises the hitter active mean from 0.75x to 0.94x, so the bench-fill premium SHRINKS; confirm new_engine hitter still ~= ERoto + a SMALLER premium, and pitchers UNCHANGED ~= ERoto) AND the SD-calibration table.
- [ ] Read the POOLED counting-cat ratio + per-cat ratios. First the sanity check: if the pooled ratio is near `sqrt(fraction_remaining)` (~0.7) rather than ~1.0, a `sqrt(fraction_remaining)` factor was dropped on one side -- fix that before adjudicating. Then:
  - **Pooled in [0.8, 1.25] (esp. pitchers, the no-fill control, tight)** -> GATE PASSES. The horizon split preserved dispersion and the `f^2` fill term is BOUNDED below the band's resolution (small bench portion -- this BOUNDS, does not prove, the `f^2` approximation; document as such). Proceed to PR.
  - **MC too WIDE (> 1.25)** -> localize: if HITTERS-only wide but PITCHERS calibrated, it's the `f^2` bench-fill term (hitters have fill, pitchers don't); if BOTH wide, re-check the Task-1 volume fix landed (cv_pt band).
  - **MC too TIGHT (< 0.8)** -> the variance horizon was flattened. This is NOT cleanly fixable by the `_ROS_DIRECT_*` flag (flipping to full-season top-k changes the clamp/churn and may not widen the SD); first investigate the `fraction_remaining` path in `_apply_variance_batch` (`var_target`, `eff_sd`). The flag is the LAST-RESORT escape (keep full-season sampling) only if ROS-direct cannot be calibrated, not a precision SD lever.
- [ ] Caveat to note: `project_team_sds(displacement=True)` reaches the displaced set via `_apply_displacement` (possibly binary in/out), while the engine uses graded `factor` (variance scaled by `factor^2`); for IL-heavy teams the analytic reference may differ on displaced bodies -- interpret per-team IL-roster ratios with that caveat; the undisplaced-dominated pooled ratio is the primary verdict.
- [ ] Write `docs/superpowers/games-mc-phase6-sd-evidence-2026-06-27.md` (+ raw table) with the verdict; update the ledger.

---

## Self-Review

**Spec coverage:** Component 4 GATE (the `fraction_remaining` horizon split -- Task 1 corrects the volume term it depends on; Task 3 validates the SD), Component 6 (validation backtest, SDs), the "Variance note" (`f^2`, bounded not proven). Task 1 is the fix the plan-review's Critical-1 demanded; the gate's apples-to-apples rationale (the `sqrt(fraction_remaining)` cancellation + matched full-season cv_pt band) is now stated explicitly, not asserted.

**Placeholder scan:** Concrete `pt_volumes` wiring, `compute_sd_calibration` with enum/string-key handling + counting-cat scope + NaN guards, typed 2-tuple return channel (no sentinel key), tolerance-not-inherited-calibration threshold. Tests pin the volume fix, the byte-identical default, the enum round-trip, the rate-cat exclusion, and the NaN guard.

**Type consistency:** `pt_volumes: np.ndarray | None` (additive, default None = legacy). `run_selection_attribution -> (arm_medians, sd_calibration | None)` (2-tuple, not a widened dict). `compute_sd_calibration` joins string batch keys to `Category`-enum `team_sds` via `Category(<str>)`, counting cats only.
