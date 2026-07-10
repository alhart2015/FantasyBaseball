# Closer Role-Mixture Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the ~2.2x under-dispersion of team saves (SV) variance in the SD-calibration backtest by replacing the SV playing-time (`cv_pt`) variance term with an explicit bimodal role-switch mixture, in both the analytic ERoto path and the Monte Carlo sampler.

**Architecture:** A single source-of-truth module `sgp/closer_mixture.py` owns (a) smooth parameter curves over projected SV, (b) a closed-form per-component `sv_role_variance(mu0, frac)`, and (c) the per-draw mean-1 multiplier the MC uses. ERoto and the MC both consume it. The mixture is **mean-neutral** (a mean-1 multiplier on each path's existing SV mean) and **in-season-aware** (the role-switch/`between` term scales with `fraction_remaining`). Calibration is a continuous mixture-regression on 2022-2025 realized-vs-projected data.

**Tech Stack:** Python, numpy, scipy (existing), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-10-closer-role-mixture-design.md` (read it fully before starting).

## Global Constraints

- **ASCII-only** in all source, log, and format strings (Windows cp1252 stdout). Use `sigma`, `->`, `--`, straight quotes.
- **Numeric defaults:** never `x or default` for numeric values; use `x if x is not None else default`.
- **Within-dispersion `r` is FIXED at `STAT_DISPERSION['sv'] = 37.757`** for BOTH mixture components. Never fit or loosen it. All excess spread flows through the `between` term.
- **Mean-neutral:** the mixture must not change either path's SV mean. `E[X] = 1` by construction; the MC keeps its `eff_mean` haircut and `+8*frac_missed` backfill.
- **Single source of truth:** all three SV-variance call sites (`scoring.py`, `simulation.py`, `backtest_sd_calibration.py`) route through `closer_mixture`. None re-derives SV dispersion locally.
- **Reference variance identity (verified):** `within + between = negbin_perf_variance(mu0) + between*(1 + 1/r)`, where `within = q*nb_var(mu0*a_m) + (1-q)*nb_var(mu0*a_s)`, `between = mu0^2*q*(1-q)*(a_m-a_s)^2`, `nb_var(m) = m + m^2/r`. Evaluating `within` at the single mean `mu0` is WRONG (omits `between/r`, ~2.4% for a 30-SV closer).
- **End-of-effort checks (repo rule):** `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` if any touched file is under `[tool.mypy].files`. Show command output; never claim "checks pass" without it.

## File Structure

- **Create** `src/fantasy_baseball/sgp/closer_mixture.py` -- curves `q(s), a_s(s), a_m(s)`; `sv_role_variance(mu0, fraction_remaining)`; `role_multiplier_draw(mu0_array, s_array, rng, fraction_remaining)`. One responsibility: the SV role-switch mixture math.
- **Modify** `src/fantasy_baseball/utils/constants.py` -- add `SV_ROLE_MIXTURE` (fitted curve coefficients).
- **Modify** `src/fantasy_baseball/scoring.py` -- SV branch in `player_category_variance` (line ~1283); thread `fraction_remaining`. Mean path (`project_team_stats`) untouched.
- **Modify** `src/fantasy_baseball/simulation.py` -- SV role draw in `_apply_variance_batch` (~781-817): SV mean/variance handled outside the shared `scales`; `r` and backfill unchanged.
- **Create** `scripts/calibrate_closer_mixture.py` -- continuous mixture-regression calibration, 2022-2025.
- **Modify** `scripts/backtest_sd_calibration.py` -- wire SV to `sv_role_variance`; per-year category set to admit 2025 SV; optional 2025 SO derivation.
- **Create** tests under `tests/test_sgp/test_closer_mixture.py` and additions to `tests/test_scoring.py`, `tests/test_integration/`.

## Milestone structure (important)

The spec's feasibility section flags that a **two-component** mixture may not reach the gate at the closer end (needs ~45% job-turnover). The plan is therefore gated:

- **Phase A (Tasks 1-3):** build the closed-form module + constant scaffold with a *provisional* hand-set curve, TDD'd against the math identity. No calibration yet.
- **Phase B (Tasks 4-5):** calibration script + backtest wiring. **Task 5 is the DECISION GATE:** run the wired backtest; if SV `SD(z)` in `[0.8,1.25]`, proceed with two components; if not, escalate to three components (Task 5b) before wiring the seams.
- **Phase C (Tasks 6-9):** ERoto seam, MC seam, parity + regression tests, full end-of-effort verification.

Do NOT wire the ERoto/MC seams (Phase C) until Task 5's gate is green, so the production paths are only ever fed a calibration that actually works.

---

### Task 1: `closer_mixture` closed-form variance (provisional curves)

**Files:**
- Create: `src/fantasy_baseball/sgp/closer_mixture.py`
- Test: `tests/test_sgp/test_closer_mixture.py`

**Interfaces:**
- Consumes: `STAT_DISPERSION['sv']` (=37.757) from `utils.constants`.
- Produces:
  - `sv_role_variance(mu0: float | np.ndarray, fraction_remaining: float = 1.0) -> float | np.ndarray`
  - `_components(s: float | np.ndarray) -> tuple[q, a_m, a_s]` (internal; reads `SV_ROLE_MIXTURE` once it exists -- for Task 1 use a provisional module-level curve so the math is testable before calibration).
  - `nb_var(m) = m + m*m/r` helper (or reuse `dispersion.negbin_variance_from_r(m, r)`).

- [ ] **Step 1: Write the failing test -- closed form equals the generative process at a shared mu0**

```python
# tests/test_sgp/test_closer_mixture.py
import numpy as np
from fantasy_baseball.sgp import closer_mixture as cm
from fantasy_baseball.utils.constants import STAT_DISPERSION

def _brute_force_var(mu0, q, a_m, a_s, r, n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    mult = np.where(rng.random(n) < q, a_m, a_s)
    m = mu0 * mult
    lam = rng.gamma(r, m / r)          # NegBin(mean=m, disp=r) via gamma-poisson
    sv = rng.poisson(lam)
    return sv.var()

def test_closed_form_matches_generative_at_shared_mu0():
    r = STAT_DISPERSION["sv"]
    mu0, q, a_m, a_s = 30.0, 0.55, 1.0 / 0.55, 0.0   # closer: a_s->0, mean-1 => a_m=1/q
    within = q * cm.nb_var(mu0 * a_m) + (1 - q) * cm.nb_var(mu0 * a_s)
    between = mu0**2 * q * (1 - q) * (a_m - a_s) ** 2
    closed = within + between
    brute = _brute_force_var(mu0, q, a_m, a_s, r)
    # tolerance tight enough to catch the ~2.4% between/r cross-term (well under it)
    assert abs(closed - brute) / brute < 0.01
    # and prove the naive single-mu0 form is WRONG (misses ~2.4%)
    naive = cm.nb_var(mu0) + between
    assert (closed - naive) / closed > 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sgp/test_closer_mixture.py::test_closed_form_matches_generative_at_shared_mu0 -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'nb_var'`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/sgp/closer_mixture.py
"""Bimodal role-switch mixture for saves (SV) variance. Single source of truth
for the SV dispersion shared by ERoto (scoring.py) and the MC (simulation.py).

The mixture is a mean-1 multiplier X on a pitcher's current SV mean mu0:
  X = a_m w.p. q(s), else a_s w.p. 1-q(s);  q*a_m + (1-q)*a_s = 1  (E[X]=1).
Variance (law of total variance over role AND NegBin), r fixed at STAT_DISPERSION['sv']:
  within  = q*nb_var(mu0*a_m) + (1-q)*nb_var(mu0*a_s)
  between = mu0^2 * q*(1-q) * (a_m - a_s)^2       (role-switch; scaled by fraction_remaining)
"""
from __future__ import annotations

import numpy as np

from fantasy_baseball.utils.constants import STAT_DISPERSION

_R = STAT_DISPERSION["sv"]

# PROVISIONAL curves (replaced by SV_ROLE_MIXTURE after Task 4 calibration).
# Closer end: high job retention, a_s->0. Vault end: small 1-q, large a_s.
def _components(s):
    s = np.asarray(s, dtype=float)
    # placeholder logistic q(s) and a_s(s); overwritten by calibration.
    q = 0.55 + 0.0 * s
    a_s = np.zeros_like(s)
    a_m = (1.0 - (1.0 - q) * a_s) / q
    return q, a_m, a_s

def nb_var(m):
    m = np.asarray(m, dtype=float)
    return m + m * m / _R

def sv_role_variance(mu0, fraction_remaining: float = 1.0):
    mu0 = np.asarray(mu0, dtype=float)
    q, a_m, a_s = _components(mu0)   # keyed on projected SV == mu0 pre-haircut; see note
    within = q * nb_var(mu0 * a_m) + (1.0 - q) * nb_var(mu0 * a_s)
    between = mu0**2 * q * (1.0 - q) * (a_m - a_s) ** 2
    out = within + fraction_remaining * between
    return float(out) if out.ndim == 0 else out
```

Note for the implementer: the curve is keyed on **projected SV** `s`. In ERoto `mu0` is the raw projection so `s == mu0`; in the MC `mu0 = base*eff_mean` but the curve must still be keyed on the raw projected SV `s` (pass `s` separately in Task 7). For Task 1 the provisional curve is `s`-flat, so this distinction does not yet bite; Task 7 threads `s` explicitly.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sgp/test_closer_mixture.py::test_closed_form_matches_generative_at_shared_mu0 -v`
Expected: PASS.

- [ ] **Step 5: Add the invariant unit tests (mean-1, between->0, non-negativity)**

```python
def test_mean_one_and_frac_scaling():
    # provisional q=0.55, a_s=0 => a_m=1/0.55; E[X]=1
    q, a_m, a_s = 0.55, 1/0.55, 0.0
    assert abs(q*a_m + (1-q)*a_s - 1.0) < 1e-9
    mu0 = 25.0
    v_full = cm.sv_role_variance(mu0, 1.0)
    v_zero = cm.sv_role_variance(mu0, 0.0)
    within = q*cm.nb_var(mu0*a_m) + (1-q)*cm.nb_var(mu0*a_s)
    assert abs(v_zero - within) < 1e-6            # between -> 0 at frac=0
    assert v_full > v_zero                         # between adds at frac=1

def test_variance_nonnegative_vectorized():
    v = cm.sv_role_variance(np.array([0.0, 1.0, 8.0, 22.0, 40.0]), 1.0)
    assert np.all(v >= 0)
```

- [ ] **Step 6: Run all Task-1 tests and commit**

Run: `pytest tests/test_sgp/test_closer_mixture.py -v`  (Expected: PASS)
```bash
git add src/fantasy_baseball/sgp/closer_mixture.py tests/test_sgp/test_closer_mixture.py
git commit -m "feat(sgp): closer role-mixture closed-form SV variance (provisional curves) (#193)"
```

---

### Task 2: `SV_ROLE_MIXTURE` constant scaffold + module wiring

**Files:**
- Modify: `src/fantasy_baseball/utils/constants.py`
- Modify: `src/fantasy_baseball/sgp/closer_mixture.py`
- Test: `tests/test_sgp/test_closer_mixture.py`

**Interfaces:**
- Produces: `SV_ROLE_MIXTURE: dict` (curve coefficients). `closer_mixture._components` reads it instead of the hard-coded provisional values.

- [ ] **Step 1: Write the failing test** -- `_components` reads coefficients from the constant and honors the mean-1 constraint across `s`.

```python
def test_components_read_constant_and_are_mean_one():
    for s in [0.5, 3.0, 12.0, 20.0, 35.0]:
        q, a_m, a_s = cm._components(s)
        assert 0.0 <= q <= 1.0
        assert a_m >= 0 and a_s >= 0
        assert abs(q * a_m + (1 - q) * a_s - 1.0) < 1e-9
```

- [ ] **Step 2: Run -> FAIL** (mean-1 not guaranteed / constant missing).
Run: `pytest tests/test_sgp/test_closer_mixture.py::test_components_read_constant_and_are_mean_one -v`

- [ ] **Step 3: Add the scaffold constant** (a documented, valid-but-provisional shape; the real values land in Task 4).

```python
# constants.py -- provisional; regenerated by scripts/calibrate_closer_mixture.py
# q(s) logistic; a_s(s) surprise multiplier: large at low s (vault), ~0 at high s (job loss).
SV_ROLE_MIXTURE: dict[str, list[float]] = {
    "q_logit": [0.2, 0.05],      # q(s) = sigmoid(b0 + b1*s)   (provisional)
    "a_s_curve": [0.0, 0.0],     # a_s(s) placeholder           (provisional)
}
```

- [ ] **Step 4: Rewrite `_components` to consume the constant with mean-1 by construction**

```python
from fantasy_baseball.utils.constants import STAT_DISPERSION, SV_ROLE_MIXTURE

def _components(s):
    s = np.asarray(s, dtype=float)
    b0, b1 = SV_ROLE_MIXTURE["q_logit"]
    q = 1.0 / (1.0 + np.exp(-(b0 + b1 * s)))
    q = np.clip(q, 1e-3, 1 - 1e-3)
    a_s = _a_s_curve(s, SV_ROLE_MIXTURE["a_s_curve"])   # >= 0 by construction
    a_m = (1.0 - (1.0 - q) * a_s) / q
    a_m = np.maximum(a_m, 0.0)                           # feasibility guard (a_m,a_s >= 0)
    return q, a_m, a_s
```

with a small `_a_s_curve(s, coeffs)` returning a non-negative surprise multiplier (e.g. `np.maximum(0, c0 + c1*s)` clamped; the exact functional form is fixed in Task 4). For the scaffold, `a_s = 0` everywhere is valid (degenerate to unimodal) and keeps tests green until calibration.

- [ ] **Step 5: Run -> PASS, then commit**
Run: `pytest tests/test_sgp/test_closer_mixture.py -v`
```bash
git add src/fantasy_baseball/utils/constants.py src/fantasy_baseball/sgp/closer_mixture.py tests/test_sgp/test_closer_mixture.py
git commit -m "feat(constants): SV_ROLE_MIXTURE scaffold + mean-1 component construction (#193)"
```

---

### Task 3: `role_multiplier_draw` for the MC (per-draw, in-season)

**Files:**
- Modify: `src/fantasy_baseball/sgp/closer_mixture.py`
- Test: `tests/test_sgp/test_closer_mixture.py`

**Interfaces:**
- Produces: `role_multiplier_draw(s, rng, size, fraction_remaining) -> np.ndarray` returning `X' = 1 + sqrt(frac)*(X-1)` where `X in {a_m, a_s}` w.p. `{q, 1-q}` -- so `E[X']=1`, `Var(X')=frac*Var(X)`, `X' >= 0`.

- [ ] **Step 1: Write the failing test** (mean 1, variance scales by frac, non-negative)

```python
def test_role_multiplier_draw_moments():
    rng = np.random.default_rng(1)
    s = np.full(50_000, 30.0)
    for frac in (1.0, 0.5, 0.25):
        x = cm.role_multiplier_draw(s, rng, fraction_remaining=frac)
        assert abs(x.mean() - 1.0) < 0.02            # E[X']=1 in-season
        assert np.all(x >= 0)                        # no negative multiplier
    x1 = cm.role_multiplier_draw(np.full(200_000, 30.0), np.random.default_rng(2), fraction_remaining=1.0)
    xh = cm.role_multiplier_draw(np.full(200_000, 30.0), np.random.default_rng(2), fraction_remaining=0.5)
    assert abs(xh.var() / x1.var() - 0.5) < 0.05     # Var scales ~ frac
```

- [ ] **Step 2: Run -> FAIL.**  Run: `pytest tests/test_sgp/test_closer_mixture.py::test_role_multiplier_draw_moments -v`

- [ ] **Step 3: Implement**

```python
def role_multiplier_draw(s, rng, fraction_remaining: float = 1.0):
    s = np.asarray(s, dtype=float)
    q, a_m, a_s = _components(s)
    pick_m = rng.random(s.shape) < q
    x = np.where(pick_m, a_m, a_s)
    x_prime = 1.0 + np.sqrt(fraction_remaining) * (x - 1.0)   # E=1, Var=frac*Var(X), >=0
    return np.maximum(x_prime, 0.0)
```

- [ ] **Step 4: Run -> PASS. Commit.**
```bash
git add src/fantasy_baseball/sgp/closer_mixture.py tests/test_sgp/test_closer_mixture.py
git commit -m "feat(sgp): in-season mean-1 role multiplier draw for the MC (#193)"
```

---

### Task 4: Calibration script (DISCOVERY -- fits the real curves)

**Files:**
- Create: `scripts/calibrate_closer_mixture.py`

This is a discovery/spike task: the exact functional forms of `q(s)` and `a_s(s)` are determined by the data, not prescribed. Deliverable = fitted `SV_ROLE_MIXTURE` coefficients pasted into `constants.py`, plus a printed fit report.

**Acceptance criteria (not fabricated code):**
- Reads 2022-2025 `(projected s, realized SV)` pairs using the same steamer+zips blend and `data/stats` actuals as `backtest_sd_calibration.py` (reuse its `blend`/`build_year` helpers; import, do not re-implement -- CLAUDE.md reuse rule).
- 2025 actuals: `SV` and `W` used directly; `SO` (only if the optional W/SO inclusion is wanted) reconstructed with the thirds fix: `ip_true = floor(IP) + (IP-floor(IP))*10/3; SO = round(K/9*ip_true/9)`.
- Fits `q(s)` (logistic) and `a_s(s)` (non-negative low-order curve) by maximum likelihood of realized SV under `NegBin(s*X, r=37.757)` with `X` the two-point mean-1 mixture, `a_m` derived from the mean-1 constraint, and an explicit `a_m, a_s >= 0` feasibility guard.
- `r` fixed at 37.757 (NOT fit).
- Prints per-`s`-decile fitted `(q, a_m, a_s)` and the effective sample support at the vault tail (spec calls this out).
- Emits the `SV_ROLE_MIXTURE` dict literal to paste into `constants.py`.

- [ ] **Step 1:** Write the calibration script per the acceptance criteria. Reuse `backtest_sd_calibration.blend` and `build_year`.
- [ ] **Step 2:** Run it: `python scripts/calibrate_closer_mixture.py`. Inspect the fit report (curves monotone-sane; `a_s` large at low `s`, ~0 at high `s`; `a_m,a_s>=0`).
- [ ] **Step 3:** Paste the emitted coefficients into `SV_ROLE_MIXTURE` in `constants.py`. Re-run `pytest tests/test_sgp/test_closer_mixture.py -v` (all invariant tests still pass with real curves).
- [ ] **Step 4: Commit**
```bash
git add scripts/calibrate_closer_mixture.py src/fantasy_baseball/utils/constants.py
git commit -m "feat(calibrate): fit SV role-mixture curves from 2022-2025 (#193)"
```

---

### Task 5: Wire the backtest + DECISION GATE (two vs three component)

**Files:**
- Modify: `scripts/backtest_sd_calibration.py` (line 44 `P_CATS`, line 85 gate, line 90 column-select, line 117 inline SV var)

**Interfaces:**
- Consumes: `closer_mixture.sv_role_variance`.

- [ ] **Step 1: Branch SV variance to the mixture.** In `team_z` (line ~117), for the `sv` key use `closer_mixture.sv_role_variance(proj, fraction_remaining=1.0)` per player summed over the team; W/K keep `negbin_perf_variance(key, proj) + proj**2 * cvp**2`. (Backtest is full-season, so `frac=1`.)

- [ ] **Step 2: Admit 2025 for SV via a per-year category set.** `build_year` currently returns `(hm, None)` unless `{"W","SO","SV"}` present (line 85) and hard-selects `["MLBAMID","W","SO","SV"]` (line 90). Change to: include SV (and W, both present in 2025) always; drop SO from the 2025 column-select and from that year's `P_CATS` when the actuals lack it. (If the optional 2025 SO derivation is implemented, add SO instead of dropping it.)

- [ ] **Step 3: Run the backtest.**
Run: `python scripts/backtest_sd_calibration.py`
Record SV `SD(z)` (both MATCHED-ONLY and DNP=0), and confirm R/HR/RBI/SB unchanged and W/SO still in `[0.8,1.25]`.

- [ ] **Step 4: DECISION GATE.**
  - **If SV `SD(z)` in `[0.8, 1.25]`** (both variants): two components suffice. Skip Task 5b. Commit and proceed to Task 6.
  - **If SV `SD(z)` out of band:** escalate -- do Task 5b before the seams. Do NOT loosen `r` or tune curves to the gate.

- [ ] **Step 5: Commit**
```bash
git add scripts/backtest_sd_calibration.py
git commit -m "feat(backtest): wire SV variance to role mixture; admit 2025 SV (#193)"
```

---

### Task 5b (CONDITIONAL): three-component escalation

Only if Task 5's gate missed. Extend `closer_mixture` to a three-point `X` (hold / job-share / lose, plus the vault path folded into the low-`s` curve), re-derive the closed-form `within`/`between` for three components (same law-of-total-variance structure, `r=37.757` all components), update `role_multiplier_draw`, re-run Task 4 calibration and Task 5 backtest until the gate is green. Update `tests/test_sgp/test_closer_mixture.py` for the three-point identity. Commit as `feat(sgp): three-component role mixture (#193)`.

---

### Task 6: ERoto seam (`scoring.py`)

**Files:**
- Modify: `src/fantasy_baseball/scoring.py` (`player_category_variance`, SV term ~line 1283; thread `fraction_remaining`)
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `closer_mixture.sv_role_variance(v, fraction_remaining)`.

- [ ] **Step 1: Write the failing test** -- ERoto SV variance now equals the mixture, and W/K are unchanged.

```python
# tests/test_scoring.py
def test_sv_variance_uses_role_mixture():
    from fantasy_baseball.sgp import closer_mixture as cm
    from fantasy_baseball.scoring import player_category_variance
    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.utils.constants import Category
    player = _pitcher_fixture(sv=30, w=4, k=70, ip=65)   # existing helper / dict
    out = player_category_variance(player)                # frac defaults to 1.0
    assert abs(out[Category.SV] - cm.sv_role_variance(30, 1.0)) < 1e-6
```

- [ ] **Step 2: Run -> FAIL.**  Run: `pytest tests/test_scoring.py::test_sv_variance_uses_role_mixture -v`

- [ ] **Step 3: Implement.** Split SV out of the `w/k/sv` loop: keep `w`,`k` on `negbin_perf_variance + v*v*cv_pt_sq`; set `result[Category.SV] = closer_mixture.sv_role_variance(_stat(player,"sv"), fraction_remaining)`. Add a `fraction_remaining: float = 1.0` parameter to `player_category_variance` and thread it from `project_team_sds` (default 1.0 preserves current callers). Leave `project_team_stats` (mean path) untouched.

- [ ] **Step 4: Run -> PASS.** Also run `pytest tests/test_scoring.py -v` to confirm no regression on W/K/hitter categories.

- [ ] **Step 5: Commit**
```bash
git add src/fantasy_baseball/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): route ERoto SV variance through role mixture (#193)"
```

---

### Task 7: MC seam (`simulation.py`)

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (`_apply_variance_batch`, ~781-817)
- Test: `tests/test_integration/test_lineup_integration.py` (or nearest MC test)

**Interfaces:**
- Consumes: `closer_mixture.role_multiplier_draw`.

- [ ] **Step 1: Write the failing test** -- an injured-closer team's simulated SV variance widens vs the pre-change baseline while its SV mean is stable (per-path stability).

```python
def test_mc_sv_variance_widens_mean_stable():
    # build a small pitcher set incl. a 30-SV closer; run _apply_variance_batch
    # with a fixed seed at frac=1; compare SV column mean (stable within tol)
    # and variance (strictly greater) vs a baseline run with the mixture disabled.
    ...
```

- [ ] **Step 2: Run -> FAIL.**

- [ ] **Step 3: Implement the SV-specific path.** For the `sv` column only:
  - `mu0 = base['sv'][None,:] * eff_mean[None,:]` (the mean location of `scales`, NOT the `z_pt*eff_sd` spread).
  - `X = closer_mixture.role_multiplier_draw(base['sv'], rng, fraction_remaining)` per (iter, player).
  - Feed `mu_mat[:,:,sv_idx] = mu0 * X` and keep `r_mat[:,:,sv_idx]` at `STAT_DISPERSION['sv']` (unchanged line 795).
  - Leave the shared `frac_missed` and the `+8*frac_missed` backfill (811-814) untouched -- SV keeps the shared backfill (mean-neutral).
  - Key the curve on the RAW projected SV `base['sv']`, not `mu0`.

- [ ] **Step 4: Run -> PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/fantasy_baseball/simulation.py tests/test_integration/test_lineup_integration.py
git commit -m "feat(simulation): SV role-switch mixture in the MC sampler (#193)"
```

---

### Task 8: Parity + regression tests

**Files:**
- Test: `tests/test_integration/test_closer_mixture_parity.py` (new)

- [ ] **Step 1: Full-season per-path stability + shared-mu0 invariant (Testing #1, #3).** Assert (a) `sv_role_variance` closed form matches a generative brute-force at a shared `mu0` within the 2.4%-catching tolerance (already in Task 1 -- re-assert at the integration level); (b) each path's SV **mean** and **SD** move only by the intended mixture delta vs a mixture-disabled baseline; NOT cross-path absolute equality (paths feed different `mu0`).
- [ ] **Step 2: In-season property (Testing #4).** The mixture's `between` contribution scales toward 0 as `fraction_remaining -> 0` and the SCALING (ratio to full-season) matches across paths.
- [ ] **Step 3: Valuation regression (Testing #5).** SGP/VAR/VONA SV values for a set of pitchers are unchanged (the mean pipeline is untouched).
- [ ] **Step 4: MC re-baseline (Testing #6).** Update deterministic-seed MC test expectations for the added Bernoulli draw; document that "only SV changes" is distributional.
- [ ] **Step 5: Run all, commit.**
```bash
git add tests/test_integration/test_closer_mixture_parity.py tests/  # + any re-baselined fixtures
git commit -m "test(sim): parity, valuation-regression, and MC re-baseline for role mixture (#193)"
```

---

### Task 9: Full end-of-effort verification

- [ ] **Step 1:** `pytest -v` (or `pytest -n auto`) -- all pass. Fix any failure (code, not tests).
- [ ] **Step 2:** `ruff check .` -- zero violations.
- [ ] **Step 3:** `ruff format --check .` -- no drift (`ruff format .` to fix).
- [ ] **Step 4:** `vulture` -- no NEW dead-code findings.
- [ ] **Step 5:** `mypy` -- if any touched file is under `[tool.mypy].files`.
- [ ] **Step 6:** Re-run `python scripts/backtest_sd_calibration.py` and paste the final SV `SD(z)` line into the PR body. Confirm R/HR/RBI/SB unchanged and W/SO/SV in `[0.8,1.25]`.
- [ ] **Step 7: Commit** any fixups; the branch is ready for PR.

---

## Spec-coverage self-check

- Bimodal mixture, mean-1, continuous curves, r-fixed -> Tasks 1-4. In-season scaling -> Tasks 1/3. ERoto seam -> Task 6. MC seam -> Task 7. Calibration (constrained, thirds-SO) -> Task 4. Backtest wiring + 2025 admit + per-year cats -> Task 5. Feasibility gate + 3-component escalation -> Task 5/5b. Tests #1-#6 -> Tasks 1,3,8. Known limitations -> documented in `closer_mixture.py` docstring (add in Task 1). #235/#236 -> untouched (out of scope).
