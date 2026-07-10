# Closer Role-Mixture Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the ~2.2x under-dispersion of team saves (SV) variance in the SD-calibration backtest by replacing the SV playing-time (`cv_pt`) variance term with an explicit bimodal role-switch mixture, in both the analytic ERoto path and the Monte Carlo sampler.

**Architecture:** A single source-of-truth module `sgp/closer_mixture.py` owns (a) smooth parameter curves over projected SV, (b) a **full-season** closed-form `sv_role_variance(s)`, and (c) a per-draw mean-1 multiplier `role_multiplier_draw` the MC uses. ERoto and the MC both consume it. The mixture is **mean-neutral** (a mean-1 multiplier on each path's existing SV mean) and **in-season-aware** via existing external machinery (ERoto's `build_team_sds` `sqrt(frac)`; the MC copula + the `X'` shrink) -- `sv_role_variance` itself is full-season. Calibration is a continuous mixture-regression on 2022-2025 realized-vs-projected data.

**Tech Stack:** Python, numpy, scipy (existing), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-10-closer-role-mixture-design.md` (read it fully before starting).

## Global Constraints

- **ASCII-only** in all source, log, format strings. Use `sigma`, `->`, `--`, straight quotes.
- **Numeric defaults:** never `x or default`; use `x if x is not None else default`.
- **Within-dispersion `r` FIXED at `STAT_DISPERSION['sv'] = 37.757`** for BOTH components. Never fit or loosen. All excess spread flows through the `between` term.
- **Mean-neutral:** `E[X] = 1` by construction; the MC keeps its `eff_mean` haircut and `+8*frac_missed` backfill.
- **`sv_role_variance` is FULL-SEASON (no `fraction_remaining` parameter).** In-season scaling is external and uniform: ERoto via `build_team_sds` (`scoring.py:1389`, `sd_scale = sqrt(frac)`); the MC via the copula (within) + `role_multiplier_draw`'s `X'` shrink (between). Do NOT thread `frac` into `player_category_variance` / `sv_role_variance` -- that double-scales (`build_team_sds` already scales).
- **Curves keyed on projected SV `s`** (NOT the MC haircut mean `base*eff_mean`). In ERoto/backtest `s == mu0`; the MC passes raw `base['sv']` to the curve and `base*eff_mean` only as the NegBin mean.
- **Single source of truth:** all SV-variance sites route through `closer_mixture`.
- **Reference variance identity (verified):** `within + between = negbin_perf_variance(s) + between*(1 + 1/r)`, `within = q*nb_var(s*a_m) + (1-q)*nb_var(s*a_s)`, `between = s^2*q*(1-q)*(a_m-a_s)^2`, `nb_var(m) = m + m^2/r`. Evaluating `within` at the single mean `s` is WRONG (omits `between/r`, ~2.4% for a 30-SV closer).
- **End-of-effort checks:** `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, `mypy` (if a touched file is under `[tool.mypy].files`). Show output.

## File Structure

- **Create** `src/fantasy_baseball/sgp/closer_mixture.py`
- **Modify** `src/fantasy_baseball/utils/constants.py` -- `SV_ROLE_MIXTURE`
- **Modify** `src/fantasy_baseball/scoring.py` -- SV branch in `player_category_variance` (~1283). Mean path untouched.
- **Modify** `src/fantasy_baseball/simulation.py` -- SV role draw in `_apply_variance_batch` (~781-817)
- **Create** `scripts/calibrate_closer_mixture.py`
- **Modify** `scripts/backtest_sd_calibration.py` -- per-year cat set to admit 2025 SV; SV wired to mixture
- **Verify (call-site audit)** `src/fantasy_baseball/lineup/delta_roto.py` (~346-352) -- consumes `player_category_variance`/`project_team_sds`; SV term switches to the mixture. Confirm intended.
- **Create/extend tests** under `tests/test_sgp/`, `tests/test_scoring.py`, `tests/test_integration/`

## Milestone structure

- **Phase A (Tasks 1-3):** `closer_mixture` module (closed form + draw) with provisional curves, TDD'd against the math identity.
- **Phase B (Tasks 4-6):** backtest 2025-admission plumbing (Task 4) -> calibration (Task 5) -> wire backtest + **DECISION GATE** (Task 6). Task 6 decides two vs three components empirically.
- **Phase C (Tasks 7-10):** ERoto seam, MC seam, parity/regression tests, verification. **Do NOT start Phase C until Task 6's gate is green.**

---

### Task 1: `closer_mixture` full-season closed-form variance

**Files:**
- Create: `src/fantasy_baseball/sgp/closer_mixture.py`
- Test: `tests/test_sgp/test_closer_mixture.py`

**Interfaces produced:**
- `nb_var(m) -> np.ndarray` = `m + m*m/r`
- `sv_role_variance(s: float | np.ndarray) -> float | np.ndarray` -- full-season `within + between`, keyed and meaned on projected SV `s`.
- `_components(s) -> (q, a_m, a_s)` -- provisional module-level curve until Task 2.

- [ ] **Step 1: Write the failing test -- closed form equals the generative process; naive form is wrong**

```python
# tests/test_sgp/test_closer_mixture.py
import numpy as np
from fantasy_baseball.sgp import closer_mixture as cm
from fantasy_baseball.utils.constants import STAT_DISPERSION

def _brute_force_var(s, q, a_m, a_s, r, n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    mult = np.where(rng.random(n) < q, a_m, a_s)
    m = s * mult
    lam = rng.gamma(r, m / r)          # NegBin(mean=m, disp=r) via gamma-poisson
    return rng.poisson(lam).var()

def test_closed_form_matches_generative():
    r = STAT_DISPERSION["sv"]
    s, q, a_m, a_s = 30.0, 0.55, 1.0 / 0.55, 0.0   # a_s=0 => a_m=1/q (mean-1)
    within = q * cm.nb_var(s * a_m) + (1 - q) * cm.nb_var(s * a_s)
    between = s**2 * q * (1 - q) * (a_m - a_s) ** 2
    closed = within + between
    assert abs(closed - _brute_force_var(s, q, a_m, a_s, r)) / closed < 0.01
    naive = cm.nb_var(s) + between                 # WRONG single-mean form
    assert (closed - naive) / closed > 0.02        # closed keeps the ~2.4% between/r cross-term
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_sgp/test_closer_mixture.py::test_closed_form_matches_generative -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'fantasy_baseball.sgp.closer_mixture'` (the module does not exist yet; the import line raises before any `cm.*` reference).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/sgp/closer_mixture.py
"""Bimodal role-switch mixture for saves (SV) variance. Single source of truth
for the SV dispersion shared by ERoto (scoring.py) and the MC (simulation.py).

X is a mean-1 multiplier on a pitcher's SV mean:
  X = a_m w.p. q(s), else a_s w.p. 1-q(s);  q*a_m + (1-q)*a_s = 1  (E[X]=1).
FULL-SEASON variance (law of total variance over role AND NegBin), r fixed:
  within  = q*nb_var(s*a_m) + (1-q)*nb_var(s*a_s)
  between = s^2 * q*(1-q) * (a_m - a_s)^2
In-season scaling is applied EXTERNALLY (ERoto: build_team_sds sqrt(frac); MC:
copula for within + role_multiplier_draw's X' for between) -- NOT here.

Known limitations (see spec): SV pulled out of the shared MC `scales` loses its
playing-time co-movement with W/K/ER; the role Bernoulli is independent of the
copula so job-loss does not co-move with high ER; handcuffed closers' anti-
correlation is not modeled. All second-order; do not touch the marginal target.
"""
from __future__ import annotations

import numpy as np

from fantasy_baseball.utils.constants import STAT_DISPERSION

_R = STAT_DISPERSION["sv"]

def _components(s):
    s = np.asarray(s, dtype=float)
    q = np.full(s.shape, 0.55)          # provisional; replaced in Task 2
    a_s = np.zeros_like(s)
    a_m = (1.0 - (1.0 - q) * a_s) / q
    return q, a_m, a_s

def nb_var(m):
    m = np.asarray(m, dtype=float)
    return m + m * m / _R

def sv_role_variance(s):
    s = np.asarray(s, dtype=float)
    q, a_m, a_s = _components(s)
    within = q * nb_var(s * a_m) + (1.0 - q) * nb_var(s * a_s)
    between = s**2 * q * (1.0 - q) * (a_m - a_s) ** 2
    out = within + between
    return float(out) if out.ndim == 0 else out
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_sgp/test_closer_mixture.py::test_closed_form_matches_generative -v`  -- Expected: PASS.

- [ ] **Step 5: Add invariant tests (mean-1, non-negativity, vectorized)**

```python
def test_variance_nonnegative_and_vectorized():
    v = cm.sv_role_variance(np.array([0.0, 1.0, 8.0, 22.0, 40.0]))
    assert v.shape == (5,) and np.all(v >= 0)

def test_provisional_components_mean_one():
    q, a_m, a_s = cm._components(np.array([5.0, 30.0]))
    assert np.allclose(q * a_m + (1 - q) * a_s, 1.0)
```

- [ ] **Step 6: Run all Task-1 tests and commit**

Run: `pytest tests/test_sgp/test_closer_mixture.py -v`  (Expected: PASS)
```bash
git add src/fantasy_baseball/sgp/closer_mixture.py tests/test_sgp/test_closer_mixture.py
git commit -m "feat(sgp): full-season closer role-mixture SV variance (provisional curves) (#193)"
```

---

### Task 2: `SV_ROLE_MIXTURE` constant + curve wiring

**Files:** Modify `constants.py`, `closer_mixture.py`; Test `tests/test_sgp/test_closer_mixture.py`

**Interfaces produced:** `SV_ROLE_MIXTURE: dict` (curve coefficients); `_components` reads it.

- [ ] **Step 1: Write the failing test -- `_components` actually reflects the constant's values**

The test must FAIL against Task 1's hard-coded `q=0.55`, so it pins behavior to the constant by asserting a value the provisional curve does NOT produce (a non-0.55 `q` at some `s`, driven by the coefficients).

```python
def test_components_reflect_constant():
    import fantasy_baseball.sgp.closer_mixture as cm
    from fantasy_baseball.utils import constants
    # A distinctive logistic that gives q(0)=~0.5 and q(40) clearly != 0.55:
    saved = constants.SV_ROLE_MIXTURE
    constants.SV_ROLE_MIXTURE = {"q_logit": [0.0, 0.1], "a_s_curve": [0.0, 0.0]}
    try:
        q0, _, _ = cm._components(0.0)
        q40, _, _ = cm._components(40.0)
        assert abs(float(q0) - 0.5) < 1e-6            # sigmoid(0)=0.5
        assert float(q40) > 0.95                        # sigmoid(4)=0.982 -- proves it reads coeffs
        # mean-1 still holds everywhere
        q, a_m, a_s = cm._components(np.array([0.0, 40.0]))
        assert np.allclose(q * a_m + (1 - q) * a_s, 1.0)
    finally:
        constants.SV_ROLE_MIXTURE = saved
```

- [ ] **Step 2: Run -> FAIL** (`_components` ignores the constant; `q40` is 0.55).
Run: `pytest tests/test_sgp/test_closer_mixture.py::test_components_reflect_constant -v`

- [ ] **Step 3: Add the scaffold constant**

```python
# constants.py -- provisional; regenerated by scripts/calibrate_closer_mixture.py (Task 5)
SV_ROLE_MIXTURE: dict[str, list[float]] = {
    "q_logit": [0.2, 0.05],    # q(s) = sigmoid(b0 + b1*s)
    "a_s_curve": [0.0, 0.0],   # a_s(s) = max(0, c0 + c1*s) (provisional: 0 everywhere)
}
```

- [ ] **Step 4: Rewrite `_components` to read the constant (mean-1 + feasibility by construction)**

```python
from fantasy_baseball.utils import constants          # module import -- NOT `from ... import SV_ROLE_MIXTURE`

def _components(s):
    s = np.asarray(s, dtype=float)
    b0, b1 = constants.SV_ROLE_MIXTURE["q_logit"]     # read via module each call
    q = np.clip(1.0 / (1.0 + np.exp(-(b0 + b1 * s))), 1e-3, 1 - 1e-3)
    c0, c1 = constants.SV_ROLE_MIXTURE["a_s_curve"]
    a_s = np.maximum(0.0, c0 + c1 * s)                       # a_s >= 0
    a_m = np.maximum(0.0, (1.0 - (1.0 - q) * a_s) / q)       # mean-1 derived, a_m >= 0 guard
    return q, a_m, a_s
```

CRITICAL: read `constants.SV_ROLE_MIXTURE` (module-qualified) each call, NOT a top-level
`from ...constants import SV_ROLE_MIXTURE` bare name -- a bare import binds the original dict
object into `closer_mixture`, so the Step-1 test's `constants.SV_ROLE_MIXTURE = {...}` rebind
would be invisible and the test would not go red->green. Keep `_R = STAT_DISPERSION["sv"]` from
the Task-1 import (STAT_DISPERSION is never monkeypatched, and Task 5 does not change `r`).

- [ ] **Step 5: Run -> PASS, then commit**

Run: `pytest tests/test_sgp/test_closer_mixture.py -v`
```bash
git add src/fantasy_baseball/utils/constants.py src/fantasy_baseball/sgp/closer_mixture.py tests/test_sgp/test_closer_mixture.py
git commit -m "feat(constants): SV_ROLE_MIXTURE curves + mean-1 component construction (#193)"
```

---

### Task 3: `role_multiplier_draw` -- 2-D per-(iter,player) draw for the MC

**Files:** Modify `closer_mixture.py`; Test `tests/test_sgp/test_closer_mixture.py`

**Interfaces produced:**
`role_multiplier_draw(s: np.ndarray, rng: np.random.Generator, fraction_remaining: float = 1.0) -> np.ndarray`
Returns `X'` the SAME SHAPE as `s`. `s` is the RAW projected SV, broadcast by the caller to the shape it wants sampled (the MC passes 2-D `(n_iter, n_players)` so the role varies per iteration -- THIS is where the between-variance comes from). `X' = 1 + sqrt(frac)*(X-1)`, `X in {a_m,a_s}` w.p. `{q,1-q}`. There is NO `size` parameter; the shape of `s` IS the draw shape.

- [ ] **Step 1: Write the failing test -- correct moments AND non-zero cross-iteration variance on a 2-D input**

```python
def test_role_multiplier_draw_2d_moments():
    rng = np.random.default_rng(1)
    # 2-D input: 40k iters x 1 player at s=30 (closer). Between-variance lives ACROSS rows.
    s2d = np.full((40_000, 1), 30.0)
    for frac in (1.0, 0.5, 0.25):
        x = cm.role_multiplier_draw(s2d, rng, fraction_remaining=frac)
        assert x.shape == s2d.shape
        assert abs(x.mean() - 1.0) < 0.02          # E[X']=1
        assert np.all(x >= 0)
    x1 = cm.role_multiplier_draw(np.full((200_000, 1), 30.0), np.random.default_rng(2), 1.0)
    xh = cm.role_multiplier_draw(np.full((200_000, 1), 30.0), np.random.default_rng(2), 0.5)
    assert x1.var() > 0.1                            # NON-ZERO between-variance (guards the F1 bug)
    assert abs(xh.var() / x1.var() - 0.5) < 0.05    # Var scales ~ frac
```

- [ ] **Step 2: Run -> FAIL.**  Run: `pytest tests/test_sgp/test_closer_mixture.py::test_role_multiplier_draw_2d_moments -v`

- [ ] **Step 3: Implement**

```python
def role_multiplier_draw(s, rng, fraction_remaining: float = 1.0):
    s = np.asarray(s, dtype=float)
    q, a_m, a_s = _components(s)                 # same shape as s
    x = np.where(rng.random(s.shape) < q, a_m, a_s)
    x_prime = 1.0 + np.sqrt(fraction_remaining) * (x - 1.0)   # E=1, Var=frac*Var(X), >=0
    return np.maximum(x_prime, 0.0)
```

- [ ] **Step 4: Run -> PASS. Commit.**
```bash
git add src/fantasy_baseball/sgp/closer_mixture.py tests/test_sgp/test_closer_mixture.py
git commit -m "feat(sgp): 2-D per-(iter,player) mean-1 role multiplier draw (#193)"
```

---

### Task 4: Backtest 2025-admission plumbing (per-year category set)

**Files:** Modify `scripts/backtest_sd_calibration.py`

Pure data plumbing, no mixture dependency -- do it BEFORE calibration so Task 5 can read 2025. Today `build_year` returns `(hm, None)` unless `{"W","SO","SV"}` all present (line 85) and hard-selects `pa[["MLBAMID","W","SO","SV"]]` (line 90); `P_CATS` (line 44) is a module global used in `run()` (zs init/print) and `team_z`. 2025 actuals lack `SO`.

- [ ] **Step 1: Make `build_year` return per-year pitcher categories.** Change it to return `(hm, pm, p_cats)` where `p_cats` lists only the pitcher `(actual_col, key)` pairs present that year: always `[("W","w"),("SV","sv")]`; add `("SO","k")` when `SO` is present OR reconstructable. Select columns dynamically (`["MLBAMID"] + [c for c in ("W","SO","SV") if c in pa.columns]`) instead of the hard 4-col select at line 90.
- [ ] **Step 2: Reconstruct 2025 SO (optional W/SO completeness)** with the thirds fix, so 2025 can carry SO too:
```python
ip = pa["IP"].to_numpy(float)
ip_true = np.floor(ip) + (ip - np.floor(ip)) * 10.0 / 3.0
pa["SO"] = np.round(pa["K/9"].to_numpy(float) * ip_true / 9.0)
```
(Guard: only when `SO` absent but `K/9` and `IP` present.)
- [ ] **Step 3: Thread `p_cats` through `run()`** -- accumulate `zs` over the union of per-year cats; in the per-year loop pass that year's `p_cats` to `team_z` instead of the global `P_CATS`.
- [ ] **Step 4: Run and verify 2025 now contributes.** Run: `python scripts/backtest_sd_calibration.py`. Confirm the SV `n` count rises vs the pre-change 2022-2024-only baseline (more team-seasons), and W/SO/SV verdicts still print.
- [ ] **Step 5: Commit**
```bash
git add scripts/backtest_sd_calibration.py
git commit -m "feat(backtest): per-year category set admits 2025 SV (and derived SO) (#193)"
```

---

### Task 5: Calibration script (DISCOVERY -- fits the real curves)

**Files:** Create `scripts/calibrate_closer_mixture.py`

Discovery task: the functional forms of `q(s)`, `a_s(s)` are fit from data. Deliverable = fitted `SV_ROLE_MIXTURE` coefficients in `constants.py` + a printed fit report.

**Acceptance criteria:**
- Reads 2022-2025 `(projected s, realized SV)` pairs. Reuse `backtest_sd_calibration.blend` and the now-2025-aware `build_year` (Task 4) -- import, do not re-implement (CLAUDE.md reuse rule). Confirm 2025 rows are present in the fitted sample.
- Fits `q(s)` (logistic, `q_logit` coeffs) and `a_s(s)` (non-negative low-order, `a_s_curve` coeffs) by maximum likelihood of realized SV under `NegBin(s*X, r=37.757)`, `X` the two-point mean-1 mixture, `a_m` derived from the constraint, with an `a_m,a_s >= 0` feasibility guard. `r` FIXED (not fit).
- Prints per-`s`-decile fitted `(q, a_m, a_s)` and the effective sample support of the vault tail (low `s`, rare high-SV events -- spec flags weak identifiability there).
- Emits the `SV_ROLE_MIXTURE` dict literal.

- [ ] **Step 1:** Write the script per the acceptance criteria.
- [ ] **Step 2:** Run: `python scripts/calibrate_closer_mixture.py`; inspect the report (a_s large at low s, ~0 at high s; a_m,a_s>=0; 2025 present).
- [ ] **Step 3:** Paste coefficients into `SV_ROLE_MIXTURE`. Re-run `pytest tests/test_sgp/test_closer_mixture.py -v` (invariants hold with real curves).
- [ ] **Step 4: Commit**
```bash
git add scripts/calibrate_closer_mixture.py src/fantasy_baseball/utils/constants.py
git commit -m "feat(calibrate): fit SV role-mixture curves from 2022-2025 (#193)"
```

---

### Task 6: Wire backtest SV variance + DECISION GATE

**Files:** Modify `scripts/backtest_sd_calibration.py` (inline SV var, line ~117)

- [ ] **Step 1: Branch SV to the mixture.** In `team_z`, for the `sv` key use `closer_mixture.sv_role_variance(proj)` summed over the team; W/K keep `negbin_perf_variance(key, proj) + proj**2 * cvp**2`. (Backtest is full-season; `sv_role_variance` is full-season -- no frac.)
- [ ] **Step 2: Run the backtest.** Run: `python scripts/backtest_sd_calibration.py`. Record SV `SD(z)` (MATCHED-ONLY and DNP=0); confirm R/HR/RBI/SB unchanged, W/SO in `[0.8,1.25]`.
- [ ] **Step 3: DECISION GATE.**
  - **SV `SD(z)` in `[0.8, 1.25]`** (both variants) -> two components suffice; skip Task 6b; commit; proceed to Task 7.
  - **Out of band** -> Task 6b (three components). Do NOT loosen `r` or tune curves to the gate.
- [ ] **Step 4: Commit**
```bash
git add scripts/backtest_sd_calibration.py
git commit -m "feat(backtest): wire SV variance to role mixture; decision gate (#193)"
```

---

### Task 6b (CONDITIONAL): three-component escalation

Only if Task 6's gate missed. Extend `closer_mixture` to a three-point `X` (hold / job-share / lose, with the vault path in the low-`s` curve): re-derive the closed-form per-component `within`/`between` (same law-of-total-variance, `r=37.757` all components), extend `role_multiplier_draw` to a 3-way `rng.choice`-style draw preserving `E[X]=1` and 2-D shape, add a third coefficient block to `SV_ROLE_MIXTURE`, re-run Task 5 calibration and Task 6 backtest until green. Update `test_closer_mixture.py` for the 3-point identity. Commit `feat(sgp): three-component role mixture (#193)`.

---

### Task 7: ERoto seam (`scoring.py`)

**Files:** Modify `scoring.py` (`player_category_variance`, SV term ~1283); Test `tests/test_scoring.py`

- [ ] **Step 1: Write the failing test**

```python
def test_sv_variance_uses_role_mixture():
    from fantasy_baseball.sgp import closer_mixture as cm
    from fantasy_baseball.scoring import player_category_variance
    from fantasy_baseball.utils.constants import Category
    player = _pitcher_fixture(sv=30, w=4, k=70, ip=65)     # existing test helper / dict
    out = player_category_variance(player)
    assert abs(out[Category.SV] - cm.sv_role_variance(30)) < 1e-6
```

- [ ] **Step 2: Run -> FAIL.**  Run: `pytest tests/test_scoring.py::test_sv_variance_uses_role_mixture -v`
- [ ] **Step 3: Implement.** Pull SV out of the shared `w/k/sv` loop: keep `w`,`k` on `negbin_perf_variance + v*v*cv_pt_sq`; set `result[Category.SV] = closer_mixture.sv_role_variance(_stat(player, "sv"))`. **No signature change** to `player_category_variance` / `project_team_sds` -- `sv_role_variance` is full-season and ERoto in-season scaling stays in `build_team_sds`. Leave `project_team_stats` (mean path) untouched.
- [ ] **Step 4: Run -> PASS**, plus `pytest tests/test_scoring.py -v` (W/K/hitters unregressed).
- [ ] **Step 5: Audit ALL `player_category_variance` / `project_team_sds` / `build_team_sds` call sites (CLAUDE.md "fix all call sites").** The wider SV SD propagates single-source-of-truth to every consumer -- no code changes needed, but confirm the behavior change (SV category outcomes become less certain) is intended on each:
  - `lineup/delta_roto.py:346-352` -- SV swap-band widths widen. Its `fraction_remaining * total` (line 352) scales the summed variance uniformly, composing correctly with the full-season mixture.
  - `models/standings.py:480` -- analytic **ProjectedStandings** (user-facing season dashboard): SV standings odds become less certain. This is the intended fix propagating; confirm it reads sane on the live dashboard during verification.
  - `web/refresh_pipeline.py` (~319/953/975/978), `draft/finalslate.py:250`, `draft/recs_integration.py:323` -- inherit the wider SV SD; no change required.
  Run the affected tests: `pytest tests/test_lineup/ -k "delta or stash" -v` then `pytest tests/test_scoring.py -v` (two separate commands -- a trailing positional after `-k` would be filtered by it). Note `tests/test_lineup/test_stash_value.py` calls `build_team_sds` (which wraps `project_team_sds`). If any test pins old SV-variance-derived widths/odds, update the expectation to the mixture value (a real behavior change, not a test bug) and note it in the commit.
- [ ] **Step 6: Commit**
```bash
git add src/fantasy_baseball/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): route ERoto SV variance through role mixture (#193)"
```

---

### Task 8: MC seam (`simulation.py`)

**Files:** Modify `simulation.py` (`_apply_variance_batch`, ~781-817); Test `tests/test_integration/test_monte_carlo_integration.py`

- [ ] **Step 1: Write the failing test** -- SV variance widens vs a mixture-disabled baseline while SV mean is stable (real assertions, NOT a stub).

```python
def test_mc_sv_variance_widens_mean_stable(monkeypatch):
    import numpy as np
    from fantasy_baseball import simulation
    from fantasy_baseball.sgp import closer_mixture
    from fantasy_baseball.models.player import PlayerType
    players = [_pitcher_dict(name=f"C{i}", sv=32, w=4, k=70, ip=65,
                             positions=["RP"]) for i in range(6)]
    def run():
        rng = np.random.default_rng(7)
        vb = simulation._apply_variance_batch(players, PlayerType.PITCHER, rng, n_iter=3000,
                                              fraction_remaining=1.0, suppress_repl=True)
        return vb.counts["sv"]     # VarianceBatch dataclass: .counts is {col: (n_iter, n_players)}
    mix = run()
    # baseline = mixture disabled by returning an all-ones multiplier (no between-component)
    monkeypatch.setattr(closer_mixture, "role_multiplier_draw",
                        lambda s, rng, fraction_remaining=1.0: np.ones(np.asarray(s).shape))
    base = run()
    # per-player SV variance strictly larger with the mixture (between-component added)
    assert mix.var(axis=0).mean() > 1.5 * base.var(axis=0).mean()
    # mean stable within tolerance (mean-neutral)
    assert abs(mix.mean() - base.mean()) / base.mean() < 0.03
```

Baseline via `monkeypatch` of `role_multiplier_draw` to all-ones (NOT a `_disable_sv_mixture`
kwarg on the production `_apply_variance_batch` signature -- keep test flags out of prod). With
ones, `mu0*X = base*eff_mean`, so both runs share the same SV mean and the mean-neutral
assertion holds exactly.

- [ ] **Step 2: Run -> FAIL.**  Run: `pytest tests/test_integration/test_monte_carlo_integration.py::test_mc_sv_variance_widens_mean_stable -v`
- [ ] **Step 3: Implement the SV-specific path.** After `scales`/`mu_mat` are built, for the `sv` column index (`PITCHER_IDX["sv"]`):
```python
sv_idx = idx_map["sv"]
s2d = np.broadcast_to(base["sv"][None, :], (n_iter, n_players))     # RAW projected SV
x = closer_mixture.role_multiplier_draw(s2d, rng, fraction_remaining)  # (n_iter, n_players)
mu_mat[:, :, sv_idx] = base["sv"][None, :] * eff_mean[None, :] * x   # mean rides eff_mean (NOT scales spread)
# r_mat[:, :, sv_idx] stays STAT_DISPERSION['sv']=37.757 (line 795, unchanged)
```
Leave the shared `frac_missed` and the `+8*frac_missed` backfill (811-814) untouched (mean-neutral). SV stays in the copula draw so its within co-moves with er/bb/h. No production test-flag kwarg -- the test disables the mixture by monkeypatching `role_multiplier_draw` to all-ones.
- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit**
```bash
git add src/fantasy_baseball/simulation.py tests/test_integration/test_monte_carlo_integration.py
git commit -m "feat(simulation): SV role-switch mixture in the MC sampler (#193)"
```

---

### Task 9: Parity + regression tests

**Files:** Test `tests/test_integration/test_closer_mixture_parity.py` (new)

- [ ] **Step 1: Shared-`mu0` invariant (Testing #1).** Re-assert at integration level: `sv_role_variance(mu0)` matches a generative brute-force `NegBin(mu0*X, r)` within the 2.4%-catching tolerance (already unit-tested in Task 1; re-assert here as the cross-path math guard).
- [ ] **Step 2: Per-path stability (Testing #3).** Each path's SV **mean** and **SD** move only by the intended mixture delta vs a mixture-disabled baseline -- NOT cross-path absolute equality (paths feed different `mu0`).
- [ ] **Step 3: In-season property (Testing #4).** `role_multiplier_draw` moments: `E[X']=1`, `Var(X') = frac*Var(X)` at `frac in {0.25,0.5,0.75}` (covered in Task 3; re-assert at integration level). No ERoto-side in-season test (external `build_team_sds`, unchanged).
- [ ] **Step 4: Valuation regression (Testing #5).** SGP/VAR/VONA SV values for a set of pitchers are unchanged (mean pipeline untouched).
- [ ] **Step 5: MC re-baseline (Testing #6).** Re-pin deterministic-seed MC test expectations for the added Bernoulli draw; document "only SV changes" is distributional (the extra RNG shifts the shared stream).
- [ ] **Step 6: Run all, commit.**
```bash
git add tests/test_integration/test_closer_mixture_parity.py tests/
git commit -m "test(sim): parity, valuation-regression, MC re-baseline for role mixture (#193)"
```

---

### Task 10: Full end-of-effort verification

- [ ] **Step 1:** `pytest -v` (or `-n auto`) -- all pass; fix code (not tests) on failure.
- [ ] **Step 2:** `ruff check .` -- zero violations.
- [ ] **Step 3:** `ruff format --check .` -- no drift (`ruff format .` to fix).
- [ ] **Step 4:** `vulture` -- no NEW dead-code findings.
- [ ] **Step 5:** `mypy` -- if any touched file is under `[tool.mypy].files`.
- [ ] **Step 6:** Re-run `python scripts/backtest_sd_calibration.py`; paste the final SV `SD(z)` line into the PR body; confirm R/HR/RBI/SB unchanged, W/SO/SV in `[0.8,1.25]`.
- [ ] **Step 7: Commit** any fixups; branch ready for PR.

---

## Spec-coverage self-check

Bimodal mixture / mean-1 / continuous curves / r-fixed -> Tasks 1-5. Full-season `sv_role_variance` + external in-season scaling (F3) -> Global Constraints, Tasks 1/7/8. ERoto seam -> Task 7. MC 2-D role draw (F1) -> Tasks 3/8. Calibration reads 2025 (F4 ordering) -> Tasks 4-5. Backtest wiring + 2025 admit + per-year cats -> Tasks 4/6. Decision gate + 3-component -> Tasks 6/6b. `delta_roto` call site (F6) -> Task 7 Step 5 + Files. Tests #1-#6 -> Tasks 1,3,9. Real MC test (F2) -> Task 8 Step 1. Known limitations -> `closer_mixture.py` docstring (Task 1). #235/#236 -> untouched.
