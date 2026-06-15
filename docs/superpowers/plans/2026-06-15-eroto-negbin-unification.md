# ERoto / Analytic Variance Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every per-stat variance consumer (the analytic ERoto/deltaRoto/standings engine and the pace color z-scores) derive from `STAT_DISPERSION` via one shared helper, then delete `STAT_VARIANCE` — so the MC and the analytic engines use the same dispersion by construction.

**Architecture:** Add `negbin_perf_variance(stat_key, mu) = mu + mu^2/r` and `negbin_perf_cv(stat_key, mu) = sqrt(1/mu + 1/r)` (r from `resolve_dispersion_r(STAT_DISPERSION[stat_key], mu)`) to `utils/dispersion.py`. Swap `scoring.py`'s flat-CV variance for `negbin_perf_variance` (counting: `+ mu^2*cv_pt^2`; rate SDs: `sqrt(sum perf_var(component))/denom`), swap the `pace.py`/`season_data.py` color z-scores to `negbin_perf_cv`, then remove `STAT_VARIANCE`.

**Tech Stack:** Python 3.12, numpy, scipy, pytest.

**Source spec:** `docs/superpowers/specs/2026-06-15-eroto-negbin-unification-design.md`

---

## Background the engineer needs

- `STAT_DISPERSION` (`utils/constants.py`): per-stat NegBin dispersion `r`, scalar or `(mu_upper, r)` bands, `float("inf")` == Poisson floor. `resolve_dispersion_r(value, mu)` (`utils/dispersion.py`) returns a per-element `r` array.
- The NegBin performance variance is `var = mu + mu^2/r` (r=inf -> `var = mu`). This is exactly the MC's `var_full` in `simulation.py::_negbin_copula_counts`.
- ERoto's current per-player variance (`scoring.py::player_category_variance`): counting `var = stat^2 * (STAT_VARIANCE[stat]^2 + cv_pt^2)`; rate components exposed as squared raw stat (`h_sq`, `er_sq`, `bb_sq`, `ha_sq`). `project_team_sds` sums them; rate SDs use `STAT_VARIANCE[...] * sqrt(sum stat^2) / denom`.
- `pace.py` / `season_data.py` color z-score: `z = (ratio - 1) / variance` with `variance = STAT_VARIANCE[stat]` (a flat per-stat CV).
- The full-season SDs are scaled by `sqrt(fraction_remaining)` in `build_team_sds` (unchanged; consistent with the MC mid-season — the spec accepts a late-season Poisson-floor divergence).
- Repo conventions: ASCII-only; no `x or default` for numerics; `pytest -n auto` runner; tools via `python -m ruff`/`mypy`/`vulture`.

---

## Phase 1: shared helpers (additive — no behavior change)

### Task 1: `negbin_perf_variance` + `negbin_perf_cv`

**Files:**
- Modify: `src/fantasy_baseball/utils/dispersion.py`
- Test: `tests/test_negbin/test_dispersion_resolver.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_negbin/test_dispersion_resolver.py`)

```python
def test_negbin_perf_variance_matches_mu_plus_mu2_over_r():
    import numpy as np
    from fantasy_baseball.utils.dispersion import negbin_perf_variance

    # k is scalar r=109.134 in STAT_DISPERSION; var = mu + mu^2/r
    mu = np.array([10.0, 100.0])
    out = negbin_perf_variance("k", mu)
    expected = mu + mu**2 / 109.134
    assert np.allclose(out, expected)


def test_negbin_perf_variance_poisson_floor_is_mu():
    import numpy as np
    from fantasy_baseball.utils.dispersion import negbin_perf_variance

    # h is float("inf") (Poisson) -> var == mu
    out = negbin_perf_variance("h", np.array([0.0, 50.0, 150.0]))
    assert np.allclose(out, [0.0, 50.0, 150.0])


def test_negbin_perf_variance_banded_uses_resolved_r():
    import numpy as np
    from fantasy_baseball.utils.dispersion import negbin_perf_variance, resolve_dispersion_r
    from fantasy_baseball.utils.constants import STAT_DISPERSION

    mu = np.array([2.0, 30.0])  # sb is banded
    r = resolve_dispersion_r(STAT_DISPERSION["sb"], mu)
    assert np.allclose(negbin_perf_variance("sb", mu), mu + mu**2 / r)


def test_negbin_perf_cv_is_sqrt_var_over_mu():
    import numpy as np
    from fantasy_baseball.utils.dispersion import negbin_perf_cv, negbin_perf_variance

    mu = np.array([5.0, 40.0])
    cv = negbin_perf_cv("sv", mu)
    assert np.allclose(cv, np.sqrt(negbin_perf_variance("sv", mu)) / mu)
    # Poisson floor: cv == sqrt(1/mu)
    assert np.allclose(negbin_perf_cv("h", mu), np.sqrt(1.0 / mu))
```

- [ ] **Step 2: Run, confirm fail** (ImportError: cannot import negbin_perf_variance)

Run: `python -m pytest tests/test_negbin/test_dispersion_resolver.py -k "perf" -v`

- [ ] **Step 3: Implement** — add to `src/fantasy_baseball/utils/dispersion.py`

Add the import at the top (verify NO circular import: `constants.py` must not import `dispersion` — it does not; `dispersion` is a leaf util):

```python
from fantasy_baseball.utils.constants import STAT_DISPERSION
```

Then append:

```python
def negbin_perf_variance(stat_key: str, mu: Any) -> np.ndarray:
    """Per-element NegBin performance variance ``mu + mu**2 / r``.

    r comes from ``resolve_dispersion_r(STAT_DISPERSION[stat_key], mu)``; an
    inf r (Poisson floor) yields ``var == mu``. This is the SAME quantity the
    MC's ``_negbin_copula_counts`` calls ``var_full`` -- the single source of
    truth for per-stat performance dispersion, shared by the MC and the
    analytic ERoto/pace engines. Conditional on realized playing time (callers
    add the playing-time variance separately for counting stats).
    """
    mu = np.asarray(mu, dtype=float)
    r = resolve_dispersion_r(STAT_DISPERSION[stat_key], mu)
    with np.errstate(divide="ignore"):
        overdispersion = np.where(np.isinf(r), 0.0, mu**2 / r)
    return np.asarray(mu + overdispersion, dtype=float)


def negbin_perf_cv(stat_key: str, mu: Any) -> np.ndarray:
    """Per-element performance CV ``sqrt(var)/mu == sqrt(1/mu + 1/r)`` (mu > 0).

    The multiplicative relative SD used by the pace color z-scores. Undefined at
    mu == 0 (callers guard expected/mu > 0).
    """
    mu = np.asarray(mu, dtype=float)
    var = negbin_perf_variance(stat_key, mu)
    return np.asarray(np.sqrt(var) / mu, dtype=float)
```

- [ ] **Step 4: Run, confirm 4 pass + full file green**

Run: `python -m pytest tests/test_negbin/test_dispersion_resolver.py -v` (expect all pass)
Also: `python -c "import fantasy_baseball.utils.dispersion, fantasy_baseball.simulation"` (no circular-import crash).

- [ ] **Step 5: Verify + commit**

Run: `python -m ruff check src/fantasy_baseball/utils/dispersion.py tests/test_negbin/test_dispersion_resolver.py && python -m ruff format --check src/fantasy_baseball/utils/dispersion.py && python -m mypy src/fantasy_baseball/utils/dispersion.py`

```bash
git add src/fantasy_baseball/utils/dispersion.py tests/test_negbin/test_dispersion_resolver.py
git commit -m "feat(sim): negbin_perf_variance/negbin_perf_cv shared dispersion helpers"
```

---

## Phase 2: migrate scoring.py (ERoto / deltaRoto / standings)

### Task 2: swap `player_category_variance` + `project_team_sds` to the NegBin helper

**Files:**
- Modify: `src/fantasy_baseball/scoring.py` (`player_category_variance` ~1213-1284, `project_team_sds` ~1287-1364, imports ~38)
- Test: `tests/test_scoring.py` (re-bless value-pins; add agree-by-construction)

- [ ] **Step 1: Add the agree-by-construction test** (append to `tests/test_scoring.py`)

```python
def test_project_team_sds_counting_variance_equals_negbin_helper_sum():
    # IMPORTANT: reconstruct cv_pt via the SAME volume function the impl uses
    # (_full_season_volume), not p["pa"] directly -- otherwise a divergence in
    # how the volume is derived would make this a false failure.
    from fantasy_baseball.scoring import _full_season_volume, project_team_sds
    from fantasy_baseball.models.player import PlayerType
    from fantasy_baseball.utils.dispersion import negbin_perf_variance
    from fantasy_baseball.utils.playing_time import playing_time_params
    from fantasy_baseball.utils.constants import Category

    roster = [
        {"player_type": PlayerType.HITTER, "name": "H", "pa": 600, "ab": 540,
         "r": 90, "hr": 25, "rbi": 85, "sb": 15, "h": 150},
        {"player_type": PlayerType.HITTER, "name": "H2", "pa": 500, "ab": 450,
         "r": 70, "hr": 12, "rbi": 60, "sb": 30, "h": 130},
    ]
    sds = project_team_sds(roster, displacement=False)
    # Rebuild SB variance from the shared helper + PT term; SD must match.
    exp_var = 0.0
    for p in roster:
        v = float(p["sb"])
        cv_pt = playing_time_params(PlayerType.HITTER, _full_season_volume(p, True))[1]
        exp_var += float(negbin_perf_variance("sb", v)) + v * v * cv_pt**2
    assert abs(sds[Category.SB] - exp_var**0.5) < 1e-9
```

- [ ] **Step 2: Run, confirm it FAILS** (old flat-CV variance != NegBin sum)

Run: `python -m pytest tests/test_scoring.py::test_project_team_sds_counting_variance_equals_negbin_helper_sum -v`

- [ ] **Step 3: Migrate `player_category_variance`** — replace its body's variance math.

Imports: remove `STAT_VARIANCE` from the `from ...constants import (...)` block (~line 38); add `from fantasy_baseball.utils.dispersion import negbin_perf_variance`.

Replace the counting + rate-component logic:

```python
    if ptype == PlayerType.HITTER:
        cv_pt_sq = playing_time_params(PlayerType.HITTER, _full_season_volume(player, True))[1] ** 2
        for stat_key, cat in [
            ("r", Category.R),
            ("hr", Category.HR),
            ("rbi", Category.RBI),
            ("sb", Category.SB),
        ]:
            v = _stat(player, stat_key)
            result[cat] = float(negbin_perf_variance(stat_key, v)) + v * v * cv_pt_sq
        # Rate-assembly: per-component PERFORMANCE variance (cv_pt cancels in a rate).
        result["h_var"] = float(negbin_perf_variance("h", _stat(player, "h")))
        result["ab"] = _stat(player, "ab")

    elif ptype == PlayerType.PITCHER:
        cv_pt_sq = (
            playing_time_params(PlayerType.PITCHER, _full_season_volume(player, False))[1] ** 2
        )
        for stat_key, cat in [
            ("w", Category.W),
            ("k", Category.K),
            ("sv", Category.SV),
        ]:
            v = _stat(player, stat_key)
            result[cat] = float(negbin_perf_variance(stat_key, v)) + v * v * cv_pt_sq
        result["er_var"] = float(negbin_perf_variance("er", _stat(player, "er")))
        result["bb_var"] = float(negbin_perf_variance("bb", _stat(player, "bb")))
        result["ha_var"] = float(negbin_perf_variance("h_allowed", _stat(player, "h_allowed")))
        result["ip"] = _stat(player, "ip")
```

Also update the function's docstring: counting `var = negbin_perf_variance(stat, mu) + mu^2*cv_pt^2`; rate components now expose `h_var`/`er_var`/`bb_var`/`ha_var` (the per-component NegBin performance variance), not the squared raw stat.

- [ ] **Step 4: Migrate `project_team_sds`** — consume the `_var` keys.

Replace the rate-assembly sum dicts + the rate-SD block:

```python
    # Rate-assembly PERFORMANCE-variance sums (playing-time-invariant).
    h_sum_var = 0.0
    p_sum_var: dict[str, float] = {k: 0.0 for k in ("er", "bb", "h_allowed")}
    total_ab = 0.0
    total_ip = 0.0
```

In the hitter branch: `h_sum_var += contrib.get("h_var", 0.0)` (was `h_sum_sq["h"] += contrib.get("h_sq", 0.0)`).
In the pitcher branch:
```python
            p_sum_var["er"] += contrib.get("er_var", 0.0)
            p_sum_var["bb"] += contrib.get("bb_var", 0.0)
            p_sum_var["h_allowed"] += contrib.get("ha_var", 0.0)
```

Rate SDs (the `STAT_VARIANCE[...]` factors are now inside the per-component variance):
```python
    if total_ab > 0:
        sds[Category.AVG] = sqrt(h_sum_var) / total_ab
    if total_ip > 0:
        sds[Category.ERA] = 9.0 * sqrt(p_sum_var["er"]) / total_ip
        whip_var = p_sum_var["bb"] + p_sum_var["h_allowed"]
        sds[Category.WHIP] = sqrt(whip_var) / total_ip
```

The counting-category SDs (`sds[cat] = sqrt(h_var[cat])` / `sqrt(p_var[cat])`) are UNCHANGED — `h_var`/`p_var` now hold the NegBin-based per-player variance sums. Update the docstring's formula block to the NegBin form.

- [ ] **Step 5: Run the agree-by-construction test (now PASS) + scoring suite**

Run: `python -m pytest tests/test_scoring.py -v 2>&1 | tail -25`
Expected: the new test passes; some value-pinned SD/score tests FAIL (the SDs legitimately changed) -> handled in Step 6.

- [ ] **Step 6: Re-bless value-pinned scoring tests**

For each failing assertion that pins a specific SD or roto/deltaRoto score: confirm the new value is explained by the dispersion switch (NegBin adds the Poisson term + uses banded r), NOT a structural break (missing key, crash, wrong category). Re-pin the expected number; keep tolerances as tight as before. Do NOT loosen to hide an unexplained shift. Document the model switch in the commit.

- [ ] **Step 7: Verify + commit**

Run: `python -m pytest tests/test_scoring.py -q && python -m ruff check src/fantasy_baseball/scoring.py && python -m ruff format --check src/fantasy_baseball/scoring.py && python -m mypy src/fantasy_baseball/scoring.py`

```bash
git add src/fantasy_baseball/scoring.py tests/test_scoring.py
git commit -m "feat(sim): ERoto/standings variance uses NegBin dispersion (drop STAT_VARIANCE in scoring)"
```

---

## Phase 3: migrate pace.py + season_data.py color z-scores

### Task 3: swap the pace color z-score CV to `negbin_perf_cv`

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py` (~135-152, imports)
- Modify: `src/fantasy_baseball/web/season_data.py` (~793-825, imports)
- Test: `tests/test_analysis/test_pace.py` + season_data tests (re-bless)

- [ ] **Step 0: Confirm the stat keys are valid STAT_DISPERSION keys** (the
  migration changes `STAT_VARIANCE.get(key, 0.0)` (tolerant) to
  `negbin_perf_cv(key, ...)` which does `STAT_DISPERSION[key]` (KeyError on an
  unknown key). Before editing, read the loop variables and the `rate_cats`
  definition in both files and list, for each call site, the exact key passed:
  - `pace.py`: the `counting` stat keys.
  - `season_data.py` counting: `cat.value.lower()` for the counting cats.
  - `season_data.py` rate: the `component` value from `rate_cats` (the most
    likely mismatch -- e.g. WHIP reduced to one component).
  Valid keys are: `r, hr, rbi, sb, h, w, k, sv, er, bb, h_allowed`. For counting
  keys (known-valid) no guard is needed, but confirm. For the rate `component`,
  the rate branch is guarded with `component in STAT_DISPERSION` (Step 3) so an
  unmapped component degrades to z=0 (old behavior) rather than crashing. Report
  the key list so the reviewer can confirm.

- [ ] **Step 1: pace.py** — replace the flat-CV with the NegBin CV.

Imports: remove `from ...constants import ... STAT_VARIANCE` (or the specific import); add `from fantasy_baseball.utils.dispersion import negbin_perf_cv`.

Replace:
```python
            ratio = actual / expected
            variance = STAT_VARIANCE.get(stat, 0.0)
            z = (ratio - 1.0) / variance if variance > 0 else 0.0
```
with:
```python
            ratio = actual / expected
            cv = float(negbin_perf_cv(stat, expected))  # expected > 0 guaranteed above
            z = (ratio - 1.0) / cv if cv > 0 else 0.0
```

- [ ] **Step 2: season_data.py counting branch** — same swap.

Imports: add `from fantasy_baseball.utils.dispersion import negbin_perf_cv` AND
`from fantasy_baseball.utils.constants import STAT_DISPERSION` (the rate branch in
Step 3 needs `STAT_DISPERSION` for its `component in STAT_DISPERSION` guard);
remove the `STAT_VARIANCE` import.

Replace the counting branch:
```python
            ratio = actual / expected
            variance = STAT_VARIANCE.get(cat.value.lower(), 0.0)
            z = (ratio - 1.0) / variance if variance > 0 else 0.0
```
with:
```python
            ratio = actual / expected
            cv = float(negbin_perf_cv(cat.value.lower(), expected))
            z = (ratio - 1.0) / cv if cv > 0 else 0.0
```

- [ ] **Step 3: season_data.py rate branch** — recover the component count, then NegBin CV.

The rate branch has `weighted = sum(v * opp for v, opp in proj_vals)` (the projected numerator total) and `component` (the underlying stat key, e.g. "er"). Recover the component COUNT and use its NegBin CV. **Guard the key lookup**: the old code used `STAT_VARIANCE.get(component, 0.0)` (an unknown key -> 0 -> z=0, no coloring), but `negbin_perf_cv` does `STAT_DISPERSION[component]` (a hard index that KeyErrors on an unknown key). Preserve the old skip-on-unknown behavior with an `in STAT_DISPERSION` guard (import `STAT_DISPERSION` into season_data.py for it):
```python
        if expected_val > 0 and actual_val > 0:
            # weighted is the projected rate-numerator total: ERA*IP = 9*er,
            # WHIP*IP = bb+ha, AVG*PA ~ h. Recover the component count for the
            # NegBin CV; guard unknown component / degenerate zero (preserves the
            # old STAT_VARIANCE.get(component, 0.0) tolerance -> z=0).
            component_count = weighted / 9.0 if rate_cat == Category.ERA else weighted
            if component in STAT_DISPERSION and component_count > 0:
                cv = float(negbin_perf_cv(component, component_count))
            else:
                cv = 0.0
            z = (actual_val - expected_val) / (cv * expected_val) if cv > 0 else 0.0
            if is_inverse:
                z = -z
        else:
            z = 0.0
```
(If `component` for AVG is the hits key, `weighted` is the projected hits-ish total -- the PA-vs-AB factor is a small display-only approximation, acceptable per spec.)

- [ ] **Step 4: Run + re-bless**

Run: `python -m pytest tests/test_analysis/test_pace.py tests/test_web -q 2>&1 | tail -20`
Re-bless any value-pinned z-score / color-class assertions that legitimately change (the CV moved from flat 0.715-style to mu-dependent). Same discipline as Task 2 Step 6.

- [ ] **Step 5: Verify + commit**

Run: `python -m ruff check src/fantasy_baseball/analysis/pace.py src/fantasy_baseball/web/season_data.py && python -m ruff format --check src/fantasy_baseball/analysis/pace.py src/fantasy_baseball/web/season_data.py && python -m mypy src/fantasy_baseball/analysis/pace.py src/fantasy_baseball/web/season_data.py`

```bash
git add src/fantasy_baseball/analysis/pace.py src/fantasy_baseball/web/season_data.py tests/test_analysis/test_pace.py tests/test_web
git commit -m "feat(sim): pace color z-scores use NegBin CV (drop STAT_VARIANCE in pace/season_data)"
```

---

## Phase 4: delete STAT_VARIANCE + validate

### Task 4: remove the constant + draft/sim re-bless + eye-test

**Files:**
- Modify: `src/fantasy_baseball/utils/constants.py` (delete `STAT_VARIANCE`)
- Modify/Note: `scripts/backtest_sd_calibration.py`, `scripts/calibrate_variance.py`, any remaining readers
- Modify: draft/recommend/simulate tests as needed (re-bless)

- [ ] **Step 1: Grep all remaining `STAT_VARIANCE` readers**

Run: `python -m ruff check . 2>&1 | head` then search:
PowerShell: `Select-String -Path (Get-ChildItem -Recurse -Filter *.py src,scripts,tests) -Pattern 'STAT_VARIANCE' | ForEach-Object { "$($_.Path):$($_.LineNumber)" }`
List every hit. Each must be either migrated (use the helper) or, for a calibration script whose sole purpose was the OLD constant, noted as superseded by `scripts/calibrate_stat_dispersion.py`.

- [ ] **Step 2: Handle the script + test readers**

- `scripts/calibrate_variance.py` (calibrated the old `STAT_VARIANCE`): it is superseded by `scripts/calibrate_stat_dispersion.py`. If it imports `STAT_VARIANCE` only as an output target, leave it but it will break on the deleted constant -> either delete the script or update it to read `STAT_DISPERSION`. Decide per its content; default: leave a one-line module docstring note that it is superseded and remove the now-broken `STAT_VARIANCE` reference (or delete the script if it has no other use). State which you did.
- `scripts/backtest_sd_calibration.py`: same triage.
- Any test asserting `STAT_VARIANCE` contents: delete/replace those assertions (the constant is gone).

- [ ] **Step 3: Delete `STAT_VARIANCE`** from `constants.py` (the per-stat Gaussian-sigma dict). Keep `STAT_DISPERSION`.

- [ ] **Step 4: Run the draft/recommend/simulate suites + re-bless**

Run: `python -m pytest tests/test_draft tests/test_web tests/test_scoring.py tests/test_analysis -n auto -q -p no:cacheprovider --ignore=tests/test_draft/test_simulate_draft.py 2>&1 | tail -20`
deltaRoto draft scores depend on the SDs, so value-pinned recommend/score tests may shift. Re-bless those explained by the dispersion switch (per Task 2 Step 6 discipline). Confirm `python -c "import fantasy_baseball.utils.constants"` has no `STAT_VARIANCE` reference errors anywhere.

- [ ] **Step 5: Eye-test — confirm ERoto now agrees with the MC** (read-only prod)

Write a temp script (delete after; do NOT commit) that, READ-ONLY against prod via `build_explicit_upstash_kv()` (no writes, no refresh, no `RENDER=true`), loads Hart of the Order's roster + projections and compares, per category: ERoto's `project_team_sds` SD vs the MC's sampled team SD (run `run_ros_monte_carlo` / aggregate `_apply_variance`). Confirm they now match (especially SV: the prior ~3x analytic-vs-MC gap should be ~gone). Save a short markdown to `docs/superpowers/eye-test-eroto-unified-2026-06-15.md` and report the per-category SD comparison.

- [ ] **Step 6: FORCED final checklist + commit**

Run at repo root:
- `python -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_draft/test_simulate_draft.py 2>&1 | tail -5` (bare collection works; all pass)
- `python -m ruff check .` (only pre-existing untracked-scratch hits, if any)
- `python -m ruff format --check src/fantasy_baseball` 
- `python -m vulture src/fantasy_baseball/utils/dispersion.py src/fantasy_baseball/scoring.py` (no new dead code)
- `python -m mypy` (the touched src files clean)

```bash
git add src/fantasy_baseball/utils/constants.py scripts/ tests/ docs/superpowers/eye-test-eroto-unified-2026-06-15.md
git commit -m "feat(sim): delete STAT_VARIANCE -- one dispersion source of truth (STAT_DISPERSION)"
```

---

## Self-review notes (author)

- **Spec coverage:** helpers (Task 1) <- spec sec.1; scoring counting+rate migration (Task 2) <- spec sec.2 + agree-by-construction test; pace/season_data incl. rate-component recovery (Task 3) <- spec sec.3; delete STAT_VARIANCE + draft/sim re-bless + eye-test (Task 4) <- spec sec.4 + Testing + downstream-impact. fraction_remaining/Poisson-floor: build_team_sds unchanged (spec accepts late-season divergence) -- no task needed, noted in Background.
- **Type consistency:** `negbin_perf_variance(stat_key, mu) -> ndarray`, `negbin_perf_cv(stat_key, mu) -> ndarray`; `player_category_variance` rate keys renamed `h_sq/er_sq/bb_sq/ha_sq` -> `h_var/er_var/bb_var/ha_var` and `project_team_sds` reads the new keys (`h_sum_var`, `p_sum_var`) consistently.
- **Known caveat:** season_data AVG component-count recovery is approximate (PA-vs-AB); display-only, accepted per spec. The pre-existing `test_simulate_draft.py` native segfault stays `--ignore`d.
