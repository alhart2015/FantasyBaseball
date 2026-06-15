# ERoto / Analytic Variance Unification onto the NegBin Dispersion

Date: 2026-06-15
Status: Approved (brainstorm); pending implementation plan
Branch: eroto-negbin-unification (stacked on june-13-group-review / PR #139)

## Problem

The NegBin counting-stat sampler (PR #139) switched the Monte Carlo's per-stat
variance to a Negative-Binomial dispersion calibrated from 2022-2024 actuals
(`STAT_DISPERSION`, mu-banded for SB/HR/RBI/ER, role-stable scalar for SV). But
the ANALYTIC engines were left on the old flat per-stat Gaussian CV constant
`STAT_VARIANCE`:

- `scoring.py` (ERoto / deltaRoto / projected standings) computes per-player and
  team-level category SDs from `STAT_VARIANCE`.
- `analysis/pace.py` and `web/season_data.py` use `STAT_VARIANCE` as a flat
  per-stat CV to standardize a pace ratio into a color z-score for the dashboard.

So the project now has **two sources of truth for per-stat variance**: the MC
uses `STAT_DISPERSION` (NegBin), the analytic band uses `STAT_VARIANCE`
(Gaussian CV). They diverge most on the banded low-mu/high-sigma stats. The
remote eye-test (`docs/superpowers/eye-test-negbin-remote-2026-06-15.md`) made
this concrete for saves: ERoto's flat `sv` sigma is 0.900 (CV ~0.90) while the
NegBin implies CV ~0.31 at a typical closer mu -- the analytic deltaRoto band
treats saves as roughly 3x more uncertain than the standings MC does.

This is the "different models / one source of truth" problem the user flagged:
the dashboard's displayed deltaRoto band and the MC standings should describe the
same dispersion, not coincidentally-similar ones.

## Goals

- **One source of truth for per-stat dispersion.** All variance consumers (MC,
  ERoto/deltaRoto/standings, pace coloring) derive from `STAT_DISPERSION` via a
  single shared helper. `STAT_VARIANCE` is deleted.
- The analytic ERoto SDs and the MC's per-stat variance **agree by
  construction** (same formula + constant), not by coincidence.
- Preserve the existing analytic structure that is correct: per-player
  independence sum, playing-time variance added in quadrature for counting stats,
  playing-time cancelling out of rate stats.

## Non-goals

- **Closer role-switch mixture for SV.** ERoto adopts the SAME role-stable NegBin
  SV dispersion the MC uses (user-decided). ERoto's SV band tightens to match the
  MC. The "setup man becomes the closer" risk is the SAME documented, deferred
  limitation that already applies to the MC -- a future role-mixture spec handles
  it for both engines at once. We do NOT add a per-engine SV width fudge (that
  would re-break one-source-of-truth).
- **The ERoto mean (mean_scale haircut).** ERoto is known to omit the playing-time
  `mean_scale` haircut the MC applies to projection MEANS (see memory
  `project_eroto_mc_mean_inconsistency_2026_06_09`; it is tiny mid-season). This
  spec unifies VARIANCE only; the mean question is separate and out of scope.
- **A fully team-level SD for the season_data team color z-score** (see the pace
  section) -- a possible future refinement, kept out of scope as display-only.
- Changing the MC sampler (`simulation.py`) behavior. It already consumes
  `STAT_DISPERSION` via `resolve_dispersion_r`; this spec only ensures the
  analytic engines consume the same source.

## Key decisions (settled in brainstorm)

1. **Role-stable SV everywhere** -- ERoto uses `STAT_DISPERSION["sv"]` exactly as
   the MC does. (Non-goal above.)
2. **Full migration + delete `STAT_VARIANCE`** -- migrate all three consumers
   (scoring.py, pace.py, season_data.py), then remove the constant. Literal one
   source of truth.

## Architecture

### 1. Shared dispersion-variance helpers (the single source)

In `src/fantasy_baseball/utils/dispersion.py` (beside `resolve_dispersion_r`):

```
negbin_perf_variance(stat_key, mu) -> ndarray
    r   = resolve_dispersion_r(STAT_DISPERSION[stat_key], mu)
    var = mu + mu**2 / r            # r == inf (Poisson floor) -> var = mu

negbin_perf_cv(stat_key, mu) -> ndarray
    # relative SD of the performance count: sqrt(var)/mu = sqrt(1/mu + 1/r)
    # (the multiplicative CV the pace color z-score wants)
```

- `mu` may be a scalar or an ndarray; the helpers vectorize (mirroring
  `resolve_dispersion_r`). `negbin_perf_variance` returns `mu` when `r` is the
  Poisson sentinel (`mu**2/inf = 0`). `negbin_perf_cv` returns `sqrt(1/mu)` (the
  Poisson floor CV) when `r` is inf, and is defined only for `mu > 0` (callers
  already guard `expected > 0` / `mu > 0`).
- `negbin_perf_variance(stat_key, mu)` is exactly the `var_full` the MC's
  `_negbin_copula_counts` computes (`mu + mu^2/r`). The MC and the analytic
  engines now share this formula AND `STAT_DISPERSION` AND `resolve_dispersion_r`
  -- the single source of truth. (The MC keeps computing `var_full` inline from
  its already-resolved `r_mat`; it is the identical quantity. A test pins the
  equality so the two cannot drift.)

This is the **conditional-on-PT performance** variance (the `r` was calibrated on
rate residuals at realized PT). The playing-time variance is added separately by
the consumers (below), exactly as `STAT_VARIANCE` + `cv_pt` were combined -- so
there is no double-count of playing-time variance.

### 2. ERoto / deltaRoto / standings (`scoring.py`)

`player_category_variance` and `project_team_sds` swap the flat-CV variance for
the NegBin form. `mu` is the player's projected stat (`_stat(player, key)`), the
same value used today.

- **Counting** (R/HR/RBI/SB for hitters; W/K/SV for pitchers):
  per-player `var_cat = negbin_perf_variance(stat, mu) + mu**2 * cv_pt_sq`.
  Team `SD_cat = sqrt(sum_i var_cat_i)` (player-independence sum, unchanged).
  vs old `mu**2 * (STAT_VARIANCE[stat]**2 + cv_pt_sq)` -- the new form adds the
  Poisson term `mu` and replaces `STAT_VARIANCE**2` with the mu-banded `1/r`.
- **Rate** (AVG/ERA/WHIP): generalize `CV * sqrt(sum stat_i^2)` to
  `sqrt(sum negbin_perf_variance(component, mu_i))` (playing-time NOT added --
  it cancels in a rate, unchanged):
  - `SD_AVG  = sqrt(sum perf_var(h_i))  / total_ab`   (h is Poisson -> perf_var = mu_h)
  - `SD_ERA  = 9 * sqrt(sum perf_var(er_i)) / total_ip`
  - `SD_WHIP = sqrt(sum [perf_var(bb_i) + perf_var(ha_i)]) / total_ip`
  `player_category_variance` returns the per-component `negbin_perf_variance`
  (e.g. `"h_var"`, `"er_var"`, `"bb_var"`, `"ha_var"`) instead of the squared
  raw stat (`"h_sq"`, ...); `project_team_sds` sums those and divides by the
  team denominator. The `total_ab`/`total_ip` denominators are unchanged.
- Remove the `STAT_VARIANCE` import from `scoring.py`.

### 3. Pace coloring (`pace.py` per-player, `season_data.py` team)

Both compute `z = (ratio - 1) / variance` where `variance` was
`STAT_VARIANCE[stat]` (a flat per-stat CV). Replace with `negbin_perf_cv`:

- `pace.py` (per-player pace z-score): `cv = negbin_perf_cv(stat, expected)` at
  the player's expected count. A per-player CV for a per-player z-score --
  semantically clean and now mu-aware.
- `season_data.py` (team totals z-score, counting): `cv = negbin_perf_cv(stat,
  expected)` at the team's expected total. The rate-category branch
  (`z = (actual - expected) / (cv * expected)`) uses `negbin_perf_cv(component,
  expected_component)`. This preserves the current "per-stat CV on a team ratio"
  structure, just sourced from the unified model.
  - **Known limitation (documented, not fixed here):** a team total's true
    relative SD is much tighter than a single-mu CV (it is the
    `project_team_sds`-style sum). Using `negbin_perf_cv` at the team total keeps
    the same rough heuristic the flat CV had (slightly tighter), and removes the
    `STAT_VARIANCE` dependency. A fully team-level SD for team coloring is a
    deferred display-only refinement.

### 4. Delete `STAT_VARIANCE`

After all three consumers migrate, delete `STAT_VARIANCE` from `constants.py`.
Grep src/ + scripts/ + tests/ to confirm no remaining readers before deletion
(known readers besides the three above: `scripts/backtest_sd_calibration.py`,
`scripts/calibrate_variance.py`, and tests). Migrate or update those readers too;
if a script's sole purpose was calibrating the old `STAT_VARIANCE`, note it
(it is superseded by `scripts/calibrate_stat_dispersion.py`).

## Edge cases / failure modes

- **`r = inf` (Poisson floor):** `negbin_perf_variance -> mu`, `negbin_perf_cv ->
  sqrt(1/mu)`. Affects h, w, and any banded cell at its Poisson floor. Correct
  (a count's variance floors at its mean).
- **`mu = 0`:** `negbin_perf_variance(.., 0) = 0` (a 0-projection contributes no
  variance). `negbin_perf_cv` is undefined at `mu = 0` (div-by-zero); callers
  must guard `mu > 0` / `expected > 0` (pace and season_data already do; the
  scoring rate-SD sums skip zero denominators via `total_ab/ip > 0`).
- **Banded `r` resolution uses the player projection `mu`**, consistent with the
  MC (which bands on `base*scale`); the analytic mu = projection is the
  full-season expectation, the right band key.
- **No PT double-count:** `negbin_perf_variance` is the conditional-on-PT
  performance variance; consumers add `cv_pt` separately for counting stats and
  omit it for rates -- identical to the old `STAT_VARIANCE + cv_pt` composition.

## Testing expectations

- **Unit:** `negbin_perf_variance` == `mu + mu^2/r` and `== mu` at `r=inf`;
  `negbin_perf_cv` == `sqrt(1/mu + 1/r)` and `== sqrt(1/mu)` at `r=inf`; both
  vectorize over a `mu` array; `mu=0` variance is 0.
- **Agree-by-construction (the load-bearing guarantee):** a test that, for a
  representative roster, ERoto's `project_team_sds` counting-category variance
  equals `sum_i (negbin_perf_variance(stat, mu_i) + mu_i^2 * cv_pt_i^2)`, AND
  that `negbin_perf_variance(stat, mu)` equals the MC's `var_full` for the same
  `(stat, mu)` (i.e. the analytic and MC dispersion are the same number). This is
  what "one source of truth" must verify.
- **Regression re-bless:** value-pinned SD/z-score tests in `test_scoring.py`,
  `tests/test_analysis/test_pace.py`, and the season_data tests WILL change (the
  SDs/CVs legitimately move). Re-pin them against the new model, citing this
  spec, per CLAUDE.md's don't-edit-tests-without-justification rule. Split
  structural assertions (must pass unchanged) from value-pins (re-blessed).
- **Eye-test:** re-run the ERoto-vs-MC per-category SD comparison on the user's
  prod roster (read-only, `build_explicit_upstash_kv`); confirm the analytic
  band now matches the MC (especially SV: the ~3x gap should close). Save the
  comparison.
- **Forced checklist:** pytest (bare `pytest -v` must collect+pass now that the
  test-package collision is fixed), ruff, ruff format, vulture, mypy on the
  touched src files (scoring.py, dispersion.py, constants.py, pace.py,
  season_data.py are likely under `[tool.mypy].files` -- check).

## Internal phasing

Each phase leaves the tree green.

1. **Helpers (additive).** Add `negbin_perf_variance` + `negbin_perf_cv` to
   `utils/dispersion.py` with unit tests. No consumer change; `STAT_VARIANCE`
   stays. Tree green.
2. **scoring.py migration.** Switch `player_category_variance` /
   `project_team_sds` to the helpers; re-bless `test_scoring.py` value-pins. Add
   the agree-by-construction test. (`STAT_VARIANCE` still present for pace/
   season_data.)
3. **pace.py + season_data.py migration.** Switch the color z-scores to
   `negbin_perf_cv`; re-bless `test_pace.py` + season_data value-pins.
4. **Delete `STAT_VARIANCE`** (after grep confirms all readers migrated; handle
   the calibration scripts) + the eye-test validation. This phase must land with
   2-3 since deleting the constant requires every consumer migrated.

## Open items to resolve in planning

- Confirm the exact set of `STAT_VARIANCE` readers (grep) and the disposition of
  `scripts/backtest_sd_calibration.py` / `scripts/calibrate_variance.py` (migrate
  vs note-as-superseded).
- Confirm `negbin_perf_cv` at the team total is the chosen season_data behavior
  (vs deferring season_data's team coloring) -- the spec chooses the team-total
  CV swap; flag if the planner finds it changes coloring more than expected.
