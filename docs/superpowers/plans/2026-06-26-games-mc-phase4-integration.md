# Games-based MC -- Phase 4 (ROS-direct hitter integration: displacement + fill + horizon split) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Wire the Phase 2 setup (`mc_roster.build_effective_roster` -> `EffectiveRoster`) and the Phase 3 allocator (`mc_fill.allocate_bench_fill`) into the HITTER path of `simulate_remaining_season_batch`, sampling ROS production DIRECTLY (not full-season-minus-YTD). Apply each active body's displacement `factor` to its baseline before sampling, compute `frac_missed`, fill bench-then-replacement, sum to team ROS, and blend `team_total = team_YTD + summed_ROS`. PITCHERS stay on the existing full-season path UNCHANGED (Phase 5). The scalar and draft paths stay UNCHANGED. Ship the before/after artifact (definition of done).

**Architecture seam (the riskiest phase).** This rewrites the core MC batch. Three independent pieces, three commits:

- **4a -- sampler plumbing (isolated).** Expose `frac_missed` and the per-body sampled counting matrix OUT of `_apply_variance_batch`, and add an opt-in `suppress_repl` flag that zeroes the built-in `repl_contrib` on the new path. Pure-additive: existing callers (scalar `simulate_season` via `_apply_variance`, the top-k path inside the batch, draft) are byte-for-byte unaffected because the new outputs are returned alongside the old dict and the flag defaults to keeping `repl_contrib`. Small, its own tests.
- **4b -- ROS-direct hitter integration + setup wiring + fill.** Build each team's `LeagueContext` + `EffectiveRoster` at MC setup (`run_ros_monte_carlo` / pipeline `_run_ros_monte_carlo`), thread it into the batch, and route hitters through a NEW body-direct helper `_simulate_team_hitters_ros_direct` (fixed active set + displacement + bench fill + ROS-direct blend) when `effective_rosters` is provided -- bypassing the flat-dict `active_cols` column mechanism entirely (the C2 fix). Resolve the `fraction_remaining` horizon split. Also patch the refresh-test fixture wrapper to accept/forward `effective_rosters=None`. The heavy change.
- **4c -- before/after evidence.** Extend `mc_selection.py` + the gated pipeline hook with a 4th arm (NEW engine) reporting all 10 categories + overall standings vs OLD top-k / fixed-topk / active-slot. Requires a Yahoo-auth'd local refresh (controller runs it).

**Tech Stack:** Python, pytest. Touched: `src/fantasy_baseball/simulation.py`, `src/fantasy_baseball/mc_selection.py`, `tests/test_web/_refresh_fixture.py` (one wrapper-signature line); new test files `tests/test_mc_integration.py` (4a + 4b), additions to `tests/test_mc_selection.py` (4c). Reuses `mc_roster.{build_effective_roster, EffectiveRoster, ActiveBody, BenchBody, PA_PER_GAME}`, `mc_fill.{ActiveSample, BenchSample, allocate_bench_fill}`, `scoring.{LeagueContext}`, and `simulation._replacement_line` (module-local at simulation.py:435; `REPLACEMENT_BY_POSITION` already imported at simulation.py:28).

**The central architecture decision (read first -- it is the rewrite).** The NEW-engine HITTER path operates DIRECTLY on the `EffectiveRoster` bodies via a new self-contained helper `_simulate_team_hitters_ros_direct(...)`. It does NOT reuse the Phase-0 `active_cols` flat-dict COLUMN mechanism. The reason this matters: `EffectiveRoster.active` is `[*il, *active]` (verified mc_roster.py:85) -- REORDERED, BENCH-EXCLUDED, and containing BOTH player types -- whereas the flat-dict `active_cols` index the roster-order, BENCH-INCLUDED, hitter-only sublist (`hitters = [p for p in players ...]`, simulation.py:752). Mapping bodies onto those columns by position MISALIGNS silently, and `active_cols` carry no factor/`g_ros_adj`/bench-pool information the new path needs. So the new path samples ITS OWN hitter bodies (active hitters + bench hitters from the EffectiveRoster) and never touches `active_cols`. `active_cols` stays the Phase-0 DIAGNOSTIC path only (the three OLD arms in `mc_selection.py`). `Player.to_flat_dict()` (verified player.py:293-295; overlays `rest_of_season`) is a Player METHOD -- it is called on the live `BenchBody.player` / `ActiveBody.player` objects, never on a flat dict.

## Global Constraints

- ASCII-only in source/strings. Numeric defaults via `is not None`, never `x or default` (the falsy-zero footgun). Especially in sort keys / index lookups.
- All imports at top of module.
- Spec: `docs/superpowers/specs/2026-06-26-games-based-availability-mc-design.md` -- Component 4 (MC integration, ROS-direct), the "Missed playing-time accounting (single authority)" section, and "Acceptance evidence (before/after)". Binding.
- `simulation.py` IS under `[tool.mypy].files` (verified pyproject.toml:93; `mc_roster.py`:89, `mc_fill.py`:88 also present). `mc_selection.py` is NOT yet listed (verified absent) -- ADD `"src/fantasy_baseball/mc_selection.py"` to `[tool.mypy].files` in 4c and run mypy on it. Run mypy on every touched file.
- **Do NOT regress the scalar/draft paths or pitchers.** `_apply_variance` (scalar, simulation.py:538) is untouched. The batch's PITCHER branch (simulation.py:775-793) and its YTD pitcher blend (the `max(actual, sim)` clamp at simulation.py:819-823 for W/K/SV and the rate recombine at :797-809) stay EXACTLY as-is. `HITTING_COUNTING` and `PITCHING_COUNTING` are disjoint (`utils/constants.py:127-128`) and the two `_apply_variance_batch` calls (:755-756) are separate, so the hitter rewrite cannot touch pitchers.
- Reuse the shared `mc_roster.PA_PER_GAME` (= 4.3) -- one constant, no second per-game constant.
- **Phase 4 tests are MECHANISM-ONLY.** The SD/horizon calibration is validated in Phase 6 (`scripts/backtest_sd_calibration.py`), NOT asserted here. Allowed Phase-4 assertions: churn-freeze (active set identical across iters), banked-YTD floor structural for hitters (`team_total >= YTD`), `repl_contrib` not double-firing on the new path, displacement factor applied to the mean, fill nonzero on a low draw / zero on a full draw, pitcher DISTRIBUTION unchanged across the dual path (model identical; per-iteration values shift benignly with the shared rng position) -- the None-path fallback IS byte-identical to pre-4b.

---

## Pre-Work (Step 0)

- [ ] **Dead-code sweep on `simulation.py`** (>300 LOC, structural refactor). Run `ruff check --select F,I src/fantasy_baseball/simulation.py` and `vulture src/fantasy_baseball/simulation.py`. Remove any unused imports / dead helpers surfaced. If clean, note it. Commit separately if anything is removed.

---

## Sub-phase 4a: sampler plumbing (`_apply_variance_batch`)

**File:** `src/fantasy_baseball/simulation.py`; test `tests/test_mc_integration.py`.

**Current signature (verified simulation.py:642-648):**
```python
def _apply_variance_batch(
    players: list, player_type: str, rng: np.random.Generator,
    fraction_remaining: float, n_iter: int,
) -> dict[str, np.ndarray]: ...
```
Today it computes `frac_missed = np.maximum(0.0, 1.0 - scales)` (line 701) and folds `repl_contrib = repl_line[None, :] * frac_missed` into every returned column (lines 704-710), then DISCARDS `frac_missed`. The new path needs (a) `frac_missed` exposed, and (b) `repl_contrib` suppressible.

**Chosen change -- additive, default-preserving.** Add two keyword-only params and a richer return, keeping the old `dict[str, np.ndarray]` shape for back-compat by returning a small result object the existing call sites unpack the same way.

```python
@dataclass
class VarianceBatch:
    counts: dict[str, np.ndarray]   # {col: (n_iter, n_players)} -- the SAME values returned today
    frac_missed: np.ndarray         # (n_iter, n_players) = max(0, 1 - scales)

def _apply_variance_batch(
    players: list, player_type: str, rng: np.random.Generator,
    fraction_remaining: float, n_iter: int,
    *,
    pt_mean_fraction: float | None = None,   # horizon split (4b); None -> use fraction_remaining
    suppress_repl: bool = False,             # zero the built-in repl backfill (new path)
) -> VarianceBatch: ...
```

- `pt_mean_fraction` is the MEAN-horizon term (Step 4b uses 1.0 for ROS-direct so the mean is NOT re-haircut). When `None`, it equals `fraction_remaining` (today's behavior). `fraction_remaining` continues to drive `eff_sd` (via `playing_time_moments`) AND `_negbin_copula_counts` dispersion -- the VARIANCE horizon, kept. See "Horizon split" below for the exact wiring.
- `suppress_repl=True` -> `repl_contrib` is identically zero, so returned `counts[col]` is `counts[:, :, idx]` (corr stats) or `base*scales` (non-corr) with NO replacement added. Default `False` reproduces today's columns exactly.
- `frac_missed` (always populated) is `np.maximum(0.0, 1.0 - scales)` (line 701) -- the per-(iter,player) stochastic shortfall fraction the fill engine consumes.

**Horizon split (the wiring inside 4a; default is a no-op).** `playing_time_moments(mean_scale, cv_pt, fraction_remaining)` currently computes BOTH `eff_mean` and `eff_sd` from one `fraction_remaining` (utils/playing_time.py:103-104). Split the call:
```python
fr_mean = pt_mean_fraction if pt_mean_fraction is not None else fraction_remaining
eff_mean[j], _ = playing_time_moments(mean_scale, cv_pt, fr_mean)        # mean horizon
_, eff_sd[j]   = playing_time_moments(mean_scale, cv_pt, fraction_remaining)  # variance horizon
```
`_negbin_copula_counts(..., fraction_remaining)` (line 698) is UNCHANGED -- it keeps the variance horizon. When `pt_mean_fraction is None`, `fr_mean == fraction_remaining`, so this is identical to today (a single moments call gives the same `eff_mean`/`eff_sd`). NO numeric change on the default path -- verify with a byte-equality test.

**LOAD-BEARING INVARIANT for the byte-equality test.** The 4a refactor must not REORDER, ADD, or REMOVE any `rng.random` / `rng.multivariate_normal` (the copula) draws on the DEFAULT path (`pt_mean_fraction=None`, `suppress_repl=False`). Splitting one `playing_time_moments` call into two is allowed ONLY because `playing_time_moments` consumes no rng (it is closed-form moments). If the refactor consumed even one extra rng draw, or moved the copula sampling relative to the per-player loop, the seed-pinned snapshot would differ for a NON-BUG reason and the test would falsely fail (or, worse, mask a real reordering). The byte-equality test is therefore an RNG-stream-stability test, not merely an arithmetic-equality test -- state this in the test docstring so a future editor does not "fix" a legitimate snapshot drift by re-pinning.

- [ ] **Step 4a.1: Write failing tests** in `tests/test_mc_integration.py`:
  - `test_variance_batch_default_matches_legacy_columns`: build a tiny hitter list; call with defaults; assert `result.counts` equals a captured snapshot of the OLD return (seed-pinned rng). This locks back-compat. (Implement by first capturing the pre-change dict; pin the exact arrays.)
  - `test_suppress_repl_removes_replacement_contribution`: with a constructed player whose `frac_missed > 0` (force a low draw via a seed or a degenerate single-quantile ladder), assert `counts[col]` under `suppress_repl=True` is strictly LESS THAN under `suppress_repl=False` for at least one col where the replacement line is positive -- mechanism: the repl backfill is gone.
  - `test_frac_missed_exposed_and_in_unit_range`: assert `0 <= frac_missed <= 1` elementwise and shape `(n_iter, n_players)`.
  - `test_pt_mean_fraction_one_lifts_mean_without_touching_sd_dispersion`: MECHANISM -- with `pt_mean_fraction=1.0` and `fraction_remaining<1`, the mean of `scales` is HIGHER than with the default (mean un-haircut) while the empirical SD of `scales` across iters is ~unchanged (not collapsed). Use a loose band (mechanism, not calibration): SD ratio in `[0.5, 2.0]`, mean strictly higher.
- [ ] **Step 4a.2: Run, confirm FAIL.**
- [ ] **Step 4a.3: Implement.** Introduce `VarianceBatch` (top of module, after the IDX maps). Update the body per the wiring above. Update the TWO internal call sites in `simulate_remaining_season_batch` (:755-756) to unpack `.counts` (so 4a alone is green before 4b touches the selection logic): `hb = _apply_variance_batch(...).counts` etc. -- a pure mechanical unpack, no behavior change.
- [ ] **Step 4a.4: Run full suite subset:** `pytest tests/test_mc_integration.py tests/test_simulation.py -v` (any existing MC tests must stay green). `ruff check`, `ruff format --check`, `mypy src/fantasy_baseball/simulation.py`, `vulture`.
- [ ] **Step 4a.5: Commit:** `feat(mc): expose frac_missed + suppressible repl + horizon split from variance batch (Phase 4a)`.

---

## Sub-phase 4b: ROS-direct hitter integration + setup wiring + fill

**Files:** `src/fantasy_baseball/simulation.py`; pipeline `src/fantasy_baseball/web/refresh_pipeline.py` (`_run_ros_monte_carlo` setup only); test `tests/test_mc_integration.py`.

### B1. Setup: build EffectiveRoster per team (LeagueContext required)

The pipeline ALREADY retains the three inputs (verified refresh_pipeline.py): `self.eos_baseline` (:417/:941), `self.team_sds` (:419/:951), `self.fraction_remaining` (:421/:895). Phase 2's `build_effective_roster(roster, league_context)` and the `LeagueContext` ctor pattern (verified standings.py:490-497) are in hand. The Player objects are LIVE in `_run_ros_monte_carlo` as `rest_of_season_mc_rosters` (:1382-1384) BEFORE the flatten in `run_ros_monte_carlo` (:948-950).

Two valid placements -- **pin the SIMULATION-LAYER placement** (so the engine is self-contained and testable without the pipeline): `run_ros_monte_carlo` accepts an OPTIONAL `effective_rosters: dict[str, EffectiveRoster] | None = None`. The pipeline builds them (it has the context) and passes them in; when `None` (scalar callers, tests without context), the batch falls ENTIRELY to top-k (whole-context fallback, spec Scope). The batch gains the same optional param.

Pipeline construction (in `_run_ros_monte_carlo`, right after `rest_of_season_mc_rosters` is assembled, before `run_ros_monte_carlo`):
```python
from fantasy_baseball.mc_roster import build_effective_roster
from fantasy_baseball.scoring import LeagueContext
effective_rosters: dict[str, EffectiveRoster] = {}
if self.eos_baseline is not None and self.team_sds is not None:
    for tname, roster in rest_of_season_mc_rosters.items():
        lc = LeagueContext(
            baseline_other_team_stats={t: s for t, s in self.eos_baseline.items() if t != tname},
            team_sds=self.team_sds,
            team_name=tname,
            fraction_remaining=self.fraction_remaining,
        )
        effective_rosters[tname] = build_effective_roster(roster, lc)
```
This is the IDENTICAL context standings build (same baseline object, same `team_sds`, same `fraction_remaining`) -> agrees with ERoto by construction (spec Component 2). DO NOT recompute a second baseline.

### B2. The body-direct hitter helper (replaces the `active_cols` column mapping)

The new hitter path lives in a NEW, self-contained helper -- NOT in the existing flat-dict `hitters`/`active_cols` selection. This is the C2 fix: it sources its inputs from the `EffectiveRoster` bodies (which carry factor, `g_ros_adj`, bench pool, eligible positions) and never aligns bodies against the misordered, bench-included, hitter-only flat columns.

```python
def _simulate_team_hitters_ros_direct(
    effective_roster: EffectiveRoster,
    fraction_remaining: float,
    rng: np.random.Generator,
    n_iter: int,
) -> dict[str, np.ndarray]:
    """Return the team's ROS-ONLY hitter arrays (each shape (n_iter,)):
    {R, HR, RBI, SB, ros_h, ros_ab}. ROS-direct: samples the effective HITTER
    bodies' rest_of_season lines, applies displacement factors, runs the bench
    injury-fill, and returns summed_ROS counting + the ros_h/ros_ab components.
    It does NOT take or use team_YTD: the CALLER owns the YTD blend
    (team_total = YTD + ROS) and the AVG recombine ((YTD_h+ros_h)/(YTD_ab+ros_ab)).
    Keeping YTD in one place (the caller) avoids a dead param (vulture) and keeps
    this helper a pure ROS sampler."""
```

Called PER TEAM by `simulate_remaining_season_batch` ONLY when `effective_rosters is not None` (and the team has an entry). When `effective_rosters is None`, the EXISTING flat-dict + top-k hitter branch runs unchanged (whole-context fallback, spec Scope). PITCHERS ALWAYS use the existing full-season path (unchanged) -- this helper covers HITTERS ONLY.

**Inputs the helper builds (all from bodies, by IDENTITY -- never bare name):**
- `active_h_bodies = [b for b in effective_roster.active if b.player.player_type == HITTER]`. Filtering `EffectiveRoster.active` (= `[*il, *active]`) to hitters drops the pitcher bodies; order is the body-list order, which is the helper's OWN column order (it never has to agree with any flat sublist).
- `bench_h_bodies = effective_roster.bench` (already HITTER-only -- `build_effective_roster` excludes bench pitchers, mc_roster.py:102-103).
- ROS flat dicts: `active_flats = [b.player.to_flat_dict() for b in active_h_bodies]` and likewise for bench. `to_flat_dict()` overlays `rest_of_season` (verified player.py:293-295), so these are ROS-direct counting lines (R/HR/RBI/SB/H/AB/PA) -- exactly the remaining-season base. (`to_flat_dict` is a Player method called on the LIVE `body.player`; NEVER on a flat dict.)

**Edge: unprojected active hitter** (waiver add, no FanGraphs line). It is an `ActiveBody` whose ROS line is ~0, so it samples ~0 and its slot shortfall routes to fill/replacement -- handled naturally, no per-player switch (spec Scope).

### B3. Inside the helper: sample, displace, fill, blend (the rewrite)

1. **Sample ROS-direct -- ACTIVE and BENCH bodies separately.** Call `_apply_variance_batch` (the 4a version) TWICE:
   - `active_vb = _apply_variance_batch(active_flats, HITTER, rng, fraction_remaining, n_iter, pt_mean_fraction=1.0, suppress_repl=True)`. ROS-direct: the projection IS the remaining mean, so `pt_mean_fraction=1.0` lifts the mean haircut (NO re-haircut), while `fraction_remaining` keeps SD + dispersion (the variance horizon). `suppress_repl=True` -- the new fill replaces the built-in backfill. Returns `.counts` (per-iter, per-active-body TOTALS) and `.frac_missed`.
   - The BENCH per-game line is DETERMINISTIC (the clean base, below) -- it does NOT need a sampled draw, so the bench bodies are NOT sampled through `_apply_variance_batch` in the default design. (See per_game_counts pin.) This keeps the bench line iteration-independent.

   Because the helper samples ITS OWN `active_flats` (already ROS-only), there is NO second re-flatten of the batch's full-season hitter sublist and NO `active_cols` indexing -- the C2 misalignment cannot occur. The batch's existing full-season flatten (:948-950) feeds ONLY the pitcher path and the `effective_rosters=None` fallback.

2. **Apply displacement factor to the SAMPLED ROS counts (ROS only, no YTD).** Per active hitter body i: `realized[col] = active_vb.counts[col][:, i] * body.factor`. The factor scales the MEAN only; see the variance note below. `frac_missed_i = active_vb.frac_missed[:, i]`; `games_missed = frac_missed_i * body.g_ros_adj` (the reduced baseline -- `allocate_bench_fill` recomputes this internally from `g_ros_adj`, so the helper passes `frac_missed_i` and lets the allocator multiply). For an undisplaced body `factor = 1.0` -> no-op.

   **Displacement-variance note (PIN the exact claim).** Applying `factor` POST-sampling scales BOTH the mean AND the SD of that body's counts by `factor` (so its variance by `factor^2`). That is INTENDED and spec-correct: the `_projected_volume` curve lookup inside `_apply_variance_batch` is UNCHANGED (it still indexes the body's FULL projected volume), so the body is sampled with the FULL-VOLUME CV BAND -- the coefficient of variation the curve assigns a full-timer. What the spec requires ("not narrowed by the factor") is that this FULL-VOLUME CV band is preserved, i.e. the displacement does NOT push the body to a lower-volume (higher-CV / more-injury-prone) curve point. Multiplying the realized counts by `factor` rescales mean and SD together, holding CV fixed -- exactly the full-volume band. (It is NOT the claim that absolute variance is unchanged; absolute variance scales by `factor^2`, which is correct because a player with fewer games has proportionally less count variance at fixed CV.) State this in the code comment.

3. **Per iteration, build fill inputs and call `allocate_bench_fill`.** Python loop over iterations (cheap: <=12 active, <=2 bench; spec sanctions a per-team/per-iter Python loop). For iteration i:
   - `actives = [ActiveSample(body, frac_missed=float(frac_missed[i, idx])) for idx, body in enumerate(active_h_bodies)]`.
   - `benches = [BenchSample(body, per_game_counts={col: base_ros_total[col]/body.g_ros_full for col in HITTING_COUNTING}) for body in bench_h_bodies]` -- built ONCE at setup (iteration-independent), reused every iteration.
   - `replacement_for = lambda ab: _replacement_line(ab.player.to_flat_dict(), is_hitter=True)` (the allocator passes the `ActiveBody`; `_replacement_line` takes a flat dict, so flatten the body's player -- ROS overlay; verified simulation.py:435).
   - `fill = allocate_bench_fill(actives, benches, replacement_for).fill_counts`.
   - Team ROS per category for iteration i = `sum_over_active(realized[col][i]) + fill[col]`, for each counting col (R/HR/RBI/SB) AND the `h`/`ab` AVG components.

   **per_game_counts source -- PIN: clean BASE projection (deterministic fill), NOT the per-iteration sampled draw.** Rationale: (a) the spec variance note (Component 3) flags that scaling a full-ROS-volume sampled draw to games-covered gives `f^2*var` (understated, and noisier); using the clean base ROS-per-game line gives a stable, defensible fill that is strictly more realistic than today's deterministic replacement and avoids injecting an under-calibrated variance term into the small fill portion. (b) It makes the bench body's `per_game_counts` ITERATION-INDEPENDENT (compute ONCE at setup), lifting it entirely out of the per-iteration loop -- a real speed win. The per-iteration sampled-draw variant (adds fill variance) is the deferred refinement gated on the Phase 6 SD backtest. `base_ros_total[col]` is the bench body's clean ROS counting from `BenchBody.player.to_flat_dict()` (ROS overlay), divided by `BenchBody.g_ros_full`.

4. **Blend (in the CALLER, after the helper returns summed_ROS) `team_total = team_YTD + summed_ROS`.** ROS-direct means `summed_ROS >= 0` for every counting cat, so `team_total >= YTD` STRUCTURALLY -- NO `max(actual, sim)` clamp for hitters (ROS>=0 makes the floor structural). Hitter outputs = `team_YTD.get(cat, 0) + summed_ROS[cat]` (R/HR/RBI/SB), with `team_YTD = actual_standings[team]`. RATE stat (AVG): recombine from YTD + ROS components using `actual_ab`/`actual_h` (already threaded, :795-796) plus the summed ROS `ab`/`h`: `total_ab = actual_ab + ros_ab`, `total_h = actual_h + ros_h`, `avg = total_h/total_ab` (no clamp -- ROS components are nonnegative). The helper returns the four counting arrays + `ros_h`/`ros_ab` for this recombine; AVG is assembled in the caller alongside the existing `actual_ab`/`actual_h` (so the helper stays pure-counting and the YTD blend lives in one place).
   - **PITCHERS: the `max(actual, sim)` clamp and the existing full-season blend are RETAINED verbatim** (W/K/SV at :819-823, ERA/WHIP at :797-809/:822-823). Pitchers never enter the helper.

### Horizon split decision (PINNED) and fallback

- **Chosen:** SEPARATE the mean-horizon from the variance-horizon (spec lean). `pt_mean_fraction=1.0` (ROS-direct: projection is the remaining mean, no re-haircut); `fraction_remaining` retained for `eff_sd` AND `_negbin_copula_counts` dispersion (remaining-season risk kept, NOT collapsed). This is implemented in 4a; 4b just passes `pt_mean_fraction=1.0` on the hitter ROS-direct call.
- **Explicitly NOT `fraction_remaining=1.0`** -- that would also collapse the dispersion (spec warns this is too blunt).
- **Fallback (if Phase 6 SD backtest shows mis-calibration):** keep FULL-season sampling for hitters (revert the ROS-direct base + `pt_mean_fraction`) and source ONLY games/displacement/fill from the setup ROS quantities. Keep this fallback a ONE-LINE switch: a module-level `_ROS_DIRECT_HITTERS = True` flag gating the ROS base + `pt_mean_fraction=1.0`; flip to False to fall back without touching the displacement/fill wiring. Documented, not asserted in Phase 4 (Phase 6 gates it).

- [ ] **Step 4b.1: Write failing tests** (MECHANISM-only -- do NOT assert absolute magnitudes tied to the `fraction_remaining`/`per_game` choices) in `tests/test_mc_integration.py`. Prefer testing the helper `_simulate_team_hitters_ros_direct` DIRECTLY with a constructed `EffectiveRoster` + a seeded rng (cheaper, isolates the mechanism); use a full `simulate_remaining_season_batch(..., effective_rosters=...)` run only for the dual-path/fallback seam tests. Build a 1-2 team roster of real `Player` objects + a hand-built `LeagueContext` (or `EffectiveRoster` directly via `build_effective_roster`). Assert:
  - `test_healthy_roster_hitter_totals_positive_no_bench_contrib`: a HEALTHY roster (no injury draw -- pin a seed/degenerate ladder so `frac_missed == 0`) yields positive hitter counting totals and the bench contributes EXACTLY 0 (fill is zero when no games are missed).
  - `test_injured_active_body_gets_bench_fill`: force one active body to a low draw (`frac_missed > 0`) with an eligible bench body; assert the bench body's per-game line contributes a NONZERO fill share to the team total (and zero in the healthy case above). Mechanism only -- no magnitude pinned to `fraction_remaining`.
  - `test_hitter_team_total_at_least_ytd`: every hitter cat output `>= team_YTD[cat]` across all iters (banked-YTD floor structural -- ROS>=0).
  - `test_churn_freeze_active_set_fixed`: the contributing active hitter bodies are identical every iteration (no per-iter re-selection). Construct a roster where a bench bat has higher raw stats than a starter: OLD top-k would seat the bench bat (varying per iter); the new engine fixes the active set -> the bench bat contributes ONLY injury-fill (zero when starters draw full).
  - `test_repl_not_double_counted_on_new_path`: with the new fill active, no hitter total reflects BOTH the built-in `repl_contrib` AND the new fill. Assert a STARVED-bench case (no eligible bench body) routes the shortfall to fill-replacement EXACTLY once (compare against a hand-computed single-replacement bound), confirming the helper called `_apply_variance_batch(..., suppress_repl=True)`.
  - `test_displacement_factor_scales_hitter_mean`: a displaced body (factor < 1) contributes a LOWER mean than the same body undisplaced -- factor applied to the sampled ROS counts (mean). Do NOT assert the absolute mean value.
  - `test_pitchers_unchanged_with_effective_rosters`: with `effective_rosters` supplied for a mixed roster, the PITCHER category outputs MATCH a run WITHOUT `effective_rosters` (same seed) in DISTRIBUTION -- mean and SD within tolerance over `n_iter`, NOT byte-identical. Why not byte-identical: the new hitter helper samples a different player count than the old flat-dict hitter path, which shifts the single shared `rng` stream that the subsequent pitcher draw reads from (`simulation.py` runs one `rng` through hitters-then-pitchers per team). The pitcher MODEL is untouched (same `_apply_variance_batch` call, same inputs), so the DISTRIBUTION is identical -- which is the correct "pitchers unchanged" invariant; the per-iteration value shift is a benign consequence of changing hitter selection, not a regression.
  - `test_whole_context_fallback_to_topk`: `effective_rosters=None` -> BYTE-identical to the pre-4b top-k batch (seed-pinned). This is the byte-anchor: the fallback runs the EXACT old code path with zero added/reordered rng draws, so it stays byte-stable. (The two tests are not in tension: the fallback path is the frozen baseline; the `effective_rosters` path is the new path whose pitcher stream legitimately shifts.)
- [ ] **Step 4b.2: Run, confirm FAIL.**
- [ ] **Step 4b.3: Implement** the setup wiring (pipeline B1), the body-direct helper `_simulate_team_hitters_ros_direct` (B2), and its sample/displace/fill/blend internals (B3), gating the per-team helper call on `effective_rosters is not None` + the `_ROS_DIRECT_HITTERS` flag. Keep the flat-dict top-k hitter branch intact for the `effective_rosters=None` fallback. The pitcher branch and `active_cols` (diagnostic) path are untouched.
- [ ] **Step 4b.3a: Update the refresh-test fixture wrapper.** `tests/test_web/_refresh_fixture.py::_scaled_ros_mc` (verified ~442-462, patched over `run_ros_monte_carlo` at ~527) forwards a FIXED kwarg list and has NO `effective_rosters` param -- once the pipeline passes `effective_rosters=` into `run_ros_monte_carlo`, the patched wrapper raises `TypeError: unexpected keyword argument`. Add `effective_rosters=None` to the wrapper signature and forward it to `_real_ros_mc(...)`. (One-line signature + one-line forward; the fixture passes `None` so the patched MC stays on the top-k fallback, which is the right behavior for the mocked integration test.)
- [ ] **Step 4b.4: Verify locally (refresh path).** Run `python scripts/run_season_dashboard.py --no-sync` against the live cache (per repo memory: `--no-sync` so not-yet-deployed code is exercised without clobbering local SQLite) and confirm `run_full_refresh` completes and the MC step produces finite standings. (Auth required -- controller runs.)
- [ ] **Step 4b.5: Full suite + lint.** `pytest -v` (or `pytest tests/test_mc_integration.py tests/test_simulation.py tests/test_scoring.py -v` if scoping; state the subset). `ruff check .`, `ruff format --check .`, `mypy` on touched files, `vulture`.
- [ ] **Step 4b.6: Commit:** `feat(mc): ROS-direct hitter integration with IL displacement + bench injury-fill (Phase 4b)`.

---

## Sub-phase 4c: before/after evidence (4th arm = NEW engine)

**Files:** `src/fantasy_baseball/mc_selection.py`; gated hook in `refresh_pipeline.py:_run_ros_monte_carlo` (:1402-1422); test additions in `tests/test_mc_selection.py`.

- Extend `run_selection_attribution` with a 4th arm `"new_engine"`. CRITICAL: this arm must use the SAME body-direct path as the engine -- it builds per-team `EffectiveRoster` (via `build_effective_roster(roster, lc)` from the Player rosters + a per-team `LeagueContext`) and calls `simulate_remaining_season_batch(..., effective_rosters=<built here>)`, which routes hitters through `_simulate_team_hitters_ros_direct`. It must NOT flatten full-season and reuse `active_cols` for this arm (the three OLD arms keep `active_cols`; the new arm bypasses it entirely -- mirroring the engine's dual path). The diagnostic has the Player rosters; it additionally needs `eos_baseline`/`team_sds`/`fraction_remaining` -- thread them in as new args from the pipeline hook (which has them on `self`). When those context inputs are absent (e.g. a slot-less synthetic test), the new arm falls to the same whole-context top-k fallback as the engine.
- Extend `format_attribution_table`: it currently hardcodes a 3-arm list `arms = ["topk_per_iter", "topk_fixed", "active_slot"]` (mc_selection.py:123) and a fixed-width header. Add the 4th arm -> `["topk_per_iter", "topk_fixed", "active_slot", "new_engine"]`, widen the header/rows for the extra column, report all 10 categories + an overall roto-standings summary (rank each arm's medians). Keep the ASCII table (no non-ASCII).
- Phase-0 hook header: the gated-hook table header that labels the arms must add the `new_engine` column too (so the pasted artifact shows 4 arms). Keep ASCII.
- Pipeline hook (:1409): pass `self.eos_baseline`, `self.team_sds`, `self.fraction_remaining` into `run_selection_attribution`; keep it gated behind `FB_SELECTION_ATTRIBUTION`.
- **AUTH-REQUIRED sub-step (controller runs):** a Yahoo-auth'd local refresh with `FB_SELECTION_ATTRIBUTION=1` against the Upstash/Render snapshot (source of truth, never stale local). Confirm DIRECTIONAL acceptance (spec Acceptance): NEW sits between active-slot (floor) and topk (ceiling) in aggregate + on the demonstrative teams' counting cats; Cavalli's seated bats (Perez/Ward/Arraez) show a SMALL injury-fill share not full ~99 RBI; SkeleThor/Hart track ERoto's displacement per-player.

- [ ] **Step 4c.1: Write a failing test** `tests/test_mc_selection.py::test_new_engine_arm_present_and_between_floor_and_ceiling` on a SMALL synthetic 2-team roster (no auth): assert the 4th arm key exists, has all 10 cats, and on a constructed bench-seating case `active_slot <= new_engine <= topk_per_iter` for the demonstrative counting cat (mechanism, not magnitudes).
- [ ] **Step 4c.2: Run, confirm FAIL.**
- [ ] **Step 4c.3: Implement** the 4th arm + table extension + hook threading.
- [ ] **Step 4c.4:** add `"src/fantasy_baseball/mc_selection.py"` to `[tool.mypy].files`; `pytest tests/test_mc_selection.py -v`; ruff; `mypy src/fantasy_baseball/mc_selection.py`; vulture.
- [ ] **Step 4c.5: Commit:** `feat(mc): add NEW-engine arm to before/after selection-attribution diagnostic (Phase 4c)`.
- [ ] **Step 4c.6 (AUTH, controller):** run the gated diagnostic on the live snapshot; paste the 4-arm all-category table + overall standings into the PR as the before/after artifact (definition of done).

---

## End-of-effort checklist (FORCED VERIFICATION)

- [ ] `pytest -v` (or the stated MC/scoring/simulation subset) -- all green.
- [ ] `ruff check .` -- zero violations.
- [ ] `ruff format --check .` -- clean.
- [ ] `vulture` -- no NEW findings (the `VarianceBatch` dataclass + new params are referenced; call out pre-existing findings).
- [ ] `mypy` on `simulation.py`, `mc_selection.py` (both under `[tool.mypy].files`).
- [ ] Paste outputs into the final message.

---

## Self-Review

**Split decision:** YES -- 4a (sampler plumbing, isolated, default-preserving), 4b (ROS-direct hitter integration + setup wiring + fill), 4c (before/after evidence). Each touches <=2-3 files and is independently committable + reviewable. 4a is a pure-additive change with byte-equality back-compat tests, de-risking the core-batch rewrite before any behavior changes in 4b.

**Horizon split (PINNED):** separate the mean-horizon from the variance-horizon. `pt_mean_fraction=1.0` lifts the mean haircut (ROS-direct: projection IS the remaining mean); `fraction_remaining` is RETAINED for `eff_sd` and `_negbin_copula_counts` dispersion (remaining-season risk kept). Explicitly NOT `fraction_remaining=1.0` (collapses dispersion). Fallback to full-season sampling is a one-line `_ROS_DIRECT_HITTERS` flag, gated on the Phase 6 SD backtest, NOT asserted in Phase 4 (tests are mechanism-only).

**per_game_counts source (PINNED):** clean BASE ROS projection (`base_ros_total / g_ros_full`), DETERMINISTIC fill -- not the per-iteration sampled draw. Simpler, defensible, avoids the spec's `f^2*var` under-calibration on the small fill portion, and lifts the bench per-game line out of the per-iteration loop (iteration-independent). Sampled-draw fill variance is the deferred refinement.

**Key signatures:** `_apply_variance_batch(..., *, pt_mean_fraction: float | None = None, suppress_repl: bool = False) -> VarianceBatch(counts, frac_missed)`; `_simulate_team_hitters_ros_direct(effective_roster: EffectiveRoster, fraction_remaining: float, rng: np.random.Generator, n_iter: int) -> dict[str, np.ndarray]` (returns the team's R/HR/RBI/SB arrays + ros_h/ros_ab; the CALLER owns the YTD blend + AVG recombine; HITTERS only); `simulate_remaining_season_batch(..., effective_rosters: dict[str, EffectiveRoster] | None = None)`; `run_ros_monte_carlo(..., effective_rosters=None)`; pipeline builds `LeagueContext(baseline_other_team_stats={t:s for t,s in self.eos_baseline.items() if t!=tname}, team_sds=self.team_sds, team_name=tname, fraction_remaining=self.fraction_remaining)` -> `build_effective_roster(roster, lc)`.

**C2 alignment avoided (the central fix):** the new hitter path is the self-contained `_simulate_team_hitters_ros_direct` helper that samples the `EffectiveRoster` bodies DIRECTLY (active hitters filtered from `[*il, *active]`, plus the hitter bench pool), via `body.player.to_flat_dict()` (ROS overlay, a Player method). It NEVER re-flattens the batch's full-season hitter sublist and NEVER uses the roster-order, bench-included `active_cols` indices -- so the `[*il, *active]`-vs-flat-sublist misalignment cannot occur. `active_cols` remains the Phase-0 DIAGNOSTIC path (the three OLD arms) only; the 4th `new_engine` diagnostic arm uses the same body-direct `effective_rosters` route.

**Pitchers untouched:** disjoint `HITTING_COUNTING`/`PITCHING_COUNTING`, separate `_apply_variance_batch` calls, the pitcher `max(actual,sim)` clamp + full-season blend retained verbatim; a dedicated test asserts pitcher DISTRIBUTIONAL equivalence (mean/SD within tolerance) across the dual path; the `effective_rosters=None` fallback is byte-identical to pre-4b (the byte-anchor).

**Scalar/draft unaffected:** `_apply_variance` (scalar) untouched; new batch params default to legacy behavior (`effective_rosters=None` -> top-k; `suppress_repl=False`, `pt_mean_fraction=None` -> identical columns). Byte-equality tests lock both.

**Mechanism-only assertions:** banked-YTD floor, churn-freeze, repl-not-double-counted, displacement-scales-mean, pitcher-byte-identity, whole-context fallback. No absolute-magnitude / SD-calibration claims (Phase 6 gates those).

**Re-key by id, never name:** the new helper sources its bodies straight from `EffectiveRoster.active`/`.bench` (already identity-keyed `Player` objects) -- no name-to-column mapping at all, so the misordered-flat-sublist hazard is gone. `build_effective_roster` already re-keys the displacement factors onto Players by identity and guards same-name collisions in the active+IL set (mc_roster.py:84-93). The 4c diagnostic's `new_engine` arm uses the identical body-direct route.
