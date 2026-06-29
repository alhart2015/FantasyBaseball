# Sampled replacement-fill line -- design (2026-06-29)

## Goal

Give the terminal replacement-fill term in the ROS-direct hitter Monte Carlo the
same NegBin rate variance the bench bodies now carry, so the hitter
counting-category SD under-dispersion (R/RBI realized SD ~0.70-0.75 of analytic)
is corrected -- while keeping the team-total means neutral.

## Background and motivation

PR #146 ("sampled bench injury-fill") made the bench bodies that cover an injured
active's missed games sample their rate + availability instead of contributing
deterministic (zero-variance) production. The Task 5 acceptance gate
(`docs/superpowers/sampled-bench-fill-sd-evidence-2026-06-29.md`) showed that
change is mean-safe and directionally correct but INSUFFICIENT: R/RBI SD ratios
rose only +0.008 / +0.009 against a required +0.08 floor.

Root cause (measured): with benches ~2 hitters deep against ~10 actives each
shedding 6-25% of games per iteration, ~77% of all injury-filled games cascade
PAST the bench to the still-deterministic replacement line. Bench-only sampling
touches only the ~23% bench portion, so the dominant fill term stays
zero-variance and team-total R/RBI dispersion barely moves. This is exactly
adversarial finding #5 in the parent spec
(`docs/superpowers/specs/2026-06-29-sampled-bench-fill-design.md`). Its named
follow-up -- sampling the replacement line -- is therefore a prerequisite for the
de-bias, not an optional refinement.

This spec covers ONLY the replacement-line sampling. The bench sampling
(PR #146, already on branch `mc-sampled-bench-fill`) stays as-is; the replacement
term is sampled ON TOP of it. Both contribute fill variance.

## Scope

- IN: sampling the hitter ROS-direct replacement-fill term in
  `_simulate_team_hitters_ros_direct` / `allocate_bench_fill`.
- OUT: pitchers (pitcher ROS-direct has no bench fill and no replacement-fill
  sampling -- unchanged). The active and bench draws are unchanged. No projection,
  standings, or web-pipeline changes.

## Chosen model: correlated per-game rate (reuse the bench model)

The residual `need` games (per active body, after the bench cascade) are filled
at a STOCHASTIC mean-neutral per-game replacement rate instead of the
deterministic `_replacement_line` per-game rate. A scrub's per-game rate is
modeled as fixed for the iteration (drawn once), so the contribution variance
scales ~`need^2`. This reuses the bench path's per-game decomposition
(`per_game_rate * games`); it is identical in form FOR THE MEAN, but NOT for the
tail (see "Tail behavior" below -- the bench's capacity cap is absent here, so the
safety bound the bench relies on does not transfer). This was chosen over (a) a
proper NegBin draw at the actual `need` volume (more faithful to the analytic
dispersion but the draw would have to happen after `need` is known, breaking the
pure allocator) and (b) an independent per-game stream (least variance, expected
insufficient). Rationale: consistency with the bench treatment, reuse of existing
machinery, a pure allocator, and it injects the most variance of the pure options
(best chance of clearing the +0.08 floor) while remaining mean-neutral (in
expectation, conditional on `scale > eps` -- see Mean-neutrality).

## Tail behavior (NOT inherited from the bench cap)

The bench path is tail-safe by construction: a bench body's total contribution is
`capacity * per_game = (g*scale) * (realized/(g*scale)) = realized_total`, so the
`1/scale` is CANCELLED by the capacity and even an extreme small-`scale` draw is
bounded (parent spec's "no division blow-up" invariant). The replacement term has
NO capacity cap (the pool is inexhaustible; it covers the full `need`), so the
`1/scale` is NOT cancelled: `fill = (realized / (repl_implied_games * scale)) *
need`, where `need` is bounded by the ACTIVE's shortfall, independent of the
scrub's `scale`. Dividing realized by `scale` removes the playing-time MEAN level
but AMPLIFIES the NegBin count noise by `1/scale` -- it is a noise amplifier, not a
clean cancellation. As `scale -> 0` a nonzero count produces an inflated
single-iteration fill spike. The eps guard (`scale <= 1e-9`) is ONLY a
division-by-zero guard -- at 1e-9 it does almost nothing for variance. The actual
over-dispersion control is the `cv_pt` band (the curve volume, below): a
near-full-time band keeps `scale` away from the moderate-small range (~0.01-0.1)
that drives the heavy tail, making spikes rare; a low-volume band widens them.
The dampers below are the explicit controls if the band alone is not enough.

This is the specific mechanism behind an over-correction (gate #1 `>= 1.20` HARD
FAIL, per the authoritative ladder). If the gate trips high, the dampers in priority order are: (1) cap the
per-iteration replacement per-game rate at a multiple (e.g. 3-5x) of the
deterministic rate; (2) switch this term to the proper-NegBin-at-need model. Do
NOT "cap replacement capacity" -- the pool is terminal (nothing cascades below
it), so a capacity cap would leave `need - capacity` UNCOVERED in heavy-injury
iterations and silently drop production (a downward mean bias, not the clean
bench cancellation). The rate cap clips the positive tail, so it introduces a
DOWNWARD mean bias (the proper-NegBin fallback is mean-faithful by contrast);
after applying EITHER damper, the gate #3 re-check (which bounds downward drift,
below) is mandatory -- a rate cap in particular must not be tuned past the point
where it drags the mean down materially. The implementer must NOT assume the
bench tail-safety transfers and skip a clamp -- the plan carries this asymmetry
explicitly.

## Mechanism

Per call to `_simulate_team_hitters_ros_direct` (HITTERS only), batched across the
active bodies:

1. For each active body, look up its replacement line:
   `_replacement_line(active.player.to_flat_dict(), is_hitter=True)`. This is
   already position-routed (a catcher's replacement gives ~0 SB, a middle
   infielder's ~15) and is a FULL-SEASON counting bundle with NO games field. NOTE
   it returns a SHARED read-only module constant (two actives at the same position
   alias the same object); the list of these forms the synthetic "players" for the
   draw -- read-only, never mutate (see the no-mutate note in test impact).
   The sampled replacement line gets NO displacement `factor` (unlike active
   bodies): replacement is the floor a healthy scrub provides, and the
   deterministic path it replaces applies no factor (`mc_fill.py:100-106`), so
   omitting it preserves mean-consistency.
2. Derive TWO distinct volumes from each replacement line's `ab` field (which is
   at-bats, NOT plate appearances):
   - `repl_implied_games = repl_ab / PA_PER_GAME` -- the implied-games
     denominator. This is the SAME heuristic the current deterministic allocator
     uses (`mc_fill.py:102`); keeping it preserves mean-consistency with the old
     path. (It is an AB/PA-per-game approximation, deliberately inherited verbatim
     so the mean does not shift relative to today.)
   - `repl_pa_volume = repl_ab / AB_PER_PA` -- the playing-time CURVE volume,
     derived the SAME way `_full_season_pt_volume` converts AB to PA
     (`simulation.py:420-422`), so the scrub is banded consistently with how other
     bodies are banded. (See Calibration levers: this band choice affects how much
     PT-scale variance is injected.)
3. ONE `_apply_variance_batch` draw over all the replacement bodies
   (PlayerType.HITTER, `pt_mean_fraction=1.0`, `suppress_repl=True`,
   `pt_volumes` = the per-body `repl_pa_volume`), APPENDED AFTER the bench draw.
   Yields realized counts + `scale` per (iter, active).
4. Decompose to a mean-neutral per-game rate, per (iter, active, col):
   `per_game_repl = realized / (repl_implied_games * scale)` when
   `repl_implied_games * scale > eps`, else `0.0`. Conditional on `scale > eps`,
   `E[per_game_repl] = repl_base / repl_implied_games` -- today's deterministic
   per-game replacement rate (the `scale` cancels exactly because the NegBin mean
   is `base*scale`). The eps-guard mass (`scale <= eps`) zeroes the rate, so the
   UNCONDITIONAL mean is `(repl_base/repl_implied_games) * P(scale > eps)`,
   i.e. a tiny DOWNWARD bias only -- never upward (see Mean-neutrality).
5. In the per-iteration fill loop, the allocator multiplies each active's residual
   `need` by THAT active's sampled per-game replacement line for THAT iteration.

### Availability

The replacement pool is inexhaustible: it always covers the full residual `need`
(no availability haircut on the contribution). The replacement body's `scale` is
sampled (it falls out of `_apply_variance_batch`); dividing realized by
`repl_implied_games * scale` removes its playing-time MEAN level (keeping the rate
mean-neutral) while amplifying the count noise by `1/scale`. It is NOT used as a
capacity cap -- unlike a bench body, a replacement body has no fill ceiling, so
the `1/scale` does NOT cancel (see "Tail behavior").

### Independence

Each active's replacement body is an independent cell in the batched draw
(different injured starters -> independent scrubs). Counting categories correlate
WITHIN each body via the existing copula (`HITTER_CORR_MATRIX`). There is no
cross-active correlation of replacement fills -- the realistic model.

## Architecture: keep `allocate_bench_fill` pure

`allocate_bench_fill` is intentionally pure (no rng, no sampler import). It stays
pure. The change:

- `replacement_for` contract changes from "return the FULL-SEASON replacement
  line (dict)" to "return the PER-GAME replacement line (dict) for this active
  body THIS iteration." The caller now owns BOTH the implied-games conversion and
  the sampling.
- The allocator's residual branch simplifies from (line ~100-106: fetch
  full-season line -> divide by `repl_ab / PA_PER_GAME` -> multiply by `need`) to:
  `repl_pg = replacement_for(a.body); fill[col] += repl_pg[col] * need`. The
  `repl_ab`/`repl_games` conversion is deleted from the allocator (it moves to the
  caller, where the sampled per-game rate is produced).
- The caller (`_simulate_team_hitters_ros_direct`) builds, per iteration, a
  closure or lookup that returns the sampled per-game replacement line for the
  given active body / index. The deterministic implied-games conversion that the
  allocator used to do is reproduced once in the caller when forming the
  per-game rate (step 4 already divides by `repl_implied_games`).

This narrows the allocator's responsibility (it no longer knows the replacement
line is full-season) and keeps the rng in the caller.

## Mean-neutrality (load-bearing -- gate #3)

`E[realized / (repl_implied_games * scale) | scale > eps] = repl_base /
repl_implied_games`. The denominator MUST be the UNCAPPED `repl_implied_games *
scale` (the same invariant the bench sampling relies on). A denominator capped at
`repl_implied_games` (i.e. dividing by the unscaled volume) would reintroduce an
upward mean bias proportional to `max(1, scale)` -- the exact bug class the
parent spec's spec-review caught.

The neutrality is EXACT only conditional on `scale > eps`; the eps-guarded mass
(`scale <= eps`, rate forced to 0) makes the unconditional mean a hair LOW, never
high. That is the load-bearing direction: gate #3 fails ONLY on UPWARD drift, and
the only mechanism that could push the mean up is a capped denominator (forbidden
above). A small downward drift is expected and acceptable (it overlaps the
intended replacement re-damping). The acceptance gate's mean-drift check (no
upward drift > +1%) is the guard; do not claim bit-exact neutrality.

## EPS guard

When a replacement body has `repl_ab <= 0` (no implied volume) or is sampled
fully unavailable (`scale == 0`), `repl_implied_games * scale <= eps` and the
per-game rate is `0.0` for all cols (the residual contributes nothing rather than
dividing by zero). `eps = 1e-9`, matching the bench path. Never `x or default`.

## RNG stream and test impact

- The replacement draw is APPENDED AFTER the active draw and the bench draw, so it
  consumes rng and SHIFTS the stream for any later per-team draws -- the same
  shared-Generator property the active and bench draws already have. This is
  acceptable (it does not bias distributions) and is the established batch design.
- IMPORTANT (retracts a parent-spec invariant): the replacement draw fires for
  EVERY team with active hitters (`n_active > 0`), regardless of bench depth. So
  the parent spec's finding-#8 guarantee that "empty-bench ROS-direct teams
  consume ZERO extra rng / stay byte-identical" is now VOID for empty-bench teams.
  What REMAINS byte-identical: (a) a team with NO active hitters early-returns
  before any draw (the `if not active_h_bodies:` return, ~`simulation.py:904-908`),
  still zero rng; (b) the direct
  byte-anchor `test_variance_batch_default_matches_legacy_columns` exercises
  `_apply_variance_batch` with legacy defaults and is UNTOUCHED (this change adds a
  new call site, not a change to the function). Do not rely on the parent's
  empty-bench byte-identical claim.
- MEAN-INVARIANT guardrails -- must still hold, never loosen:
  `test_repl_not_double_counted_on_new_path`,
  `test_displacement_factor_scales_hitter_mean`. These pin values only on HEALTHY
  iterations (`frac_missed == 0` -> `need == 0` -> no fill fires regardless of
  sampling) and on mean/structural properties that mean-neutrality preserves. If
  one breaks, it is a real regression -- STOP and fix the code, do not re-pin.
- SD-CEILING guardrail -- ONLY the ceiling line moves; re-pin it WITH analysis:
  `test_ros_direct_uses_full_season_volume_for_cv_pt` asserts an ABSOLUTE upper
  bound on `out["R"].std()` (around `tests/test_mc_integration.py:695`) calibrated
  for the OLD deterministic fill. This change injects variance into that fill BY
  DESIGN, so `helper_sd` rises and the ceiling can trip. That is the INTENDED
  effect, NOT a regression: re-derive the `:695` bound with the new fill variance,
  re-pin it WITH the recorded analysis, and confirm it still reflects a real upper
  bound (not just "bump it until green"). CRITICAL: re-pin ONLY the `:695`
  ceiling. The healthy-iteration exact-match assertions in the SAME test (around
  `:690-691`) stay BYTE-IDENTICAL (those iters have `need == 0` -> no fill, and the
  active draw is unchanged) and must NOT be loosened. This test is explicitly NOT
  in the mean-invariant set above for the `:695` line only.
- Our own de-bias SD test and any pinned post-bench-draw team-total goldens will
  shift (new sampling) and are re-pinned with justification, as in PR #146.
- The synthetic replacement flat-dicts returned by `_replacement_line` are SHARED
  module-level constants (`REPLACEMENT_BY_POSITION[...]` / `_GENERIC_HITTER_REPL`).
  `_apply_variance_batch` only READS them (`p.get(col)`), so aliasing is benign --
  but they must be treated read-only; never mutate a flat-dict on this path or the
  shared constant corrupts for all consumers.

## Acceptance gate (re-run Task 5)

Re-run the SD-calibration diagnostic (`FB_SELECTION_ATTRIBUTION=1`) BEFORE
(current branch HEAD, bench-only) vs AFTER (this change), same seed / n_iter /
fraction_remaining, and confirm ALL of:

1. R and RBI SD ratios: record actual values and classify per the
   PASS/PARTIAL/FAIL ladder below (the merge decision lives there, not here). The
   target is `[0.85, 1.20)`; neither may reach `1.20` (i.e. `>= 1.20` is
   over-correction, per the ladder).
2. Pooled ratio stays in `[0.8, 1.25]`.
3. Hitter category team-total MEANS (the new_engine MC arrays' MEAN, NOT the
   median, and NOT the bench-independent ERoto `standings_breakdown` projection).
   Mean-neutrality is a MEAN property, so the gate must check the mean: this change
   replaces a point-mass deterministic fill with a right-skewed `1/scale` term
   whose MEDIAN sits below its mean, so the new_engine MEDIAN would drift down even
   with the mean exactly preserved (a skew artifact, not real drift). The
   diagnostic currently emits per-cat medians; this gate run must additionally
   compute the per-cat MEAN of the new_engine team totals (before/after) and check
   it. The per-iter arrays are already in hand where `compute_sd_calibration`
   reads `np.std(batch[t][cat])`, so this is a trivial `np.mean(...)` add via the
   same evidence-only instrument-then-revert pattern the gate-#4 share counter
   used -- not a tooling gap. Criteria: NO upward drift `> +1%` (hard fail -- the only mechanism that
   pushes the mean UP is a capped denominator, forbidden) AND no downward drift
   `> ~5%` (surface/investigate -- a small downward drift is the intended
   re-damping, but a large one signals a damper over-clipping the tail or a bug,
   and must be explained before merge, especially on a post-damper re-run).
4. Replacement-fill share recorded (expected ~unchanged from the ~77% baseline;
   the change samples that term, it does not change how much of it fires).

### Calibration outcomes (surface, do not paper over)

Because R and RBI are scored per-cat and can DIVERGE (one over-shoots while the
other under-shoots), the outcomes are NOT mutually exclusive on their own.
Evaluate in this strict PRECEDENCE order and take the FIRST that matches (so a
divergent run, e.g. R=1.22 / RBI=0.80, has exactly one defined action):

0. **Gate #2 or #3 breach (independent hard fail, checked FIRST):** if pooled
   leaves `[0.8, 1.25]`, OR mean drift is upward `> +1%`, OR downward `> ~5%`
   unexplained -> BLOCKS merge regardless of the R/RBI ladder. Mean/pooled
   calibration is a precondition for any PASS.
1. **OVER-CORRECTION (blocks merge):** EITHER R or RBI `>= 1.20`. The uncapped
   `1/scale` tail (see "Tail behavior") over-disperses. Apply a damper in the
   priority order named there (rate cap -> proper-NegBin-at-need), re-run the
   gate (including the gate #3 downward-drift re-check). Do NOT ship an
   over-correction.
2. **FAIL (blocks merge):** EITHER R or RBI rose `< +0.08` over baseline. The
   model is insufficient; STOP and re-design. Do NOT loosen the `+0.08` floor
   (the parent evidence doc explicitly warns against this).
3. **PARTIAL (does NOT auto-merge):** neither of the above, but at least one of
   R/RBI stalls in `[baseline+0.08, 0.85)`. Record it and SURFACE to the user for
   an explicit go/no-go -- a PARTIAL never merges on the implementer's own
   judgment.
4. **PASS (auto-merge eligible):** both R and RBI land in `[0.85, 1.20)` (and
   gates #2/#3 already held at step 0). Proceed to merge.

There is NO one-line "off switch"; the bench sampling (PR #146) and this
replacement sampling share the per-game-rate machinery, so the fallback for an
over-correction is a model adjustment (above), surfaced through the gate, not a
silent revert.

### Calibration levers (for PARTIAL/over-correction tuning)

The dominant injected variance is the NegBin count dispersion (`STAT_DISPERSION`,
a function of `mu = base*scale`), so the model should move R/RBI materially. Two
levers if it lands off-target, recorded so tuning is principled, not flailing:

- **Curve volume (`repl_pa_volume`)** sets the scrub's playing-time `cv_pt` band.
  Sampling at the replacement line's near-full-time `repl_ab` (~423-520 for the
  Core-8 hitter replacement lines) gives a LOW PT-availability variance -- realistic replacement scrubs (call-ups/demotions)
  have more PT volatility. If the result is a PARTIAL (under-injection), a smaller
  effective volume (higher `cv_pt` band) is the lever to widen it; if it
  over-corrects, this same band (with the tail in "Tail behavior") is the first
  suspect. The plan picks the `repl_ab`-derived band as the principled default and
  treats it as the tuning knob.
- **The `1/scale` tail** (see "Tail behavior") -- the rate cap (and the
  proper-NegBin fallback) named there are the over-correction levers. A capacity
  cap is the named ANTI-PATTERN there (terminal pool), NOT a lever -- do not use
  it.

## Reuse (no new sampling math)

- `_apply_variance_batch` -- the batched NegBin + copula draw (called once more,
  for the replacement bodies). Feeding it the synthetic `_replacement_line` dicts
  is correct ONLY because all three of these hold simultaneously: (a) `pt_volumes`
  is supplied explicitly (so the `:777` per-player volume read is bypassed -- the
  replacement dict has no volume field of its own); (b) `suppress_repl=True` (so
  the unconditional `_replacement_line(p, ...)` at `:810`, which would return
  `_GENERIC_HITTER_REPL` for a positionless synthetic dict, is computed but
  discarded -- no replacement-of-a-replacement double count); (c) the counting
  cols are read via `safe_float(p.get(col))`, which the replacement dict carries.
  The plan must keep all three; dropping any one silently corrupts the draw.
- `_replacement_line` -- position-routed replacement bundle (unchanged; returns
  SHARED read-only module constants -- see the no-mutate note in test impact).
- `PA_PER_GAME` -- implied-games denominator constant (unchanged). `AB_PER_PA` --
  the AB->PA conversion for the curve volume (matches `_full_season_pt_volume`).
- The mean-neutral per-game decomposition pattern -- mirrors the bench path in
  `_simulate_team_hitters_ros_direct` exactly (`realized / (vol * scale)`).
- `HITTER_CORR_MATRIX` / `_negbin_copula_counts` -- the category copula.

## Files (anticipated)

- `src/fantasy_baseball/mc_fill.py`: change `replacement_for` to a per-game
  contract; simplify the residual branch (delete the `repl_ab/PA_PER_GAME`
  conversion at `~100-106`). REWRITE the `allocate_bench_fill` docstring
  (`~44-55`), which currently describes the deleted conversion in detail
  ("residual -> replacement per-game (replacement total / (repl_ab / PA_PER_GAME)
  ...)") -- that sentence becomes actively false and misleading after the flip;
  pin it to the new "caller supplies a per-game replacement line" contract, do not
  leave a generic "update docstrings".
- `src/fantasy_baseball/simulation.py`: in `_simulate_team_hitters_ros_direct`,
  add the batched replacement-body draw (after the bench draw) + per-(iter,active)
  per-game rate decomposition; pass the sampled per-game `replacement_for` into
  `allocate_bench_fill`; update the docstring. The single production caller is
  `_repl_for` (`~simulation.py:927`); it must flip from full-season to per-game in
  the SAME change.

### The `replacement_for` contract flip is TYPE-INVISIBLE -- enumerate every consumer

Both the old and new contracts are `dict[str, float]`, so mypy/ruff CANNOT catch a
half-applied flip: a consumer still returning a full-season line while the
allocator assumes per-game silently over-fills by ~`repl_ab/PA_PER_GAME` (~120x).
Every consumer below must flip atomically; the listed tests are the only safety
net.

- `tests/test_mc_fill.py`:
  - `_flat_replacement` / `_realistic_replacement` factories (`~51-58`) return
    FULL-SEASON lines -> convert to per-game (or have the tests pass per-game lines
    directly) so they match the new contract.
  - `test_replacement_per_game_not_overscaled` (`~96-106`) exists SOLELY to test
    the allocator's now-deleted `repl_ab/PA_PER_GAME` conversion. Under the new
    contract it is vacuous: RETARGET it to the caller (where the conversion now
    lives) or DELETE it with the reason recorded -- do not leave it asserting a
    conversion the allocator no longer does.
  - `test_position_mismatch_routes_to_replacement_not_bench` (`~83-93`) and
    `test_residual_goes_to_replacement_when_bench_exhausted` (`~172-180`) feed
    `replacement_for` lambdas and assert on the residual magnitude -> update their
    lambdas + expected values to the per-game contract. If the lambda stays
    full-season but the allocator stops converting, they break (~120x too high) --
    that is the contract flip working, not a regression; fix the test inputs, do
    NOT loosen the assertion.
- `tests/test_mc_integration.py`: new replacement-sampling de-bias /
  mean-neutrality tests; re-pin shifted goldens with justification; handle the
  SD-ceiling guardrail per "RNG stream and test impact".

### Caller wiring note (avoid an unhashable-key trap)

The allocator calls `replacement_for(a.body)` with the `ActiveBody`. `ActiveBody`
wraps a `Player` and may not be hashable, so a `{body: line}` dict is unsafe.
PREFER index alignment: the replacement draw is built in active-body order, so the
sampled per-game lines are an `(n_iter, n_active)`-shaped structure indexed by the
active body's position; the per-iteration `replacement_for` closure resolves the
active body to its index (the same enumeration used to build `ActiveSample`s) and
returns that column. `id(active_body)` keying also works (the bodies outlive the
loop, so no id reuse) but index alignment is strictly safer and is the default.

### AVG components

The residual fill loops ALL `HITTING_COUNTING`, including `h` and `ab`, so sampling
the replacement term also adds noise to the `ros_h` / `ros_ab` AVG components (as
the bench fill already does). `ab` stays mean-exact (`out["ab"] = base*scale`, so
`realized_ab/(repl_implied_games*scale) = base_ab/repl_implied_games = PA_PER_GAME`
deterministically); `h` carries count noise. This mirrors the active/bench AVG
asymmetry and needs no special handling -- noted so the AVG impact is not a
surprise.

## ASCII-only

All source, comments, strings, and this doc are ASCII (Windows cp1252 stdout).
