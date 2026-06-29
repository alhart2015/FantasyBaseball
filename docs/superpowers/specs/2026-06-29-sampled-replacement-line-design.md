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
scales ~`need^2` -- identical in form to how bench bodies are already sampled
(`per_game_rate * assigned_games`). This was chosen over (a) a proper NegBin draw
at the actual `need` volume (more faithful to the analytic dispersion but the draw
would have to happen after `need` is known, breaking the pure allocator) and (b)
an independent per-game stream (least variance, expected insufficient). Rationale:
consistency with the bench treatment, reuse of existing machinery, a pure
allocator, and it injects the most variance of the pure options (best chance of
clearing the +0.08 floor) while remaining mean-neutral.

## Mechanism

Per call to `_simulate_team_hitters_ros_direct` (HITTERS only), batched across the
active bodies:

1. Build one synthetic replacement flat-dict per active body:
   `_replacement_line(active.player.to_flat_dict(), is_hitter=True)`. This is
   already position-routed (a catcher's replacement gives ~0 SB, a middle
   infielder's ~15) and is a FULL-SEASON counting bundle with NO games field.
2. Compute each replacement body's implied full-season volume from its own line:
   `repl_implied_games = repl_ab / PA_PER_GAME` (the same implied-games heuristic
   the current deterministic path uses), and pass `repl_ab` (PA) as the
   playing-time curve volume.
3. ONE `_apply_variance_batch` draw over all the replacement bodies
   (PlayerType.HITTER, `pt_mean_fraction=1.0`, `suppress_repl=True`,
   `pt_volumes` = the per-body replacement PA volumes), APPENDED AFTER the bench
   draw. Yields realized counts + `scale` per (iter, active).
4. Decompose to a mean-neutral per-game rate, per (iter, active, col):
   `per_game_repl = realized / (repl_implied_games * scale)` when
   `repl_implied_games * scale > eps`, else `0.0`. In expectation
   `E[per_game_repl] = repl_base / repl_implied_games` -- exactly today's
   deterministic per-game replacement rate.
5. In the per-iteration fill loop, the allocator multiplies each active's residual
   `need` by THAT active's sampled per-game replacement line for THAT iteration.

### Availability

The replacement pool is inexhaustible: it always covers the full residual `need`
(no availability haircut on the contribution). The replacement body's `scale` is
sampled (it falls out of `_apply_variance_batch`) only to drive rate noise and is
DIVIDED BACK OUT in step 4 (the mean-neutral denominator). It is NOT used as a
capacity cap -- unlike a bench body, a replacement body has no fill ceiling.

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

`E[realized / (repl_implied_games * scale)] = repl_base / repl_implied_games`.
The denominator MUST be the UNCAPPED `repl_implied_games * scale` (the same
invariant the bench sampling relies on). A denominator capped at
`repl_implied_games` (i.e. dividing by the unscaled volume) would reintroduce an
upward mean bias proportional to `max(1, scale)` -- the exact bug class the
parent spec's spec-review caught. The acceptance gate's mean-drift check (no
upward drift > +1%) is the guard.

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
- The all-empty byte-anchor `test_variance_batch_default_matches_legacy_columns`
  is UNTOUCHED (it exercises `_apply_variance_batch` directly with the legacy
  defaults; this change does not alter that path).
- The active-only "empty-bench" guardrail tests
  (`test_repl_not_double_counted_on_new_path`,
  `test_displacement_factor_scales_hitter_mean`,
  `test_ros_direct_uses_full_season_volume_for_cv_pt`) route their active
  `need` through the NOW-SAMPLED replacement term. Their MEAN / structural
  assertions must still hold (mean-neutral); any value pinned to the old
  deterministic replacement magnitude will be re-captured DELIBERATELY (with the
  reason recorded), never loosened to pass. A guardrail asserting a true
  invariant (e.g. replacement not double-counted, displacement scales the mean)
  that breaks is a real regression, not a golden to re-pin -- STOP and fix.
- Our own de-bias SD test and any pinned post-bench-draw team-total goldens will
  shift (new sampling) and are re-pinned with justification, as in PR #146.

## Acceptance gate (re-run Task 5)

Re-run the SD-calibration diagnostic (`FB_SELECTION_ATTRIBUTION=1`) BEFORE
(current branch HEAD, bench-only) vs AFTER (this change), same seed / n_iter /
fraction_remaining, and confirm ALL of:

1. R and RBI SD ratios each rise to `>= 0.85` (target band `[0.85, 1.20)`),
   with neither exceeding `1.20` (no over-correction). Record actual values.
2. Pooled ratio stays in `[0.8, 1.25]`.
3. Hitter category team-total means (the new_engine MC medians -- the quantity
   the fill actually moves, NOT the bench-independent ERoto `standings_breakdown`
   projection): NO upward drift `> +1%` vs the BEFORE run (hard fail -- the
   mean-neutral decomposition must hold). A downward drift up to ~5% is acceptable
   (the intended replacement re-damping).
4. Replacement-fill share recorded (expected ~unchanged from the ~77% baseline;
   the change samples that term, it does not change how much of it fires).

### Calibration outcomes (surface, do not paper over)

- If R/RBI land in `[0.85, 1.20)` and gate #3 holds: PASS -> proceed to merge.
- If R/RBI OVERSHOOT `1.20`: the correlated `need^2` model over-disperses; the
  plan must surface this (candidate dampers: a partial-correlation factor, or
  switching this term to the proper-NegBin-at-need model). Do NOT ship an
  over-correction.
- If R/RBI rose `>= +0.08` but stall in `[baseline+0.08, 0.85)`: PARTIAL --
  record it, surface to the user, decide whether a further refinement is
  warranted before merge.
- If R/RBI still fail the `+0.08` floor: the model is insufficient; STOP and
  re-design. Do NOT loosen the floor.

There is NO one-line "off switch" required; the bench sampling (PR #146) and this
replacement sampling share the per-game-rate machinery, so the fallback is a model
adjustment (above), surfaced through the gate, not a silent revert.

## Reuse (no new sampling math)

- `_apply_variance_batch` -- the batched NegBin + copula draw (called once more,
  for the replacement bodies).
- `_replacement_line` -- position-routed replacement bundle (unchanged).
- `PA_PER_GAME` -- implied-games conversion constant (unchanged).
- The mean-neutral per-game decomposition pattern -- mirrors the bench path in
  `_simulate_team_hitters_ros_direct` exactly (`realized / (vol * scale)`).
- `HITTER_CORR_MATRIX` / `_negbin_copula_counts` -- the category copula.

## Files (anticipated)

- `src/fantasy_baseball/mc_fill.py`: change `replacement_for` to a per-game
  contract; simplify the residual branch; update docstrings.
- `src/fantasy_baseball/simulation.py`: in `_simulate_team_hitters_ros_direct`,
  add the batched replacement-body draw (after the bench draw) + per-(iter,active)
  per-game rate decomposition; pass the sampled per-game `replacement_for` into
  `allocate_bench_fill`; update the docstring.
- `tests/test_mc_fill.py`, `tests/test_mc_integration.py`: per-game `replacement_for`
  contract update; new replacement-sampling de-bias / mean-neutrality tests;
  re-pin shifted goldens with justification.

## ASCII-only

All source, comments, strings, and this doc are ASCII (Windows cp1252 stdout).
