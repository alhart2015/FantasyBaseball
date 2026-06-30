# Sampled bench injury-fill -- SD-calibration acceptance evidence (2026-06-29)

## Verdict: FAIL (gate #1). Mean-safe, but the de-bias is ~1/10th of what the gate requires.

The sampled bench fill is directionally correct and mean-neutral, but it moves
the R and RBI SD ratios by only +0.008 / +0.009 -- far short of the required
+0.08 floor. Root cause: ~77% of all injury-filled games are served by the
still-deterministic replacement line (only ~23% by the now-sampled bench), so
bench-only sampling cannot inject enough variance. This is exactly adversarial
finding #5 ("increased replacement firing") in the spec, and its named follow-up
-- sampling the replacement line -- is now a prerequisite for passing gate #1,
not an optional refinement.

## Run conditions

- Diagnostic: `FB_SELECTION_ATTRIBUTION=1`, written to `phase0_attribution.txt`
  (selection-attribution table + SD-calibration table + an added
  replacement-fill-share line, see Instrumentation).
- seed=42, n_iter=1000, fraction_remaining=0.4919 (identical for both runs --
  same live horizon, so the comparison is apples-to-apples).
- Entry point: a local refresh via `run_full_refresh()` against the LOCAL SQLite
  KV (RENDER unset), driven by a throwaway `scripts/_task5_diag.py`. NOT
  `run_season_dashboard.py` (that launches a blocking Flask server and does not
  itself trigger a refresh) and NOT `scripts/refresh_remote.py` (that writes to
  REMOTE Upstash -- would clobber production). The direct `run_full_refresh()`
  call is the same pattern `scripts/run_lineup.py` uses; it writes local-only and
  emits the attribution file. Reported per the brief's "say what you used".
- BEFORE = commit 94f48f3 (Task 2 stopgap, byte-identical-deterministic fill).
  AFTER = branch `mc-sampled-bench-fill` HEAD 53ac787 (sampled fill).
- BEFORE was run twice (once concurrently with AFTER's tail, once clean and
  sequential after AFTER fully finished). Both BEFORE runs produced IDENTICAL
  numbers (seed=42 deterministic, same inputs), so concurrency had no effect;
  the clean sequential BEFORE is reported below.

## Gate 1 + 2: SD calibration (mc_sd / analytic_sd), cross-team mean

| cat | BEFORE | AFTER | delta |
|-----|-------:|------:|------:|
| R   | 0.739 | 0.747 | +0.008 |
| HR  | 0.850 | 0.859 | +0.009 |
| RBI | 0.704 | 0.713 | +0.009 |
| SB  | 0.918 | 0.933 | +0.016 |
| W   | 1.272 | 1.254 | -0.017 |
| K   | 1.007 | 1.001 | -0.006 |
| SV  | 1.172 | 1.164 | -0.008 |
| POOLED (median of finite team-cats) | 0.914 | 0.935 | +0.021 |

Per-team R / RBI SD ratios (the two gate-#1 cats):

| team | R bef | R aft | RBI bef | RBI aft |
|------|------:|------:|--------:|--------:|
| Boston Estrellas | 0.653 | 0.653 | 0.644 | 0.644 |
| Hart of the Order | 0.756 | 0.764 | 0.704 | 0.739 |
| Hello Peanuts! | 0.745 | 0.748 | 0.683 | 0.687 |
| Jon's Underdogs | 0.754 | 0.786 | 0.751 | 0.752 |
| Send in the Cavalli | 0.780 | 0.790 | 0.736 | 0.725 |
| SkeleThor | 0.731 | 0.736 | 0.716 | 0.709 |
| Spacemen | 0.699 | 0.732 | 0.663 | 0.698 |
| Springfield Isotopes | 0.729 | 0.704 | 0.685 | 0.689 |
| Tortured Baseball Department | 0.819 | 0.806 | 0.745 | 0.761 |
| Work in Progress | 0.723 | 0.748 | 0.717 | 0.728 |

The pre-change baseline cross-checks against the published Phase-6 evidence
(R 0.72, RBI 0.70, pooled 0.92) within run-to-run noise.

### Gate #1 (R and RBI rise >= +0.08, neither > 1.20, target >= 0.85): FAIL

- R: 0.739 -> 0.747 = +0.008. Required floor +0.08. FAIL (improvement is ~10x
  too small). Lands at 0.747, below both the 0.85 target and the 0.819
  partial-success floor (baseline+0.08).
- RBI: 0.704 -> 0.713 = +0.009. Required floor +0.08. FAIL (same magnitude).
- Neither ratio exceeds 1.20 (both ~0.71-0.75): the no-over-correction bound
  holds; this is an under-correction, not an over-correction.
- This is NOT even the partial-success window `[baseline+0.08, 0.85)` -- the
  ratios did not clear the +0.08 floor at all.

### Gate #2 (pooled in [0.8, 1.25]): PASS

Pooled 0.935 (was 0.914), comfortably inside [0.8, 1.25]. The small pooled rise
is carried mostly by the SB/HR hitter cats and a slight softening of the
over-wide pitcher W (1.272 -> 1.254), not by R/RBI.

## Gate 3: hitter category team-total mean drift (new_engine median)

Means taken as the new_engine median team totals from the attribution table
(the MC central tendency that the bench fill actually moves; the ERoto
`standings_breakdown` means are bench-fill-independent and identical before/after
by construction -- see note). League-sum drift:

| cat | BEFORE | AFTER | delta | pct |
|-----|-------:|------:|------:|----:|
| R   | 9562.6 | 9560.9 | -1.8 | -0.02% |
| HR  | 2692.4 | 2694.1 | +1.7 | +0.06% |
| RBI | 9155.1 | 9154.7 | -0.4 | -0.00% |
| SB  | 1494.0 | 1493.4 | -0.6 | -0.04% |

Worst per-team UPWARD drift across all 10 teams: R +0.1%, HR +0.3%, RBI +0.2%,
SB +0.5%. All far below the +1% hard-fail bound. Downward drifts are all tiny
(<<5%).

### Gate #3 (no upward drift > +1%; downward up to ~5% OK): PASS

The mean-neutral decomposition holds -- no hitter cat shows an upward mean shift
anywhere near the +1% hard-fail threshold. The intended finding-#5 downward
re-damping is present but negligible at this fraction_remaining (the fill premium
itself is small). Note: because gate #1 fails, the small downward pressure is not
the concern here; the headline is that the fill barely adds variance.

Note on the means source: the brief suggested reading
`cache:standings_breakdown` (team_ytd[col] + sum(player.contribution_stats[col])).
That payload is the deterministic ERoto projection, which the bench-fill change
does NOT touch -- it is byte-identical before/after, so a drift check against it
is trivially 0% and uninformative. The meaningful, gate-relevant mean is the MC
new_engine team total (ERoto + bench-fill premium), which the attribution table
emits as a per-cat median; that is what is compared above. The ERoto means were
also captured (`standings_breakdown`, effective horizon 2026-07-07) and confirm
new_engine ~= ERoto + a small premium, matching Phase-6.

## Gate 4: replacement-fill share

Fraction of all injury-filled games served by the deterministic replacement line
(team-summed `replacement_games / total_filled_games` across the new_engine arm,
8... 10 teams x 1000 iters):

| run | replacement_games | total_filled_games | share |
|-----|------------------:|-------------------:|------:|
| BEFORE | 673,043 | 880,489 | 0.7644 |
| AFTER  | 678,294 | 879,360 | 0.7713 |

delta share = +0.0069 (+0.69 percentage points).

### Gate #4 assessment

The share rose only modestly (+0.69pp), so the rise itself is not "material" in
the narrow sense. But the ABSOLUTE level is the story: ~77% of all filled games
are served by the still-deterministic replacement line in BOTH runs. The sampled
change touches only the ~23% bench portion, which is why R/RBI move by less than
a hundredth. The small upward delta is the spec's predicted finding-#5
re-damping (sampled capacity g_ros_full*scale averages slightly below the old
deterministic g_ros_full), and it works against the de-bias.

## Conclusion and recommendation

- The change is SAFE (mean-neutral, no over-dispersion, pooled stays calibrated)
  but INSUFFICIENT: it fails gate #1's +0.08 improvement floor for both R and RBI
  by a factor of ~10.
- The mechanism is understood and was anticipated by the spec: with benches this
  thin (~2 hitters) versus ~10 actives each shedding 6-25% of games every
  iteration, ~77% of the shortfall cascades past the bench to the deterministic
  replacement line. Sampling only the bench leaves the dominant fill term
  zero-variance, so team-total R/RBI dispersion barely rises.
- RECOMMENDATION: do NOT merge as the acceptance-gate-passing fix. Implement the
  spec's deferred follow-up -- sample the terminal replacement line (give it the
  same NegBin + availability treatment as bench bodies) -- which is where ~77% of
  the missing variance lives. Re-run this gate afterward. Loosening the +0.08
  floor post hoc is the wrong call (the spec explicitly warns against it).

## Instrumentation (evidence-only; reverted)

Gate #4 needed a replacement-fill-share counter that the stock diagnostic does
not emit. Added behind `FB_SELECTION_ATTRIBUTION`:
- `mc_fill.FillResult.replacement_games` (default 0.0) accumulated in
  `allocate_bench_fill` (the residual `need` routed to the replacement line).
- `simulation._REPL_FILL_ACCUM` (module-level [repl_games, total_filled_games])
  summed inside `_simulate_team_hitters_ros_direct`'s fill loop when the env is
  set; total_filled = sum over actives of frac_missed * g_ros_adj.
- `web/refresh_pipeline.py` resets the accumulator before
  `run_selection_attribution` and writes the share line to the attribution file.

This instrumentation was applied to BOTH 94f48f3 and HEAD for the runs, then
REVERTED -- the working tree is left clean on branch `mc-sampled-bench-fill`.
The throwaway driver scripts (`scripts/_task5_*.py`) were removed.
