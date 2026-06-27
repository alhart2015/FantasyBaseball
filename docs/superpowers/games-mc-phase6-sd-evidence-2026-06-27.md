# Phase 6 SD-calibration gate evidence (2026-06-27) -- PASS

Run: local refresh, FB_SELECTION_ATTRIBUTION=1, seed=42, n_iter=1000,
frac=0.5027. Raw 4-arm means + SD table: -raw.txt.

## Verdict: GATE PASSES. Pooled SD ratio = 0.923 (calibrated, in [0.8, 1.25]).

The cv_pt-volume bug the plan-review caught (ROS-direct looked up the PT curve at
ROS volume -> SDs ~1.5x too wide) is FIXED (Task 1, commit cbcc868). Post-fix,
mc_sd / analytic_sd (analytic = project_team_sds, validated vs realized 2022-2025
by backtest_sd_calibration.py) per counting cat, mean across 8 teams:

| cat | R | HR | RBI | SB | W | K | SV |
|---|---|---|---|---|---|---|---|
| ratio | 0.72 | 0.85 | 0.70 | 0.92 | 1.27 | 1.01 | 1.13 |

Pooled median 0.923. Hitter median 0.77; pitcher median 1.08.

## Means re-validated (the Task-1 fix did not regress them)
- HITTERS: new_engine ~= ERoto + a SMALL bench-fill premium (SkeleThor RBI +14,
  Hart +11, Cavalli +4 -- smaller than Phase-4's +22 because the fix raised the
  active mean 0.75->0.94, so the fill restores less). Still fixes the top-k
  over-credit (Cavalli R 993 top-k -> 811 ~= ERoto 804).
- PITCHERS: ~= ERoto EXACTLY (W within 0.5, K within 4.6) -- unchanged by the fix
  (pt_mean_fraction=0 -> volume-independent mean).

## Documented residuals (within acknowledged/deferred approximations -- NOT new bugs)
- **R/RBI ~0.70 (MC ~30% tighter than analytic).** The bench injury-fill restores
  injured games deterministically, damping team-total variance -- the spec's
  acknowledged `f^2` partial-fill variance understatement ("Variance note":
  bounded, still strictly MORE realistic than the old DETERMINISTIC replacement
  fill which had ZERO fill variance). The deferred refinement (per spec) is
  partial-volume re-sampling of the fill body; not done here. HR/SB (0.85/0.92)
  are in band. (May also reflect the analytic reference's active-set/displacement
  definition vs the engine's graded factor -- the noted Task-3 caveat.)
- **W ~1.27 (slightly wide).** Low-count pitcher stat; the pitcher copula
  correlation vs the analytic's player-independence assumption. Pre-existing
  (the full-season MC shares the same copula), not introduced by this engine.
  K (1.01) and SV (1.13) are in band.

## Bottom line
The gross SD bug is fixed and the engine's SDs are broadly calibrated (pooled
0.92). Residual per-cat structure is within the spec's acknowledged-and-deferred
approximations. Full engine (Phases 0-6) validated: means (hitters = ERoto +
small premium; pitchers = ERoto) AND SDs (pooled calibrated).
