# Barrel-anchored HR level for the breakout diagnostic

Issue: #262 follow-on (the confirm-gate backtest's real finding). Feature line: keeper
breakout/mirage diagnostic (`src/fantasy_baseball/analysis/breakout.py`). Builds on
`docs/superpowers/specs/2026-07-26-hr-confirmation-backtest-design.md` and its harness
`src/fantasy_baseball/analysis/hr_confirm.py`.

## Problem

The #262 backtest tested barrels/xHR only as a *confirmation multiplier* (a bounded
lever on the shared `surface -> Marcel` blend) and found a tie. But a direct-level
test (Spearman of a current-year signal vs next-year HR/PA, held-out 2021-2024,
bootstrap CIs) told a different story:

| signal | ALL n=1034 | TOP-100/yr | TOP-50/yr |
|--------|-----------|-----------|-----------|
| surface HR/PA | 0.512 | 0.346 | 0.291 |
| xHR/PA | 0.525 | 0.356 | 0.347 |
| **barrel-expected HR/PA** | **0.569** | **0.458** | **0.396** |
| shipped surface->Marcel line | 0.552 | 0.406 | 0.362 |

barrel-expected HR (a calibrated function of barrels-per-PA) **significantly** beats
raw surface HR (CI on the diff excludes 0 at every cut) and beats xHR; xHR only beats
surface directionally (CI includes 0). The gain is largest exactly among the
keeper-relevant sluggers. The confirm-gate architecture is too weak a lever to capture
this; the fix is to use barrel-expected as the HR *level/regression target*, not a
confirm gate.

## Goal

For HR in the breakout diagnostic, regress the surface HR rate toward a
barrel-expected HR rate (a skill anchor) instead of toward the Marcel prior, and ship
it **only if** that barrel-anchored forecast beats the current shipped HR line on
held-out years (bootstrap CI on the Spearman difference excludes 0). The change is to
the HR *level estimate* (and the mirage/breakout label's HR baseline that rides on
it), not the confirm mechanism.

## Non-goals

- Any stat other than HR. Only HR's anchor/weight change.
- xHR (loses to barrels; not wired in). The #262 xHR fetcher stays as-is (unused by the
  live path; still exercised by the backtest).
- Multi-year barrel history. The anchor is **single-year** `brl_pa` (barrels settle
  fast, and single-year `brl_pa` is what won the direct-level test). Multi-year is a
  possible follow-up.
- Separately validating the mirage/breakout *label boundary*. It stays "unvalidated"
  as it already is; the label's HR baseline shifting to barrel-expected is a
  sign-consistent consequence of the level change, not a separately backtested claim.
- 3-way (surface/Marcel/barrel) blends. Rejected in brainstorming (overfit risk on a
  small corpus); the anchor is barrel-expected with a Marcel fallback only.

## Approach

### The estimate (weighted-average form)

For HR, when `brl_pa` is available:

```
barrel_expected = A + B * brl_pa                       # frozen calibration
adj_hr = w_b * barrel_expected + (1 - w_b) * surface_hr
```

`w_b in [0,1]` is the weight on the barrel skill anchor (`w_b=1` -> pure barrels,
`w_b=0` -> pure surface). In `adjust_line`'s existing shrinkage idiom this is the
algebraically identical `adj_hr = barrel_expected + w_s * (surface_hr - barrel_expected)`
with `w_s = 1 - w_b`; the code keeps the shrinkage shape (minimal change to the loop),
the spec/report state the weighted-average form (clearer).

`A`, `B`, and the weight are **frozen constants** fit by the gating backtest (below),
living in `breakout.py` alongside the existing stabilization constants.

### Weight form

- **Primary:** a single flat `w_b` tuned by grid search on fit-years.
- **Reported variant:** a reliability-scaled weight (more surface weight for higher-PA
  players), `w_s = reliability * cw`. Ship whichever generalizes better on held-out
  years; default to flat unless the scaled one clearly wins (fewer knobs).

### Fallback (no barrels)

When `brl_pa` is absent (low-PA players with no batted-ball data on the Savant barrels
leaderboard), HR keeps the **current** `surface -> Marcel` confirm blend unchanged.
Both the weight and the anchor key on `brl_pa is not None`, so the fallback is a clean
either/or, and no player loses a valid HR estimate.

### What is removed

The HR branch of the confirm gate (`_confirm_gap(slg, xslg, 0.150)`) is superseded for
barrel-covered players (it demonstrably added nothing). It remains only on the Marcel
fallback path.

## The gating backtest (extends `hr_confirm.py`, committed -- not scratch)

Two additions to the harness + a new report script section (or a sibling script
`scripts/backtest_hr_level.py` reusing the same corpus builder):

1. **Direct-level test (captures the finding):** Spearman of `barrel_expected`,
   `surface_hr`, `xhr_rate` vs next-year HR/PA, with bootstrap CIs on the pairwise
   differences, on the full population + top-100/top-50 per year. This is the evidence
   for the change and must be reproducible in-repo.

2. **Level-blend gate:** forward HR `= barrel_expected + w_s*(surface - barrel_expected)`,
   `w_s` (or `w_b`) tuned on fit-years by max fit-year Spearman (grid search, ordinal
   `_spearman`), scored on held-out report years against the **shipped** HR line
   (`prior + reliability*((1-cw)+cw*confirm_xslg)*(surface - prior)` -- i.e. the
   harness's existing `xslg_shipped` forward). Metrics: Spearman + rate-MAE + a
   **seeded** bootstrap CI on `Spearman(barrel-anchored) - Spearman(shipped)`.

**Gate rule:** barrel-anchored ships **only if** its Spearman-difference CI vs the
shipped line **excludes 0** (lower bound > 0), on the full applied population (that is
where the wire-in acts). Top-100/top-50 are reported as robustness (the edge was
largest there; it must not invert). Also report the flat-vs-reliability-scaled weight
comparison; freeze the winner's `A`, `B`, `w`.

**Population:** the same `build_hr_records` common-support + PA-floor + HR-mover
filters as #262 (barrel-covered players only, which is exactly the wire-in's
applicability set). Fit 2016-2020, report 2021-2024.

## Freeze & wire-in (Phase 2, gated on the backtest)

Only if the gate clears:

- **Constants in `breakout.py`:** `HR_BARREL_SLOPE (B)`, `HR_BARREL_INTERCEPT (A)`,
  `HR_BARREL_WEIGHT` (flat `w_b`) or the reliability `cw` -- fit on all available
  seasons (2016..latest), frozen with a one-line provenance comment (the backtest that
  produced them), analogous to the existing seed constants.
- **`adjust_line` HR override:** when `row.brl_pa is not None`, set HR's baseline
  `p_rate = barrel_expected(row.brl_pa)` and HR's weight to the frozen barrel weight,
  so `adj_hr`, the believed/surface deviation, and the label's HR contribution all use
  barrel-expected as the HR baseline consistently. Else, current behavior.
- **`w_for_stat("hr")`:** returns the barrel weight when `brl_pa` present, else the
  current `reliability*confirm_xslg`.
- **Report (`run_breakout_report.py`):** surface `barrel_expected` for HR next to
  surface HR so the driver of the HR adjustment is visible.

## Testing

- Unit: `barrel_expected` from frozen constants; the weighted-average identity
  (`w_b*barrel + (1-w_b)*surface == barrel + (1-w_b)*(surface-barrel)`); the tuning
  picks the grid argmax; the gate rule (CI excludes 0) pinned on a synthetic result.
- `adjust_line` HR path: a barrels-backed HR surge holds (adj near surface) vs a
  barrels-thin HR surge regresses toward barrel_expected; the **fallback** path
  (`brl_pa is None`) reproduces the pre-change adjusted line exactly (a pinned
  characterization test, so non-barrel players are provably unaffected).
- Determinism: the gate CI is reproducible (seeded bootstrap), pinned.
- Existing `breakout`/`hr_confirm` suites stay green.

## Risks / open points

- **Smoothing confound:** `barrel_expected` is a regression-fit (hence pre-smoothed)
  estimator, so part of its raw-level edge is regression-to-mean. The gate controls for
  this by comparing against the **shipped** line, which *also* regresses (toward
  Marcel) -- not against raw surface. If barrel-anchored can't beat the shipped
  regressed line on held-out years, it does not ship.
- **Calibration vintage:** `A`, `B` are frozen from historical seasons; the Savant
  barrel definition is stable, but the constants should carry a provenance comment and
  a note to refresh if Statcast redefines barrels.
- **Label not separately validated:** the HR label baseline moving to barrel-expected
  is a consequence, not a gated claim; the label boundary remains flagged unvalidated.
- **Small-sample top-50** CIs are wide; top-N is robustness, not the gate.
- If the gate does **not** clear (barrel-anchored fails to beat the shipped line on the
  full held-out population despite the direct-level edge), we do NOT wire in; the
  deliverable is the committed direct-level + gate backtest and that negative result.
