# Closer Role-Mixture Model for Saves Variance

**Issue:** #193 -- Saves variance vs realized: re-validate under the unified NegBin model
**Date:** 2026-07-10
**Status:** Design approved, spec-review hardened

## Problem

`scripts/backtest_sd_calibration.py` computes standardized team-category residuals
`z = (actual_team_total - projected_team_total) / eroto_SD` across the seasons in `YEARS`.
Under the unified NegBin dispersion model, every counting category is calibrated
(`SD(z)` in 0.9-1.3) EXCEPT saves:

```
== MATCHED-ONLY ==            == DNP=0 ==
  SV  mean +0.80  SD 2.43       SV  mean +0.56  SD 2.07
```

**Sample note:** these SV numbers are measured on **2022-2024 only** (`build_year` currently
excludes 2025 pitchers). The "~2.2x" is a 3-year figure.

Two distinct defects:

1. **Variance ~2.2x too tight.** Realized team SV totals scatter ~twice as wide as modeled.
2. **Positive mean bias (+0.6 to +0.8 sigma).** Realized team SV exceeds projected.

This design fixes **(1) only** (see Scope).

### Root cause

For a projected 30-SV closer the model variance splits ~74% playing-time (`proj^2 * cv_pt^2`,
cv_pt=0.417) and ~26% NegBin performance (`negbin_perf_variance('sv', proj)`, r=37.757).
Closing the gap via `cv_pt` needs cv_pt ~= 1.0 -- absurd. The diagnosis: SV is not
over-dispersed *unimodal* data; it is a **discrete role switch** (a closer keeps the job
~35 SV or loses it ~5; a setup man can vault in). The distribution is **bimodal**; a smooth
`cv_pt` matches the second moment but not the shape.

## Scope

**In scope:** fix the SV **variance** via an explicit bimodal role-switch mixture, in both
the analytic ERoto path and the Monte Carlo sampler.

**Out of scope:** the **mean bias** (a projection-quality problem; #235). This design is
**mean-neutral** -- it changes only variance and leaves *each path's existing SV mean
exactly as today*. In-season scaling is handled by existing machinery in both paths
(`build_team_sds` in ERoto, the copula + `X'` in the MC), so `sv_role_variance` is
full-season and this design adds no new in-season plumbing (see In-season).

## Design decisions (resolved during brainstorming + spec review)

1. **Bimodal two-component mixture**, keyed on projected SV.
2. **Independent per-pitcher** (keeps the per-player quadrature and ERoto/MC unification;
   handcuff caveat in Known limitations).
3. **Continuous parameters, not bands** -- the mixture's knobs are smooth functions of
   projected SV, fit on the whole pitcher population, pooling the thin closer data
   (~23-33 established closers/season) and avoiding parameter cliffs. Bimodal at every `s`.
4. **Mean-neutral** -- the mixture is a mean-1 multiplier on each pitcher's *current* SV
   mean; it replaces the SV playing-time (`cv_pt`) variance term with the mixture's
   role-switch variance and does not touch the mean machinery. Valuation (SGP/VAR/VONA)
   reads the projection and is untouched.
5. **In-season aware** -- the role-switch variance scales down with remaining season.

## The model

Two-component latent role state per pitcher, expressed as a **mean-1 multiplier** `X` on the
pitcher's *current* per-iteration SV mean `mu0` (whatever the existing model already produces
-- projection, playing-time haircut, and, in the MC, injury backfill). Let `s` = projected
SV. The two components are the **modal** outcome (role unchanged, probability `q(s)`) and the
**surprise** outcome (role change, probability `1-q(s)`):

```
X ~ { a_m(s) with prob q(s);  a_s(s) with prob 1 - q(s) }
constraint:  q*a_m + (1-q)*a_s = 1        # E[X] = 1  ->  mean preserved exactly
per-draw SV mean = mu0 * X
```

The **surprise multiplier `a_s(s)` changes direction across the SV spectrum**, and this is
the crux the earlier drafts got wrong:

- **Established closer (high `s`):** the surprise is *losing the job*, so `a_s -> 0` (few
  saves), with `1-q` the job-loss rate; `a_m` (holds) is slightly `> 1` (a held closer beats
  his loss-hedged projection).
- **Middle reliever (low `s`):** the surprise is *vaulting into the job*, so `a_s >> 1` (a
  large multiple of a near-zero projection), with `1-q` the small vault rate; `a_m` (stays)
  is `<= 1`.

So `a_s(s)` is a smooth curve running from large at low `s`, crossing 1 near mid-`s`, to ~0
at high `s`; `q(s)` and `a_m(s)` follow from the mean-1 constraint. One continuous
parameterization covers both regimes -- there is no hard closer/non-closer switch.

### Variance -- MUST match the MC's generative process (invariant, not just a formula)

The MC samples `SV ~ NegBin(mu0 * X, r)`. The analytic ERoto variance must equal the **full**
law of total variance over BOTH the role state and the NegBin, or the shared-`mu0` invariant test
(#1) fails. With `nb_var(m) = m + m^2/r`:

```
within  = q * nb_var(mu0*a_m) + (1-q) * nb_var(mu0*a_s)     # per COMPONENT, not at mu0
between = mu0^2 * q*(1-q) * (a_m - a_s)^2
Var(SV) = within + between
```

**Both components use the same within-dispersion `r = 37.757`.** Evaluating `within` at the
single mean `mu0` (as a prior draft did) is WRONG -- it omits the `between/r` cross-term
(verified: ~2.4% of the total for a 30-SV closer, a systematic gap the shared-`mu0` unit test
(#1) detects).
The reference identity is `within + between = negbin_perf_variance(mu0) + between*(1 + 1/r)`;
`closer_mixture.sv_role_variance` implements the per-component form so it is bit-consistent
with what the MC generates. `r` is **fixed** at 37.757 (not fit): the modal/hold component IS
the role-stable population that value was fit on (`constants.py:158-159`), so all excess
spread flows through `between`, never through inflating `r`. (This retracts the iteration-2
"loosen r_primary" idea -- the survivors are the modal component.)

### In-season (`fraction_remaining`)

The role-switch (`between`) variance must vanish as the season settles (a held role stops
being a coin flip). `sv_role_variance` is defined **full-season** -- exactly mirroring the
`cv_pt` playing-time term it replaces, which was also full-season -- and the
`fraction_remaining` scaling is applied **externally and uniformly** by machinery that
already exists in both paths:

- **ERoto:** `build_team_sds` multiplies each category's team SD by `sd_scale =
  sqrt(fraction_remaining)` (`scoring.py:1389`), i.e. scales the whole SV variance (within AND
  between) by `frac`. Unchanged by this design; the full-season `sv_role_variance` slots in as
  the per-player term exactly where the `cv_pt` term was, and inherits the same external
  scaling.
- **MC:** the copula scales the within term by `frac` (`_negbin_copula_counts`,
  `var_target = frac*var_full`), and the role multiplier `X' = 1 + sqrt(frac)*(X - 1)` scales
  the between term by `frac`.

So both paths scale within+between by `frac` uniformly and consistently -- there is **no**
ERoto/MC in-season divergence to reconcile (an earlier draft wrongly claimed a within-term
divergence; ERoto's external `build_team_sds` handles it). Consequently **`sv_role_variance`
takes NO `fraction_remaining` parameter** -- only the MC's `role_multiplier_draw` does, for the
`X'` shrink. The success guard (Testing #4) is a property test on `role_multiplier_draw`:
`Var(X') = frac*Var(X)` and `E[X'] = 1`.

At `frac = 1`, cross-path *math* consistency is guarded by the shared-`mu0` invariant
(Testing #1) and each path's own SD+mean **stability** by Testing #3 -- not cross-path
absolute equality (the paths' `mu0` differ).

## Integration seams

New module `src/fantasy_baseball/sgp/closer_mixture.py` -- single source of truth: the smooth
curves `q(s), a_m(s), a_s(s)`, the closed-form full-season `sv_role_variance(s)`
(per-component `within + between`), and the per-draw multiplier the sampler uses. All three
SV call sites route through it.

### A. ERoto analytic (`scoring.py`: `player_category_variance`)

The SV variance term (`scoring.py:1283`, currently `negbin_perf_variance('sv', v) +
v*v*cv_pt_sq`) becomes `closer_mixture.sv_role_variance(v)` (the full-season per-component
`within + between`, keyed on the raw projected SV `v`). W/K keep their `negbin_perf_variance +
cv_pt` term. **No `fraction_remaining` threading** -- ERoto's in-season scaling continues to
flow through the existing external `build_team_sds` `sqrt(frac)` (see In-season). **The mean path
(`project_team_stats`) is untouched.**

### B. Monte Carlo (`simulation.py`: `_apply_variance_batch`, ~781-817)

The cleanest mechanism, given that `scales` is a single array multiplying every correlated
stat (line 794) and also drives `frac_missed = 1 - scales` (804) and thus the backfill
(811-814): **handle SV's variance outside the shared `scales`/copula pipeline.**

1. Compute SV's current-mean base `mu0` = the existing per-iteration SV mean *without* its
   `cv_pt` spread -- i.e. `base['sv'] * eff_mean` (the `eff_mean` location of `scales`, not
   the `z_pt*eff_sd` spread). This is the mean we preserve.
2. Draw the role state per (iter, pitcher): `X` from the mixture (`E[X]=1`).
3. Draw `SV ~ NegBin(mu0 * X, r=37.757)`. The `between` (role-switch) term is scaled by
   `fraction_remaining` by **shrinking the multiplier's two-point support toward its mean of
   1**: draw from `X' = 1 + sqrt(frac)*(X - 1)`, so `Var(X') = frac*Var(X)` and thus
   `between` scales by `frac` -- matching ERoto's `between*frac`. The within-component uses
   the copula's existing `fraction_remaining` variance scaling; do NOT additionally frac-scale
   `X`'s within contribution (that would double-scale within). Testing #4 guards that the
   added-variance scaling matches across paths.
4. **Keep the injury backfill unchanged.** Backfill uses the *shared* `frac_missed` (from W/K
   playing time), not an SV-specific one, so it is untouched by step 1 -- mean-neutral (the
   `+8*frac_missed` term is part of the current mean we preserve).

**Consequences, documented as limitations (not bugs):**
- Pulling SV's mean out of the shared `scales` **drops SV's within-component playing-time
  correlation** with W/K/ER (the common `scales` factor). This is a *new* second-order
  correlation loss beyond the copula one below; it does not affect SV's marginal or team-total
  variance (the target).
- The role Bernoulli is independent of the copula latent `all_z`, so the dominant
  between-component does not co-move with er/bb/h -- the model understates "lost the job
  because ERA spiked." Second-order; deferred.
- The `max(0, .)` clamp on `scales` (786) makes today's SV mean sit a hair above
  `base*eff_mean` in the deep-injury tail; defining `mu0 = base*eff_mean` introduces a tiny
  (closers: `eff_mean~0.9`, negligible) downward shift. The mean-parity test (#3) uses a
  tolerance that accommodates this; the plan should confirm it is within tolerance, not zero.

Guards: the **shared-`mu0` invariant (#1)** catches mixture-math errors and the **per-path
mean+SD stability test (#3)** catches any error in this mechanism.

## Calibration

New script `scripts/calibrate_closer_mixture.py`.

- **Source:** 2022-2025 realized-vs-projected (same blends and `data/stats` the backtest uses).
- **2025 data.** `pitchers-2025.csv` has `SV, W, IP, K/9`, no raw `SO/ER/BB/H`. **The SV work
  needs only SV, present directly.** Reconstructing `SO` (for the optional 2025 W/SO inclusion
  you requested) needs the thirds-notation fix (`195.1` = 195 1/3):
  `ip_true = floor(IP) + (IP - floor(IP))*10/3; SO = round(K/9 * ip_true / 9)`. This is the
  definitional inverse of how K/9 was formed (not an independent validation; 2025 carries no
  raw SO to check against), exact to integer precision. H/WHIP stay unavailable (#236).
- **Fitting (continuous, constrained, pooled).** Fit low-order curves for the mixture over ALL
  pitchers' `(projected s, realized SV)` pairs by maximum likelihood. Parameterize so the
  mean-1 constraint holds *by construction* -- e.g. free `q(s)` (logistic) and the surprise
  level `a_s(s)`, with `a_m(s) = (1 - (1-q)*a_s)/q` derived -- rather than fitting freely and
  rescaling afterward (which is not the MLE and can yield an invalid `a_m < 0`). Add an
  explicit `a_m, a_s >= 0` feasibility guard. `r` is fixed at 37.757 (not fit). The low-`s`
  vault tail is weakly identified (rare events); the smooth curve borrows strength from the
  whole population, but the calibration must report the vault-tail's effective sample support.
- **Backtest is validation, not fitting** -- no curve parameter is tuned to `SD(z)`.
- **Output:** `SV_ROLE_MIXTURE` (fitted curve coefficients) in `utils/constants.py`.

### Feasibility (team-total, both ends of the SV spectrum)

Gate: `SD(z)` in `[0.8, 1.25]`; from 2.43 the 1.25 edge needs team SV variance ~3.8x
(`(2.43/1.25)^2`, verified). A decomposition of realized 2022-24 team SV residual variance
(9-pitcher random teams, matching the backtest) attributes **~44% to pitchers projected < 15
SV** (vault-ins) and ~56% to projected >= 15 (job losers) -- so both ends must contribute.

- **Job-loss end (closers):** with `a_s -> 0`, `between ~= mu0^2 * (1-q)/q`. For a single
  30-SV closer, 3.8x is reached at `q ~= 0.55` (verified numerically), i.e. a ~45% job-turn
  rate on the *random-roster* backtest population; not reached at `q ~= 0.7` (~2.1x). Whether
  the *calibrated* `q(s)` lands there is the empirical question the backtest answers.
- **Vault-in end (relievers):** the small-`q`-tail, large-`a_s` component supplies this via the
  smooth low-`s` curve; a mean-1 multiplier can produce large `between` for tiny `mu0` while
  holding the mean at projection.

**Escalation commitment.** Calibrate the two-component curves from data. If the backtest SV
`SD(z)` lands in `[0.8, 1.25]`, done. If it misses at either end, escalate to a
**three-component** mixture (hold / job-share / lose, plus the vault path) -- decided as the
first plan milestone via the wired backtest, never by tuning `r` or the curves to the gate.
The 44/56 split and the `q`-feasibility figures are design-analysis inputs (not reproducible
from committed artifacts); the wired backtest is the authoritative check. Note the design's own
numbers make three-component escalation a **live likelihood**: `q ~= 0.55` (45% job-turnover)
for a projected-30 closer is high -- established closers turn over well below that, so the
calibrated `q(s)` at the closer end may land nearer 0.7-0.85 (~2.1x, a gate-miss), leaning on
the vault-in end to make up the aggregate. The plan should budget for the three-component
milestone rather than treat it as a remote contingency.

## Backtest changes (`scripts/backtest_sd_calibration.py`)

1. **Wire SV variance to the mixture.** The backtest re-derives SV variance inline
   (`negbin_perf_variance(key, proj) + proj**2 * cvp**2`, line 117); branch SV to
   `closer_mixture.sv_role_variance(proj)`. Required for the gate to be measurable. (The
   backtest is full-season, so no frac enters here regardless.)
2. **Admit 2025 for SV.** `build_year` returns `(hm, None)` unless `{"W","SO","SV"}` are all
   present (line 85), AND line 90 selects `pa[["MLBAMID","W","SO","SV"]]` (a hard `SO`
   reference that KeyErrors on 2025), AND `P_CATS` (line 44) drives per-year `team_z`. So
   admitting 2025-for-SV requires a **per-year category set**: include SV (and W, both present)
   for 2025 while dropping SO from the 2025 column-select and its cat list -- not merely
   relaxing the line-85 gate. (Edit #2 is therefore NOT independent of the SO question; it
   needs either #3 or this per-year restructure.)
3. **(Optional, your request) 2025 W/SO** via the thirds-corrected SO derivation, so W/SO also
   gain 2025.

**Success criterion.**
- **R/HR/RBI/SB** (hitters): `SD(z)` unchanged (2025 hitters already included).
- **W**: present directly in the 2025 actuals, so it gains a 2025 sample as soon as edit #2
  admits 2025 pitchers (independent of the SO derivation) -- `SD(z)` shifts but must stay in
  `[0.8,1.25]` (record before/after).
- **SO**: gains 2025 only if edit #3 (thirds-corrected derivation) is applied; else unchanged.
  When added, must stay in `[0.8,1.25]`.
- **SV**: `SD(z)` into `[0.8, 1.25]`. The **raw** bias `sum(actual)-sum(projected)` is
  unchanged (mean-neutral); the *standardized* `mean z` **shrinks** as the SD grows (e.g.
  +0.80 -> ~+0.4) -- this is expected, not a regression, and `mean z` is not a tuning target.

## Testing

1. **Unit / mixture-math invariant (shared `mu0`)** -- at a common `mu0`, the full-season
   closed-form `sv_role_variance(mu0)` matches a brute-force sample of the generative process
   `NegBin(mu0*X, r)` over the role draw, to a tolerance **tight enough to catch the
   `between/r` cross-term** (~2.4% for a 30-SV closer). This is THE check that ERoto and the MC
   use identical mixture math (both consume `closer_mixture`), independent of each path's `mu0`.
   Also: mean-1 `E[X] == 1`; `a_m, a_s >= 0` over the fitted `s` range. (Do NOT assert cross-`s`
   monotonicity -- the vault curve is legitimately non-monotone at low `s`.)
2. **Integration (target)** -- backtest SV `SD(z)` in `[0.8, 1.25]`; R/HR/RBI/SB unchanged;
   W/SO stay in `[0.8, 1.25]`; SV raw bias unchanged (standardized `mean z` shrinks, expected).
3. **Per-path stability (frac = 1), REQUIRED** -- each path's SV **mean** AND **SD** change only
   by the intended mixture contribution vs that path's OWN pre-change baseline. NOT cross-path
   absolute equality, for either moment: the two paths feed the mixture different `mu0` -- ERoto
   the raw ROS projection `v` (`project_team_stats`/`player_category_variance`, no haircut), the
   MC `base*eff_mean` (haircut) plus the `+8*E[frac_missed]` backfill. That is a pre-existing
   ~2 SV/closer mean gap and a corresponding ~8-10% SD gap (variance scales with `mu0^2`); the
   old cv_pt term had the same asymmetry, and this design inherits and preserves it rather than
   removing it. Cross-path *math* consistency is guaranteed structurally instead: both paths
   compute SV variance through the same `closer_mixture` module (Testing #1), so they cannot
   diverge in the mixture formula. Per-path tolerance accommodates the `max(0,.)` clamp gap.
4. **In-season property (frac in {0.25,0.5,0.75})** -- `role_multiplier_draw` satisfies
   `E[X'] = 1` and `Var(X') = frac*Var(X)` (the MC between-scaling). ERoto's between scaling is
   delivered by the existing external `build_team_sds` `sqrt(frac)` and is not re-implemented,
   so nothing new to test on that side. (Scoped to the MC's added term -- see In-season.)
5. **Valuation regression (guard)** -- SGP/VAR/VONA SV values do not move.
6. **MC re-baseline** -- deterministic-seed MC tests re-pinned for the added Bernoulli.

**Success = (2), (3), and (4).**

## Known limitations

- **Handcuff anti-correlation.** Independent per-pitcher role draws overstate team SV variance
  for an owner who handcuffs both closers of one MLB bullpen (anti-correlated saves). The
  mixture is calibrated on the backtest's independent random rosters; not modeled.
- **Within-component correlation losses** (all second-order, don't touch the marginal target):
  SV pulled out of the shared `scales` loses its playing-time co-movement with W/K/ER; the role
  Bernoulli is independent of the copula, so job-loss doesn't co-move with high ER; and because
  SV's own mean no longer rides `scales` while its `+8*frac_missed` backfill still does, the
  intra-iteration anti-correlation between a closer's own SV dropping and his backfill firing is
  lost (tiny vs the `between` term).
- **Skew / mean-variance separability.** Realized SV is floored at 0 and right-skewed. `SD(z)`
  is mean-centered so the variance target is measurable independent of the mean bias; if the
  fix still leaves `SD(z)` out of band, that is the signal (Integration test) to revisit
  (three-component, or coordinate with #235).

## Files touched

- `src/fantasy_baseball/sgp/closer_mixture.py` (new).
- `src/fantasy_baseball/utils/constants.py` -- `SV_ROLE_MIXTURE`.
- `src/fantasy_baseball/scoring.py` -- SV term in `player_category_variance` (full-season
  `sv_role_variance(v)`, no frac threading); mean path untouched.
- `src/fantasy_baseball/simulation.py` -- SV role draw + out-of-`scales` SV mean/variance in
  `_apply_variance_batch`; backfill and `r` unchanged.
- `scripts/calibrate_closer_mixture.py` (new) -- constrained continuous calibration, 2022-2025.
- `scripts/backtest_sd_calibration.py` -- SV wired to mixture; per-year cat set to admit 2025
  SV; optional 2025 SO derivation.
- **Call-site audit (CLAUDE.md "fix all call sites"):** `player_category_variance` /
  `project_team_sds` are also consumed by `lineup/delta_roto.py` (SV swap-band widths). Its SV
  term switches to the mixture automatically; confirm the wider swap bands are intended and that
  its own `fraction_remaining * total` scaling still composes correctly (it does -- it scales the
  full team variance uniformly, same as `build_team_sds`).
- Tests under `tests/`.

## Out-of-scope / related issues

- **#235** -- all-positions projection-accuracy backtest (owns the SV mean bias).
- **#236** -- re-export 2025 pitcher actuals with the standard counting template.
