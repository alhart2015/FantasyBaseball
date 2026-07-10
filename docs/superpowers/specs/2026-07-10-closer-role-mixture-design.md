# Closer Role-Mixture Model for Saves Variance

**Issue:** #193 -- Saves variance vs realized: re-validate under the unified NegBin model
**Date:** 2026-07-10
**Status:** Design approved, in spec-review hardening (iteration 4)

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
exactly as today*. Also out of scope: the pre-existing in-season divergence between ERoto
(full-season variance) and the MC (variance scaled by `fraction_remaining`) for the
*within/performance* term across all counting stats -- this design does not try to reconcile
that; it only requires the *new* role-switch variance to scale consistently (see In-season).

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
law of total variance over BOTH the role state and the NegBin, or the required SD-parity test
(#3) fails. With `nb_var(m) = m + m^2/r`:

```
within  = q * nb_var(mu0*a_m) + (1-q) * nb_var(mu0*a_s)     # per COMPONENT, not at mu0
between = mu0^2 * q*(1-q) * (a_m - a_s)^2
Var(SV) = within + between
```

**Both components use the same within-dispersion `r = 37.757`.** Evaluating `within` at the
single mean `mu0` (as a prior draft did) is WRONG -- it omits the `between/r` cross-term
(verified: ~2.4% of the total for a 30-SV closer, a systematic gap the parity test detects).
The reference identity is `within + between = negbin_perf_variance(mu0) + between*(1 + 1/r)`;
`closer_mixture.sv_role_variance` implements the per-component form so it is bit-consistent
with what the MC generates. `r` is **fixed** at 37.757 (not fit): the modal/hold component IS
the role-stable population that value was fit on (`constants.py:158-159`), so all excess
spread flows through `between`, never through inflating `r`. (This retracts the iteration-2
"loosen r_primary" idea -- the survivors are the modal component.)

### In-season (`fraction_remaining`)

The `between` (role-switch) term is scaled by `fraction_remaining` so it vanishes as the
season settles (a held role stops being a coin flip). This is the ONLY in-season behavior
this design adds, and it must be applied **identically in both paths**. The success guard is
a property test (Testing #4): `between -> 0` as `fraction_remaining -> 0`, and the mixture's
*added* variance matches between the analytic and MC paths at partial seasons. It is NOT a
full ERoto-vs-MC SD-parity test at `frac < 1` -- that would trip the pre-existing
within-term divergence (ERoto raw, MC frac-scaled) that predates this work and is out of
scope (Scope). Full SD+mean parity is asserted at `frac = 1` (Testing #3).

## Integration seams

New module `src/fantasy_baseball/sgp/closer_mixture.py` -- single source of truth: the smooth
curves `q(s), a_m(s), a_s(s)`, the closed-form `sv_role_variance(mu0, fraction_remaining)`
(per-component `within + between`), and the per-draw multiplier the sampler uses. All three
SV call sites route through it.

### A. ERoto analytic (`scoring.py`: `player_category_variance`)

The SV variance term (`scoring.py:1283`, currently `negbin_perf_variance('sv', v) +
v*v*cv_pt_sq`) becomes `closer_mixture.sv_role_variance(v, frac)` (the full per-component
`within + between`). W/K keep their `negbin_perf_variance + cv_pt` term. `fraction_remaining`
must be threaded into `player_category_variance` (today it uses raw cv_pt). **The mean path
(`project_team_stats`) is untouched.**

### B. Monte Carlo (`simulation.py`: `_apply_variance_batch`, ~781-817)

The cleanest mechanism, given that `scales` is a single array multiplying every correlated
stat (line 794) and also drives `frac_missed = 1 - scales` (804) and thus the backfill
(811-814): **handle SV's variance outside the shared `scales`/copula pipeline.**

1. Compute SV's current-mean base `mu0` = the existing per-iteration SV mean *without* its
   `cv_pt` spread -- i.e. `base['sv'] * eff_mean` (the `eff_mean` location of `scales`, not
   the `z_pt*eff_sd` spread). This is the mean we preserve.
2. Draw the role state per (iter, pitcher): `X` from the mixture (`E[X]=1`).
3. Draw `SV ~ NegBin(mu0 * X, r=37.757)` with the `fraction_remaining` variance scaling the
   copula already applies to within-stat variance; apply the `frac` scaling to `between` via
   the mixture (Testing #4 guards consistency).
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

Guards: the **mean-parity AND SD-parity tests (#3)** catch any error in this mechanism.

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
from committed artifacts); the wired backtest is the authoritative check.

## Backtest changes (`scripts/backtest_sd_calibration.py`)

1. **Wire SV variance to the mixture.** The backtest re-derives SV variance inline
   (`negbin_perf_variance(key, proj) + proj**2 * cvp**2`, line 117); branch SV to
   `closer_mixture.sv_role_variance(proj, frac)`. Required for the gate to be measurable.
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
- **W, SO**: if #3 applied, gain a 2025 sample -- `SD(z)` shifts but must stay in `[0.8,1.25]`
  (record before/after). If #3 skipped, unchanged.
- **SV**: `SD(z)` into `[0.8, 1.25]`. The **raw** bias `sum(actual)-sum(projected)` is
  unchanged (mean-neutral); the *standardized* `mean z` **shrinks** as the SD grows (e.g.
  +0.80 -> ~+0.4) -- this is expected, not a regression, and `mean z` is not a tuning target.

## Testing

1. **Unit** -- `sv_role_variance` matches a brute-force sample of the generative process
   (NegBin(mu0*X, r) over the role draw); mean-1 check `E[X] == 1`; `between -> 0` as
   `fraction_remaining -> 0`; `a_m, a_s >= 0` over the fitted `s` range. (Do NOT assert
   cross-`s` monotonicity -- the vault curve is legitimately non-monotone at low `s`.)
2. **Integration (target)** -- backtest SV `SD(z)` in `[0.8, 1.25]`; R/HR/RBI/SB unchanged;
   W/SO stay in `[0.8, 1.25]`; SV raw bias unchanged (standardized `mean z` shrinks, expected).
3. **Full-season parity (frac = 1), REQUIRED** -- analytic ERoto SV **mean** == MC SV mean AND
   analytic SV **SD** == MC SV SD, within MC tolerance. Also assert each path's SV mean is
   unchanged from a pre-change baseline (guards the `mu0` decomposition; tolerance accommodates
   the `max(0,.)` clamp gap).
4. **In-season property (frac in {0.25,0.5,0.75})** -- the mixture's *added* (`between`) SV
   variance scales to 0 with `fraction_remaining` and is applied identically in both paths.
   (Scoped to the added term, NOT full SD parity -- see In-season / Scope.)
5. **Valuation regression (guard)** -- SGP/VAR/VONA SV values do not move.
6. **MC re-baseline** -- deterministic-seed MC tests re-pinned for the added Bernoulli.

**Success = (2), (3), and (4).**

## Known limitations

- **Handcuff anti-correlation.** Independent per-pitcher role draws overstate team SV variance
  for an owner who handcuffs both closers of one MLB bullpen (anti-correlated saves). The
  mixture is calibrated on the backtest's independent random rosters; not modeled.
- **Within-component correlation losses** (both second-order, don't touch the marginal target):
  SV pulled out of the shared `scales` loses its playing-time co-movement with W/K/ER; the role
  Bernoulli is independent of the copula, so job-loss doesn't co-move with high ER.
- **Skew / mean-variance separability.** Realized SV is floored at 0 and right-skewed. `SD(z)`
  is mean-centered so the variance target is measurable independent of the mean bias; if the
  fix still leaves `SD(z)` out of band, that is the signal (Integration test) to revisit
  (three-component, or coordinate with #235).

## Files touched

- `src/fantasy_baseball/sgp/closer_mixture.py` (new).
- `src/fantasy_baseball/utils/constants.py` -- `SV_ROLE_MIXTURE`.
- `src/fantasy_baseball/scoring.py` -- SV term in `player_category_variance` (+ thread
  `fraction_remaining`); mean path untouched.
- `src/fantasy_baseball/simulation.py` -- SV role draw + out-of-`scales` SV mean/variance in
  `_apply_variance_batch`; backfill and `r` unchanged.
- `scripts/calibrate_closer_mixture.py` (new) -- constrained continuous calibration, 2022-2025.
- `scripts/backtest_sd_calibration.py` -- SV wired to mixture; per-year cat set to admit 2025
  SV; optional 2025 SO derivation.
- Tests under `tests/`.

## Out-of-scope / related issues

- **#235** -- all-positions projection-accuracy backtest (owns the SV mean bias).
- **#236** -- re-export 2025 pitcher actuals with the standard counting template.
