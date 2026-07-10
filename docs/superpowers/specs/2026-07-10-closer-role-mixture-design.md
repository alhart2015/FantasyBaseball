# Closer Role-Mixture Model for Saves Variance

**Issue:** #193 -- Saves variance vs realized: re-validate under the unified NegBin model
**Date:** 2026-07-10
**Status:** Design approved, in spec-review hardening (iteration 3)

## Problem

`scripts/backtest_sd_calibration.py` computes standardized team-category residuals
`z = (actual_team_total - projected_team_total) / eroto_SD` across the seasons in `YEARS`.
Under the unified NegBin dispersion model, every counting category is calibrated
(`SD(z)` in 0.9-1.3) EXCEPT saves:

```
== MATCHED-ONLY ==            == DNP=0 ==
  SV  mean +0.80  SD 2.43       SV  mean +0.56  SD 2.07
```

**Sample note:** these SV numbers are measured on **2022-2024 only** -- `build_year`
currently excludes 2025 pitchers (see the 2025-data section). The quoted "~2.2x" is a
3-year figure.

Two distinct defects:

1. **Variance ~2.2x too tight.** Realized team SV totals scatter roughly twice as wide as
   the model predicts.
2. **Positive mean bias (+0.6 to +0.8 sigma).** Realized team SV *exceeds* projected --
   projection systems under-total league saves.

This design fixes **(1) only**. See Scope.

### Root cause

For a projected 30-SV closer, the model variance splits ~74% playing-time term
(`proj^2 * cv_pt^2`, cv_pt=0.417) and ~26% NegBin performance term
(`negbin_perf_variance('sv', proj)`, r=37.757). Closing the 2.2x gap via `cv_pt` alone
needs cv_pt ~= 1.0 -- physically absurd. The absurdity is the diagnosis: SV is not
over-dispersed *unimodal* data. It is a **discrete role switch** -- a closer keeps the job
(~35 SV) or loses it (~5); a setup man can vault in. The distribution is **bimodal**; a
wide unimodal `cv_pt` matches the second moment but not the shape.

## Scope

**In scope:** fix the SV **variance** via an explicit bimodal role-switch mixture, in both
the analytic ERoto path and the Monte Carlo sampler.

**Out of scope:** the positive **mean bias** (a projection-quality problem; systems
under-total saves). It belongs to #235. This design is **mean-neutral** -- it changes only
variance and leaves *each path's existing SV mean exactly as it is today* (see the
mean-neutrality subsection). The backtest measures `z` against *projected* totals, so the
SV `mean z` is unaffected by this work regardless.

## Design decisions (resolved during brainstorming + spec review)

1. **Bimodal two-component mixture**, keyed on projected SV. Every pitcher gets an explicit
   "hold/stay" component and a "lose/gain" component -- the bimodality a smooth `cv_pt`
   cannot represent.
2. **Independent per-pitcher**, not league-conserving (keeps the per-player quadrature and
   the ERoto/MC unification; handcuff caveat in Known limitations).
3. **Continuous parameters, not bands.** The mixture's knobs -- `P(hold)`, the lose/gain
   floor, the between-component spread -- are **smooth functions of projected SV**, fit on
   the whole pitcher population at once. This is the same band-vs-smooth choice the repo
   already makes for `STAT_DISPERSION` (which uses `(mu_upper, r)` bands); here we choose
   smooth to (a) pool the thin closer data -- there are only ~23-33 established closers per
   season -- and (b) avoid parameter cliffs at arbitrary SV cutoffs. The mixture stays
   bimodal at every projected-SV value; only the parameters vary smoothly.
4. **Mean-neutral.** The mixture adds between-component variance while leaving each path's
   current SV mean unchanged. It is a *variance swap*: it replaces the SV playing-time
   variance term (`cv_pt`, the failed smooth role proxy) with the mixture's
   between-component, and does not touch the mean machinery. Player valuation
   (SGP/VAR/VONA) reads the projection directly and is untouched.
5. **In-season aware.** The role-switch (between-component) variance scales down with
   remaining season: a closer who has already held the job most of the year is unlikely to
   lose it in the time left. See the fraction_remaining subsection.

## The model

Per-pitcher two-component latent role state. Expressed as a **mean-1 multiplier** on the
pitcher's existing per-iteration SV mean `mu0` (whatever the current model already produces
for that pitcher -- projection, playing-time haircut, and, in the MC, injury backfill --
so the mean is preserved by construction):

```
role ~ Bernoulli(p(s))                     # s = projected SV; p = P(primary), a smooth fn of s
mult | primary   = a_hi(s)                 # >= 1 : holds the job / vaults in
mult | alternate = a_lo(s)                 # ~0  : loses the job / stays a non-closer
p(s)*a_hi(s) + (1 - p(s))*a_lo(s) = 1      # mean-1 constraint  ->  E[mult] = 1  ->  mean unchanged

effective SV mean this draw = mu0 * mult
```

Because `E[mult] = 1`, the expected SV is `mu0` -- identical to today. The **spread** of
`mult` is the role-switch variance:

```
Var(mult) = p(s) * (1 - p(s)) * (a_hi(s) - a_lo(s))^2
```

### Variance (law of total variance), and what it replaces

Per-pitcher SV variance becomes:

```
Var(SV) = within  +  between
within  = negbin_perf_variance('sv', mu0)                      # performance, r stays 37.757 (see below)
between = mu0^2 * Var(mult) * frac_scale(fraction_remaining)   # role switch, replaces the cv_pt term
```

The `between` term **replaces** the old `proj^2 * cv_pt^2` playing-time term for SV -- it
is not added on top, so role variation is not double-counted. `cv_pt` continues to apply to
W/K/IP unchanged; only SV swaps its playing-time term for the mixture. Equivalently, the
mixture supplies SV an *effective* bimodal "cv" of `sqrt(Var(mult))` (e.g. `(1-p)/p` in
variance terms when `a_lo -> 0`), replacing the smooth `cv_pt=0.417`.

**`r_primary` stays at the fitted 37.757.** The primary (hold) component IS the role-stable
closer population that `STAT_DISPERSION['sv']=37.757` was fit on (`constants.py:158-159`,
"role-stable... thin n=43"); pulling job-losers into the alternate component leaves the
primary matching that population, so its within-dispersion should not be loosened. All the
extra spread comes from the `between` term, not from inflating `r`. (This retracts the
earlier "loosen r_primary as survivorship correction" argument, which was directionally
wrong: the survivors are exactly the primary component.)

### fraction_remaining scaling (in-season)

The `between` (role-switch) variance is scaled by `frac_scale(fraction_remaining)`, so it
vanishes as the season ends -- a settled closer role stops being a coin flip. First cut:
`frac_scale = fraction_remaining` (linear, mirroring how the MC already scales within-stat
variance via `var_target = fraction_remaining * var_full` in `_negbin_copula_counts`).

**This surfaces a pre-existing inconsistency the plan must resolve:** ERoto's
`player_category_variance` uses **raw** `cv_pt` (`scoring.py:1260,1268,1283`) with **no**
`fraction_remaining` scaling, while the MC scales within-stat variance by
`fraction_remaining`. So today the two paths' playing-time variance already diverges
in-season. The plan must make the SV `between` term scale identically in both paths (and
document how the within terms are meant to scale), guarded by an **in-season parity test**
(Testing #4) at `fraction_remaining < 1`, not just full-season.

## Integration seams

A new module `src/fantasy_baseball/sgp/closer_mixture.py` is the single source of truth:
the smooth parameter curves `p(s)`, `a_hi(s)`, `a_lo(s)`, the closed-form
`sv_role_between_variance(mu0, fraction_remaining)`, and the per-draw multiplier the sampler
uses. All three SV call sites (ERoto, MC, backtest) route through it; none re-derives SV
dispersion locally.

### A. ERoto analytic (`scoring.py`: `player_category_variance`)

The SV variance term (`scoring.py:1283`, currently
`negbin_perf_variance('sv', v) + v*v*cv_pt_sq`) becomes:

```
result[Category.SV] = negbin_perf_variance('sv', v) + closer_mixture.sv_role_between_variance(v, frac)
```

W and K keep their `negbin_perf_variance + cv_pt` term. **The mean path
(`project_team_stats`) is untouched** -- mean-neutrality means ERoto's SV mean does not
move. (`player_category_variance` will need `fraction_remaining` threaded in; today it uses
raw cv_pt -- see the in-season subsection.)

### B. Monte Carlo (`simulation.py`: `_apply_variance_batch`, lines ~781-817)

The current SV path: `mu_mat[:,:,j] = base['sv'] * scales` (794), `r_mat` from
`STAT_DISPERSION['sv']` (795), copula draw (799), `+8*frac_missed` backfill (811-814).
Mean-neutral changes:

1. **Draw the role state** per (iter, pitcher): Bernoulli(`p(s)`) off the existing RNG,
   selecting `a_hi` or `a_lo`.
2. **Keep the mean, swap the spread.** SV's mean stays driven by the existing `eff_mean`
   haircut and the backfill (mean preserved). But SV's *cv_pt-driven spread* (the `eff_sd`
   part of `scales`) is **removed** and replaced by the role multiplier: SV's per-draw mean
   is `mu0 * mult` where `mu0` carries `eff_mean` but not the `z_pt*eff_sd` wiggle. Since
   `E[mult]=1`, the SV mean is unchanged; the spread is now the bimodal role switch.
3. **`r` stays `STAT_DISPERSION['sv']=37.757`** for the within-component (per the r_primary
   decision) -- no change to line 795 for SV.
4. **Keep the injury backfill.** Mean-neutrality means we do NOT exempt SV from the `+8`
   backfill (reversing a prior draft) -- it is part of the current mean we are preserving.
   It adds only its existing small variance.

W/K/ER/BB/H are untouched. SV's within-component still draws off the correlated copula
latent `all_z` (799), so it stays correlated with er/bb/h **within a component** (see
correlation limitation). The exact decomposition of `scales` into its mean part
(`eff_mean`, kept) and spread part (`eff_sd`, replaced) for the SV column is a plan-level
correctness task; the **SD parity test and the mean-parity test (Testing #3) are the
guards** that catch any error.

**Correlation limitation.** The copula's `sv<->er/bb/h` couplings (~ -0.34,
`constants.py:309`) apply only to the within-component. The dominant between-component
(role switch) is an independent Bernoulli, so the model understates the "lost the job
*because* ERA spiked" co-movement. This does not affect SV's marginal or team-total
variance (the target); it affects only the joint SV-vs-ERA category-win correlation, a
second-order effect. Deferred.

## Calibration

New script `scripts/calibrate_closer_mixture.py`.

- **Source:** 2022-2025 realized-vs-projected, same steamer+zips blends and `data/stats`
  actuals the backtest uses.
- **2025 data.** `pitchers-2025.csv` has `SV, W, IP, K/9` but not raw `SO/ER/BB/H`. **The
  SV work needs only SV, which is present directly** -- so including 2025 for SV requires
  only relaxing the backtest's `{"W","SO","SV"}` gate to admit SV (below), NOT any SO
  reconstruction. Separately, to also include 2025 in the W/SO backtest (a completeness
  nice-to-have you asked for), `SO` is reconstructed -- but `IP` is stored in **baseball
  thirds-notation** (`195.1` = 195 1/3), so the naive `K/9 * IP / 9` misrounds ~76 of 873
  pitchers. Correct derivation:

  ```
  ip_true = floor(IP) + (IP - floor(IP)) * 10/3
  SO      = round(K/9 * ip_true / 9)
  ```

  This recovers SO to integer precision (K/9 is stored to ~6 decimals; the residual
  rounding is 0 for MLB-range IP -- it is the definitional inverse of how K/9 was formed,
  not an independent validation, since 2025 carries no raw SO to check against). H/WHIP
  remain unavailable for 2025 (tracked in #236).
- **Fitting (continuous, pooled).** Fit the smooth curves `p(s)`, `a_hi(s)`, `a_lo(s)` over
  ALL pitchers' `(projected s, realized SV)` pairs by maximum likelihood -- a parametric
  mixture regression (the mixing probability and component levels are low-order functions of
  `s`, e.g. logistic `p(s)` and linear/`sqrt` level curves), so the whole population
  constrains a handful of curve parameters rather than ~30-per-band independent fits. `r`
  for the within-component is fixed at 37.757 (not fit), further reducing free parameters.
  After the fit, the mean-1 constraint `p*a_hi + (1-p)*a_lo = 1` is enforced exactly. This
  is a mean-neutral variance model; the `+bias` is deliberately not fit (that is #235).
- **Backtest is validation, not fitting.** No curve parameter is tuned to the backtest
  `SD(z)`.
- **Output:** a new constant (`SV_ROLE_MIXTURE`, the fitted curve coefficients) in
  `utils/constants.py`, consumed only by `closer_mixture.py`.

### Feasibility (team-total, all projected-SV levels)

The gate is `SD(z)` in `[0.8, 1.25]`. From `SD(z)=2.43`, the 1.25 edge needs team SV
variance to grow ~3.8x. A decomposition of realized 2022-24 team SV residual variance
(9-pitcher random teams, matching the backtest) shows **~44% comes from pitchers projected
< 15 SV** (vault-ins) and ~56% from projected >= 15 (job losers). Both must be reached:

- **Job-loss (projected >= 15):** supplied by the `a_lo -> 0` alternate with retention
  `p(s)`. Whether it reaches the target is an empirical question about `p(s)`: a single-30
  closer needs ~3.8x variance, reachable at `p ~ 0.5-0.6`, not at `p ~ 0.7` (2.1x). The
  calibration -- not tuning -- decides.
- **Vault-in (projected < 15):** supplied by the `a_hi >> 1` primary with small `1-p`. A
  mean-1 multiplier CAN produce this: between-variance `~ mu0^2 * (1-p) * a_hi^2` with a
  small vault probability reproduces the realized ~2.2/pitcher while keeping the mean at
  projection. This is why continuous curves matter -- the vault behavior is a smooth
  extension of the low-`s` end, not a separately-fit thin band.

**Escalation commitment.** Calibrate the two-component curves from data. If the backtest SV
`SD(z)` lands in `[0.8, 1.25]`, done. If it misses, escalate to a **three-component**
mixture (hold / job-share / lose *and* the vault path), decided as the first plan milestone
via the wired backtest -- not by tuning `r` or the curves to the gate. The escalation
covers both the closer and the vault ends, since the feasibility gap could be in either.

## Backtest changes (`scripts/backtest_sd_calibration.py`)

1. **Wire SV variance to the mixture.** The backtest computes SV variance inline
   (`var = negbin_perf_variance(key, proj) + proj**2 * cvp**2`, line 117). Branch SV to
   `negbin_perf_variance('sv', proj) + closer_mixture.sv_role_between_variance(proj, frac)`;
   W/K keep the inline formula. **Required for the gate to be measurable** (the backtest
   re-derives SV variance independently of `scoring.py`).
2. **Admit 2025 for SV** by relaxing the `{"W","SO","SV"}` gate (`build_year`, line 85) so
   SV is included when present even if SO is absent. **This alone** gets 2025 SV into the
   gate.
3. **(Optional, per your request) 2025 W/SO** via the thirds-corrected SO derivation above,
   so W/SO also gain the 2025 season. Separable from the SV target.

**Success criterion.**
- **R/HR/RBI/SB** (hitters): `SD(z)` unchanged (2025 hitters were already included).
- **W, SO**: if edit 3 is applied they gain a 2025 sample -- `SD(z)` shifts but must stay
  in `[0.8, 1.25]` (record before/after). If edit 3 is skipped, unchanged.
- **SV**: `SD(z)` moves into `[0.8, 1.25]`; `mean z` unchanged (by design, mean-neutral).

## Testing

1. **Unit** -- `sv_role_between_variance` matches a brute-force sample of the mixture;
   mean-1 check `E[mult] == 1` within tolerance; between-variance is non-negative and
   scales linearly to 0 as `fraction_remaining -> 0`. (Do NOT assert cross-`s` monotonicity
   of variance -- the vault-in curve legitimately makes low-`s` variance non-monotone.)
2. **Integration (target)** -- backtest SV `SD(z)` in `[0.8, 1.25]`; R/HR/RBI/SB unchanged;
   W/SO stay in `[0.8, 1.25]`; SV `mean z` unchanged.
3. **Mean + SD parity (full season)** -- analytic ERoto SV **mean** == MC SV **mean** AND
   analytic SV **SD** == MC SV **SD**, within MC tolerance. Mean parity is a REQUIRED
   test (not just SD): mean-neutrality is the core promise and nothing else guards it.
   Also assert each path's SV mean is unchanged from a pre-change baseline (guards the
   `scales` mean/spread decomposition).
4. **In-season parity** -- ERoto SV SD == MC SV SD at `fraction_remaining` in {0.25, 0.5,
   0.75}, not only 1.0. This catches the between-component being scaled inconsistently
   across paths (the in-season concern) and forces the pre-existing raw-cv_pt-vs-frac
   discrepancy to be resolved.
5. **Valuation regression (guard)** -- SGP/VAR/VONA player SV values do not move
   (guaranteed by construction; the guard prevents silent future breakage).
6. **MC re-baseline** -- deterministic-seed MC tests re-pinned for the added Bernoulli.

**Success = (2), (3), and (4).**

## Known limitations

- **Handcuff anti-correlation (train/serve mismatch).** The backtest builds synthetic
  rosters by `rng.choice` (independent players); the mixture is calibrated on that
  population. A real manager who handcuffs both closers of one MLB bullpen holds
  *anti*-correlated save sources; independent role draws model that as additive variance,
  overstating team SV variance for handcuff owners. Accepted; not modeled.
- **Role state uncorrelated with run-prevention** (see MC correlation limitation):
  understates SV-vs-ERA joint category correlation; second-order; deferred.
- **Skew / mean-variance separability.** Realized SV is floored at 0 and right-skewed, so
  mean and variance are not fully independent. `SD(z)` is mean-centered, so the variance
  target is measurable independent of the mean bias; if the fix nonetheless leaves `SD(z)`
  out of band, that is the signal (Integration test) to revisit -- three-component, or
  coordinate with #235.

## Files touched

- `src/fantasy_baseball/sgp/closer_mixture.py` (new) -- smooth parameter curves,
  `sv_role_between_variance`, per-draw multiplier.
- `src/fantasy_baseball/utils/constants.py` -- new `SV_ROLE_MIXTURE` curve coefficients.
- `src/fantasy_baseball/scoring.py` -- SV variance term in `player_category_variance`
  (+ thread `fraction_remaining`); mean path untouched.
- `src/fantasy_baseball/simulation.py` -- SV role-state draw in `_apply_variance_batch`:
  mean-neutral multiplier replacing SV's cv_pt spread; `r` and backfill unchanged.
- `scripts/calibrate_closer_mixture.py` (new) -- continuous mixture-regression calibration
  from 2022-2025.
- `scripts/backtest_sd_calibration.py` -- SV variance wired to the mixture; gate relaxed to
  admit 2025 SV; optional 2025 SO derivation.
- Tests under `tests/` per the Testing section.

## Out-of-scope / related issues

- **#235** -- all-positions projection-accuracy backtest (owns the SV mean bias).
- **#236** -- re-export 2025 pitcher actuals with the standard counting template.
