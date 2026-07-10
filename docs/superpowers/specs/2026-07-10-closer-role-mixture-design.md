# Closer Role-Mixture Model for Saves Variance

**Issue:** #193 -- Saves variance vs realized: re-validate under the unified NegBin model
**Date:** 2026-07-10
**Status:** Design approved, pending spec review

## Problem

`scripts/backtest_sd_calibration.py` computes standardized team-category residuals
`z = (actual_team_total - projected_team_total) / eroto_SD` across 2022-2025. Under the
unified NegBin dispersion model, every counting category is calibrated (`SD(z)` in
0.9-1.3) EXCEPT saves:

```
== MATCHED-ONLY ==            == DNP=0 ==
  SV  mean +0.80  SD 2.43       SV  mean +0.56  SD 2.07
```

Two distinct defects:

1. **Variance ~2.2x too tight.** Realized team SV totals scatter roughly twice as wide
   as the model predicts.
2. **Positive mean bias (+0.6 to +0.8 sigma).** Realized team SV *exceeds* projected --
   projection systems under-total league saves (they hedge across committees).

The ERoto/MC variance unification (June 2026) only made the analytic ERoto SD match the
MC SD; it did not re-settle calibration vs reality, and SV remained broken.

### Root cause

For a projected 30-SV closer, the variance splits ~74% playing-time term
(`proj^2 * cv_pt^2`, cv_pt=0.417) and ~26% NegBin performance term
(`negbin_perf_variance('sv', proj)`, r=37.757). To close the 2.2x gap via `cv_pt` alone
would require cv_pt ~= 1.0 (a closer's innings scattering 100% around projection), which
is physically absurd. The absurdity is the diagnosis: SV is not over-dispersed continuous
data. It is a **discrete role switch** -- a closer keeps the job (~35 SV) or loses it
(~5), and a setup man can vault in. That distribution is bimodal; a wide unimodal `cv_pt`
can match the second moment but not the shape.

## Scope

**In scope:** fix the SV **variance** (the 2.2x) via an explicit role-change mixture, in
both the analytic ERoto path and the Monte Carlo sampler, keyed on projected SV.

**Out of scope:** the positive **mean bias**. It is a projection-quality problem
(systems under-total saves), not a dispersion problem, and "fixing" it means shifting the
SV point estimate that also feeds player valuation (SGP/VAR/VONA). It belongs to the
all-positions projection-accuracy backtest, #235. This design is deliberately
**mean-preserving**.

## Design decisions (resolved during brainstorming)

1. **Independent per-pitcher**, not league-conserving. Each pitcher draws its own role
   outcome; a lost job's saves do not reappear on a teammate. Keeps the analytic
   per-player SD quadrature and the ERoto/MC unification intact. Saves are not conserved
   league-wide (acceptable simplification).
2. **Keyed on projected SV, banded.** Projected SV is present in both ERoto and MC per
   player and in the historical backtest data. Bands (e.g. 0-2, 3-14, 15-24, 25+) get
   calibrated parameters.
3. **Mean-preserving.** The mixture reproduces each pitcher's projected SV as its mean and
   only inflates variance. Player valuation is untouched; the mean bias is left for #235.

## The model

Per-pitcher two-component latent role state, parameters selected by the pitcher's
projected-SV band:

```
role ~ Bernoulli(p)                       # p = calibrated retention/acquisition prob
SV | primary   ~ NegBin(mu = m_primary, r_primary)
SV | alternate ~ NegBin(mu = m_alt,     r_alt)
```

Mean-preserving constraint, anchored to the projection `s`:

```
p * m_primary + (1 - p) * m_alt = s       # so E[SV] = projected SV, unchanged
```

- **Closer bands (high s):** primary = "hold job" (mu above s, full save volume),
  alternate = "lose job" (mu collapses toward 0). This is the dominant, load-bearing
  direction -- it is where rostered value sits.
- **Low bands (small s):** mirror image -- primary = "stays a middle reliever" (mu near
  its small s), alternate = low-probability "gains the job" (mu = an absolute closer-line,
  ~20-30 SV, NOT a multiple of the near-zero projection). Still obeys the mean-preserving
  constraint.

### Variance (law of total variance)

```
Var(SV) = E[Var(SV | role)]               # within:  p*negbin_var(m_primary) + (1-p)*negbin_var(m_alt)
        + Var(E[SV | role])               # between: p*(1-p)*(m_primary - m_alt)^2
```

The **between-component term** supplies the fat, role-switch variance the smooth `cv_pt`
could not represent, and it is what fixes the 2.2x.

**SV drops the `proj^2 * cv_pt^2` playing-time term entirely.** The mixture now owns SV's
extra-Poisson variance, so keeping the pt term would double-count role variation. The
`cv_pt` curve continues to apply to W/K/IP unchanged; only SV changes.

### Calibration-sufficiency risk (explicit)

At realistic retention (`p ~ 0.65-0.70`) with a lose-component near 0, a pure two-state
mean-preserving mixture reaches ~1.3-1.5x per closer, not the full 2.2x. Closing the rest
may require calibration to also loosen `r_primary` -- the current `r=37.757` was fit on
*role-stable* closers and understates hold-state performance spread. If two states cannot
reach the target with defensible params, that is the signal a third "mid-season job-share"
state is needed. **We start with two states and let the backtest decide**; the design does
not assume two is sufficient.

## Integration seams

Both consumers already read shared per-stat dispersion, so the fix lands in both. A new
module `src/fantasy_baseball/sgp/closer_mixture.py` owns the shared math: band lookup, the
closed-form `sv_role_variance(proj_sv)`, and the component means used by the sampler.

### A. ERoto analytic (`scoring.py`: `project_team_sds` / `score_roto`)

Replace only the SV per-player term. Today:

```
sv_var = negbin_perf_variance('sv', proj) + proj**2 * cv_pt**2
```

Becomes:

```
sv_var = closer_mixture.sv_role_variance(proj)   # within + between, no cv_pt
```

Every other category term is untouched.

### B. Monte Carlo (`simulation.py`: batched copula sampler)

SV lives inside the correlated pitcher copula (`PITCHER_CORR_MATRIX`). Chosen approach --
**modulate mu inside the copula (Option B1):**

- Draw the role state per (iter, pitcher) as an extra Bernoulli off the existing RNG.
- Pick `m_primary` or `m_alt`, and feed that as SV's `mu` into the existing copula draw
  (`mu = proj_sv * scale` becomes `mu = role_mean * scale`).
- SV stays correlated with W/ER/BB/H within each component (preserving the `-0.34`
  couplings); the between-component spread appears for free because different iterations
  draw different components.

Rejected alternative (B2): pull SV out of the copula into a standalone sampler. Loses SV's
cross-stat correlations and desyncs the two paths' structure. More code, worse fidelity.

**Consequence:** the added per-pitcher Bernoulli consumes RNG, so deterministic-seed MC
tests that pin exact outputs shift and must be re-baselined. Expected, not a regression.

## Calibration

New script `scripts/calibrate_closer_mixture.py`, sibling to
`scripts/calibrate_stat_dispersion.py`.

- **Source:** 2022-2025 realized-vs-projected, same steamer+zips blends and `data/stats`
  actuals the backtest uses.
- **2025 data (rectified):** `data/stats/pitchers-2025.csv` was exported with an advanced
  column template and lacks raw `SO/ER/BB/H`, but `SV, W, IP` are present and `SO` is
  exactly reconstructable as `SO = K/9 * IP / 9`. The calibration and backtest **derive
  `SO` for 2025 and include the year** -- giving four full seasons of SV. (H/WHIP remain
  unavailable for 2025, but this issue never touches them. A full re-export of the 2025
  file is tracked separately in #236, tied to #235.)
- **Per band, fit the shape:** retention/acquisition probability `p`, within-component
  dispersion, and the component-mean ratios. Then **anchor to the projection** so the
  components average back to each pitcher's own `s`. This is what makes it mean-preserving
  and deliberately absorbs the `+bias` out (we are not fixing the mean here).
- **Parameterization asymmetry:** closer bands express component means as multiples of `s`
  (a lost projected-40 closer bled more early saves than a lost projected-25); the low-band
  gain component uses an absolute closer-line, since a multiple of a near-zero projection
  cannot vault.
- **Output:** a new banded constant `SV_ROLE_MIXTURE` in `utils/constants.py`, consumed
  only by `closer_mixture.py`.

### Backtest gate change

`backtest_sd_calibration.py::build_year` currently returns `None` for pitchers unless
`{"W","SO","SV"}` are all present, which excludes 2025. Loosen this: derive `SO` for 2025
from `K/9` and include the year for the W/SO/SV categories.

## Testing

1. **Unit** -- closed-form `sv_role_variance(s)` matches a brute-force sample of the
   mixture (property test); mean-preserving check `E[mixture] == s` within tolerance;
   variance rises monotonically with band.
2. **Integration (the real target)** -- `backtest_sd_calibration.py` SV `SD(z)` lands in
   **[0.8, 1.25]**; the other six categories' `SD(z)` are **unchanged**; SV mean z is
   **unchanged** (still ~+0.6-0.8, by design).
3. **Unification** -- analytic ERoto SV SD == MC SV SD within tolerance (extend the
   existing parity guard).
4. **Valuation regression** -- SGP/VAR/VONA player SV values **do not move**
   (mean-preserving guarantees it; assert it so a future edit cannot silently break it).
5. **MC re-baseline** -- deterministic-seed MC tests re-pinned for the added Bernoulli
   draw.

**Success = all of (2) and (3).** If a two-state mixture cannot reach [0.8, 1.25] with
defensible calibrated params, the backtest surfaces it and we escalate to a third
(job-share) state.

## Files touched

- `src/fantasy_baseball/sgp/closer_mixture.py` (new) -- band lookup, `sv_role_variance`,
  component means for the sampler.
- `src/fantasy_baseball/utils/constants.py` -- new `SV_ROLE_MIXTURE` banded constant.
- `src/fantasy_baseball/scoring.py` -- SV term in `project_team_sds` / `score_roto`.
- `src/fantasy_baseball/simulation.py` -- role-state draw modulating SV `mu` in the copula.
- `scripts/calibrate_closer_mixture.py` (new) -- calibration from 2022-2025.
- `scripts/backtest_sd_calibration.py` -- 2025 SO derivation + gate loosening.
- Tests under `tests/` per the Testing section.

## Out-of-scope / related issues

- **#235** -- all-positions projection-accuracy backtest (owns the SV mean bias).
- **#236** -- re-export 2025 pitcher actuals with the standard counting template (restores
  H/WHIP parity; not blocking this issue).
