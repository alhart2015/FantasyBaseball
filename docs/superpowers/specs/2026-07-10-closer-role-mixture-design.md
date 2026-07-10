# Closer Role-Mixture Model for Saves Variance

**Issue:** #193 -- Saves variance vs realized: re-validate under the unified NegBin model
**Date:** 2026-07-10
**Status:** Design approved, in spec-review hardening

## Problem

`scripts/backtest_sd_calibration.py` computes standardized team-category residuals
`z = (actual_team_total - projected_team_total) / eroto_SD` across the seasons in
`YEARS`. Under the unified NegBin dispersion model, every counting category is calibrated
(`SD(z)` in 0.9-1.3) EXCEPT saves:

```
== MATCHED-ONLY ==            == DNP=0 ==
  SV  mean +0.80  SD 2.43       SV  mean +0.56  SD 2.07
```

**Sample note:** these SV numbers are measured on **2022-2024 only** -- `build_year`
currently excludes 2025 pitchers (see the 2025-data section below), so the quoted "~2.2x"
is a 3-year figure. After this work adds 2025, the post-fix gate is measured on a 4-year
sample; the before/after comparison therefore spans slightly different populations, which
is expected and called out here so it is not mistaken for a regression.

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
**mean-preserving** (E[SV] = projected SV in both paths; see the mean-preservation
subsection under Integration for why this holds in the MC too).

Note the backtest measures `z` against *projected* team totals, not against the model's
internal mean. So the SV **mean z is unaffected by this work regardless** -- it depends
only on projections and the SD. Fixing the variance moves `SD(z)`, not `mean z`; the
mean bias stays until #235.

## Design decisions (resolved during brainstorming)

1. **Independent per-pitcher**, not league-conserving. Each pitcher draws its own role
   outcome; a lost job's saves do not reappear on a teammate. Keeps the analytic
   per-player SD quadrature and the ERoto/MC unification intact. Saves are not conserved
   league-wide (acceptable simplification; see Known limitations for the handcuff caveat).
2. **Keyed on projected SV, banded.** Projected SV is present in both ERoto and MC per
   player and in the historical backtest data. Bands (defined below) get calibrated
   parameters.
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

### Bands and regime assignment (concrete, not illustrative)

Four bands on projected SV, each assigned to a regime and a parameterization. Band edges
use the same `searchsorted(..., side="left")` convention as `resolve_dispersion_r`
(`dispersion.py:25`), so a value on an edge takes the upper band; a continuous projection
(e.g. 2.5) falls in the band whose range contains it.

| Band (proj SV) | Regime | primary | alternate | Component means |
|---|---|---|---|---|
| `[0, 3)`   | middle reliever | stays (mu near s) | **gains** the job (low p) | primary = ratio of s; alt = **absolute** closer-line |
| `[3, 15)`  | speculative / committee | stays | gains/loses (two-sided) | blend; see calibration |
| `[15, 25)` | fringe closer | **holds** (mu >= s) | **loses** (mu -> ~0) | both as ratios of s |
| `[25, inf)`| established closer | holds | loses | both as ratios of s |

Rationale for the `s`-ratio vs absolute split: for a lost projected-40 closer the
pre-demotion saves scale with his projection (ratio of `s`), but a middle reliever's
*gain* to a closer role is an absolute level (~20-30 SV), since a multiple of a near-zero
projection cannot vault. Every band still obeys the mean-preserving constraint.

**Reconciliation with `CLOSER_SV_THRESHOLD = 20` (`constants.py:127`):** that threshold
is used elsewhere only to rank pitchers into active save-slots, a different purpose. The
mixture deliberately does not reuse it as a hard closer/non-closer switch, because a
single 20-SV cliff is exactly the coarse binary this design rejects (a 19-SV and a 21-SV
projection should not get opposite regimes). The `[15,25)` fringe band spans the threshold
and blends the two regimes continuously. Band edges are a calibration output and may be
refined by `calibrate_closer_mixture.py`; the four-band structure and the ratio/absolute
split are the fixed design.

### Variance (law of total variance)

```
Var(SV) = E[Var(SV | role)]               # within:  p*negbin_var(m_primary) + (1-p)*negbin_var(m_alt)
        + Var(E[SV | role])               # between: p*(1-p)*(m_primary - m_alt)^2
```

The **between-component term** supplies the fat, role-switch variance the smooth `cv_pt`
could not represent, and it is what fixes the 2.2x.

**SV drops the `proj^2 * cv_pt^2` playing-time term entirely.** The mixture now owns SV's
extra-Poisson variance, so keeping the pt term would double-count role variation. The
`cv_pt` curve continues to apply to W/K/IP unchanged; only SV changes. **Both paths must
make the identical drop** -- see the MC section for how the sampler is exempted from the
shared `scales` term for SV, without which the two paths diverge.

### Feasibility (corrected arithmetic + escalation commitment)

The gate is `SD(z)` in `[0.8, 1.25]`, matching the backtest's own verdict band
(`backtest_sd_calibration.py:155`). Starting from `SD(z) = 2.43`, reaching the **1.25 gate
edge** needs the model SV variance to grow by `(2.43/1.25)^2 ~= 3.8x`; reaching `SD(z)=1.0`
needs `~5.9x`.

A mean-preserving two-state mixture reaches this **only at low retention**. Worked for a
projected-30 closer with `m_alt -> 0`, `m_primary = s/p`, `r` unchanged at 37.757:

| p (retention) | variance multiple | resulting SD(z) |
|---|---|---|
| 0.70 | ~2.1x | ~1.68 (fails) |
| 0.65 | ~2.6x | ~1.51 (fails) |
| 0.50 | ~4.7x | ~1.12 (passes) |

So whether two states suffice is an **empirical question about the calibrated retention
rate** -- if projected closers historically keep the job ~50-60% of the time, two states
pass; at ~70% they fall short.

The second lever is `r_primary`. Loosening it from 37.757 is **not** the rejected
`cv_pt~=1.0` absurdity: the code comment at `constants.py:158-159` states 37.757 was fit on
the "role-stable closer population" with "thin (n=43) data." That is a survivorship-biased
subset -- closers who *kept and held* the job. The true hold-state population (all closers
who retained the role, including those who scuffled) genuinely scatters wider, so a
calibrated `r_primary < 37.757` is **correcting that survivorship**, not fabricating
variance. It is bounded by what the realized hold-state data actually shows, not tuned to
hit the gate.

**Escalation commitment (plan branch, not a hope):** the plan will calibrate the two-state
mixture with `p`, component means, and `r_primary`/`r_alt` all fit from realized data
(never tuned to the gate). If the resulting backtest `SD(z)` for SV lands in `[0.8, 1.25]`,
done. If it does not, the plan escalates to a **three-state mixture** (hold / mid-season
job-share / lose) rather than forcing implausible parameters. The two-state build is the
first plan milestone precisely so this fork is decided by measurement early.

## Integration seams

Both consumers already read shared per-stat dispersion, so the fix lands in both. A new
module `src/fantasy_baseball/sgp/closer_mixture.py` is the **single source of truth**: band
lookup, the closed-form `sv_role_variance(proj_sv)`, and the per-component `(p, m_primary,
m_alt, r_primary, r_alt)` the sampler draws from. Both paths call it; neither re-derives SV
dispersion locally.

### A. ERoto analytic (`scoring.py`: `player_category_variance` / `project_team_sds`)

Replace only the SV per-player term. Today (`scoring.py:1283`, inside the loop over
`w/k/sv`):

```
result[Category.SV] = negbin_perf_variance('sv', v) + v*v*cv_pt_sq
```

Becomes an SV-specific branch:

```
result[Category.SV] = closer_mixture.sv_role_variance(v)   # within + between, no cv_pt
```

W and K keep their existing `negbin_perf_variance + cv_pt` term. Every other category term
is untouched.

### B. Monte Carlo (`simulation.py`: `_apply_variance_batch`, lines ~781-817)

SV needs a dedicated path, **not** a one-line `mu` swap. The batch sampler currently does,
for every correlated column including `sv`:

- `mu_mat[:,:,j] = base[col] * scales` (line 794) -- shared playing-time scale
- `r_mat[:,:,j]  = resolve_dispersion_r(STAT_DISPERSION[col], mu)` (line 795) -- fixed r
- adds `repl_line * frac_missed` injury backfill to the count (lines 811-814); for RP,
  `REPLACEMENT_BY_POSITION['RP']['sv'] = 8` (`constants.py:331`)

A coherent role mixture for SV must change **all three**, or the MC and ERoto diverge:

1. **Draw the role state** per (iter, pitcher): a Bernoulli(`p`) off the existing RNG,
   selecting the primary or alternate component.
2. **Set SV's mu to the drawn component mean directly -- WITHOUT `* scales`.** The role
   mixture *is* SV's volume/role model now; the lose-job component already encodes the
   innings collapse. Multiplying by the shared `scales` (which carries the IP-based
   playing-time variance) on top would double-count exactly the role variation ERoto drops
   via the cv_pt removal, and would make the MC SV variance larger than ERoto's -> parity
   failure (this was the critical inconsistency in the prior draft).
3. **Set SV's r to the drawn component's `r_primary`/`r_alt`**, from `closer_mixture.py`,
   not `STAT_DISPERSION['sv']`. Otherwise the MC within-component dispersion stays at
   37.757 while ERoto uses the calibrated component r -> parity failure the moment
   calibration loosens `r_primary`.
4. **Exempt SV from the injury backfill** (`repl_contrib = 0` for the `sv` column). The
   lose-job component already models the downside; adding `+8 * frac_missed` on top would
   re-inflate the mean (breaking mean-preservation) and double-count the role downside.

W/K/ER/BB/H keep `scales`, their `STAT_DISPERSION` r, and the backfill unchanged. SV's
within-component draw still uses the correlated copula latent `all_z` (line 799), so SV
stays correlated with er/bb/h **within a component** (see the correlation limitation
below).

**Mean-preservation holds in the MC because of steps 2 and 4.** With no `scales` haircut
and no backfill, `E[SV] = p*m_primary + (1-p)*m_alt = s` exactly -- matching ERoto. (This
does raise the MC's SV mean relative to today's behavior, which currently applies the
~0.786 volume haircut and the +8 backfill to SV. That is intended: setting SV's mean to
the projection is the mean-preserving target, and it aligns MC with ERoto. It does not
touch player valuation, which reads the projection directly.)

**Correlation limitation (was over-claimed before).** The copula's `sv<->er/bb/h`
couplings (~ -0.34, `constants.py:309`) now apply only to the **within-component** wiggle
(~26% of SV variance). The dominant **between-component** move (job gain/loss) is driven by
an independent Bernoulli, uncorrelated with run-prevention -- so the model understates the
real "lost the job *because* ERA spiked" co-movement. This does **not** affect SV's
marginal variance (the target of this issue) or team-total SV variance; it affects only the
joint SV-vs-ERA category-win correlation, a second-order effect on standings-win
probabilities. Correlating the role state with run-prevention is deferred as a known
limitation, not attempted here. The prior claim that B1 "preserves the -0.34 couplings" was
wrong and is corrected to this.

**Rejected alternative (B2):** pull SV out of the copula into a standalone sampler. Loses
even the within-component correlation and desyncs the two paths' structure. Worse on every
axis.

**Consequence:** the added per-pitcher Bernoulli consumes RNG before the `multivariate_normal`
draw (line 799), so every subsequent draw shifts. Deterministic-seed MC tests re-baseline;
"only SV changes" means *distributionally* (same distribution for W/K/ER, different sample
stream), not sample-identically. Expected, not a regression.

## Calibration

New script `scripts/calibrate_closer_mixture.py`, sibling to
`scripts/calibrate_stat_dispersion.py`.

- **Source:** 2022-2025 realized-vs-projected, same steamer+zips blends and `data/stats`
  actuals the backtest uses.
- **2025 data (rectified) -- SO derivation is NOT the naive formula.** `pitchers-2025.csv`
  stores `SV, W, IP, K/9` but not raw `SO/ER/BB/H`. `SV` and `W` are used directly. `SO`
  is reconstructed, but **IP is in baseball thirds-notation** (`195.1` = 195 1/3), so the
  naive `SO = K/9 * IP / 9` rounds to the wrong integer for ~76 of 873 pitchers (verified;
  Skubal/Skenes/Crochet among them). The correct derivation converts IP first:

  ```
  ip_true = floor(IP) + (IP - floor(IP)) * 10/3
  SO      = round(K/9 * ip_true / 9)          # exact: mean |error| = 0.0 (verified)
  ```

  This yields a fourth full season of W/SO/SV for calibration. (H/WHIP remain unavailable
  for 2025; this issue never touches them. A full re-export of the 2025 file is tracked in
  #236.)
- **Fitting procedure (per band, explicit).** For each band, collect the historical
  `(projected s, realized SV)` pairs of pitchers whose projected SV falls in the band. Fit
  the two-component mixture by **maximum likelihood (EM)** over those pairs: estimate the
  mixing probability `p`, the component-mean structure (ratio-of-`s` for closer bands, an
  absolute level for the low-band gain component), and the within-component dispersions
  `r_primary`, `r_alt`. `r_primary` is a *free fit parameter here* -- there is no conflict
  with the role-stable 37.757, which was a narrower fit on a survivorship-biased subset;
  this fit uses the full realized hold-state population and supersedes it for SV. After the
  MLE fit, **rescale the component means to satisfy the mean-preserving constraint** exactly
  (`p*m_primary + (1-p)*m_alt = s`), preserving the fitted spread ratio. This deliberately
  absorbs the `+bias` out (we do not fix the mean here).
- **Backtest is validation, not fitting.** No parameter is tuned to the backtest `SD(z)`;
  the EM fit is on individual SV outcomes, and the backtest is the independent check that
  decides two-state-vs-three-state (see Feasibility).
- **Output:** a new banded constant `SV_ROLE_MIXTURE` in `utils/constants.py`, consumed
  only by `closer_mixture.py`.

## Backtest changes (`scripts/backtest_sd_calibration.py`)

Three edits, all required for the SV gate to be both *reachable* and *measurable*:

1. **Wire SV variance to the mixture.** The backtest currently computes SV variance inline
   (`var = negbin_perf_variance(key, proj) + proj**2 * cvp**2`, line 117) for all P_CATS.
   Branch SV to `closer_mixture.sv_role_variance(proj)` instead; W/K keep the inline
   formula. **Without this the backtest still uses the old SV formula and shows no
   improvement -- the gate would be unmeasurable.**
2. **2025 SO derivation + gate loosening.** `build_year` returns `(hm, None)` for pitchers
   unless `{"W","SO","SV"}` are all present (line 85), excluding 2025. Derive `SO` for 2025
   via the thirds-corrected formula above and include the year.
3. (Bookkeeping, follows from 2.) 2025 pitchers now enter the W/SO/SV samples.

**Success-criterion correction.** Because edit 2 adds a whole 2025 season to the pitcher
pool, **W's and SO's `SD(z)` and mean will change** (new samples) -- they are *not*
"unchanged." The corrected criterion:

- **R/HR/RBI/SB** (hitters): `SD(z)` unchanged (untouched by this work; 2025 hitters were
  already included).
- **W, SO**: gain a 2025 sample; their `SD(z)` shifts but must **remain calibrated**
  (stay within `[0.8, 1.25]`). Record before/after.
- **SV**: `SD(z)` moves into `[0.8, 1.25]`; `mean z` unchanged (still ~+0.6-0.8, by
  design).

## Testing

1. **Unit** -- closed-form `sv_role_variance(s)` matches a brute-force sample of the
   mixture (property test); mean-preserving check `E[mixture] == s` within tolerance.
   (Do **not** hard-assert cross-band monotonicity of variance: with independently-fit
   per-band params the `[0,3)` gain term `(s - ~25)^2` can legitimately exceed a poorly
   separated neighbor. Instead assert variance is non-negative and that within a fixed band
   it rises with `s`.)
2. **Integration (the real target)** -- `backtest_sd_calibration.py` SV `SD(z)` lands in
   **[0.8, 1.25]**; R/HR/RBI/SB `SD(z)` unchanged; W/SO remain in `[0.8, 1.25]` (before/after
   recorded); SV `mean z` unchanged (by design).
3. **Unification** -- analytic ERoto SV SD == MC SV SD within Monte Carlo tolerance
   (extend the existing parity guard). This is the test that catches any of the four MC-seam
   steps being done inconsistently with ERoto.
4. **Valuation regression (guard, not proof)** -- SGP/VAR/VONA player SV values do not move.
   This is guaranteed by construction (the change never touches the projection/mean
   pipeline); the test exists so a future edit cannot silently break the invariant.
5. **MC re-baseline** -- deterministic-seed MC tests re-pinned for the added Bernoulli draw.

**Success = (2) and (3).** If two-state cannot reach [0.8, 1.25] with data-fit params, the
backtest surfaces it and the plan escalates to a three-state (job-share) mixture.

## Known limitations

- **Handcuff anti-correlation (train/serve mismatch).** The backtest builds synthetic
  rosters by `rng.choice` (`backtest_sd_calibration.py:100`), so two rostered closers are
  independent players from different MLB teams -- independence is valid *there*, and the
  mixture is calibrated on that population. But a real manager who handcuffs both closers of
  one MLB bullpen holds two *anti*-correlated save sources (if A saves, B does not).
  Independent per-pitcher role draws model that as additive variance, **overstating** team
  SV variance for handcuff owners. Accepted: the fix targets the calibration population;
  handcuff correlation is a separate, smaller effect not modeled here.
- **Role state uncorrelated with run-prevention** (see MC correlation limitation): the
  between-component job-switch does not co-move with sampled ER/BB/H, understating the
  SV-vs-ERA joint category correlation. Second-order for standings-win probabilities;
  deferred.
- **Skew / mean-variance separability.** Realized SV is floored at 0 and right-skewed, so
  mean and variance are not fully independent. `SD(z)` is a *mean-centered* statistic, so
  the variance target is measurable independent of the residual mean bias; but if fixing
  the variance while leaving the mean bias leaves `SD(z)` outside the band, that is itself
  the signal (captured by the Integration test) to revisit -- either three-state, or
  coordinating with #235 on the mean.

## Files touched

- `src/fantasy_baseball/sgp/closer_mixture.py` (new) -- band lookup, closed-form
  `sv_role_variance`, and per-component params for the sampler.
- `src/fantasy_baseball/utils/constants.py` -- new `SV_ROLE_MIXTURE` banded constant.
- `src/fantasy_baseball/scoring.py` -- SV branch in `player_category_variance`.
- `src/fantasy_baseball/simulation.py` -- SV role-state draw in `_apply_variance_batch`:
  component `mu` (no `scales`), component `r` (not `STAT_DISPERSION`), and SV exempted from
  the injury backfill.
- `scripts/calibrate_closer_mixture.py` (new) -- EM calibration from 2022-2025 (2025 via
  thirds-corrected SO derivation).
- `scripts/backtest_sd_calibration.py` -- SV variance wired to `sv_role_variance`; 2025 SO
  derivation + gate loosening.
- Tests under `tests/` per the Testing section.

## Out-of-scope / related issues

- **#235** -- all-positions projection-accuracy backtest (owns the SV mean bias).
- **#236** -- re-export 2025 pitcher actuals with the standard counting template (restores
  H/WHIP parity; not blocking this issue).
