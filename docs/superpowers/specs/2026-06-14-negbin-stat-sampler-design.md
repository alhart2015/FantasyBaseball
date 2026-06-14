# Negative-Binomial Counting-Stat Sampler (Gaussian Copula)

Date: 2026-06-14
Status: Approved (brainstorm); pending implementation plan
Branch: june-13-group-review

## Problem

The Monte Carlo performance sampler perturbs each counting stat with a clipped,
zero-mean Gaussian multiplier (`simulation.py`, `_apply_variance`):

    perf = max(0, 1.0 + draw)          # draw ~ N(0, sigma), sigma from STAT_VARIANCE
    row[col] = base * perf * scale + repl_contrib

`draw` is symmetric and zero-mean, so the multiplier is *meant* to average 1.0.
The `max(0, ...)` clip chops the entire left tail onto a single point at 0,
producing two artifacts:

1. **Upward mean bias.** Clipping a symmetric distribution raises the surviving
   mean above 1.0. For SB (sigma=0.715) the realized multiplier averages ~1.026
   -- every SB total comes out ~2.6% high. SV is worse (sigma=0.900). The bias
   is small per player but systematic and one-directional across the whole
   roster, and SB/SV are tight roto categories where fractional points flip
   standings places.
2. **A probability spike at exactly zero.** ~8-11% of the mass lands on a
   literal 0. The model claims a 19-SB projection (Oneil Cruz) has an ~8.7%
   chance of finishing with 0 steals -- not a real baseball outcome, just the
   clipped tail piling up at the floor.

Worked example (400k draws, full variance), current vs the proposed
mean-preserving model:

    player    proj method      mean   med    P5   P25   P75   P95  bias%  P(0)%
    Caminero     4 current     4.10   4.0   0.0   2.1   5.9   8.7   +2.6   11.1
    Caminero     4 proposed    4.00   3.3   1.1   2.1   5.0   9.4   +0.1    0.2
    Soto        12 current    12.33  12.0   0.0   6.2  17.8  26.1   +2.7    9.0
    Soto        12 proposed   12.00   9.8   3.4   6.3  15.1  28.1   -0.0    0.0
    Cruz        19 current    19.51  19.0   0.0   9.8  28.2  41.4   +2.7    8.7
    Cruz        19 proposed   19.01  15.5   5.4  10.0  23.9  44.5   +0.1    0.0

This sampler feeds the draft Monte Carlo, the in-season ROS sim, the season
dashboard, and deltaRoto -- the numbers that drive real draft and lineup
decisions. It is also the root of MEDIUM #5 ("saves variance too low"): you
cannot know whether the saves variance is right without comparing to actuals.

## Goals

- Remove the upward mean bias (mean of draws sits on the projection).
- Remove the zero-spike; give counts a realistic non-negative, right-skewed
  shape.
- Calibrate the spread (dispersion) per stat against real season-to-season
  outcomes, so "is the variance right" (esp. saves) is answered from data, not
  ported from a constant fit for a different distribution.
- Preserve the existing inter-stat correlation structure.

## Non-goals

- **Closer role-switch mixture for saves.** Decided out of scope: in-season
  projections update with role, so a NegBin centered on the updated mean tracks
  it adequately. Documented consequence: historical save dispersion includes
  job-loss events, so the save `r` fit either conditions on role being stable or
  accepts a mildly inflated `r` (still better than today's too-low variance).
  Flagged for a future spec, not this one.
- Changing the playing-time model (it already owns PT variance; see below).
- Changing the correlation matrices (they are rate-residual correlations and a
  copula consumes them directly).

## Key decisions

1. **Distribution: Negative Binomial** for all correlated counting stats
   (hitters: r, hr, rbi, sb, h; pitchers: w, k, sv, er, bb, h_allowed).
   Rationale: counting stats are overdispersed (var >> mean: Cruz SB mean 19,
   target var ~185, ratio ~10). Poisson forces var = mean (CV would be ~3x too
   tight); Binomial is underdispersed (var < mean), tighter still. NegBin is the
   only one of the three whose variance can exceed its mean, tunable via `r`,
   and it arises as a Poisson with a Gamma-distributed (uncertain) rate -- which
   is exactly the talent/usage uncertainty that drives fantasy variance. As
   `r -> inf` it collapses to Poisson, so it strictly generalizes the
   alternatives.

2. **Correlation: Gaussian copula.** Keep drawing correlated normals (the
   existing multivariate-normal machinery), but draw from the *correlation*
   matrix (unit variance) as the copula latent, then map each component
   `z -> u = Phi(z) -> count = NegBin_ppf(u; mu, r)`. Preserves the calibrated
   correlations (pitcher er/bb/h_allowed coherence at 0.729, sv's -0.341 vs
   them) while giving NegBin marginals. Independent per-stat draws were rejected
   because they would break ERA/WHIP coherence.

3. **Scope: all counting stats** (uniform copula+NegBin), not just SB/SV. Avoids
   a mixed two-codepath sampler and removes the (small) bias everywhere. `ab`/
   `ip` (sigma 0.0, not in the correlation set) stay on the existing
   `base * scale` path.

4. **Calibrate dispersion up front from 2022-2025 actuals** (no interim
   ported-constant ship). The existing `STAT_VARIANCE` was fit for the clipped
   Gaussian and does not transfer cleanly; a constant CV is also mathematically
   incompatible with counts at low means (Poisson floor CV = 1/sqrt(mu)).

5. **Both sim paths in this spec** (full-season draft/preseason AND in-season
   fractional ROS). Chosen over a full-season-first split to avoid leaving the
   high-stakes dashboard on the biased sampler and to avoid a mixed codepath.

## Architecture

Three pieces (two new, one rewrite).

### 1. Calibration script (new): `scripts/calibrate_stat_dispersion.py`

Offline, version-controlled output -- mirrors `scripts/calibrate_playing_time.py`.

- Inputs: historical projections `data/projections/{2022..2025}/` and actuals
  `data/stats/{hitters,pitchers}-{year}.csv`.
- Reuse `calibrate_playing_time.py`'s population handling: volume floors,
  RP-requires-MLB-appearance, phantom-projection exclusion, and the NaN-or-0
  trap fix (a projected player with no actuals row played ~0, not dropped --
  but see PT conditioning below for who is included in the dispersion fit).
- **Condition on playing time (the load-bearing methodology).** The
  playing-time model already owns the variance of `actual_PT / projected_PT`.
  The performance dispersion must therefore be measured conditional on realized
  PT, or the two variance sources double-count (the `_apply_variance`
  double-count noted in prior audits). Concretely, for each player-season fit
  the dispersion of `actual_count` around

      mu = (proj_count / proj_PT) * actual_PT          # proj rate * realized PT

  i.e. the per-event (rate) residual, holding playing time at its realized
  value. Players with `actual_PT = 0` are excluded from the performance fit
  (that tail belongs to the PT model).
- Fit one `r` per stat by maximum likelihood (`scipy.stats.nbinom`). Then run a
  **diagnostic across projected-count buckets** (does a single `r` reproduce the
  observed per-bucket variance?). Default to a single `r` per stat (YAGNI);
  escalate to a mean-dependent dispersion only if the diagnostic clearly
  requires it.
- Enforce the **Poisson floor**: where the data is under-dispersed at low means
  (target var <= mean), clamp to Poisson (`r -> inf`). NegBin cannot represent
  var < mean.
- Saves caveat (from Non-goals): fit `sv` dispersion conditional on role being
  stable, or accept a mildly inflated `r`; document which.
- Output: a `STAT_DISPERSION` dict (`r` per stat, with Poisson sentinels) plus a
  reviewable band table, printed for paste into `constants.py`. Fail loud on
  missing year files.

### 2. Sampler rewrite: `_apply_variance` in `simulation.py`

- Draw the correlated Gaussian latent from the **correlation matrix**
  (unit-variance), not the sigma-baked covariance. The correlation matrices
  (`HITTER_CORRELATION`, `PITCHER_CORRELATION`) are reused unchanged.
- For each correlated stat: `u = Phi(z)`, then
  `count = scipy.stats.nbinom.ppf(u, n_param, p_param)` with the NegBin
  parameterized to `mean = mu`, `dispersion = r` (see Math appendix), where
  `mu = base * scale` (full-season path) or the remaining-horizon mean (in-season
  path, below).
- Delete the `max(0, 1.0 + draw)` mapping.
- `ab`/`ip` and any non-correlated stat keep the existing `base * scale` path.
- Replacement backfill (`repl_contrib`) for missed playing time is unchanged in
  intent; confirm it composes with integer NegBin counts.

### 3. Constants: `simulation.py` / `constants.py`

- Replace `STAT_VARIANCE` (per-stat Gaussian sigmas) with `STAT_DISPERSION`
  (per-stat NegBin `r`, with Poisson sentinels). Update `_build_cov_matrix`
  usage: the copula latent needs only the correlation matrix, so the
  sigma-scaled covariance precompute is removed or repurposed.

## fraction_remaining handling (the subtle part)

The full-season path (`fraction_remaining = 1.0`, draft/preseason) maps directly:
`NegBin(mu = base * scale, r)`.

The in-season path currently simulates a full season with covariance scaled by
`fraction_remaining`, then subtracts YTD actuals to get the remainder
(`simulation.py:300-329`). This does not port to NegBin: a NegBin's variance is
tied to its mean via `r`, so it cannot be independently shrunk to
`fraction_remaining * full_variance` at a fixed full-season mean -- by late
season that target falls below the Poisson floor (var < mean).

Reframe: **simulate the remaining horizon directly**, then add YTD actuals back
(the existing rate-recombination at lines 317-329 is retained -- it still
combines YTD component totals with simulated remaining components to re-derive
AVG/ERA/WHIP). For the remaining horizon:

- Mean: the remaining expectation `mu_rem` (the same quantity the current
  `sim_full - actual_YTD` targets in expectation). EXACT definition depends on
  how `base` is constructed in the in-season caller (YTD+ROS updated full-season
  projection vs ROS-only); this is the one item to pin during planning -- read
  the caller and confirm before coding.
- Dispersion: scale the talent term by `fraction_remaining` so
  `CV_rem^2 = 1/mu_rem + fraction_remaining / r`. This is a correctness
  improvement over the current `sqrt(fraction_remaining)` SD scaling: the
  irreducible Poisson term `1/mu_rem` naturally grows as the remaining sample
  shrinks (more relative noise over a short window), while the talent-uncertainty
  term shrinks as the season is increasingly observed.

## Math appendix

NegBin parameterization (mean/dispersion form). For target mean `mu` and
dispersion `r` (`r > 0`):

    var = mu + mu^2 / r
    CV^2 = 1/mu + 1/r
    scipy: nbinom(n = r, p = r / (r + mu))   # mean = mu, var = mu + mu^2/r

Deriving `r` from a target relative SD (CV), used only for sanity checks /
diagnostics, not for the shipped values:

    r = mu / (CV^2 * mu - 1)        # requires CV^2 * mu > 1 (else Poisson floor)

Poisson floor: a count's minimum achievable CV is `1/sqrt(mu)`. If a target
CV < 1/sqrt(mu), NegBin cannot represent it; clamp to Poisson (`r -> inf`).

Gaussian copula step per stat:

    z  ~ correlated standard normal (from the correlation matrix)
    u  = Phi(z)                     # standard normal CDF -> Uniform(0,1)
    x  = nbinom.ppf(u; n=r, p=r/(r+mu))

Note: a Gaussian copula preserves rank correlation exactly; Pearson correlation
on the NegBin scale shifts slightly from the input matrix. Acceptable; flag in
the calibration diagnostic if it matters.

## Error handling

- Calibration: exclude `actual_PT = 0` from the performance fit; enforce the
  Poisson floor; fail loud on missing inputs; guard divide-by-zero in
  `proj_count / proj_PT`.
- Sampler: `mu = 0` -> 0 count (no draw); `r` Poisson-sentinel -> draw Poisson
  via the same copula `u`; ensure integer, non-negative outputs.

## Testing

- Property tests on the sampler:
  - mean of draws ~= projection (no bias) across low/mid/high means;
  - no probability spike at 0 (mass at 0 ~= NegBin P(X=0), not ~10%);
  - realized correlation matches the input matrix within tolerance;
  - outputs are integer and non-negative.
- Golden regression: the Caminero/Soto/Cruz SB table above -- assert bias ~0 and
  zero-spike gone.
- fraction_remaining: remaining-horizon variance grows in relative terms as
  `fraction_remaining -> 0`; full-season path (`= 1.0`) unchanged in
  distributional shape from the new model.
- Calibration script: a small synthetic fixture with known dispersion recovers
  `r`; Poisson-floor clamp triggers on under-dispersed input.
- Existing MC/standings/deltaRoto tests must still pass (rerun the draft and
  dashboard suites).

## Performance

`nbinom.ppf` per stat/player/sim is heavier than the current vectorized numpy
op. Keep it vectorized over the player axis; benchmark the dashboard refresh and
the draft MC before/after and confirm no material regression (the free-tier
refresh budget is a known constraint).

## Internal phasing within this spec

1. Calibration script + `STAT_DISPERSION` constants (+ diagnostic).
2. Sampler rewrite for the full-season path (copula + NegBin marginals) with
   property/golden tests.
3. In-season fraction_remaining reframe + rate recombination + dashboard/
   deltaRoto regression.

Each phase touches <= 5 files per the repo's phased-execution rule; verify
(pytest/ruff/format/vulture/mypy) between phases.

## Open items to resolve in planning

- Exact `mu_rem` definition: confirm how `base` is built in the in-season caller
  (YTD+ROS vs ROS-only) before implementing the remaining-horizon mean.
- Single `r` vs mean-dependent dispersion: decided by the calibration
  diagnostic.
- Saves `r`: role-stable conditioning vs accept inflated `r` -- decide from the
  fit.
