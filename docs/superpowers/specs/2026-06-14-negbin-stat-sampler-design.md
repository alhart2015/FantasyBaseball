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
  ported from a constant fit for a different distribution. **Success is
  measured by the out-of-sample interval-coverage test in Testing** (nominal vs
  empirical 50%/80% coverage within +/- 10 pp per stat), not by the fit alone.
- Preserve the existing inter-stat correlation structure.

## Non-goals

- **Closer role-switch mixture for saves.** Decided out of scope: in-season
  projections update with role, so a NegBin centered on the updated mean tracks
  it adequately. Documented consequence: historical save dispersion includes
  job-loss events, so the save `r` is fit on the role-stable population (see
  Calibration) rather than absorbing the job-loss tail into a single inflated
  `r`. That tail is a flagged, accepted limitation for a future role-mixture
  spec, not this one.
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
  requires it. **The emitted dispersion is consumed as a lookup keyed by stat
  (and optionally by projected-count band), so the scalar-vs-mean-dependent
  outcome of the diagnostic does not change the sampler's interface** -- a scalar
  `r` is just a one-entry band. The sampler is written against this lookup from
  the start.
- Enforce the **Poisson floor**: where the data is under-dispersed at low means
  (target var <= mean), clamp to Poisson (`r -> inf`). NegBin cannot represent
  var < mean.
- **Saves (decided, given the role-mixture non-goal):** fit `sv` dispersion
  **conditional on role being stable** -- i.e. restrict the `sv` fit to
  pitcher-seasons whose projected and realized role agree (a closer who stayed a
  closer), so the single-NegBin `r` is not inflated to absorb job-loss events it
  cannot shape-model. Acceptance bar for `sv`: the role-stable population's
  predictive interval coverage (see Testing) meets the same tolerance as the
  other stats; the job-loss tail is a **documented, accepted limitation** owned
  by the future role-mixture spec, not by this one. State the role-agreement
  rule used (e.g. projected-IP/SV bucket vs realized) in the script.
- Output: a `STAT_DISPERSION` dict (`r` per stat, with Poisson sentinels) plus a
  reviewable band table, printed for paste into `constants.py`. Fail loud on
  missing year files.

### 2. Sampler rewrite: `_apply_variance` in `simulation.py`

- Draw the correlated Gaussian latent from the **correlation matrix**
  (unit-variance), not the sigma-baked covariance. The correlation matrices
  (`HITTER_CORRELATION`, `PITCHER_CORRELATION`) are reused unchanged.
- For each correlated stat: `u = Phi(z)`, **clamped to `[eps, 1 - eps]`** (see
  Math appendix), then
  `count = scipy.stats.nbinom.ppf(u, n_param, p_param)` with the NegBin
  parameterized to `mean = mu`, `dispersion = r_eff` (looked up per stat/band,
  then variance-scaled by `fraction_remaining`; see Math appendix and the
  fraction_remaining section), where `mu = base * scale` (the full-season mean,
  for BOTH callers -- see below).
- Delete the `max(0, 1.0 + draw)` mapping.
- `ab`/`ip` and any non-correlated stat keep the existing `base * scale` path.
- **Replacement backfill composition (explicit).** Keep today's decomposition:
  the NegBin mean is the *played-fraction* expectation `mu = base * scale`, the
  integer NegBin draw is the player's own production, and the unchanged
  fractional `repl_contrib = repl[col] * frac_missed` (with
  `frac_missed = max(0, 1 - scale)`) is added on top for the missed fraction:
  `row[col] = nbinom_draw(mu, r) + repl_contrib`. Only the player term changes
  shape (clipped-Gaussian multiplier -> NegBin); the replacement term is
  identical to today. The resulting total is fractional (as it is today), which
  is fine -- integer-ness of the player draw is not a requirement of downstream
  scoring.

### 3. Constants: `simulation.py` / `constants.py`

- Add `STAT_DISPERSION` (per-stat NegBin `r`, with Poisson sentinels) in the
  calibration phase **without removing `STAT_VARIANCE`** (the old sampler still
  reads it until the rewrite lands). `STAT_VARIANCE` is removed in the same phase
  that rewrites `_apply_variance`, so the tree never references a deleted
  constant. The correlation matrices stay. `_build_cov_matrix`'s sigma-scaled
  covariance precompute is removed/repurposed in the rewrite phase too (the
  copula latent needs only the correlation matrix).

## fraction_remaining handling (internal to _apply_variance; no caller change)

Both callers are left UNCHANGED. `_apply_variance` keeps producing per-player
*full-season* simulated counts; the in-season caller
(`simulate_remaining_season`) keeps its team-level "subtract YTD actuals, floor
at zero, re-derive rates" logic (`simulation.py:308-349`) verbatim.

Why this works (the key code fact): the actuals subtraction is at the
**team-aggregate** level (`:308-315`), and the per-team actual is a **constant**
across MC draws. Subtracting a constant does not change variance, so ALL of the
remaining-season variance comes from one line inside `_apply_variance`:
`cov = base_cov * fraction_remaining` (`:501`). The model reduces to
`final_total = max(team_actual, simulated_full_season)` per category. A
per-player "simulate the remainder directly" reframe was rejected: the team
actual is not the sum of the current roster's YTD (rosters change), so it cannot
be reproduced per player, and the reframe would silently alter the
`max(actual, sim_full)` behavior on the dashboard path.

So the ONLY change is how `_apply_variance` realizes the full-season count
variance. Today: scale the Gaussian covariance by `fraction_remaining`. New:
scale the NegBin **variance** by `fraction_remaining` at the fixed full-season
mean `mu = base * scale`, by solving for an effective dispersion `r_eff`:

    var_full   = mu + mu^2 / r                      # r from STAT_DISPERSION
    var_target = fraction_remaining * var_full      # mirrors today's cov scaling
    r_eff      = mu^2 / (var_target - mu)           # if var_target > mu
               = Poisson (r_eff -> inf)             # if var_target <= mu (floor)

This is the count-native analog of `cov *= fraction_remaining` (variance scales
linearly with the season fraction), with a Poisson-floor clamp for the rare
late-season case where the target variance drops below `mu`. That clamp only
bites in roughly the final two weeks (e.g. a 30-SB full-season player with
`r ~= 2.2` stays above the floor until `fraction_remaining ~= 0.07`), where a
slightly-too-wide Poisson is harmless and conservative; `run_monte_carlo`
already short-circuits to pure actuals at `fraction_remaining <= 0`.

The full-season path (`fraction_remaining = 1.0`) is the special case
`r_eff = r` (no scaling), so both callers share one code path.

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

fraction_remaining variance scaling (effective dispersion). At fixed full-season
mean `mu` and calibrated dispersion `r`:

    var_full   = mu + mu^2 / r
    var_target = fraction_remaining * var_full
    r_eff      = mu^2 / (var_target - mu)   if var_target > mu   (NegBin)
               = Poisson(mu)                if var_target <= mu  (floor clamp)

`fraction_remaining = 1.0` gives `var_target = var_full` and `r_eff = r`.

Gaussian copula step per stat (using `r_eff`):

    z  ~ correlated standard normal (from the correlation matrix)
    u  = clip(Phi(z), eps, 1 - eps) # std normal CDF -> Uniform, clamped
    x  = nbinom.ppf(u; n=r_eff, p=r_eff/(r_eff+mu))   # or poisson.ppf(u, mu) at the floor

**Tail clamp (required).** `nbinom.ppf(1.0, ...)` returns `inf`, and `Phi(z)`
reaches ~0 or ~1 for extreme latent draws; an unclamped extreme draw injects
`inf`/degenerate-0 into a team stat sum and silently corrupts a simulation.
Clamp `u` to `[eps, 1 - eps]` with `eps = 1e-9` (caps the realized stat near the
NegBin's ~1-in-1e9 quantile -- far beyond any plausible season, so the clamp
never bites realistic outcomes but guarantees finite output). A test must assert
finite, non-negative output under deliberately extreme latent draws.

Note: a Gaussian copula preserves rank correlation exactly; Pearson correlation
on the NegBin scale shifts slightly from the input matrix. Acceptable; flag in
the calibration diagnostic if it matters.

## Error handling

- Calibration: exclude `actual_PT = 0` from the performance fit; enforce the
  Poisson floor; fail loud on missing inputs; guard divide-by-zero in
  `proj_count / proj_PT`.
- Sampler: `mu = 0` -> 0 count (no draw); `r` Poisson-sentinel -> draw Poisson
  via the same copula `u`; player draw is a non-negative integer (the added
  fractional `repl_contrib` makes the stored total fractional, as today).

## Testing

- Property tests on the sampler:
  - mean of draws ~= projection (no bias) across low/mid/high means;
  - no probability spike at 0 (mass at 0 ~= NegBin P(X=0), not ~10%);
  - realized correlation matches the input matrix within tolerance;
  - player draws are non-negative integers; finite under extreme latent draws
    (tail-clamp test).
- Golden regression: the Caminero/Soto/Cruz SB table above -- assert bias ~0 and
  zero-spike gone.
- fraction_remaining: the realized full-season-count variance scales ~linearly
  with `fraction_remaining` (assert `var(sim) ~= fraction_remaining * var_full`
  in the supra-floor regime); the Poisson-floor clamp engages and stays finite
  at very small `fraction_remaining`; `r_eff = r` exactly at `= 1.0`. Both
  callers (`simulate_season`, `simulate_remaining_season`) are otherwise
  unchanged -- assert the in-season team-level subtraction/recombination code is
  untouched (it is not part of this change).
- Calibration script: a small synthetic fixture with known dispersion recovers
  `r`; Poisson-floor clamp triggers on under-dispersed input.
- **Out-of-sample calibration validation (acceptance test for "variance is
  right").** Leave-one-season-out over 2022-2025: fit dispersion on three
  seasons, then on the held-out season check per-stat **predictive-interval
  coverage** -- the fraction of held-out actual season totals (conditional on
  realized PT) that fall inside the model's central interval. Target: empirical
  coverage of the nominal 50% and 80% intervals within +/- 10 percentage points
  per stat, averaged over the four held-out folds. `sv` is evaluated on the
  role-stable population only (per the saves decision); record its coverage but
  do not block on the job-loss tail it deliberately excludes. This test gates the
  shipped `STAT_DISPERSION`; record the coverage table in the calibration output.
- Regression policy (the model intentionally changes MC output, so split this):
  - **Structural/contract tests must pass unchanged** -- roster selection,
    scoring wiring, rate recombination, no-crash on the draft and dashboard
    suites.
  - **Value-pinned MC golden tests are EXPECTED to change** -- re-bless them
    against the new model, citing this spec's model change as the justification
    (per CLAUDE.md's don't-edit-tests-without-justification rule). Do NOT loosen
    a test to hide an unexpected shift; only re-pin values whose change is
    explained by the NegBin/copula switch.

## Performance

`nbinom.ppf` per stat/player/sim is heavier than the current vectorized numpy
op. Keep it vectorized over the player axis. Concrete budgets (benchmark
before/after on the same machine, same iteration count):
- **Dashboard MC refresh** (the free-tier-constrained path): post-change
  wall-clock stays within **1.25x** the current baseline AND under the existing
  free-tier refresh time budget the pipeline already targets. If either is
  exceeded, reduce per-draw cost (e.g. precompute NegBin params per player once,
  reuse the `ppf` across the vectorized player axis) before merging.
- **Draft MC** (offline, latency-tolerant): within **1.5x** the current
  baseline; a regression beyond that is a finding to record, not necessarily a
  blocker.
Record the before/after numbers in the PR.

## Internal phasing within this spec

Each phase must leave the tree green (the repo's phased-execution rule). Because
`_apply_variance` is a single function shared by both callers, the sampler
rewrite and both caller adaptations land together -- they cannot be split by
caller.

1. **Calibration (additive).** `scripts/calibrate_stat_dispersion.py` +
   diagnostic + out-of-sample interval-coverage validation; emit and commit
   `STAT_DISPERSION` ALONGSIDE the still-present `STAT_VARIANCE`. No sampler
   change yet, so the tree stays green.
2. **Sampler (internal to `_apply_variance`).** Rewrite `_apply_variance` to the
   copula+NegBin marginals with the `fraction_remaining` -> `r_eff` variance
   scaling and Poisson-floor clamp; remove `STAT_VARIANCE` and the sigma-scaled
   covariance precompute. **No caller changes** -- both `simulate_season` and
   `simulate_remaining_season` keep their existing aggregation/subtraction/
   recombination logic verbatim. Land with the property/golden tests, the
   re-blessed value-pinned goldens, the dashboard/deltaRoto regression, and the
   perf benchmark.

Phase 2 is now contained to `_apply_variance` plus its constants; keep edits
within the <= 5-files-per-step rule (sampler core, then test re-bless) and verify
(pytest/ruff/format/vulture/mypy) at each step, but it ships as one coherent
milestone since partial states break the in-season path.

## Open items to resolve in planning

- In-season horizon: **resolved** -- no caller reframe. `_apply_variance` keeps
  producing full-season counts; variance is scaled by `fraction_remaining` via
  `r_eff` (see the fraction_remaining section). The team-level subtraction in
  `simulate_remaining_season` is left verbatim. (Earlier drafts proposed a
  per-player `mu_rem` reframe; dropped after reading `simulation.py:308-349`
  showed the subtraction is team-level and constant-valued.)
- Single `r` vs mean-dependent dispersion: decided by the calibration diagnostic.
  Resolved structurally -- the sampler consumes a stat/band dispersion lookup
  either way (scalar = one band), so the diagnostic's outcome cannot force a
  sampler redesign.
- Saves `r`: **resolved** -- fit on the role-stable population (see Calibration
  and Non-goals); job-loss tail is an accepted limitation deferred to a future
  role-mixture spec.
