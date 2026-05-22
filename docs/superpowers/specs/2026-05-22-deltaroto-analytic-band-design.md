# deltaRoto Confidence Band -- Analytic Redesign

Date: 2026-05-22
Branch: feat/deltaroto-analytic-band
Supersedes the Monte-Carlo computation in `2026-05-21-deltaroto-confidence-design.md`
(the surfaces and the crosses-zero verdict are kept; only the math and the
per-surface display change).

## Problem

The MC confidence band shipped in PR #88 broke production and muddied all four
surfaces it touched:

1. **It kills the daily refresh on free-tier Render.** `audit_roster` runs a
   300-draw MC band *per FA candidate* (~30-50) and `optimize_hitter_lineup` /
   `optimize_pitcher_lineup` run one *per starter* (~21) -- ~15-20k full-team
   variance draws added to every refresh. Render logs (2026-05-22) show both
   refreshes dying at `Running roster audit...`; the worker is hard-killed
   before `JobLogger.finish()`, so no job log is written and
   `cache:roster_audit` is never updated. The roster-audit page therefore shows
   stale/empty deltaRoto.
2. **Lineup is too cluttered**: a `mean +/- sd` chip on every active starter
   row (~21) plus the move chips.
3. **Compare shows two numbers that disagree**: it renders both the EV
   `deltaRoto` (`score_roto` with `team_sds`, Gaussian pairwise) *and* the band
   (`mean` from MC scored by *rank*, `team_sds=None`). Different scoring
   regimes, so they never match.
4. **Trade**: the single-player Trade Finder (`/api/trade-search`) never got a
   band; the multi-player builder runs a 400-draw MC per evaluate, doubled by a
   redundant `evaluate_multi_trade` call inside `/api/optimize-trade-lineup` --
   unusably slow on free tier, and a throw inside the MC surfaces as "breaks
   when evaluating".

## What the variance model actually is (scope of honesty)

We do **not** have per-player performance variance. We have:

- `STAT_VARIANCE` (constants.py): one coefficient of variation **per stat**,
  league-calibrated (Steamer+ZiPS vs actuals 2022-24). Stable: H 0.10, K 0.14,
  R 0.16, RBI 0.19. Noisy: HR 0.34, W 0.42, SB 0.72, SV 0.90. Applied
  proportionally to every player's projection.
- A playing-time `cv_pt` that varies by a player's projected **volume** (PA/IP)
  -- the only player-specific differentiation, and it is playing-time risk, not
  performance boom/bust.

So the band is honest as **"is this swap's edge bigger than the standings'
natural, category-by-category noise?"** -- an SB/SV swap reads wider than an
R/K swap, and a swap in a contested category swings more than one in a locked
category. It is **not** a per-player risk profile and must not be presented as
one. This bounds the design: no false precision (one decimal, as today), and we
do not claim a player is "steadier" than another with the same projection and
volume.

## Goal

Replace the Monte-Carlo band with a closed-form analytic band built from the
per-category variances we already compute (`project_team_sds`). Requirements:

- **mean == the EV deltaRoto** (`compute_delta_roto(...).total`) exactly, so the
  band is consistent with the roster-audit sort key and the rest of the app.
  This dissolves symptom 3: there is only one number, so nothing can disagree.
- Cheap enough to run inline in the refresh (audit + lineup) and on demand
  (compare + trade) on free-tier hardware -- no sampling.
- Deterministic (no RNG).
- Same `DeltaRotoBand(mean, sd, p_positive)` shape and the same crosses-zero
  `band_class` verdict, so templates and `band_format.py` are unchanged.

## The method (per-category Gaussian propagation)

For a swap that removes players `OUT` and adds players `IN` to the user team:

**Mean** -- reuse the existing EV path. `mean = compute_delta_roto(...).total`
(the change in `score_roto`'s Gaussian-pairwise expected points with
`team_sds`). No recomputation; identical to today's point estimate.

**SD** -- propagate the swap's per-category stat uncertainty through the
roto-points curve, one category at a time:

1. **Swap stat-change variance** `sigma2_delta_c`. In category `c`, the swap
   changes the user's category total by `dX_c = sum(IN stat_c) - sum(OUT
   stat_c)`. Its variance is the sum of the involved players' own per-category
   variances -- exactly the per-player terms `project_team_sds` already sums
   (`STAT_VARIANCE`-CV plus `cv_pt`), restricted to the IN and OUT players, and
   scaled by `fraction_remaining` (variance scales linearly). Counting
   categories use the counting formula; rate categories (AVG/ERA/WHIP) use the
   same component recombination `project_team_sds` uses (never average a rate).
2. **Propagate through the points curve.** The user's category-`c` points as a
   function of a realized total `x` is
   `pts_c(x) = sum_j Phi((x - mu_j) / s_cj)` over the 9 fixed opponents, with
   `s_cj` the same combined SD `score_roto` uses at the baseline operating
   point. The realized per-category delta is
   `dpts_c = pts_c(mu_b + dX_c) - pts_c(mu_b)`, where `mu_b` is the user's
   baseline category mean and `dX_c ~ N(d_mu_c, sigma2_delta_c)`. Holding the
   shared players at their mean is the closed-form expression of the MC's
   common-random-numbers trick: the rest of the roster is identical before and
   after, so it cancels and the band stays tight.
3. **Integrate** `Var(dpts_c)` with a small fixed **Gauss-Hermite** node set
   (~9 nodes): `dX_c = d_mu_c + sqrt(2)*sigma_delta_c*z_k`, accumulate weighted
   `dpts_c` and `dpts_c^2`, take `Var_c = E[dpts^2] - E[dpts]^2`. Deterministic,
   ~9 evals/category.
4. **Combine across categories**: `sd = sqrt(sum_c Var_c)` (categories treated
   as independent -- see Out of scope).

**p_positive** = `Phi(mean / sd)` (`sd == 0` -> 1.0 if `mean > 0` else 0.0).
**verdict** = unchanged `band_class(mean, sd)`.

Cost: ~90 normal-CDF/PDF evals per swap vs ~300 full-team re-scorings -- runs
inline everywhere.

## API

`src/fantasy_baseball/lineup/delta_roto.py`:

- Rewrite `compute_delta_roto_band(...)` to the analytic method. New signature
  drops `n_draws` / `seed`; it needs the before/after rosters (or drop/add +
  roster), the projected standings / field, `team_name`, `team_sds`, and
  `fraction_remaining`. Returns `DeltaRotoBand(mean, sd, p_positive)` unchanged.
- `compute_one_for_one_band(...)` stays as the 1-for-1 wrapper (audit, compare),
  signature updated to drop `n_draws` / `seed`.
- `DeltaRotoBand.to_dict()` and `band_format.band_class` are unchanged.
- Delete `_sum_realized` and the `_apply_variance` / numpy sampling imports.

## Surfaces

1. **Roster audit** (`roster_audit.py`, `roster_audit.html`): band stays
   per-candidate (now cheap, so the refresh stops dying and the page
   repopulates). No template change beyond what already reads `.band`.
2. **Lineup** (`optimizer.py`, `_lineup_*_tbody.html`, `lineup.html`): stop
   rendering a band chip on settled starter rows. Show the band **only on the
   recommended moves**. The optimizer computes a band only for the moves it
   surfaces, not every starter. Settled rows keep their plain `roto_delta`.
3. **Compare** (`season_data.compute_comparison_standings`, `players.html`):
   show **band `mean +/- sd` only**; delete the separate "deltaRoto: X roto
   pts" EV line in `renderDeltaRoto`. Stop returning the now-redundant separate
   `delta_roto` total for display (keep per-category deltas for the standings
   table). Since `mean == EV delta`, nothing disagrees.
4. **Trade**:
   - Single (`/api/trade-search`): compute and return the band per candidate so
     it matches the other surfaces; render `mean +/- sd` on each card.
   - Multi (`/api/optimize-trade-lineup`): drop the redundant
     `evaluate_multi_trade` call -- it only needs a legality/size check, not the
     band. `/api/evaluate-trade` keeps the band (now cheap).

## Error handling and edge cases

- Missing/NaN projection for a swapped player -> skip the band (return `None`),
  fall back to the mean only; never crash. Reuse `safe_float` for stat reads.
- Identity swap (same player in and out) -> `mean ~= 0`, `sd ~= 0`.
- Zero category variance (e.g., no playing time) -> that category contributes 0
  to the band; no divide-by-zero (guard `sd == 0`).
- Determinism is structural (no RNG).

## Testing

`tests/test_lineup/test_delta_roto_band.py` is rewritten for the analytic path:

- `mean` equals `compute_delta_roto(...).total` within a tight tolerance.
- `sd > 0` for a real swap; `0 <= p_positive <= 1`.
- **Honest-signal lock**: an SB- or SV-driven swap produces a wider `sd` than an
  R- or K-driven swap of equal `mean`. (Encodes the only player-independent
  thing the band legitimately says.)
- Identity swap -> `mean ~= 0`, `sd ~= 0`.
- Determinism: same inputs -> same band twice.
- `band_class` mapping: clears / straddles / below zero -> real / coin-flip /
  downgrade.
- Performance smoke: an audit-sized batch (~50 candidates) computes well under a
  sane time budget (no sampling).
- Integration: `cache:roster_audit` entries still carry `sd`/`p_positive`;
  `/roster-audit`, `/api/players/compare`, `/api/evaluate-trade`, and
  `/api/trade-search` render without error.

The MC-specific CRN-variance test is removed (the mechanism is gone).

## Out of scope / future refinements

- **Cross-category correlation.** Summing per-category variances assumes
  independence; a breakout hitter is up in R/HR/RBI together, so the true band
  is slightly wider. The MC captured this via the calibrated correlation
  matrix; the closed form does not. Acceptable for a v1 "edge vs noise" signal;
  revisit if bands read too tight in practice.
- **Field uncertainty.** The 9 opponents are held at their point estimates
  (same deliberate choice as the MC design).
- **Per-player variance.** Not available; out of scope by data, not by effort.
