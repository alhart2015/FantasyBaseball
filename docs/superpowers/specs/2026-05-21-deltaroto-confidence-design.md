# deltaRoto Confidence Indicator -- Design

Date: 2026-05-21
Branch: feat/deltaroto-confidence

## Problem

deltaRoto is currently a single point estimate (the change in EV roto points
from a swap). Users cannot tell a coin-flip swap -- one whose edge is smaller
than the projection noise -- from a genuinely good one. This causes churn on
noise: e.g. the Murakami/Burger and Pages/Freeman moves each scored about
+0.4 deltaRoto at the time (statistically indistinguishable from zero) and
later turned negative. The fix is to show the uncertainty around deltaRoto so
the size of the edge can be judged against the size of the noise.

## Goal

Surface a `+/- SD` band on every deltaRoto, colored by whether the band
crosses zero, across the roster audit, trade builder, player comparison, and
the lineup page -- the same band card on all four.

Decisions (all user-confirmed):
- Lead signal: edge vs noise, shown as a `+/- SD` band.
- Computation: Monte Carlo the swap, reusing the validated variance model.
- Core notation: `mean +/- SD`, e.g. `+1.9 +/- 2.3`.
- Coloring: keyed to whether the +/-1 SD band crosses zero.

## The metric

New shared function in `src/fantasy_baseball/lineup/delta_roto.py`:

```
compute_delta_roto_band(
    drop_name, add_player, user_roster, projected_standings, team_name,
    *, team_sds, n_draws=400, seed=...,
) -> DeltaRotoBand   # (mean, sd, p_positive, plus the existing per-category point deltas)
```

Method -- Monte Carlo with common random numbers (CRN):

1. For each of `n_draws` iterations, sample the user's team players' realized
   stats with `simulation._apply_variance` (performance variance from
   `STAT_VARIANCE` + playing-time `cv_pt` + the calibrated correlations,
   covariance scaled by `fraction_remaining`).
2. CRN: draw the per-player variance ONCE per iteration and apply it to BOTH
   the before-roster and after-roster. The two rosters differ by exactly the
   swapped player(s); all shared players get identical realized stats in both.
   So `dRoto_i = score(after_i) - score(before_i)` isolates the swap's marginal
   effect (plus the nonlinear boundary interaction) -- a low-variance estimate
   that needs few draws.
3. Score each realized roster with the existing `score_roto` (EV / Gaussian
   pairwise) against the FIXED field: the other nine teams stay at their
   point-estimate `projected_standings` totals.
4. Aggregate: `mean = avg(dRoto_i)` (this should track the existing point
   `compute_delta_roto.total`), `sd = std(dRoto_i)`, `p_positive = mean(dRoto_i > 0)`.

Field handling (deliberate): the field is held at its point estimate. dRoto is
the swap's marginal effect; sampling the field too would add largely-cancelling
noise and complicate CRN. The user team's own positional uncertainty relative
to a fixed field is still captured (shared players are sampled). Revisit only
if the band looks too tight in practice.

`compute_delta_roto` (the cheap point estimate) stays for callers that don't
need the band.

## Coloring (the upgrade)

Keyed to the displayed +/-1 SD band vs zero:

- band clears zero  (`mean - sd > 0`,  ~P(help) >= 84%) -> green  / "real"
- band below zero   (`mean + sd < 0`)                    -> red    / "downgrade"
- band straddles zero (otherwise)                        -> amber  / "coin-flip"

This replaces the roster-audit magnitude threshold (currently `>= 1.0` green,
`0..1.0` amber, `< 0` red), which is exactly the "treats a noisy +1.2 as solid"
bug this feature exists to fix.

## Surfaces and integration

One source of truth: every surface calls `compute_delta_roto_band` (or the
trade equivalent). No divergent copies.

1. Roster audit
   - Backend: `lineup/roster_audit.py` computes the band per candidate it
     surfaces. The `delta_roto` dict on each candidate gains `sd` and
     `p_positive` (mean stays as today).
   - Where computed: the refresh pipeline (it already runs the audit and the
     standings MC). CRN keeps the per-candidate cost low. Cached in
     `cache:roster_audit`.
   - Template `web/templates/season/roster_audit.html`: the gap-badge shows
     `mean +/- sd` and recolors by the crosses-zero rule. Reuse the existing
     traffic-light CSS classes (`gap-positive` / `gap-marginal` / `gap-negative`),
     re-pointed at the new rule.

2. Trade builder
   - Backend: `/api/evaluate-trade` (`web/season_routes.py`) ->
     `trades/evaluate.py`. Return the band on the trade total and per-category.
     On-demand (single trade), so the MC runs per request.
   - Template `web/templates/season/waivers_trades.html` + its JS: render
     `+/- sd` on the result; the roomier layout can also show the visual bar.

3. Player comparison
   - Backend: `/api/players/compare` (`web/season_routes.py`) ->
     `compute_delta_roto_band`. On-demand.
   - Template `web/templates/season/players.html`: render `+/- sd` in the
     standings-impact panel; room for the visual bar.

4. Lineup
   - Backend: the optimizer's output assignments (`HitterAssignment` /
     `PitcherStarter` in `lineup/optimizer.py`) and recommended moves carry the
     band -- their `roto_delta` gains `sd` + `p_positive`. Computed in the
     refresh pipeline, cached in `cache:lineup_optimal`.
   - Templates `web/templates/season/lineup.html` plus the
     `_lineup_hitters_tbody` / `_lineup_pitchers_tbody` partials and the move
     chips: show the same `mean +/- sd` card and crosses-zero color as the
     other surfaces (reuse `.roto-chip`, re-pointed at the new band + rule).
   - Only the optimizer's OUTPUT needs bands (the final per-player `roto_delta`
     cells + the handful of recommended moves), not every candidate the
     Hungarian assignment evaluated -- so the cost is bounded like the audit.

## Performance

- CRN + ~300-500 draws per swap.
- Roster audit: ~30-50 candidates x ~400 draws, precomputed offline in the
  refresh and cached -- no interactive cost. `score_roto` over 12 teams x 10
  categories is cheap; the draw loop dominates. If too slow, lower `n_draws`
  or limit to the top FA per slot (the audit already focuses there).
- Trade / compare: a single swap on-demand -- fast.
- Lineup: precomputed offline in the refresh alongside the optimal lineup;
  bounded to the output assignments + recommended moves.

## Error handling and edge cases

- Determinism: seed the rng (consistent with the rest of the MC) so cached
  bands are reproducible.
- Missing projection for the add player: band undefined -> fall back to the
  point estimate / no band; never crash.
- Rate stats: reuse `score_roto`'s component-based recombination (never average
  AVG/ERA/WHIP).
- NaN: use `safe_float` for stat reads (avoid the `x or 0` NaN trap that bites
  `_apply_variance:450`).

## Testing

- Unit (`tests/test_lineup/` or alongside existing delta_roto tests):
  - `mean` from the band tracks `compute_delta_roto.total` within tolerance.
  - `sd > 0` for a real swap; `0 <= p_positive <= 1`.
  - Determinism with a fixed seed (same band twice).
  - CRN sanity: variance of the band estimate is lower than independent
    (non-CRN) draws for the same `n_draws`.
  - Identity swap (same player in and out) -> `mean ~= 0`, tiny `sd`.
  - Coloring rule: clears / straddles / below zero map to green / amber / red.
- Integration:
  - `cache:roster_audit` entries carry `sd` + `p_positive`; `/roster-audit`
    renders without error.
  - `/api/evaluate-trade` and `/api/players/compare` return the band fields.
- Performance smoke: audit band compute stays within a sane time budget.

## Out of scope / follow-ups

- Sampling the field's uncertainty (deliberately fixed for now).
- The visual bar treatment for trade/compare is optional polish, not required
  for the first cut.
