# Games-based availability MC -- design

Date: 2026-06-26 (model corrected 2026-06-26 after the Phase 0 gate)
Status: Phase 0 gate = GO. Building the full engine. Phases 1-6 below.
Branch: mc-games-based-availability

## Problem

The in-season ROS Monte Carlo (`simulate_remaining_season_batch` in
`simulation.py`) picks each team's active roster **position-blind and
slot-blind**: top `h_slots` hitters by raw `R+HR+RBI+SB`, top `p_slots` pitchers
by `closer-bonus + W+K+SV`, **re-selected every iteration**, no eligibility/bench
check, IL-status-blind (`simulation.py:749-765`). Three things are wrong with it,
quantified by the Phase 0 gate (see `docs/superpowers/games-mc-phase0-attribution-2026-06-26.md`):

1. **Healthy-bench seating.** The MC seats benched bats at full whenever their raw
   stats are top-k -- with no injury required. League-wide this is the dominant
   over-credit (e.g. Send in the Cavalli: Perez + Ward + Arraez ~99 RBI seated
   off the bench; ~371 RBI of bench-seating across the league vs ~85 of churn).
2. **Per-iteration re-selection churn (~85 RBI league-wide).** Re-picking the
   top-k on each iteration's *sampled* stats is best-ball inflation.
3. **IL seated at full, no displacement.** The MC seats IL players at full ROS
   immediately AND keeps the bodies they should displace -- double-counting slots.

Note on the original framing: the "~94-RBI SkeleThor" example that first motivated
this was a MISATTRIBUTION. SkeleThor's MC-vs-ERoto RBI gap is almost entirely
**IL displacement** (its IL bats Hicks/Acuna return and ERoto scales down active
Okamoto/Bauers to make room), not bench-seating (~2 RBI). The gate corrected the
target model (below).

## Corrected model (the three fixes, as the manager actually plays)

1. **IL players: full ROS + one-for-one displacement.** An IL player's ROS
   projection already bakes in the injury (reduced games), so count it at full
   ROS. Because it returns to a slot with no opening, it displaces ONE eligible
   active body by the IL player's expected ROS playing time -- a slot-PT-conserving
   swap. This is exactly what ERoto's `_apply_displacement` already does
   (`scoring.py`); the MC does not do it at all today. **Deterministic, computed at
   setup on the ROS means.**
2. **Healthy bench: fills SIMULATED injury games.** The MC's playing-time sampling
   IS the injury simulation. When an active starter draws a low-PT (injured)
   stretch in an iteration, an eligible healthy-bench body fills those missed
   player-games at its own (typically lower) rate -- nonnegative, capped at one
   body, replacement-level only when no bench body is free. This is the
   injury-insurance value of depth that neither today's MC (seats bench at full,
   no injury needed) nor ERoto/bench-exclusion (zero) models. **Stochastic,
   per-iteration.**
3. **Churn freeze.** The active set is fixed at setup (classification +
   displacement), NOT re-selected per iteration. Per-iteration variation comes
   only from the stochastic stat/PT draws and the injury-fill they trigger -- not
   from best-ball re-selection. Folded into the engine, not a separate ship.

These compose: ERoto-correct IL handling, plus a stochastic bench-fill layer ERoto
lacks, on a churn-free fixed active set.

## Data-path reality (read before the design)

`run_ros_monte_carlo` flattens each Player via `_flatten_full_season` ->
`Player.to_flat_dict_full_season()` (`simulation.py:929-931`, `player.py:294-303`),
which overlays **`full_season_projection`** (= YTD + ROS), NOT `rest_of_season`.
So the batch samples **full-season** stats and recovers the remainder as
`max(0, sim - actual_YTD)` at the team level (`simulation.py:781-805`). A `G`
field flattened the same way would arrive as **full-season G**, not ROS games.

The engine therefore operates in **ROS (remaining) terms**, sourcing ROS games
and ROS stat means from `Player.rest_of_season` at MC setup (Player objects are
in hand in `run_ros_monte_carlo` before the flatten). See Component 4.

## Components

1. **Games data plumbing.** Add `g` to `HitterStats` and `g`/`gs` to
   `PitcherStats`; thread from the projection CSVs (`G`, `GS` present in the
   FanGraphs exports) through the blend. PA/IP unchanged. Foundational.

   Audit (grep-every-call-site):
   - New fields MUST NOT enter SGP. `calculate_player_sgp` reads only explicitly
     named fields, so this holds by construction -- the audit confirms it.
   - Forward serialization round-trips stay stable.
   - BACKWARD compat: already-persisted JSON (`draft_state*.json`, dashboard
     state) lacks `G`, so a round-trip materializes `g=0`/`gs=0` via `from_dict`'s
     `or 0` -- the classic falsy-zero footgun. Audit that no consumer trusts
     pre-change `g`/`gs`, or gate use behind a presence check.

2. **Setup: classification + IL displacement (deterministic, reuse ERoto).** In
   `run_ros_monte_carlo`, on the Player rosters (before flatten), per team:
   - `_classify_roster` -> (active, il, healthy-bench).
   - `_compute_displacement_factors(active, il, league_context)` -> per-active-body
     displacement factor (IL players activate at full ROS; the worst eligible
     active match is scaled by the IL player's expected ROS PT). These are the
     SAME functions ERoto's projected standings use, so the MC's IL handling
     agrees with ERoto by construction. Build the `LeagueContext` the same way
     `ProjectedStandings.from_rosters` does (it already ran earlier in the
     pipeline; reuse `build_eos_baseline` + `build_team_sds` or thread the context
     through rather than recomputing).
   - Output per team: the **effective active set** = active-slot bodies (each with
     its displacement factor, mostly 1.0) + IL bodies (at full ROS), and the
     **healthy-bench fill pool** (each with eligible positions, ROS games `g_ros`,
     per-game value, ROS stat means). `g_ros` is `rest_of_season.g` (the ROS CSVs
     carry a ROS-scaled `G`; a full-timer reads ~75 G mid-season); where a body
     has ROS PA/IP but no `G`, derive games via the shared per-game constant
     (Open questions).

3. **Per-iteration: availability draw + bench injury-fill (stochastic).** On the
   sampled ROS stats, per team, per iteration:
   - Each effective-active body's stats are sampled with the existing playing-time
     variance (`_apply_variance_batch`) -- this is the simulated injury. Apply the
     body's displacement factor (from Component 2) to its contribution.
   - The body's simulated shortfall = its expected ROS games minus its sampled
     games (the missed games this iteration). For each shortfall, fill from the
     eligible **healthy-bench** pool, at the bench body's own per-game rate.
   - **Value rule (per-game).** Order fill bodies by per-game ROS value =
     `calculate_player_sgp(rest_of_season) / g_ros`, guarded
     `value/g_ros if g_ros>0 else 0`. (Per-game, not total SGP: filling N games is
     a per-game-quality decision; total SGP would seat a full-time mediocre body
     over a part-time better one. Minor caveat: SGP's rate terms use fixed
     full-season `team_ab=5500`/`team_ip=1450`, so the AVG sub-term is slightly
     off-horizon -- 1 of 5 hitter cats, far smaller than the volume bias of total
     SGP.) Per-game value computed once per body at setup (Player objects in
     hand); NOT read off flat dicts (`to_dict` emits `sgp` only when set).
   - **One-body capacity.** Each bench body has a finite ROS games pool; covering a
     shortfall decrements it, so one bench bat cannot cover two simultaneous
     injuries beyond one body's worth. Tie-break deterministically (higher
     per-game value, then player-id ascending).
   - **Replacement last.** Games still uncovered after the bench pool is exhausted
     fall to `_replacement_line` (`simulation.py:435-461`), expressed per-game by
     dividing by the shared per-game constant (Open questions; it has no games
     field of its own, so this is honestly derived, not recoverable from the
     `REPLACEMENT_BY_POSITION` calibration).

   **Rate-stat handling (AVG/ERA/WHIP).** Not filled separately. Every contributing
   body (active, IL, bench-fill, replacement residual) adds recovered counting
   *components* -- hitters `h`, `ab`; pitchers `er`, `ip`, `bb`, `h_allowed` --
   scaled to games covered, into the team component sums; rates recombine from team
   totals as today. A fill body contributes its own (lower) rate by volume,
   dragging the team rate the realistic direction.

   **Variance note (acknowledged).** The NegBin copula samples a body at full ROS
   volume; scaling those counts to games-covered by fraction `f` gives variance
   `f^2*var` vs ~`f*var` for genuine partial play -- understates partial-fill
   variance. Bounded (applies only to the small fill portion) and still strictly
   more realistic than today's *deterministic* replacement fill (zero variance).
   The SD backtest is the gate; partial-volume re-sampling is the deferred
   refinement.

   Allocation quality: <=2 healthy-bench bodies (BN2) per team -- a tiny
   assignment; greedy is near-optimal, error averaged over 1000 iters.

4. **MC integration (ROS-direct).** For the hitter path, stop sampling full-season
   and recovering by subtraction; sample **ROS production** directly (from
   `rest_of_season`), apply displacement factors, run the per-iteration injury-fill
   allocation, sum to team ROS, blend `team_total = team_YTD + summed_ROS` (rates
   recombined from `YTD + ROS` components, using actual_ab/actual_ip threaded from
   Yahoo). Wins: (a) horizon-consistent (games, stats, damping all ROS); (b) the
   banked-YTD floor becomes structural for hitters (ROS contributions >=0, so
   `team_total >= YTD`; the `max(actual, sim)` clamp is unnecessary for hitters).
   The active set is fixed at setup -> no per-iteration re-selection (churn freeze).

   Reconcile the PT-scale `fraction_remaining` damping under ROS-direct so
   remaining-season risk is applied ONCE (likely: pass `fraction_remaining=1.0` to
   `playing_time_moments` for the ROS-direct draw; confirm via SD backtest).

   Heavy NegBin sampling stays vectorized; the light per-team/per-iteration fill
   allocation may be a Python loop (cheap at this scale).

5. **Pitchers.** Mirror ERoto's pitcher handling: classification + IL displacement
   via the pitcher pool model (`_compute_pitcher_pool_factors`), and exclude
   healthy bench pitchers. This gives pitchers the SAME IL-correct, bench-excluded
   treatment as hitters (mechanisms 1 + churn-freeze), keeping 5x5 standings
   coherent (no tilt toward pitching-deep teams). DEFER the stochastic
   healthy-bench injury-fill for pitchers (mechanism 2) and closer-role SV
   modeling (`SV -> 0` on job loss): pitcher streaming and the bullpen-fill dynamic
   differ enough to warrant their own design, and the measured effect is on the
   hitter side. Do NOT add a `gs`/`g` volume scaler on top of the IP-calibrated PT
   scale (double-discounts; no-op for ERA/WHIP). The remaining hitter-vs-pitcher
   asymmetry (bench injury-fill on hitters only) is explicit and surfaced by the
   all-categories acceptance evidence.

6. **Validation + before/after evidence.** See Acceptance evidence below, plus a
   backtest of category means AND SDs against realized outcomes
   (`scripts/backtest_sd_calibration.py` + the ROS-haircut TODO). The SD check is
   the gate on the PA-vs-games and variance-scaling approximations.

## Acceptance evidence (before/after) -- REQUIRED

Not shipped on green tests alone. Extend the Phase 0 diagnostic
(`scripts/compare_mc_active_selection.py` / the gated hook) to add the finished
engine as a fourth arm:

- On the **same cached snapshot**, report per-team medians for **all ten
  categories** + overall roto standings under: (1) OLD top-k, (2) bench-exclusion
  (active-slot), (3) **NEW engine** (IL displacement + bench injury-fill +
  churn-freeze), and ERoto, with run conditions in the header.
- The NEW engine must land BETWEEN bench-exclusion (floor) and OLD top-k (ceiling):
  above bench-exclusion (healthy bench earns nonnegative injury-fill credit) and
  below OLD top-k (no full-time seating of healthy bench / no IL double-count). On
  IL-driven cases (SkeleThor) it must track ERoto's displacement (e.g. Springer/
  Bauers scaled, IL bats at full); on healthy-bench cases (Cavalli) the seated
  bats (Perez/Ward/Arraez) contribute a small injury-fill share, not their full
  ~99 RBI.
- Report pitcher categories + overall standings so the hitter-fill/pitcher-no-fill
  asymmetry is bounded and visible.
- Real cached data (Upstash/Render source of truth; never stale local cache).

Definition of done for the integration phase.

## Scope

- In-season ROS path only: `run_ros_monte_carlo` -> `simulate_remaining_season_batch`.
  Draft `simulate_season` and scalar `simulate_remaining_season` untouched.
- Fallback granularity **whole-context, never per-player within a run.** In-season
  uses the new model for ALL teams; legacy top-k only for entirely slot-less
  contexts (draft/preseason, slot-less test dicts).
- Rostered-but-unprojected players (waiver adds, fresh call-ups, no FanGraphs
  line): zero projected production, zero per-game value -> contribute nothing,
  never chosen as fill. If in an ACTIVE slot, the slot's shortfall is the full slot
  games -> pool then replacement (honest estimate with no projection). NOT a
  per-player top-k switch.

## Roster context (this league)

`config/league.yaml roster_slots`: C1, 1B1, 2B1, 3B1, SS1, IF1, OF4, UTIL2 (12
active hitter slots), P9, BN2, IL2. The healthy-bench fill pool is small (BN2
shared across hitters/pitchers), which bounds the per-iteration allocation and
makes same-day collisions rare.

## Testing

- IL displacement (mirror ERoto): an IL hitter is counted at full ROS and the
  worst eligible active body is scaled by the IL player's expected ROS PT; slot PT
  is conserved; matches `_apply_displacement` on the same roster.
- Healthy-bench injury-fill: a healthy bench bat contributes ZERO when its
  position's starters draw full availability, but a NONZERO share when a starter
  draws a low-PT (injured) iteration -- capped at the bench body's own ROS games.
- One-body capacity: two starters draw low in the same iteration and one bench
  body is eligible for both -- its total contributed games do not exceed its ROS
  games.
- Replacement last: bench pool exhausted -> residual to replacement, not an
  over-extended bench body.
- Rate-stat fill: filling with a lower-rate body moves team AVG/ERA/WHIP the
  realistic direction; recombined rate equals the volume-weighted component sum.
- Per-game value ordering: a part-time higher-per-game body is chosen over a
  full-time lower-per-game body for a small same-position shortfall.
- Churn freeze: the active set is identical across iterations (no per-iteration
  re-selection); only stats/PT and the injury-fill vary.
- Unprojected active player: zero contribution; slot filled by pool then
  replacement; never a per-player top-k switch.
- Determinism / tie-break: two equal-per-game-value eligible fill bodies -> a
  SPECIFIC asserted allocation, not mere seed-stability.
- Whole-context fallback: in-season prices every team with the new model; slot-less
  input falls entirely to top-k.
- Pitcher IL displacement + bench-exclusion: an IL pitcher displaces via the pool
  model; a healthy benched pitcher is excluded; matches ERoto.
- Regression: existing MC/integration tests pass; any fixture relying on bench
  seating is flagged and justified, not silently changed.

## Implementation phasing

Phase 0 (attribution gate) is COMPLETE -> GO (commits 71e435c, 3a3ce66, 038848b;
note `docs/superpowers/games-mc-phase0-attribution-2026-06-26.md`). The diagnostic
machinery (`mc_selection.py`, the `active_cols` override, the gated hook) is reused
by later phases and the acceptance artifact. Phases 1-6 each their own plan / PR:

1. Games data plumbing (`g`, `gs`) + serialization/SGP/backward-compat audit.
2. Setup: classification + IL displacement -- reuse `_classify_roster` +
   `_compute_displacement_factors` on the Player rosters in `run_ros_monte_carlo`;
   produce the effective active set (+ factors) and the healthy-bench fill pool
   (+ per-game value, `g_ros`, eligible positions). Build/thread the LeagueContext
   like `ProjectedStandings.from_rosters`.
3. Per-iteration bench injury-fill engine (hitters): shortfall = expected ROS
   games - sampled games; per-game-value ordering; one-body capacity;
   deterministic tie-break; rate-stat component fill; replacement per-game.
4. MC integration (ROS-direct blend; fixed active set + displacement factors +
   injury-fill; pitcher displacement/bench-exclusion), plus the before/after
   artifact (definition of done).
5. Pitcher IL displacement (pool model) + bench-exclusion confirmed coherent with
   the hitter path; bench injury-fill + closer SV deferred and documented.
6. Validation backtest (means AND SDs).

## Open implementation questions (for the plan, not blocking the design)

- Exact vectorization vs. per-team Python loop for the allocation (math-identical).
- `g`/`gs` blend across systems (default: same weights; when a system omits `G`,
  drop it from the `G` blend rather than zeroing).
- The single shared PA-per-game / IP-per-appearance constant used for BOTH the
  missing-`g_ros` derivation and the replacement per-game conversion -- pinned in
  Phase 3; one constant, not two.
- The PT-scale `fraction_remaining` damping reconciliation under ROS-direct (likely
  `fraction_remaining=1.0` for the draw); confirm via SD backtest -- pinned in
  Phase 4.
- Whether to thread the already-built `LeagueContext`/displacement from the
  `_build_projected_standings` step into `_run_ros_monte_carlo` rather than
  recomputing it (cheaper; the pipeline runs standings before the MC).
