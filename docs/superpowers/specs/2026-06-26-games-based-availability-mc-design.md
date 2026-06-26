# Games-based availability MC -- design

Date: 2026-06-26
Status: approved (design); implementation phased
Branch: mc-games-based-availability

## Problem

The in-season ROS Monte Carlo (`simulate_remaining_season_batch` in
`simulation.py`) picks each team's active roster **position-blind and
slot-blind**: it takes the top `h_slots` hitters by raw `R+HR+RBI+SB` and the
top `p_slots` pitchers by `closer-bonus + W+K+SV`, re-selected every iteration,
with no eligibility or bench check (`simulation.py:749-765`). ERoto's projected
standings instead select slot/position-aware via `_classify_roster` /
`_apply_displacement` (`scoring.py`): active-slot players at face value, IL
players displace the worst eligible active match, healthy bench excluded.

The MC's blindness over-credits deep-but-blocked rosters. Measured 2026-06-26:
SkeleThor MC RBI median **1020 (2nd)** vs ERoto **926 (4th)** -- a ~94-RBI gap
attributed to bench bats the MC seats and ERoto won't. This re-ranks teams (it
demoted Hart from 1st to 3rd in RBI by inflating an opponent's bench past him).

Note on the measurement: the 1020-vs-926 figure was an eyeball read off the
Distributions view; the exact run conditions (seed, `fraction_remaining`,
iteration count) and the bench-vs-re-selection attribution were not recorded
rigorously. The "Acceptance evidence" section below makes reproducing AND
attributing this gap a required deliverable, precisely because the original
diagnosis is not yet airtight (a separate suspect is the per-iteration
re-selection churn, not bench seating per se).

## Why not the obvious fix (mirror ERoto)

The first design considered was to mirror ERoto exactly: bench-exclusion +
slot-aware classification + IL displacement, fixed on projected means. That
converges the two engines but encodes a model both engines get wrong on one
specific dimension: **roster depth as injury insurance.**

- Today's top-k MC: depth is fully stacked -- it seats SS2's whole season on
  top of SS1. Massive over-credit.
- ERoto / bench-exclusion: depth is worth zero -- SS2 never plays; an injury to
  SS1 is filled by a replacement-level (waiver) line.
- Reality: depth is worth its injury-insurance value -- when SS1 misses time you
  start your benched, eligible SS2 (at SS2's level), capped at one body's worth
  of games. Replacement-level is the *last* resort, only when no eligible bench
  body is free.

Both the current MC and ERoto are wrong, in opposite directions. The chosen
design models the correct middle.

## Chosen design: games-based availability with capacity-correct fill

### Core idea

Express every player's remaining workload in **player-games** -- the unit with a
clean, structural maximum: a body plays at most one slot per game and at most the
games it is healthy for. Injuries remove games. A vacated slot-game is filled
**bench-first** (position-eligible, ordered by the value rule defined in
Component 3), **replacement-level last**, with each fill body capped at its own
available games (the one-body capacity constraint).

`remaining_games` is the player's **rest-of-season** games projection, NOT
`full_season_G * fraction_remaining`. `fraction_remaining` is a single global
league scalar; multiplying a full-season `G` by it assumes every player has
already played `(1 - fraction_remaining)` of their games uniformly, which is
false for the exact population this design exists to handle -- a player currently
on the IL, a mid-season call-up, a returning-from-injury starter. The MC already
distinguishes ROS from full-season (`to_flat_dict_full_season`,
`_flatten_full_season`), and the in-season pipeline already loads ROS
projections (`refresh_pipeline._load_projections` swaps `*_proj` to ROS).
`remaining_games` therefore comes from the ROS projection's games field
(`g_ros`), consistent with how the rest of the ROS blend is derived. Where a ROS
games projection is unavailable, derive it from ROS PA/IP via a typical
per-game rate (PA/game ~4.3; IP per start / per appearance) -- see the fallback
rule in Scope.

This is an intentional divergence from ERoto. ERoto stays the slot-legal
snapshot view; the MC becomes the realistic-outcomes engine that prices depth.
Bringing ERoto along later is out of scope.

### Time / capacity model

There is no rest-of-season day-by-day calendar available. The only schedule
artifact in the repo is `weekly_schedule.json`, a single-week snapshot
(`games_per_team` ~6-8 for the current week) used by the lineup/matchup code; it
carries neither the ROS horizon nor per-player granularity, and the MC does not
and will not read it. So a calendar-aware daily simulation is not supported and
would be extrapolated regardless.

We therefore use **abstracted available-games**: each player has
`remaining_games` available (ROS games, per Core idea); an availability draw
removes a fraction (Component 2); capacity is enforced as a pooled allocation (a
body contributes at most its available games total across every slot it fills),
*without* modeling the literal calendar. This ignores the rare case of two needs
colliding on one eligible body on the same literal day -- bounded and
conservative given the 4-body fill pool (see Roster context), and far more
tractable. No new schedule data is plumbed.

### Components

1. **Games data plumbing.** Add `g` to `HitterStats` and `g`/`gs` to
   `PitcherStats`; thread from the projection CSVs (`G`, `GS` already present in
   the FanGraphs exports) through the blend into the dataclasses. PA/IP stay as
   they are. Foundational -- every later component needs it.

   Audit required (per the repo's grep-every-call-site rule): both dataclasses
   build kwargs from `{f.name for f in fields(cls)}` in `from_dict`/`to_dict`,
   so adding fields changes serialization round-trips and any persisted JSON
   (`draft_state*.json`, dashboard state). The new fields MUST NOT enter the SGP
   computation (`calculate_player_sgp`) or the recovered-component rate math.
   The phase explicitly verifies: SGP unchanged, serialization round-trips
   stable, no `fields()`-driven consumer silently picking up `g`/`gs`.

2. **Per-player availability draw (reuse the PT scale, with an explicit
   approximation caveat).** Reuse the existing, calibrated playing-time scale as
   the per-player available-games fraction:

       available_games = remaining_games * pt_scale_draw

   What actually changes vs. today: in the current code each player's stats are
   scaled by `scales` and the missed fraction `frac_missed = max(0, 1 - scales)`
   is **already backfilled at replacement level** (`repl_contrib = repl_line *
   frac_missed`, `simulation.py:701-710`). The missed mass does not "vanish" --
   it is filled by a waiver-quality line. This design **redirects that same
   missed fraction** to bench-fill first, replacement-level only as the residual.
   So the diff is surgically the *destination of `frac_missed`*, not a new
   sampling distribution.

   KNOWN APPROXIMATION (load-bearing -- do not gloss): `pt_scale_draw` is
   calibrated on *actual / projected PA (or IP)*, which conflates two distinct
   causes of lost volume -- (a) missed *games* (injury, demotion, late call-up),
   where a bench body genuinely starts, and (b) reduced *PA-within-games*
   (batting lower, platoon split, pinch-hit), where the starter still plays and
   no slot opens. Treating `(1 - pt_scale_draw)` as vacated slot-games credits
   bench fill for some volume that was really (b), which risks re-inflating the
   very bench credit this design suppresses. Mitigations: (i) for hitters the
   dominant driver of actual/projected PA variance is games played, not
   PA-per-game (which is relatively stable ~4.1-4.4), so the approximation leans
   small; (ii) the bench-fill body contributes at *its own* (typically lower)
   rate, capped at one body, so even over-triggered fill is bounded by bench
   quality and capacity; (iii) the before/after evidence and the validation
   backtest are the gate -- if bench fill re-inflates category means toward the
   old top-k numbers, that is a visible failure, not a silent one. A stint-based
   games-missed model (discrete IL blocks, separating (a) from (b)) is the
   principled refinement and is explicitly deferred to a follow-on.

3. **Slot-assignment / fill engine (hitters).** Per team, per iteration, on the
   sampled stats:

   - Each active-slot starter contributes its available games to its slot at its
     own sampled per-game stat line.
   - For each slot, the shortfall `(slot_games - starter_available_games)` is
     offered to the fill pool. The fill pool is the team's bench + IL bodies
     (each with its own sampled available games), restricted to those
     position-eligible for the slot.
   - **Value rule (single, explicit).** Process the open shortfalls in
     descending order of *value at risk* = `shortfall_games *
     candidate_marginal_value`, and for each shortfall choose the
     highest-value eligible fill body still holding available games. "Value" of
     a body is its **projected per-game roto contribution** (its ROS SGP per
     game; SGP is already computed on every Player and is the repo's standard
     cross-category value unit -- reuse it rather than the MC's raw counting-sum
     or ERoto's playing-time order). Ties are broken deterministically by
     (higher SGP, then `yahoo_id`/player-id ascending) so the allocation is
     reproducible independent of dict iteration order.
   - A chosen body's available-games pool is decremented by the games it covers,
     so one bench bat cannot cover two simultaneous injuries beyond one body's
     worth (the one-body capacity constraint).
   - Anything still uncovered after the bench/IL pool is exhausted falls to a
     replacement-level line for the residual games (see Rate-stat handling for
     how the replacement line is expressed per-game).

   Allocation quality: with at most ~4 fill bodies against 12 hitter slots this
   is a tiny assignment problem. The greedy above is near-optimal at this size
   and its residual error is averaged over 1000 iterations; an exact small
   assignment is an acceptable alternative if a pathological instance is found.
   This is a deliberate, bounded approximation, not an unexamined one.

   **Rate-stat handling (AVG/ERA/WHIP) -- explicit.** The three rate categories
   are NOT filled separately. Every contributing body (starter, bench-fill, and
   the replacement residual) adds its recovered counting *components* -- hitters:
   `h`, `ab`; pitchers: `er`, `ip`, `bb`, `h_allowed` -- scaled to the games it
   actually covered, into the team component sums. AVG/ERA/WHIP are then
   recombined from the team totals exactly as today (`total_h/total_ab`, etc.,
   `simulation.py:781-790`). A fill body contributes its *own* rate components by
   volume; because a benched body is in practice weaker than the starter it
   replaces, its lower rate correctly drags the team rate the realistic
   direction when it fills an injury. The replacement-level residual is expressed
   as a per-game component line: divide the existing `_replacement_line`
   (`simulation.py:435-461`, a per-stat counting bundle) by its implied games to
   get a per-game rate, then multiply by the residual games -- the replacement
   line's games denominator is pinned in this phase, not invented at
   implementation time (it must match how `REPLACEMENT_BY_POSITION` was
   calibrated).

4. **MC integration.** Replace the per-iteration top-k selection + replacement-
   only fill in `simulate_remaining_season_batch` (hitters) with: sample
   availability + counting stats (vectorized as today via the NegBin copula),
   then run the fill allocation and sum. The heavy NegBin sampling stays
   vectorized; the light per-team / per-iteration allocation may be a Python loop
   (10 teams x 1000 iters of a ~12-slot greedy is cheap). Exact vectorization of
   the allocation is an implementation detail for the plan, not a design
   constraint.

   The team-level per-component floor `total_x = actual_x + max(0, sim_x -
   actual_x)` (== `max(actual, sim)`, `simulation.py:781-805`) stays as the
   banked-YTD backstop. Interaction note: because `sim` is `actual_YTD +
   simulated_remaining` and simulated remaining counting production is
   non-negative, the floor is inert for counting categories in the normal case;
   it can bind only for an over-performing team/category where the ROS
   projection lags realized pace (per-component clamp). The before/after evidence
   MUST report, for the teams whose ranking changed, whether the floor binds on
   the affected categories -- so we confirm the fill change actually flows
   through and is not silently masked by the floor.

5. **Pitchers (v1 = unchanged this round).** Hitters get the full games-based
   fill model (that is where the measured bug is). For pitchers, v1 leaves the
   current path **unchanged** -- the existing IP-based PT scale and top-k. We do
   NOT add a `gs`/`g`-based games scaling on top, because: (a) it would
   double-discount volume already scaled by the IP-calibrated PT curve
   (`_curve_key` is IP-based, `playing_time.py:29`); (b) scaling `er`/`ip`/`bb`/
   `h_allowed` by a common games factor leaves ERA/WHIP unchanged (numerator and
   denominator scale together), so it would be a no-op for two of the four
   pitcher categories; and (c) SV depends on the closer *role* (lose the job ->
   SV goes to zero, not pro-rata), which a games scaler does not model. A proper
   pitcher games-fill -- generic-slot fill, closer-role modeling for SV, and
   ERA/WHIP component fill -- is a deliberate follow-on. Consequence acknowledged:
   for this round the engine prices hitter depth but not pitcher depth; the
   asymmetry is temporary and explicit, and the `g`/`gs` plumbing from Component
   1 still lands so the follow-on has its inputs.

6. **Validation + before/after evidence.** A validation phase that produces
   concrete evidence the bug is fixed (see "Acceptance evidence" below), plus a
   backtest checking the new model's category means and SDs against realized
   outcomes, tying into `scripts/backtest_sd_calibration.py` and the ROS
   playing-time-haircut TODO. This phase confirms the fill model does not drift
   category means (the key check on the Component 2 approximation); it does not
   attempt to re-settle the haircut-vs-reality or SV-variance questions (separate
   TODOs).

### Acceptance evidence (before/after) -- REQUIRED

The change does not ship on green tests alone. It must produce a reproducible
before/after artifact demonstrating the over-crediting is fixed:

- A small, committed script (e.g. `scripts/compare_mc_active_selection.py`) that
  runs the in-season ROS MC on the **same cached league snapshot** under (1) the
  OLD selection (top-k, position-blind, replacement-only fill), (2) the NEW
  games-based bench fill, and -- for attribution -- (3) a pure bench-exclusion
  variant (ERoto-style, no bench fill). It prints a per-team, per-category table
  of median totals for all three, side by side with the ERoto projected-standings
  figures, and records the run conditions (seed, `fraction_remaining`, iteration
  count) in the output header.
- Attribution requirement: the three-way comparison must show *where the old
  94-RBI gap comes from* -- if the NEW (bench-fill) number sits between the
  bench-exclusion floor and the old top-k, the gap was bench seating (the
  diagnosis holds); if bench-exclusion already closes most of it, the gap was
  re-selection churn and the design's framing is corrected in the note. Either
  way the artifact settles the attribution that the original eyeball read did
  not.
- The artifact must reproduce the originally-measured regression and show it
  closing: SkeleThor RBI median was **~1020 (2nd)** under the old MC vs ERoto's
  **~926 (4th)**; the new model's SkeleThor RBI median must land materially below
  1020 (between the slot-legal ~926 floor and 1020, reflecting only the
  legitimate bench-fill insurance), and the Hart RBI re-ranking (1st -> 3rd
  caused by an inflated opponent bench) must no longer occur. For the affected
  teams, report whether the banked-YTD floor binds on those categories (per
  Component 4).
- The before/after must be driven from real cached data (Upstash/Render is the
  source of truth; never stale local cache) so the evidence reflects production
  inputs, per repo convention.

This artifact is the definition of done for the integration phase, not an
optional extra. "Tests pass" is necessary but not sufficient.

### Scope

- In-season ROS path only: `run_ros_monte_carlo` -> `simulate_remaining_season_batch`.
- The draft `simulate_season` and scalar `simulate_remaining_season` are
  untouched (no Yahoo slots preseason).
- Fallback granularity is **whole-context, never per-player within a run.** The
  in-season ROS path (where every rostered player has a Yahoo slot and a ROS
  projection) uses the new games-based model for ALL teams. The legacy top-k
  path is retained only for entirely slot-less contexts (draft/preseason, tests
  that pass slot-less dicts). It is a correctness hazard to price some teams or
  players with the fill model and others with top-k inside one standings
  computation, so the model choice is made once per run, not per player. An
  individual in-season player missing a `G` projection gets a *derived* `G`
  (from ROS PA/IP, per Core idea), never a per-player switch to top-k.
- Side effect to VERIFY (hypothesis, not a promised outcome): fixing on a
  games-available allocation removes the per-iteration top-k re-selection churn,
  which *may* shrink some distribution width (related TODO: "decompose why one
  team's MC distribution is wider"). This is stated as something the before/after
  artifact and backtest should check, not an assumed result -- the width
  mechanism is admittedly not yet understood.

### Roster context (this league)

`config/league.yaml roster_slots`: C1, 1B1, 2B1, 3B1, SS1, IF1, OF4, UTIL2 (12
active hitter slots), P9, BN2, IL2. Total roster 25. Beyond the 21 active slots
the extra bodies are at most 4 (2 BN + 2 IL); the IL bodies are by definition
low-availability, so the *effective* healthy fill pool is usually smaller than 4.
This bounds the assignment problem and makes same-day collisions rare.

## Testing

- Bench-exclusion-but-fill: a healthy bench bat that out-rates an active starter
  contributes *zero* while the starter is healthy, but *does* contribute when
  the starter's availability draw is low -- capped at the bench bat's own
  available games.
- One-body capacity: when two eligible starters both draw low availability and a
  single bench body is eligible for both, that body's total contributed games do
  not exceed its available games (no double-count).
- Replacement-last: with the bench pool exhausted, residual shortfall falls to
  the replacement line, not to an over-extended bench body.
- Rate-stat fill: filling a starter's missed games with a lower-rate bench/
  replacement body moves team AVG/ERA/WHIP the realistic direction; a constructed
  case asserts the recombined team rate matches the volume-weighted component sum
  (no separate-rate shortcut, no sign flip).
- IL: an IL player is part of the fill/active pool per its (low) games
  availability; not stacked on top of a full active lineup.
- Determinism / tie-break: a constructed instance with two equal-value eligible
  fill bodies asserts a SPECIFIC allocation (the documented tie-break: higher
  SGP, then player-id ascending), not merely that a fixed seed reproduces a run.
- Whole-context fallback: an in-season run prices every team with the new model;
  a slot-less (draft-style) input falls entirely to top-k -- never mixed within
  one run.
- Regression: existing MC/integration tests still pass. Any fixture that relied
  on bench bats being seated at full season gets flagged and justified, not
  silently changed (per the no-modifying-failing-tests rule).

## Implementation phasing

Each phase is its own plan / PR:

1. Games data plumbing (`g`, `gs` into the dataclasses + blend), with the
   serialization/SGP audit from Component 1.
2. Per-player availability draw in games units (ROS games; reuse PT scale;
   redirect `frac_missed` destination).
3. Fill engine (hitters): capacity-correct allocation with the explicit value
   rule, deterministic tie-break, and rate-stat component fill.
4. MC integration into `simulate_remaining_season_batch`, plus the before/after
   comparison artifact (the "Acceptance evidence" deliverable -- definition of
   done for this phase).
5. Pitchers: confirm the path is left unchanged and the hitter/pitcher split is
   coherent (no double-scaling); document the temporary depth-pricing asymmetry.
6. Validation backtest.

## Open implementation questions (for the plan, not blocking the design)

- Exact vectorization vs. per-team Python loop for the fill allocation
  (math-identical either way; loop cost already bounded as cheap).
- How `g`/`gs` blend across projection systems (default: same weights as the
  rest of the blend; fallback when a system omits `G`: drop that system from the
  `G` blend, do not zero it).
- The exact games denominator used to convert `_replacement_line` to a per-game
  rate (must match the `REPLACEMENT_BY_POSITION` calibration basis) -- pinned in
  Phase 3, listed here only as the open numeric detail.
