# Games-based availability MC -- design

Date: 2026-06-26
Status: approved (design); implementation phased, gated on a Phase 0 diagnostic
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
SkeleThor MC RBI median **~1020 (2nd)** vs ERoto **~926 (4th)** -- a ~94-RBI gap
attributed to bench bats the MC seats and ERoto won't. This re-ranks teams (it
demoted Hart from 1st to 3rd in RBI by inflating an opponent's bench past him).

Diagnosis is not yet airtight. The 1020-vs-926 figure was an eyeball read off the
Distributions view; the exact run conditions (seed, `fraction_remaining`,
iteration count) were not recorded, and there is a competing explanation -- the
per-iteration top-k *re-selection churn* (a fresh best-ball pick every iteration)
rather than bench seating per se. **Phase 0 (below) settles this before any
expensive engine is built**, because the right fix differs: bench seating ->
build the fill engine; re-selection churn -> just freeze selection.

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
design models the correct middle -- but only after Phase 0 confirms bench seating
(not re-selection churn) is the dominant driver.

## Chosen design: games-based availability with capacity-correct fill

### Core idea

Express every player's remaining workload in **player-games** -- the unit with a
clean, structural maximum: a body plays at most one slot per game and at most the
games it is healthy for. Injuries remove games. A vacated slot-game is filled
**bench-first** (position-eligible, ordered by the value rule in Component 3),
**replacement-level last**, with each fill body capped at its own available
games (the one-body capacity constraint).

`remaining_games` is the player's **rest-of-season** games projection, NOT
`full_season_G * fraction_remaining`. `fraction_remaining` is a single global
league scalar; multiplying a full-season `G` by it assumes every player has
uniformly already played `(1 - fraction_remaining)` of their games, which is
false for the exact population this design exists to handle (IL players,
mid-season call-ups, returning starters). The in-season pipeline already loads
ROS projections (`refresh_pipeline._load_projections` swaps `*_proj` to ROS), and
the ROS CSVs carry a ROS-scaled `G` (verified: a full-timer reads ~75 G / ~343 PA
at mid-season). `remaining_games` therefore comes from the ROS projection's games
field (`g_ros`), riding the same ROS path PA/IP already ride. Where a ROS `G` is
absent for a player who still has a ROS PA/IP projection, derive games from those
via a typical per-game rate (PA/game ~4.3; IP per start / per appearance). The
rostered-but-entirely-unprojected case is handled in Scope.

This is an intentional divergence from ERoto. ERoto stays the slot-legal
snapshot view; the MC becomes the realistic-outcomes engine that prices depth.
Bringing ERoto along later is out of scope.

### Time / capacity model

There is no rest-of-season day-by-day calendar available. The only schedule
artifact in the repo is `weekly_schedule.json`, a single-week snapshot
(`games_per_team` ~6-8 for the current week) used by the lineup/matchup code; it
carries neither the ROS horizon nor per-player granularity, and the MC does not
and will not read it. A calendar-aware daily simulation is therefore unsupported
and would be extrapolated regardless.

We use **abstracted available-games**: each player has `remaining_games`
available (ROS games); an availability draw removes a fraction (Component 2);
capacity is enforced as a pooled allocation (a body contributes at most its
available games total across every slot it fills), *without* modeling the literal
calendar. This ignores the rare case of two needs colliding on one eligible body
on the same literal day -- bounded and conservative (it can only under-credit
fill) given the 4-body fill pool (Roster context). No new schedule data is
plumbed.

### Components

1. **Games data plumbing.** Add `g` to `HitterStats` and `g`/`gs` to
   `PitcherStats`; thread from the projection CSVs (`G`, `GS` already present in
   the FanGraphs exports) through the blend into the dataclasses. PA/IP stay as
   they are. Foundational -- every later component needs it.

   Audit required (per the repo's grep-every-call-site rule). Both dataclasses
   build kwargs from `{f.name for f in fields(cls)}` in `from_dict`/`to_dict`, so
   adding fields changes serialization. The phase verifies:
   - The new fields MUST NOT enter the SGP computation. (`calculate_player_sgp`
     reads only explicitly named fields, so this holds by construction -- the
     audit *confirms* it rather than assuming it.)
   - Forward round-trips stay stable.
   - BACKWARD compat: any already-persisted JSON (`draft_state*.json`, dashboard
     state) written before this change lacks the `G` key, so a round-trip
     materializes `g=0`/`gs=0` via `from_dict`'s `or 0`. No consumer may trust
     `g`/`gs` read off pre-change persisted state as a real value -- this is the
     classic falsy-zero footgun (CLAUDE.md). The audit checks no such consumer
     exists, or gates `g`/`gs` use behind a presence check.

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
   For the four pure counting categories the diff is surgically the *destination
   of `frac_missed`*. For the rate categories it is more than that -- see the
   variance note in Component 3.

   KNOWN APPROXIMATION (load-bearing -- do not gloss): `pt_scale_draw` is
   calibrated on *actual / projected PA (or IP)*, which conflates (a) missed
   *games* (injury, demotion, late call-up), where a bench body genuinely starts,
   with (b) reduced *PA-within-games* (batting lower, platoon, pinch-hit), where
   the starter still plays and no slot opens. Treating `(1 - pt_scale_draw)` as
   vacated slot-games credits bench fill for some volume that was really (b),
   risking re-inflation of bench credit. Mitigations: (i) for hitters the dominant
   driver of actual/projected PA variance is games played, not PA-per-game (stable
   ~4.1-4.4), so the (b) term is small; (ii) the bench body contributes at its own
   (lower) rate, capped at one body, so over-triggered fill is bounded; (iii) the
   before/after evidence and validation backtest are the gate -- visible failure,
   not silent. A stint-based games-missed model separating (a) from (b) is the
   principled refinement, explicitly deferred.

3. **Slot-assignment / fill engine (hitters).** Per team, per iteration, on the
   sampled stats:

   - Each active-slot starter contributes its available games to its slot at its
     own sampled per-game stat line.
   - For each slot, the shortfall `(slot_games - starter_available_games)` is
     offered to the fill pool: the team's bench + IL bodies (each with its own
     sampled available games), restricted to those position-eligible for the
     slot.
   - **Value rule (single, explicit).** Process open shortfalls in descending
     `shortfall_games`; for each, choose the eligible fill body with the highest
     **ROS SGP** that still holds available games. ROS SGP is `calculate_player_sgp`
     on the body's `rest_of_season` stats -- the repo's standard cross-category
     value unit, already used this way by `roster_audit` and `il_return_planner`.
     We use the season/ROS-horizon SGP directly; we do NOT divide it by games
     ("per-game SGP" is incoherent because SGP's rate terms are normalized against
     fixed full-season team constants `team_ab=5500`/`team_ip=1450`, which do not
     track a player's `g_ros`, so the division would distort the ordering). Ties
     break deterministically by (higher SGP, then player-id ascending) so the
     allocation is reproducible independent of dict iteration order. Because the
     fill pool is tiny (<=4 bodies) and usually eligibility-bound, this ordering
     rarely changes a team total; the before/after artifact surfaces any
     sensitivity. ROS SGP is computed once per body at MC setup (where Player
     objects are in hand) and carried into the per-iteration allocation -- it is
     NOT assumed to be pre-populated on the flat dicts the batch consumes
     (`to_dict` emits `sgp` only when set).
   - A chosen body's available-games pool is decremented by the games it covers,
     so one bench bat cannot cover two simultaneous injuries beyond one body's
     worth (the one-body capacity constraint).
   - Residual games still uncovered after the bench/IL pool is exhausted fall to a
     replacement-level line (see Rate-stat handling).

   Allocation quality: at most ~4 fill bodies vs 12 hitter slots -- a tiny
   assignment. The greedy is near-optimal at this size with residual error
   averaged over 1000 iterations; an exact small assignment is an acceptable
   alternative if a pathological instance is found.

   **Rate-stat handling (AVG/ERA/WHIP) -- explicit.** The three rate categories
   are NOT filled separately. Every contributing body (starter, bench-fill,
   replacement residual) adds its recovered counting *components* -- hitters `h`,
   `ab`; pitchers `er`, `ip`, `bb`, `h_allowed` -- scaled to the games it
   covered, into the team component sums; AVG/ERA/WHIP are recombined from team
   totals exactly as today (`total_h/total_ab`, etc., `simulation.py:781-790`). A
   fill body contributes its *own* rate components by volume; a benched body is in
   practice weaker than the starter it replaces, so its lower rate correctly drags
   the team rate the realistic direction. The replacement residual is expressed
   per-game by dividing the existing `_replacement_line` (`simulation.py:435-461`,
   a per-stat counting bundle with NO games field of its own) by a games
   denominator derived from the SAME PA-per-game / IP-per-appearance heuristic
   used for the missing-`g_ros` fallback (Core idea). This is an honestly *derived*
   denominator applied consistently, not a value recoverable from the
   `REPLACEMENT_BY_POSITION` calibration (which has no games basis).

   **Variance note (acknowledged approximation).** The NegBin copula samples each
   body at its full projected volume; the fill engine then scales those sampled
   counts to the games covered. Scaling a draw by fraction `f` yields variance
   `f^2 * var`, whereas a body genuinely playing `f` of its games has variance
   ~`f * var` -- so component-scaling *understates* the variance of a partial-games
   fill contribution. This is a real but bounded second-order effect: it applies
   only to the (usually small) fill portion of a team total, and it is still
   strictly more realistic than today's replacement fill, which is *deterministic*
   (`repl_line * frac_missed`, zero variance). The validation backtest checks
   category SDs precisely to catch deflation; correct re-sampling of fill bodies
   at their partial (games-covered) volume is the principled refinement, deferred.

4. **MC integration.** In `simulate_remaining_season_batch`, replace the
   per-iteration top-k + replacement-only fill: sample availability + counting
   stats (vectorized as today via the NegBin copula), then run the fill
   allocation and sum. Hitters use the full fill engine (Component 3); pitchers
   use slot-based bench-exclusion (Component 5). The heavy NegBin sampling stays
   vectorized; the light per-team / per-iteration allocation may be a Python loop
   (10 teams x 1000 iters of a ~12-slot greedy is cheap). Exact vectorization is
   a plan detail, not a design constraint.

   Banked-YTD floor. The per-component floor `total_x = actual_x + max(0, sim_x -
   actual_x)` (== `max(actual, sim)`, `simulation.py:781-805`) stays. It is inert
   for counting categories in the normal case (simulated remaining is
   non-negative, so `sim = actual_YTD + sim_remaining >= actual`); it can bind
   only for an over-performing team/category where the ROS projection lags
   realized pace. Decision rule for the evidence (not just reporting): if the
   floor binds on a re-ranked team's affected category, the fill change is
   mathematically invisible there and the acceptance comparison is **inconclusive**
   for that category -- report it as such and demonstrate the fill effect on a
   team/category where the floor is provably inert, rather than claiming success.

5. **Pitchers (v1 = slot-based bench-exclusion, no rich fill).** Hitters get the
   full games-based fill model (where the measured bug is). Pitchers get the
   *classification* half only: select active-slot pitchers at face value, exclude
   healthy bench pitchers, keep the existing IP-based PT scale and replacement-
   level injury fill for that active set. We do NOT add a `gs`/`g` games scaler
   (it would double-discount volume already scaled by the IP-calibrated PT curve
   (`playing_time.py:29`), and is a no-op for ERA/WHIP since a common factor
   cancels in num/denom), and we do NOT build the bench-fill insurance pool for
   pitchers (closer-role SV modeling -- lose the job, SV -> 0 not pro-rata -- and
   pitcher rate fill are deferred). Rationale for doing bench-exclusion rather
   than leaving pitchers fully unchanged: if hitters lose their bench over-credit
   but pitchers keep theirs (deep bullpen seated by top-k), the 5x5 roto totals
   tilt toward pitching-deep teams -- a NEW, uncompensated standings bias.
   Bench-exclusion removes the gross pitcher over-credit symmetric to the hitter
   bug, keeping standings coherent; only the richer pitcher injury-insurance fill
   is deferred. The `g`/`gs` plumbing from Component 1 still lands for that
   follow-on.

6. **Validation + before/after evidence.** A validation phase producing concrete
   evidence the bug is fixed (Acceptance evidence below), plus a backtest checking
   the new model's category means AND SDs against realized outcomes
   (`scripts/backtest_sd_calibration.py` + the ROS playing-time-haircut TODO). The
   SD check is the explicit gate on both the Component 2 (PA-vs-games) and
   Component 3 (variance-scaling) approximations. It does not re-settle the
   haircut-vs-reality or SV-variance questions (separate TODOs).

### Acceptance evidence (before/after) -- REQUIRED

The change does not ship on green tests alone. It must produce a reproducible
before/after artifact:

- A committed script (e.g. `scripts/compare_mc_active_selection.py`) that runs the
  in-season ROS MC on the **same cached league snapshot** under (1) OLD top-k
  (position-blind, replacement-only fill), (2) NEW games-based bench fill, and
  (3) a pure bench-exclusion variant (ERoto-style, no bench fill). It prints a
  per-team table of median totals for **all ten categories** plus overall roto
  standings for all three variants, side by side with ERoto, and records run
  conditions (seed, `fraction_remaining`, iteration count) in the header.
- Attribution: the three-way comparison shows where the ~94-RBI gap comes from --
  NEW between bench-exclusion floor and old top-k => bench seating (diagnosis
  holds); bench-exclusion alone closes most of it => re-selection churn (and the
  framing is corrected). This is the same diagnostic as Phase 0, re-run with the
  finished engine as the third arm.
- Pitcher-side check: because pitcher depth is handled by bench-exclusion (not
  fill), the artifact must report pitcher categories and overall standings so the
  hitter-fill/pitcher-no-fill asymmetry is bounded and visible, not shipped
  undetected.
- Reproduce-and-close: SkeleThor RBI median lands materially below 1020 (between
  the ~926 floor and 1020) and the Hart RBI re-rank (1st -> 3rd) no longer occurs.
  For affected teams, report whether the banked-YTD floor binds on those
  categories (Component 4); if it binds, treat that category as inconclusive and
  demonstrate the effect elsewhere.
- Driven from real cached data (Upstash/Render source of truth; never stale local
  cache), per repo convention.

This artifact is the definition of done for the integration phase. "Tests pass"
is necessary but not sufficient.

### Scope

- In-season ROS path only: `run_ros_monte_carlo` -> `simulate_remaining_season_batch`.
  Draft `simulate_season` and scalar `simulate_remaining_season` untouched.
- Fallback granularity is **whole-context, never per-player within a run.** The
  in-season ROS path uses the new model for ALL teams; the legacy top-k is
  retained only for entirely slot-less contexts (draft/preseason, slot-less test
  dicts). Pricing some teams/players with the fill model and others with top-k
  inside one standings computation is a correctness hazard, so the model choice is
  made once per run.
- Rostered-but-unprojected players (waiver adds, fresh call-ups with no FanGraphs
  ROS line) are explicit: such a player has no ROS stats and no PA/IP to derive
  `g_ros` from, so it enters the engine with zero projected production and zero
  ROS SGP -- it contributes nothing and is never chosen as fill (correct for a
  true unprojected scrub). If such a player is in an ACTIVE slot, its slot's
  shortfall is the full slot games, filled by the eligible bench/IL pool then
  replacement-level -- the honest estimate when we have no projection. This is
  NOT a per-player switch to top-k (which the whole-context rule forbids); it is
  the fill engine operating on a zero-projection body.
- Side effect to VERIFY (hypothesis, not promised): removing per-iteration top-k
  re-selection churn *may* shrink some distribution width (related TODO). Phase 0
  and the before/after artifact check this; it is not an assumed result.

### Roster context (this league)

`config/league.yaml roster_slots`: C1, 1B1, 2B1, 3B1, SS1, IF1, OF4, UTIL2 (12
active hitter slots), P9, BN2, IL2. Total roster 25. Beyond the 21 active slots
the extra bodies are at most 4 (2 BN + 2 IL); IL bodies are by definition
low-availability, so the *effective* healthy fill pool is usually smaller than 4.
This bounds the assignment problem and makes same-day collisions rare.

## Testing

- Bench-exclusion-but-fill: a healthy bench bat that out-rates an active starter
  contributes *zero* while the starter is healthy, but *does* contribute when the
  starter's availability draw is low -- capped at the bench bat's own available
  games.
- One-body capacity: two eligible starters both draw low availability and a single
  bench body is eligible for both -- that body's total contributed games do not
  exceed its available games.
- Replacement-last: with the bench pool exhausted, residual shortfall falls to the
  replacement line, not to an over-extended bench body.
- Rate-stat fill: filling a starter's missed games with a lower-rate body moves
  team AVG/ERA/WHIP the realistic direction; a constructed case asserts the
  recombined team rate equals the volume-weighted component sum (no separate-rate
  shortcut, no sign flip).
- Unprojected active player: a rostered active-slot player with no ROS projection
  contributes zero and its slot is filled by pool then replacement -- never a
  per-player top-k switch.
- IL: an IL player is in the fill/active pool per its (low) games availability;
  not stacked on a full active lineup.
- Determinism / tie-break: a constructed instance with two equal-value eligible
  fill bodies asserts a SPECIFIC allocation (higher SGP, then player-id
  ascending), not merely seed-stability.
- Whole-context fallback: an in-season run prices every team with the new model; a
  slot-less input falls entirely to top-k -- never mixed within one run.
- Pitcher bench-exclusion: a benched pitcher is excluded from the active set
  (matching slot-based classification), not seated by raw `w+k+sv`.
- Regression: existing MC/integration tests still pass. Any fixture relying on
  bench bats being seated at full season is flagged and justified, not silently
  changed (no-modifying-failing-tests rule).

## Implementation phasing

Phase 0 is a GATE; Phases 1-6 each their own plan / PR.

0. **Attribution diagnostic (gate, cheap).** On a real cached snapshot, run the
   OLD top-k MC vs a bench-exclusion variant (a selection tweak only -- NO fill
   engine, NO games plumbing) and record per-category medians + run conditions.
   Confirm bench-exclusion closes a large share of the ~94-RBI SkeleThor gap (=>
   bench seating is the driver, proceed to build the fill engine). If instead the
   gap persists under bench-exclusion (=> re-selection churn dominates), STOP and
   revisit: the cheap fix is to freeze selection, not build the engine. This
   inverts the build risk -- validate the premise before the expensive work.
1. Games data plumbing (`g`, `gs` into dataclasses + blend) with the
   serialization/SGP/backward-compat audit.
2. Per-player availability draw in games units (ROS games; reuse PT scale;
   redirect `frac_missed`).
3. Fill engine (hitters): capacity-correct allocation, ROS-SGP value rule,
   deterministic tie-break, rate-stat component fill, replacement per-game
   conversion.
4. MC integration into `simulate_remaining_season_batch` (hitter fill + pitcher
   bench-exclusion), plus the before/after comparison artifact (definition of
   done for this phase).
5. Pitcher bench-exclusion confirmed coherent with the hitter path (no
   double-scaling); document the temporary depth-pricing asymmetry.
6. Validation backtest (means AND SDs).

## Open implementation questions (for the plan, not blocking the design)

- Exact vectorization vs. per-team Python loop for the fill allocation
  (math-identical either way; loop cost already bounded as cheap).
- How `g`/`gs` blend across projection systems (default: same weights as the rest
  of the blend; when a system omits `G`, drop it from the `G` blend rather than
  zeroing it).
- The exact PA-per-game / IP-per-appearance constants used for the derived games
  denominator (both for missing-`g_ros` and the replacement per-game conversion)
  -- pinned in Phase 3; they must be one shared constant, not two divergent ones.
