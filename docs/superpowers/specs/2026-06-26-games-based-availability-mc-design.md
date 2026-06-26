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
almost entirely from bench bats the MC seats and ERoto won't. This re-ranks
teams (it demoted Hart from 1st to 3rd in RBI by inflating an opponent's bench
past him) and the per-iteration re-selection is itself a spurious variance
source.

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

Express every player's remaining workload in **player-games**:

    remaining_games = projected_G * fraction_remaining

This is the unit with a clean, structural maximum -- a body plays at most one
slot per game and at most the games it is healthy for. Injuries remove games. A
vacated slot-game is filled **bench-first** (position-eligible, value-ordered),
**replacement-level last**, with each fill body capped at its own available
games (the one-body capacity constraint).

This is an intentional divergence from ERoto. ERoto stays the slot-legal
snapshot view; the MC becomes the realistic-outcomes engine that prices depth.
Bringing ERoto along later is out of scope.

### Time / capacity model

We have a *weekly* schedule (`weekly_schedule.json: games_per_team`), not a
day-by-day rest-of-season calendar. So a fully calendar-aware daily simulation
is not supported and would be extrapolated regardless. We use **abstracted
available-games**: each player has `remaining_games` available; an availability
draw removes a fraction; capacity is enforced as a pooled allocation (a body
contributes at most its available games total across every slot it fills),
*without* modeling the literal calendar. This ignores rare same-day collisions
between two needs on one eligible body -- acceptable, and far more tractable.

### Components

1. **Games data plumbing.** Add `g` to `HitterStats` and `g`/`gs` to
   `PitcherStats`; thread from the projection CSVs (`G`, `GS` already present in
   the FanGraphs exports) through the blend into the dataclasses. PA/IP stay as
   they are. Foundational -- every later component needs it.

2. **Per-player availability draw (reuse PT calibration).** Reuse the existing,
   already-validated playing-time scale as the per-player available-games
   fraction:

       available_games = remaining_games * pt_scale_draw

   The innovation is not a new injury distribution -- it is what happens to the
   missed `(1 - pt_scale_draw)` games. Today they vanish into a replacement
   line; here they flow to bench-fill first. This isolates the change to the
   fill mechanic and keeps continuity with the calibration we trust. A
   stint-based injury model (discrete IL blocks) is a later refinement, not part
   of this build.

3. **Slot-assignment / fill engine (hitters).** Per team, per iteration:
   - Each active starter contributes its available games to its slot.
   - The shortfall `(slot_games - starter_available)` is filled by eligible
     bench bodies in value order, drawing down each body's *shared* available-
     games pool so one bench bat cannot cover two injuries beyond one body.
   - Anything still uncovered falls to a replacement-level line.
   - Multi-position eligibility (UTIL, IF, a 2B/SS body) makes this a small
     flow; with ~4 fill candidates and 12 hitter slots a greedy waterfall
     (fill the highest value-lost shortfall first, decrement the chosen body's
     pool) is simple and near-optimal. This is where the one-body maximum lives.

4. **MC integration.** Replace the per-iteration top-k selection + PT-multiplier
   in `simulate_remaining_season_batch` with: sample availability + counting
   stats (vectorized as today via the NegBin copula), then run the fill
   allocation and sum. The heavy NegBin sampling stays vectorized; the light
   per-team / per-iteration allocation may be a Python loop (10 teams x 1000
   iters of a ~12-slot greedy is cheap). Exact vectorization of the allocation
   is an implementation detail for the plan, not a design constraint. The
   team-level `np.maximum(actuals, sim)` floor stays as the banked-YTD backstop.

5. **Pitchers (simpler v1).** Hitters get the full games-based fill model (that
   is where the measured bug is). Pitchers ship a simpler v1: games
   (starts/appearances)-available scaling via `gs`/`g`, *without* the rich
   bench-fill -- pitcher slots are generic (9 P), SV is closer-role-driven, and
   ERA/WHIP/SV variance are their own open TODOs. Full pitcher fill is a
   follow-on.

6. **Validation.** A backtest phase checking the new model's category means and
   SDs against realized outcomes, tying into `scripts/backtest_sd_calibration.py`
   and the ROS playing-time-haircut TODO. This phase confirms the fill model
   does not drift category means; it does not attempt to re-settle the
   haircut-vs-reality or SV-variance questions (separate TODOs).

### Scope

- In-season ROS path only: `run_ros_monte_carlo` -> `simulate_remaining_season_batch`.
- The draft `simulate_season` and scalar `simulate_remaining_season` are
  untouched (no Yahoo slots preseason).
- Keep the current top-k as a documented fallback when a roster lacks
  `selected_position` / games data (preseason, tests), so nothing breaks where
  slot info is absent.
- Side effect, not goal: fixing on a games-available allocation removes the
  per-iteration top-k re-selection churn, so it should also shrink some
  distribution width (TODO item: "decompose why one team's MC distribution is
  wider"); the full per-team variance decomposition stays a separate task.

### Roster context (this league)

`config/league.yaml roster_slots`: C1, 1B1, 2B1, 3B1, SS1, IF1, OF4, UTIL2 (12
active hitter slots), P9, BN2, IL2. Total roster 25. The fill pool beyond the 21
active is at most 4 bodies (2 BN + 2 IL), which bounds the assignment problem and
makes same-day collisions rare.

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
- IL: an IL player is part of the fill/active pool per its games availability;
  not stacked on top of a full active lineup.
- Determinism: same seed reproduces the same allocation.
- Regression: existing MC/integration tests still pass. Any fixture that relied
  on bench bats being seated at full season gets flagged and justified, not
  silently changed (per the no-modifying-failing-tests rule).

## Implementation phasing

Each phase is its own plan / PR:

1. Games data plumbing (`g`, `gs` into the dataclasses + blend).
2. Per-player availability draw in games units (reuse PT scale).
3. Fill engine (hitters): greedy capacity-correct allocation.
4. MC integration into `simulate_remaining_season_batch`.
5. Pitcher v1 (games-available scaling, no rich fill).
6. Validation backtest.

## Open implementation questions (for the plan, not blocking the design)

- Exact vectorization vs. per-team Python loop for the fill allocation.
- How `g`/`gs` blend across projection systems (same weights as the rest, or a
  dedicated rule) and the fallback when a system omits `G`.
- Whether to derive a fallback `G` from PA/IP for players missing it (PA per
  game ~4.3; IP per start / per appearance) or treat missing-`G` as the top-k
  fallback path.
- Replacement-line semantics in games units (the existing `_replacement_line` is
  per-stat; map it to a per-game rate for the fill).
