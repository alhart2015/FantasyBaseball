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
Distributions view; run conditions (seed, `fraction_remaining`, iterations) were
not recorded, and there is a competing explanation -- the per-iteration top-k
*re-selection churn* (a fresh best-ball pick every iteration) rather than bench
seating. **Phase 0 settles this before any expensive engine is built.**

## Data-path reality (read before the design)

A critical fact that shapes the whole integration: `run_ros_monte_carlo` flattens
each Player via `_flatten_full_season` -> `Player.to_flat_dict_full_season()`
(`simulation.py:929-931`, `player.py:294-303`), which overlays
**`full_season_projection`** (= YTD + ROS), NOT `rest_of_season`. So the counting
stats `simulate_remaining_season_batch` samples are **full-season**, and the
remaining portion is recovered downstream as `max(0, sim - actual_YTD)` at the
team level (`simulation.py:781-805`). Consequence: a `G` field flattened the same
way would arrive as **full-season G**, not ROS games -- so the availability model
cannot simply read the flat dict's games.

This design therefore operates the availability/fill engine in **ROS (remaining)
terms**, sourcing ROS games and ROS stat means from `Player.rest_of_season` at MC
setup (where Player objects are still in hand), not from the full-season flat
dicts. See Component 4 for the integration shape.

## Why not the obvious fix (mirror ERoto)

Mirroring ERoto exactly (bench-exclusion + slot-aware classification + IL
displacement, fixed on means) converges the two engines but encodes a model both
get wrong on one dimension: **roster depth as injury insurance.**

- Today's top-k MC: depth fully stacked -- seats SS2's whole season on top of SS1.
  Massive over-credit.
- ERoto / bench-exclusion: depth worth zero -- SS2 never plays; an injury to SS1
  is filled by a replacement-level (waiver) line.
- Reality: depth worth its injury-insurance value -- when SS1 misses time you
  start your benched, eligible SS2 (at SS2's level), capped at one body's worth of
  games. Replacement-level is the last resort, only when no eligible bench body is
  free.

Both incumbents are wrong in opposite directions. The chosen design models the
correct middle -- after Phase 0 confirms bench seating (not re-selection churn) is
the dominant driver.

## Chosen design: games-based availability with capacity-correct fill

### Core idea

Express every player's remaining workload in **player-games** -- the unit with a
clean, structural maximum: a body plays at most one slot per game and at most the
games it is healthy for. Injuries remove games. A vacated slot-game is filled
**bench-first** (position-eligible, ordered by the value rule in Component 3),
**replacement-level last**, with each fill body capped at its own available games
(one-body capacity).

`remaining_games` is the player's **rest-of-season** games projection, taken from
`Player.rest_of_season` (the new `g` field, Component 1) at MC setup. It is NOT
`full_season_G * fraction_remaining` (a global scalar that wrongly assumes uniform
elapsed games and breaks for IL players / call-ups / returning starters), and it
is NOT the full-season `G` that the full-season flatten would deliver (see
Data-path reality). The ROS CSVs carry a ROS-scaled `G` (verified: a full-timer
reads ~75 G / ~343 PA at mid-season), so `rest_of_season.g` is the right source.
Where a player has a ROS PA/IP projection but no ROS `G`, derive games from those
via a typical per-game rate (PA/game ~4.3; IP per start / per appearance). The
rostered-but-entirely-unprojected case is handled in Scope.

This is an intentional divergence from ERoto. ERoto stays the slot-legal snapshot;
the MC becomes the realistic-outcomes engine that prices depth. Bringing ERoto
along later is out of scope.

### Time / capacity model

There is no rest-of-season day-by-day calendar. `weekly_schedule.json` is a
single-week snapshot (`games_per_team` ~6-8) used by lineup/matchup code; it
carries neither the ROS horizon nor per-player granularity, and the MC does not
and will not read it. A calendar-aware daily simulation is unsupported and would
be extrapolated regardless.

We use **abstracted available-games**: each player has `remaining_games` available
(ROS games); an availability draw removes a fraction (Component 2); capacity is a
pooled allocation (a body contributes at most its available games across every
slot it fills), *without* the literal calendar. This ignores the rare collision of
two needs on one eligible body the same day -- bounded and conservative (can only
under-credit fill) given the 4-body pool (Roster context). No new schedule data is
plumbed.

### Components

1. **Games data plumbing.** Add `g` to `HitterStats` and `g`/`gs` to
   `PitcherStats`; thread from the projection CSVs (`G`, `GS` present in the
   FanGraphs exports) through the blend into the dataclasses. PA/IP unchanged.
   Foundational.

   Audit (grep-every-call-site rule):
   - New fields MUST NOT enter SGP. `calculate_player_sgp` reads only explicitly
     named fields, so this holds by construction -- the audit confirms it.
   - Forward serialization round-trips stay stable.
   - BACKWARD compat: already-persisted JSON (`draft_state*.json`, dashboard state)
     lacks `G`, so a round-trip materializes `g=0`/`gs=0` via `from_dict`'s `or 0`.
     No consumer may trust `g`/`gs` read off pre-change persisted state -- the
     classic falsy-zero footgun. Audit that none does, or gate use behind a
     presence check.

2. **Per-player availability draw (reuse the PT scale, on ROS stats).** Reuse the
   calibrated playing-time scale as the per-player available-games fraction:

       available_games = remaining_games * pt_scale_draw

   Because the engine now operates in ROS terms (Component 4), the draw and the
   counting stats are sampled from the ROS projection (`rest_of_season`), and the
   PT-scale's `fraction_remaining` damping (`playing_time_moments`) must be
   reconciled so the remaining-season playing-time risk is applied ONCE, not
   double-counted against an already-ROS projection -- pinned in Phase 2.

   What changes vs. today: currently the missed fraction `frac_missed =
   max(0, 1 - scales)` is already backfilled at replacement level (`repl_contrib =
   repl_line * frac_missed`, `simulation.py:701-710`); the missed mass does not
   vanish. This design redirects that same missed fraction to bench-fill first,
   replacement only as residual.

   KNOWN APPROXIMATION (load-bearing): `pt_scale_draw` is calibrated on
   actual/projected PA(IP), conflating (a) missed *games* (a bench body starts)
   with (b) reduced *PA-within-games* (starter plays, no slot opens). Crediting
   `(1 - pt_scale_draw)` as vacated slot-games over-counts (b). Mitigations: (i)
   for hitters, games-played dominates PA variance (PA/game stable ~4.1-4.4), so
   (b) is small; (ii) fill body contributes at its own lower rate, capped at one
   body; (iii) the before/after evidence and the SD backtest are the gate --
   visible failure, not silent. A stint-based games-missed model separating (a)/(b)
   is the deferred refinement.

3. **Slot-assignment / fill engine (hitters).** Per team, per iteration, on the
   sampled ROS stats:

   - Each active-slot starter contributes its available games to its slot at its
     own sampled per-game line.
   - For each slot, the shortfall `(slot_games - starter_available_games)` is
     offered to the fill pool: the team's bench + IL bodies (each with its own
     sampled available games), restricted to those position-eligible.
   - **Value rule (per-game, single, explicit).** Filling a slot for N games is a
     per-game quality decision, so fill bodies are ordered by **per-game ROS
     value**, NOT total ROS SGP. (Total SGP would seat a full-time mediocre body
     ahead of a part-time higher-quality one for a small shortfall -- wrong for
     filling; this realistic case arises with OF4/UTIL2 flexibility, so it is not
     hypothetical.) Per-game value = the body's ROS roto value per ROS game.
     Compute it from `calculate_player_sgp` on the body's `rest_of_season` divided
     by `g_ros`, guarded as `value / g_ros if g_ros > 0 else 0` so a zero-game
     body is never chosen as fill. Caveat (acknowledged, minor): SGP's rate terms
     normalize against
     fixed full-season constants (`team_ab=5500`/`team_ip=1450`), so the AVG
     sub-term is slightly off-horizon; this affects 1 of 5 hitter categories and
     is far smaller than the volume bias of total SGP, so per-game value is the
     better ordering. (This supersedes the earlier total-SGP choice.) Process
     shortfalls largest-first; for each pick the highest per-game-value eligible
     body with games remaining; tie-break deterministically (higher per-game
     value, then player-id ascending) for reproducibility. Per-game value is
     computed once per body at MC setup (Player objects in hand), carried into the
     allocation; it is NOT assumed pre-populated on flat dicts (`to_dict` emits
     `sgp` only when set).
   - A chosen body's available-games pool decrements by games covered (one-body
     capacity: one bench bat cannot cover two simultaneous injuries beyond one
     body).
   - Residual uncovered games fall to a replacement-level line (Rate-stat
     handling).

   Allocation quality: <=4 fill bodies vs 12 hitter slots -- a tiny assignment;
   greedy is near-optimal at this size, error averaged over 1000 iters; exact
   small assignment is an acceptable alternative if a pathological case appears.

   **Rate-stat handling (AVG/ERA/WHIP).** Not filled separately. Every contributing
   body (starter, bench-fill, replacement residual) adds its recovered counting
   *components* -- hitters `h`, `ab`; pitchers `er`, `ip`, `bb`, `h_allowed` --
   scaled to games covered, into the team component sums; rates recombine from team
   totals as today. A fill body contributes its own (typically lower) rate by
   volume, dragging the team rate the realistic direction. The replacement residual
   is expressed per-game by dividing `_replacement_line` (`simulation.py:435-461`,
   a per-stat bundle with NO games field) by a games denominator from the SAME
   PA/IP-per-game heuristic as the missing-`g_ros` fallback -- one shared constant
   (Open questions), honestly derived, not recoverable from the
   `REPLACEMENT_BY_POSITION` calibration (which has no games basis).

   **Variance note (acknowledged).** The NegBin copula samples each body at its
   full ROS volume; scaling those counts to games-covered by fraction `f` yields
   variance `f^2 * var`, vs ~`f * var` for a body genuinely playing `f` of its
   games -- so component-scaling understates partial-fill variance. Bounded: it
   applies only to the (small) fill portion, and is still strictly more realistic
   than today's *deterministic* replacement fill (zero variance). The SD backtest
   is the gate; correct partial-volume re-sampling is the deferred refinement.

4. **MC integration (ROS-direct).** This is the part the Data-path reality forces.
   For the hitter fill path, stop sampling full-season and recovering remaining by
   subtraction; instead sample **ROS production** directly (from `rest_of_season`),
   run the per-iteration fill allocation, sum to team ROS totals, and blend
   `team_total = team_YTD + summed_ROS` (rates recombined from `YTD + ROS`
   components, using the actual_ab/actual_ip already threaded from Yahoo). Two
   wins beyond fixing the games source: (a) horizon-consistent (games, stats, and
   damping all ROS); (b) the banked-YTD floor becomes **structural** for the
   hitter path -- ROS contributions are non-negative, so `team_total >= YTD`
   automatically and the `max(actual, sim)` clamp is unnecessary *for hitters*,
   which dissolves the earlier "floor binds -> inconclusive" concern. NOTE the
   seam: pitchers stay on the existing full-season-minus-YTD blend (Component 5),
   so the `max(actual, sim)` clamp is RETAINED for pitcher categories. Within the
   one batch function this is a clean dual path -- hitting and pitching counting
   stats are disjoint and already sampled by separate `_apply_variance_batch`
   calls -- but the implementer must not drop the pitcher clamp when removing the
   hitter one.

   Classification/attribution happens at **setup on Player objects** (in
   `run_ros_monte_carlo`, which receives Player lists), reusing the existing
   Player-typed `_classify_roster`. It produces, per team: the active starters per
   slot, the bench/IL fill pool, and per body its eligible positions + ROS games +
   per-game value + ROS stat means. The batch consumes this attributed structure
   and does the per-iteration sampling + allocation -- NO reimplementation of the
   classifier on flat dicts. Heavy NegBin sampling stays vectorized; the light
   per-team/per-iteration greedy may be a Python loop (cheap at this scale).

5. **Pitchers (v1 = active-slot bench-exclusion only).** Hitters get the full fill
   model (where the bug is). Pitchers get classification only: select the
   manager's active-slot pitchers, exclude healthy bench AND IL-slotted pitchers,
   keep the existing IP-based PT scale and replacement-level injury fill for that
   active set. We do NOT add a `gs`/`g` scaler (double-discounts IP-calibrated PT
   volume; no-op for ERA/WHIP since a common factor cancels), do NOT build a
   pitcher fill pool, and do NOT model IL-pitcher return (the displacement/pool
   model -- `_compute_pitcher_pool_factors` -- and closer-role SV (`SV -> 0` on job
   loss) are deferred). Rationale for bench-exclusion rather than leaving pitchers
   on top-k: if hitters lose their bench over-credit but pitchers keep theirs
   (deep bullpen seated by top-k), 5x5 totals tilt toward pitching-deep teams -- a
   NEW uncompensated bias. Accepted v1 limitation: excluding IL-slotted pitchers
   under-credits a team with an arm about to return; this is conservative
   (vs. today's over-seat) and is surfaced by the all-categories evidence. The
   `g`/`gs` plumbing still lands for the deferred pitcher fill.

6. **Validation + before/after evidence.** A validation phase producing the
   acceptance artifact plus a backtest of category means AND SDs against realized
   outcomes (`scripts/backtest_sd_calibration.py` + the ROS-haircut TODO). The SD
   check is the gate on both the PA-vs-games (Component 2) and variance-scaling
   (Component 3) approximations. It does not re-settle the haircut-vs-reality or
   SV-variance questions (separate TODOs).

### Acceptance evidence (before/after) -- REQUIRED

Not shipped on green tests alone:

- A committed script (e.g. `scripts/compare_mc_active_selection.py`) runs the
  in-season ROS MC on the **same cached snapshot** under (1) OLD top-k, (2) NEW
  games-based bench fill, (3) pure bench-exclusion (ERoto-style). Prints per-team
  medians for **all ten categories** + overall roto standings for all three, side
  by side with ERoto, with run conditions (seed, `fraction_remaining`, iterations)
  in the header.
- Attribution: NEW between the bench-exclusion floor and old top-k => bench seating
  (diagnosis holds); bench-exclusion alone closing most of it => re-selection churn
  (framing corrected). Same diagnostic as Phase 0, re-run with the finished engine
  as the third arm.
- Pitcher-side: report pitcher categories + overall standings so the
  hitter-fill/pitcher-bench-exclusion asymmetry is bounded and visible.
- Reproduce-and-close: SkeleThor RBI median lands materially below 1020 (between
  the ~926 floor and 1020); the Hart RBI re-rank (1st -> 3rd) no longer occurs.
- Real cached data (Upstash/Render source of truth; never stale local cache).

Definition of done for the integration phase. "Tests pass" is necessary but not
sufficient.

### Scope

- In-season ROS path only: `run_ros_monte_carlo` -> `simulate_remaining_season_batch`.
  Draft `simulate_season` and scalar `simulate_remaining_season` untouched.
- Fallback granularity **whole-context, never per-player within a run.** In-season
  uses the new model for ALL teams; legacy top-k only for entirely slot-less
  contexts (draft/preseason, slot-less test dicts). Mixing models within one
  standings computation is a correctness hazard, so the choice is made once per
  run.
- Rostered-but-unprojected players (waiver adds, fresh call-ups, no FanGraphs line):
  no ROS stats and no PA/IP to derive `g_ros` => zero projected production, zero
  per-game value => contributes nothing, never chosen as fill (correct for a true
  scrub). If such a player is in an ACTIVE slot, its slot shortfall is the full
  slot games, filled by pool then replacement-level (the honest estimate with no
  projection). This is the fill engine operating on a zero-projection body, NOT a
  per-player top-k switch.
- Side effect to VERIFY (hypothesis): removing re-selection churn *may* shrink some
  distribution width (related TODO). Phase 0 and the artifact check it; not assumed.

### Roster context (this league)

`config/league.yaml roster_slots`: C1, 1B1, 2B1, 3B1, SS1, IF1, OF4, UTIL2 (12
active hitter slots), P9, BN2, IL2. Total 25. Beyond the 21 active slots the extra
bodies are at most 4 (2 BN + 2 IL); IL bodies are low-availability, so the
effective healthy fill pool is usually < 4. Bounds the assignment; same-day
collisions rare.

## Testing

- Bench-exclusion-but-fill: a healthy bench bat out-rating an active starter
  contributes zero while the starter is healthy, but contributes when the
  starter's availability draw is low -- capped at its own available games.
- One-body capacity: two eligible starters both draw low, one bench body eligible
  for both -- its total contributed games do not exceed its available games.
- Replacement-last: bench pool exhausted -> residual to replacement, not an
  over-extended bench body.
- Rate-stat fill: filling with a lower-rate body moves team AVG/ERA/WHIP the
  realistic direction; a constructed case asserts the recombined rate equals the
  volume-weighted component sum.
- Per-game value ordering: a part-time higher-per-game body is chosen over a
  full-time lower-per-game body for a small same-position shortfall (pins the
  per-game, not total-SGP, rule).
- Unprojected active player: zero contribution, slot filled by pool then
  replacement, never a per-player top-k switch.
- IL hitter in fill pool per its low availability; not stacked on a full lineup.
- Determinism / tie-break: two equal-value eligible fill bodies -> a SPECIFIC
  asserted allocation, not mere seed-stability.
- Whole-context fallback: in-season prices every team with the new model; slot-less
  input falls entirely to top-k.
- Pitcher bench-exclusion: a benched (or IL-slotted) pitcher is excluded from the
  active set, not seated by raw `w+k+sv`.
- Regression: existing MC/integration tests pass; any fixture relying on bench
  seating is flagged and justified, not silently changed.

## Implementation phasing

Phase 0 is a GATE; Phases 1-6 each their own plan / PR.

0. **Attribution diagnostic (gate).** Reuse the existing Player-typed
   `_classify_roster` at MC setup to build a bench-excluded active set, and run a
   batch variant that sums that set instead of top-k. NO games plumbing, NO
   dataclass changes, NO fill engine -- only the existing classifier + threading
   the active partition + a sum-vs-top-k toggle. Re-measure the SkeleThor gap under
   recorded seed/`fraction_remaining`/iterations (the eyeballed 94 is not the
   baseline). PASS CRITERION: bench-exclusion alone closes >= 50% of the
   re-measured gap => bench seating dominates, proceed. If < 50% => re-selection
   churn dominates; STOP and freeze selection instead of building the engine.
1. Games data plumbing (`g`, `gs`) + serialization/SGP/backward-compat audit.
2. Per-player availability draw in ROS games (reuse PT scale; reconcile the
   `fraction_remaining` damping; redirect `frac_missed`).
3. Fill engine (hitters): capacity-correct allocation, per-game-value rule,
   deterministic tie-break, rate-stat component fill, replacement per-game
   conversion.
4. MC integration (ROS-direct blend; hitter fill + pitcher bench-exclusion;
   setup-time classification on Player objects), plus the before/after artifact
   (definition of done).
5. Pitcher bench-exclusion confirmed coherent with the hitter path (no
   double-scaling, IL-pitcher exclusion documented).
6. Validation backtest (means AND SDs).

## Open implementation questions (for the plan, not blocking the design)

- Exact vectorization vs. per-team Python loop for the allocation (math-identical).
- `g`/`gs` blend across systems (default: same weights; when a system omits `G`,
  drop it from the `G` blend rather than zeroing).
- The single shared PA-per-game / IP-per-appearance constant used for BOTH the
  missing-`g_ros` derivation and the replacement per-game conversion -- pinned in
  Phase 3; must be one constant, not two.
- The exact reconciliation of the PT-scale `fraction_remaining` damping under the
  ROS-direct framing (avoid applying remaining-season risk twice) -- pinned in
  Phase 2. Likely resolution: pass `fraction_remaining=1.0` to
  `playing_time_moments` for the ROS-direct draw, since the ROS projection already
  encodes the remaining horizon; confirm against the SD backtest.
