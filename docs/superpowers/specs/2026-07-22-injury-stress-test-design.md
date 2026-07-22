# Injury Stress-Test -- Design

- Date: 2026-07-22
- Status: Approved (design); pending implementation plan
- Related code: `src/fantasy_baseball/simulation.py` (`run_ros_monte_carlo`,
  `simulate_remaining_season_batch`, `_apply_variance_batch`, `_playing_time_scales`),
  `src/fantasy_baseball/utils/playing_time.py`, `src/fantasy_baseball/mc_roster.py`
  (`build_effective_rosters`), `src/fantasy_baseball/web/refresh_pipeline.py`
  (roster/standings assembly the report reuses).

## Problem

If every player on the roster stays healthy, the deterministic rest-of-season (ROS)
projected standings show a comfortable win. That point estimate hides the real
question: how likely is "everyone stays healthy," and how fast does the lead erode
when it does not?

The ROS Monte Carlo already samples missed time and folds it into the win
probability, so the MC win% is partly this answer already. But the machinery has
two gaps relative to the questions being asked:

1. It is never surfaced as interpretable scenarios. There is no "if you lose your
   biggest contributor, win% goes from X to Y" output, and no ranking of which
   players the season leans on hardest.
2. Missed-time risk is a volume-band aggregate (`cv_pt` by projected PA/IP + role),
   identical for every player in a band. Injury-prone players and iron-men draw the
   same distribution. (This gap is acknowledged and deferred -- see Non-goals.)

## Goals

Answer four concrete questions, as a standalone re-runnable script:

1. How likely is it that everyone stays healthy (no active contributor loses
   significant time)?
2. What does losing one player cost the win probability?
3. What does losing two cost?
4. Which player is the season most exposed to (the "star" case)?

All numbers must reconcile with the dashboard's MC win% (same engine, same inputs,
same seed).

## Non-goals (deferred)

- **Per-player injury propensity.** v1 uses the existing generic volume/role
  playing-time model for the health-probability draws. Every player in a PA/IP band
  gets the same downside. Fattening the tails for injury-prone players from real
  IL/days-missed history (MLB StatsAPI, keyed on `MLBAMID`) is a fast-follow, gated
  on first prototyping whether that history is even predictive. See Future work.
- **No new UI surface.** Terminal report (optionally a written markdown file); no
  dashboard panel, no refresh-pipeline wiring.
- **No injury-news / IL-transaction ingestion.** Current-IL players are handled
  exactly as the sim already handles them (slot classification + displacement +
  bench fill via `effective_rosters`).

## Data source

Reads **live Upstash** (source of truth for season state; set `RENDER=true` so
`get_kv()` resolves to the remote store, mirroring `scripts/refresh_remote.py`).
The script reuses the refresh pipeline's roster + standings assembly so the MC
inputs (`team_rosters`, `actual_standings`, `fraction_remaining`, `h_slots`,
`p_slots`, `effective_rosters`) are byte-for-byte what the dashboard MC consumes.
No reimplementation of data loading -- extract/reuse the pipeline loaders.

## Engine

Single source of truth: the existing ROS Monte Carlo (`run_ros_monte_carlo`,
`n_iterations=1000`, `seed=42`). `first_pct` in its `team_results` is P(finish 1st)
= win%. The `effective_rosters` path already backfills a vacated hitter slot from
the bench, then replacement level -- this is exactly the "replaced" loss model.

Three ways the engine is driven:

- **Baseline run** -- the roster as-is. Yields the real win%.
- **Injury-variance-off run** -- same roster, but playing-time variance suppressed
  (see "Required sim change"). Yields "win% if health lands at its expected level"
  -- the attribution anchor for the headline.
- **Counterfactual re-runs** -- drop a player (or pair) from the user's roster, let
  `effective_rosters` backfill, re-run, read the drop in `first_pct`.

The health-probability figures in Section 2 do NOT need the joint sim: they reuse
the same playing-time primitive (`playing_time.scale_from_uniform`) to sample each
active player's missed-time fraction directly, independent of standings. The
actionable "erosion" is delivered concretely by the counterfactual sections (3-4)
for the specific players, which is more useful than a generic "given one random
injury" conditional.

## The report (five sections)

### 1. Headline -- what injury risk costs you

Three numbers side by side:
- Deterministic projected roto margin ("you win by N points" -- the point estimate).
- Full MC win% (the real number).
- Injury-variance-off MC win% (same expected roster, injury luck removed).

The gap `injury_off_win% - full_win%` is the price of injury/playing-time risk,
isolated from performance variance (which is present in both runs).

### 2. How likely is "everyone stays healthy"?

From standalone playing-time sampling of the active roster (reusing
`scale_from_uniform`, large sample, fixed seed):
- P(no active contributor loses significant time)
- P(exactly one does)
- P(two or more)

"Significant time" = missing at least a threshold fraction of a player's remaining
games. Default threshold: **0.20** (roughly a 4-week IL stint at mid-season),
exposed as a tunable module constant. The threshold is aligned in spirit with the
sim's existing `_NOTABLE_PT_LOSS = 0.15` notable-injury flag; 0.20 is chosen as a
slightly stricter "real injury, not a routine day off" bar. Reported for the
default and can be re-run at another threshold.

### 3. Who are you most exposed to? (single-player counterfactuals)

Every active contributor ranked by the win% (and roto-point) cost of losing them for
the ROS, replaced by realistic backfill. Stars surface at the top automatically; the
top row is the "what if it is a star" answer ("the season leans hardest on ___").

### 4. Losing two

Exhaustive over the top-K active contributors by single-player exposure
(default **K = 8** -> 28 pairs), ranked by combined win% cost. Flags pairs whose
joint cost exceeds the sum of the two singles (worse-than-additive), which surfaces
when both players stack in the same thin category (e.g. SB or SV).

### 5. (Reported as a note) v1 uses generic injury risk

State plainly in the report that Section 2's draws are volume-band generic, not
per-player, and point at the Future-work upgrade. No silent caps.

## Required sim change (small, additive)

Add an opt-in flag to the batch path that suppresses playing-time variance while
preserving expected volume. Mechanism: in `_apply_variance_batch` (and the
ROS-direct body samplers it feeds), when the flag is set, force the standardized
playing-time draw `z_pt = 0` so `scale = eff_mean` (deterministic expected volume,
zero spread) instead of drawing `z(u)`. Performance NegBin variance is untouched.

Threaded through `run_ros_monte_carlo` as a new keyword defaulting to the current
behavior (variance on). No existing caller changes. This is the only production-code
change; everything else lives in the new script.

Note: this refines the mechanism sketched during brainstorming (originally "return
per-iteration arrays"). Suppressing playing-time variance for one comparison run is
cleaner and less invasive than exposing per-iteration internals, and it yields the
same headline attribution.

## Runtime

~1 baseline + 1 injury-off + ~23 single counterfactuals + 28 pair counterfactuals
~= ~53 vectorized 1k-iteration runs, plus the standalone health sampling (cheap).
Benchmark one run first. If total wall-clock is uncomfortable, pre-rank singles and
pairs with the fast analytic deltaRoto (`lineup/delta_roto.py`) and MC-confirm only
the top ranks; the exhaustive-MC path stays the default until measured otherwise.
Fixed seed (42) throughout for reproducibility.

## Testing

- **Health sampler** (`test`): deterministic seed + a hand-built two-player roster
  with known band params -> assert P(all healthy), P(1), P(2+) match a computed
  expectation; assert they sum to 1.
- **Counterfactual delta**: dropping a zero-projection player yields ~0 win% change;
  dropping a high-value player yields a positive, larger change. Monotonicity: a
  strictly-more-valuable player is at least as costly to lose.
- **Injury-variance-off flag**: with the flag on, the playing-time contribution to
  spread collapses (per-player `frac_missed` variance ~ 0 / scales pinned to
  `eff_mean`); performance-stat spread is unchanged vs. a performance-only baseline.
  Guard that existing callers are byte-identical with the flag defaulted off.
- **Reuse existing fixtures** under `tests/test_simulation.py`, `tests/test_mc_*`,
  and the refresh fixture for a synthetic small-league integration smoke test.

## Future work (deferred, not in v1)

Per-player injury propensity. Prototype first: pull each rostered player's recent
IL stints / days-missed from MLB StatsAPI (already wired in
`data/mlb_game_logs.py` / `data/mlb_schedule.py`; join on `MLBAMID`), and measure
whether prior missed time predicts next-season missed time well enough to beat the
band-generic baseline. If yes, add a per-player risk multiplier / tier that widens
the playing-time downside in both the MC (`_playing_time_scales`) and the analytic
ERoto path (`scoring.player_category_variance`) through the shared
`utils/playing_time.py` primitives -- so all downstream numbers improve, not just
this report.

## Open questions

None blocking. The two flagged judgment calls -- the 0.20 significant-time threshold
and the injury-variance-off flag as the attribution mechanism -- are settled above
and both are tunable / additive.
