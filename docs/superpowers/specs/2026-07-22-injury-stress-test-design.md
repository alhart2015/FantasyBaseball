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

Reconciliation (operationalized): the baseline run, given the **same input
snapshot** the dashboard MC used (identical `team_rosters`, `actual_standings`,
`fraction_remaining`, `effective_rosters`, `seed=42`, `n_iterations=1000`), must
reproduce the dashboard's stored `first_pct` **exactly**. This is a verification
step (run baseline against the dashboard's stored MC inputs and diff), not a
tolerance. Divergence when the script is run later against a fresher live Upstash
snapshot is expected and is NOT a reconciliation failure -- the inputs differ.

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
= win%. Every run in this report reuses `seed=42`, so the baseline and each
counterfactual share the same random draws (common random numbers): the win%
*delta* between two runs is then low-MC-noise, which is what the rankings need.

### Player set (defined once, used by every section)

"Active contributor" = the active hitters + active pitchers from
`scoring._classify_roster` (the same active/IL/bench partition the MC and ERoto
already use), with IL bodies excluded. Bench players are NOT counterfactual targets
and NOT counted in the health probabilities; they participate only as fill (hitters)
per the existing engine. This one set drives Sections 2, 3, and 4.

### Counterfactual mechanism ("lose player X, replaced")

Do NOT model the loss by deleting X from the roster. Two engine facts make deletion
wrong: (a) the internal bench-fill is **hitters-only** -- `build_effective_roster`
drops healthy bench pitchers and `_simulate_team_pitchers_ros_direct` runs with no
fill (`mc_roster.py:109-112`, `simulation.py:1078-1080, 1104-1105`), so a deleted
pitcher would be a raw hole, not "replaced"; and (b) deleting a hitter shrinks the
active set rather than generating the missed-games that trigger
`mc_fill.allocate_bench_fill`, so even hitters would not backfill as intended.

Instead, model "lose X" as **position-matched replacement-level substitution**:
replace X's roster entry with a synthetic replacement-level line at X's slot
(reuse `simulation._replacement_line` / `REPLACEMENT_BY_POSITION`, scaled to a ROS
volume comparable to X's projected ROS games so the slot is fully manned by a
replacement, not left partial), then re-run the full MC and read the drop in
`first_pct`. This is uniform across hitters and pitchers and directly expresses
value-over-replacement in win%. Residual, intentional asymmetry to state in the
report: for hitters your real bench-fill still operates on top of the substituted
line (so the hitter cost credits your actual bench), whereas the pitcher cost
credits only the generic replacement arm (the engine has no pitcher bench-fill --
this matches how the sim already treats a lost pitcher). Pairs ("lose two") apply
the same substitution to both players simultaneously.

Implementation invariant: `run_ros_monte_carlo` consumes two derived-but-separate
inputs -- the `team_rosters` player dicts AND `effective_rosters` (built from
`Player` objects; hitters route through the ROS-direct engine off THIS input). A
counterfactual must rebuild **both** from the same substituted roster in lockstep:
substitute the replacement line into the `Player` list, rebuild `effective_rosters`
via `build_effective_rosters`, and flatten the same list into the `team_rosters`
dicts. Substituting only `team_rosters` and passing a stale `effective_rosters`
would leave the ROS-direct hitter path simulating the original player -- a silent
wrong number. Baseline and every counterfactual use the same `effective_rosters`-on
configuration (the dashboard's), so they reconcile.

Three ways the engine is driven:

- **Baseline run** -- the roster as-is. Yields the real win%.
- **Availability-variance-off run** -- same roster, but availability variance
  suppressed (see "Required sim change"). Yields "win% if availability lands at its
  expected level" -- the attribution anchor for the headline.
- **Counterfactual re-runs** -- substitute a player (or pair) with a replacement-level
  line per above, re-run, read the drop in `first_pct`.

The health-probability figures in Section 2 do NOT need the joint sim: they reuse
the same playing-time primitive (`playing_time.scale_from_uniform`) to sample each
active player's missed-time fraction directly, independent of standings. The
actionable "erosion" is delivered concretely by the counterfactual sections (3-4)
for the specific players, which is more useful than a generic "given one random
injury" conditional.

## The report (five sections)

### 1. Headline -- what injury risk costs you

Three numbers side by side:
- Deterministic projected roto **margin** -- the signed gap between the user and the
  projected leader (positive = ahead, "you win by N"; negative = behind). Computed by
  scoring the same `eos_baseline` the MC consumes (the dashboard's projected standings
  IS the scored `eos_baseline`), so it reproduces the dashboard's projected margin up
  to the minor un-persisted team-AB-attribution drift (which the reconciliation note
  documents). This avoids a second, blob-shape-fragile read of the stored projected
  standings for a number the baseline already has in hand.
- Full MC win% (the real number).
- Availability-variance-off MC win% (same expected roster, availability luck removed).

The gap `availability_off_win% - full_win%` is the price of injury/availability risk
(missed time plus closer role loss), isolated from performance variance (which is
present in both runs).

### 2. How likely is "everyone stays healthy"?

From standalone playing-time sampling of the active roster (reusing
`scale_from_uniform`, large sample, fixed seed):
- P(no active contributor loses significant time)
- P(exactly one does)
- P(two or more)

"Significant time" is measured as a shortfall below the player's **own expected
remaining playing time** (`eff_mean`), NOT below the raw projection. This matters:
the MC's hitter playing-time model centers on a systematic `mean_scale` haircut
(~0.75-0.94 -- projection optimism that the engine restores via bench-fill), so a
raw `frac_missed = 1 - scale` would exceed any reasonable threshold for nearly every
hitter from the haircut alone, not from injury. Measuring the downside tail relative
to `eff_mean` isolates the injury/availability shortfall from that baseline haircut.

Concretely, per active player the sampler mirrors the MC's own moments (reusing
`playing_time.playing_time_params` / `playing_time_shape` / `playing_time_moments`
and `simulation._full_season_pt_volume` for the curve's full-season volume basis):
`eff_mean` uses the MC's mean horizon (`pt_mean_fraction=1.0` for hitters ->
`eff_mean = mean_scale`; `=0.0` for pitchers -> `eff_mean = 1.0`) and `eff_sd` uses
the variance horizon (`fraction_remaining`). A draw yields
`scale = max(0, eff_mean + z(u) * eff_sd)`, and the player "loses significant time"
when `scale <= eff_mean * (1 - threshold)` (i.e. realized playing time at least
`threshold` below expected). Default threshold: **0.20** (roughly a 4-week IL stint
at mid-season), exposed as a tunable module constant. Reported for the default and
can be re-run at another threshold.

### 3. Who are you most exposed to? (single-player counterfactuals)

Every active contributor (see "Player set") ranked by the win% (and roto-point) cost
of losing them for the ROS, via the replacement-level substitution defined in
"Counterfactual mechanism." Stars surface at the top automatically; the top row is
the "what if it is a star" answer ("the season leans hardest on ___").

### 4. Losing two

Exhaustive over the top-K active contributors by single-player exposure
(default **K = 8** -> 28 pairs), ranked by combined win% cost. Flags pairs whose
joint cost exceeds the sum of the two singles (worse-than-additive), which surfaces
when both players stack in the same thin category (e.g. SB or SV).

### 5. (Reported as a note) v1 uses generic injury risk

State plainly in the report that Section 2's draws are volume-band generic, not
per-player, and point at the Future-work upgrade. No silent caps.

## Required sim change (small, additive)

Add one opt-in flag that suppresses **availability variance** while preserving
expected volume. `_apply_variance_batch` is the single choke point -- both
ROS-direct body samplers (`_simulate_team_hitters_ros_direct` via
`_sample_hitter_bodies`, and `_simulate_team_pitchers_ros_direct`) and the top-k
path all route their availability draws through it. When the flag is set, inside
`_apply_variance_batch`:
- force the standardized playing-time draw `z_pt = 0` so `scale = eff_mean`
  (deterministic expected volume, zero spread) instead of drawing `z(u)`; and
- pin the SV closer-role draw (`closer_mixture.role_multiplier_draw`) to its
  expected multiplier instead of sampling the role switch.

Performance NegBin variance is untouched. Because the flag lives in the one shared
function, it uniformly covers hitters, pitchers, and SV in a single place.

Threaded through `run_ros_monte_carlo` as a new keyword defaulting to the current
behavior (variance on). No existing caller changes. This is the only production-code
change; everything else lives in the new script.

Note: this refines the mechanism sketched during brainstorming (originally "return
per-iteration arrays"). Suppressing availability variance for one comparison run is
cleaner and less invasive than exposing per-iteration internals, and it yields the
same headline attribution.

## Runtime

~1 baseline + 1 availability-off + ~23 single counterfactuals + 28 pair
counterfactuals ~= ~53 vectorized 1k-iteration runs, plus the standalone health
sampling (cheap). Fixed seed (42) throughout for reproducibility and common-random-
number deltas.

v1 scope is **exhaustive MC only**. The first implementation task benchmarks one
1k-iteration run; if the projected total (runs x per-run time) exceeds ~3 minutes,
STOP and surface the measurement to the user for a scope decision rather than
silently building an optimization. The analytic deltaRoto pre-rank
(`lineup/delta_roto.py`) is explicitly out of v1 scope -- listed under Future work.

## Testing

- **Health sampler** (`test`): deterministic seed + a hand-built two-player roster
  with known band params -> assert P(all healthy), P(1), P(2+) match a computed
  expectation; assert they sum to 1. Also assert the eff_mean-relative definition: a
  hitter whose only shortfall is the systematic `mean_scale` haircut (realized ==
  `eff_mean`) is NOT counted as losing significant time (guards the haircut-vs-injury
  bug this section fixes).
- **Counterfactual delta** (substitution mechanism): substituting a player who is
  already replacement-level yields ~0 win% change; substituting a high-value player
  yields a positive, larger change. Monotonicity: a strictly-more-valuable player is
  at least as costly to lose. Assert the substitution actually mans the slot (the
  team's category totals do not collapse to a raw hole), and that it works for a
  pitcher as well as a hitter (the asymmetry check that motivated the mechanism).
- **Availability-variance-off flag**: with the flag on, the availability
  contribution to spread collapses (per-player `frac_missed` variance ~ 0 / scales
  pinned to `eff_mean`, and the SV role multiplier pinned to expected); performance-
  stat spread is unchanged vs. a performance-only baseline. Guard that existing
  callers are byte-identical with the flag defaulted off.
- **Reconciliation**: baseline run on a fixed stored input snapshot reproduces a
  known `first_pct` exactly (regression-locks the "same inputs -> same number"
  contract).
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

Analytic deltaRoto pre-rank. If the exhaustive-MC runtime proves uncomfortable in
practice, pre-rank single/pair counterfactuals with `lineup/delta_roto.py` and
MC-confirm only the top ranks. Out of v1 scope (see Runtime); listed here so it is
not silently built.

## Open questions

None blocking. The judgment calls -- the 0.20 significant-time threshold, the
replacement-level substitution as the counterfactual mechanism, and the
availability-variance-off flag as the headline attribution -- are settled above and
are all tunable / additive.
