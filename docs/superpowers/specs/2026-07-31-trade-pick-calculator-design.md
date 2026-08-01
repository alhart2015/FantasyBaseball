# Trade-for-Future-Pick Calculator -- Design

Date: 2026-07-31
Status: approved design (brainstorming complete)
Author: session (Hart)

## Problem

In a keeper league you sometimes get offered a future draft pick for a current
player -- e.g. "trade Julio Rodriguez to SkeleThor for their 2027 5th-round
pick." The first three rounds are keeper rounds, so a nominal 5th-round pick is
really the 2nd pick you actually make in the draft. To judge the offer you need
two numbers on different footings:

1. **This year's cost:** how your win% / top-3% / per-category odds move once
   Julio is off your roster (and on a rival's).
2. **Next year's benefit:** what an extra 2nd-round-equivalent pick is worth.

There is no tool for this today. The trades module (`trades/evaluate.py`) scores
one-for-one swaps in ROS roto points, not win% deltas, and does not touch future
picks at all.

## Goals

- A **CLI** that, given a player you send, the partner you send them to, and a
  nominal pick round, prints two clearly separated views: the this-year Monte
  Carlo impact and the next-year marginal pick value.
- **General and parameterized** -- any player, any partner, any round. Julio /
  SkeleThor / round 5 is just the example.
- The this-year half rides the **existing ROS Monte Carlo** so every number
  reconciles with the season dashboard and the injury stress-test.
- The next-year half is an **honest marginal estimate** in draft currency (VAR),
  not a fabricated next-year win%.
- Calculation logic lives in a **reusable library module** so a future dashboard
  tab can call the same functions without rework.

## Non-goals

- **No full 2027 league simulation.** We do not project every team's 2027
  keepers or simulate a 2027 draft. (Rejected in brainstorming: one mid-round
  pick barely moves a full-season win%, so the result would be dominated by
  modeling assumptions.)
- **No combined single "worth it" verdict.** The two halves stay separate; a
  one-line framing sentence is the only bridge.
- **No decomposition of the this-year cost** into "you-only" vs "rival gets
  stronger." The user chose the full-swing model; we report the full swing only.
- **No dashboard/web UI in this phase.** CLI only; the library is structured so a
  dashboard tab is a later, separate change.
- **No new projection sourcing.** We reuse the frozen 2026 draft-day board for
  the pick-value curve and the stored Upstash state for the MC. We do not fetch
  Yahoo or FanGraphs.

## Chosen approach

Two deliverables plus one small refactor:

1. **`src/fantasy_baseball/analysis/trade_pick.py`** -- pure library: scenario
   construction, the two MC runs, delta extraction, and the pick-value lookup.
2. **`scripts/trade_pick_calc.py`** -- thin CLI: arg parsing, load state, call
   the library, render an ASCII report.
3. **`src/fantasy_baseball/analysis/draft_value.py`** -- add a public
   `projected_par_curve(config=None) -> ParCurve` that reproduces the preseason
   par-curve construction standalone and returns it. `run_draft_value` is **left
   unmodified** -- no behavior-preserving refactor, so no regression risk to the
   draft-value metric that feeds the dashboard. The new function repeats the
   board / anchor / reconstruct / index / assign setup sequence; that duplication
   is deliberate (de-risking over DRY) and is guarded by a cross-check test (see
   Testing). A future DRY pass could extract a shared private setup helper.

Invocation:

```
python scripts/trade_pick_calc.py --send "Julio Rodriguez" --to "SkeleThor" --pick-round 5
```

Optional flags: `--pick-slot early|mid|late` (narrow within the drafted round),
`--player-type hitter|pitcher` (disambiguate two same-normalized-name roster
players of different types), `--iterations N` (MC iterations, default 2000),
`--seed N` (default 42).

## Component detail

### A. This-year half (real ROS Monte Carlo)

**Data load.** Reuse `injury_stress.load_mc_inputs_from_upstash()` -> `McInputs`.
It reads the stored (last-refresh vintage) Upstash blobs -- no Yahoo call -- and
carries `team_rosters` (`dict[str, list[Player]]`), `actual_standings`,
`fraction_remaining`, `h_slots`, `p_slots`, `eos_baseline`, `team_sds`, `denoms`,
`user_team_name`.

**Scenario construction** (`build_trade_scenario`):
- Locate the sent player on the user's roster by normalized name. If more than
  one roster player normalizes to the same name AND they are different player
  types, require `--player-type hitter|pitcher` to disambiguate; otherwise error.
- **User roster:** drop the sent player, then append one **replacement-level
  filler** at the sent player's position/volume (see "Replacement filler"
  below). Net: roster size is unchanged; the vacated active slot is filled by the
  best remaining eligible player, and the filler starts only if nothing better is
  available.
- **Partner roster:** append the intact sent Player (both lines real), then drop
  the partner's **worst-projected player of the sent player's type** -- ranked by
  lowest full-season projected value (the same `calculate_player_sgp` /
  `to_flat_dict_full_season` value used elsewhere) -- to keep the partner's
  roster size constant (symmetric with the user-side filler). The drop
  is second-order -- it removes a benched player, not an active one -- but the
  rule is stated so the model is not left to planner invention. The partner's
  active lineup re-optimizes to include the star, benching their next-worst
  active player of that type. The partner must be a real team in `team_rosters`
  (match by normalized name; error listing valid teams otherwise).

**Replacement filler.** A synthetic `Player` with a **distinct name**
(`"Replacement (<pos>)"`, so it never aliases the real sent player now on the
partner and reads clearly in the report), the sent player's **positions**, and
**both its ROS line and full-season line neutralized to replacement level** at
the sent player's position, scaled to the sent player's ROS volume (AB for
hitters, IP for pitchers). Built from the existing
`injury_stress._replacement_ros` (ROS line) and `simulation._replacement_line`
(full-season line) machinery. Both lines must be neutralized because the MC reads
the ROS-direct hitter path off the rebuilt `EffectiveRoster` (ROS line) but the
top-k / pitcher path off the flattened full-season line -- neutralizing only one
would leak the sent player's real production on the other path.

**Two MC runs** (`run_scenario` helper, one call per roster set):
- For each of {baseline rosters, scenario rosters}: rebuild
  `mc_roster.build_effective_rosters(team_rosters, inputs.eos_baseline,
  inputs.team_sds, inputs.fraction_remaining, denoms=inputs.denoms)`, then call
  `simulation.run_ros_monte_carlo(...)` with the same `actual_standings`,
  `fraction_remaining`, `h_slots`, `p_slots`, `user_team_name`, and the **same
  seed** (common random numbers) so the delta isolates the trade, not MC noise.
- **Defaults:** `n_iterations = 2000` (per-category `first_pct` is noisy at low
  counts; only two runs are needed, so a higher count than the dashboard's 1000
  is affordable and stabilizes the per-category deltas), `seed = 42` (matches
  `injury_stress.SEED`). Both overridable via `--iterations` / `--seed`.
- `eos_baseline` / `team_sds` are **held fixed** across both runs (reuse
  `McInputs`), mirroring the injury stress-test. First-order effect (Julio's
  production leaving Hart and joining the partner) flows through the rosters and
  the rebuilt effective rosters; only the second-order active-set-selection
  scaling uses the pre-trade baselines. Documented as a deliberate modeling
  choice.

**Deltas extracted** (`this_year_impact`):
- Overall: `team_results[user]["first_pct"]` (win%) and `["top3_pct"]`, baseline
  -> scenario.
- Per-category (all 10): from `category_risk[cat]` -- `first_pct` and
  `top3_pct`, baseline -> scenario. `category_risk` is already user-only, which
  is exactly what we want. (Per-category `median_pts` is available in the same
  dict but is intentionally **not** reported -- the per-category view is
  first%/top-3% only, matching the output table.)

### B. Next-year half (marginal pick value)

**Par curve.** Add `draft_value.projected_par_curve(config=None) -> ParCurve`
that runs the existing pure steps: `reproduce_draft_day_board` ->
`_frozen_var_by_player_id` -> `_anchor_board_var_to_frozen` ->
`reconstruct_draft` -> `_board_index` -> `_assign_pick_types` ->
`build_par_curve(typed_picks, bindex, fraction=1.0)`. This is preseason
(full-season, f=1) VAR, the right horizon for a future draft. It reads only local
files (projection CSVs, `player_positions.json`, `draft_state_board.json`,
`draft_state.json`, `league.yaml`) -- no Upstash. `run_draft_value` is **not
modified**; `projected_par_curve` re-runs the setup standalone. A cross-check
test (see Testing) asserts the helper's `drafted_pars` equal the par curve built
from the same `typed_picks`/`bindex` sequence, so the standalone path cannot
silently drift from the metric's construction.

**Nominal -> drafted-round -> ordinal mapping** (`pick_value`):
- `keeper_rounds = len(config.keepers) // config.num_teams` (30 // 10 = 3).
  Validate that `len(config.keepers)` is divisible by `num_teams`; if not, error
  (the "first K rounds are keepers" assumption does not hold).
- `drafted_round = nominal_round - keeper_rounds`. If `drafted_round < 1`, error:
  "Round N is a keeper round; drafted picks start at round {keeper_rounds+1}."
- Ordinal range for the drafted round (1-based, in the VAR-sorted par curve):
  `[(drafted_round - 1) * num_teams + 1, drafted_round * num_teams]`
  (round 2 -> ordinals 11..20). Clamp the upper bound to `len(drafted_pars)`; if
  the whole range is beyond the curve, error (round too deep to have par data).
- **Value = mean of `par.par_for_slot(k)` over the ordinal range.**
  `--pick-slot early|mid|late` narrows to the top / middle / bottom third of the
  range (early = lower ordinals = higher value, since the curve is sorted
  descending). Default is the full-round average.

**Reported** (`next_year_value`): the expected pick VAR, the keeper-average VAR
(`par.keeper_par`, NaN-guarded) and an early-drafted-round VAR for context, and a
one-line note that VAR is value above a replacement-level roster spot, so it is
also the pick's approximate marginal roto-point value next year.

### C. CLI + report

`scripts/trade_pick_calc.py`:
- Inject `src/` into `sys.path` (repo convention) and
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (player names may
  carry non-ASCII).
- Parse args, call the library, render an ASCII (cp1252-safe) report with two
  labeled sections and a one-line, **sign-aware** framing sentence keyed on the
  win% delta:
  - loss (delta < 0): "You give up ~X.X win% / ~Y.Y top-3% this year to gain a
    2027 R{nominal} pick (drafted round {drafted}) worth ~Z.Z VAR."
  - neutral/gain (delta >= 0): "This year is roughly neutral to positive (+X.X
    win% / +Y.Y top-3%), and you also gain a 2027 R{nominal} pick worth ~Z.Z
    VAR." (Signs shown explicitly for both metrics.)
- The per-category section is a table: category, baseline first%, scenario
  first%, delta, and the same for top-3%.

## Output shape (illustrative)

```
========================================================================
TRADE-FOR-PICK CALCULATOR
  Send: Julio Rodriguez  ->  SkeleThor
  For:  2027 Round 5 pick  (keeper rounds: 3  ->  drafted round 2)
========================================================================

1. THIS YEAR WITHOUT Julio Rodriguez  (full swing: he joins SkeleThor)
------------------------------------------------------------------------
  Win%   : 62.1%  ->  57.9%   (-4.2)
  Top-3% : 91.0%  ->  88.2%   (-2.8)

  Per-category odds (your team):
    Cat    1st% base  1st% new   d1st    top3 base  top3 new  dTop3
    R       ...
    HR      ...
    ...
------------------------------------------------------------------------

2. NEXT YEAR -- extra 2027 pick (drafted round 2)
------------------------------------------------------------------------
  Expected pick value : ~4.2 VAR
  (early in the round : ~5.1 VAR ; keeper-average keeper : ~18.4 VAR)
  VAR is value above a replacement roster spot, so ~4.2 VAR is roughly
  the pick's marginal roto-point value next year.
  Estimate is the 2026 draft-day value distribution at that slot; the
  specific 2027 player is unknown.
------------------------------------------------------------------------

You give up ~4.2 win% / ~2.8 top-3% this year to gain a pick worth ~4.2 VAR.
MC: n_iter=2000, seed=42 (common random numbers across both runs).
```

(The framing line above is the loss case; a non-negative delta uses the
neutral/gain wording from Component C.)

## Edge cases and failure modes

- Sent player not on the user's roster -> error naming the player.
- Ambiguous sent player (same normalized name, two types) -> require
  `--player-type`.
- Partner name not found -> error listing valid team names.
- Partner resolves to the user's own team (`--to` names your team) -> error (you
  cannot trade to yourself).
- Sent player is a pitcher -> filler is a pitcher (both lines neutralized);
  feasibility is judged against `p_slots`. Symmetric to the hitter case.
- Sent player has no ROS line (already out for the season) -> removing him has
  little effect; the filler's volume floors to 0. Handle without crashing.
- `nominal_round <= keeper_rounds` -> error (it names a keeper round, not a
  pick).
- `drafted_round` beyond the par curve -> clamp the ordinal range; if entirely
  beyond, error.
- `len(keepers)` not divisible by `num_teams` -> error (keeper-round assumption
  broken).
- `keeper_par` is NaN (no keeper matched the board) -> render as "n/a", never
  crash.
- Upstash missing blobs -> `load_mc_inputs_from_upstash` already raises a clear
  "run a refresh first" error; let it propagate.
- Frozen draft board missing/unreadable -> `_anchor_board_var_to_frozen` raises;
  surface the message (do not ship a pick value on the drifted rebuilt VAR).

## Modeling choices and caveats (stated in the report where load-bearing)

- **Full swing:** the sent player is added to the named partner, so the this-year
  cost includes a rival getting stronger. The magnitude of that component depends
  on the partner's own standing (a star to a non-contender barely moves the
  user's win%).
- **Fixed eos_baseline / team_sds:** controlled-comparison choice; first-order
  correct, second-order approximate (see A).
- **Roster sizes stay constant on both sides:** the user replaces the sent
  player with a replacement-level filler (models signing a waiver add; also
  restores bench injury-insurance depth, which very slightly reduces the measured
  downside vs playing a man down -- the intended behavior, not overstating the
  cost). The partner drops its worst-projected player of the sent player's type
  to fit the star. Both are second-order (they touch benched players), but keep
  the two teams symmetric and the model explicit.
- **Pick value from the 2026 curve:** assumes the 2027 value distribution at that
  slot resembles 2026's. Draft value-by-slot curves are stable year to year, but
  this is an estimate, flagged as such.

## Testing expectations

Unit (no network; synthetic fixtures + reuse of existing draft_value / injury
fixtures):
- Keeper-round -> drafted-ordinal math: nominal 5 / keepers 3 / teams 10 ->
  ordinals 11..20; nominal 4 -> 1..10; nominal 3 -> error; non-divisible keepers
  -> error.
- `pick_value` round-average and early/mid/late slicing over a synthetic
  `ParCurve`; upper-bound clamp when the round exceeds the curve.
- `projected_par_curve` cross-check: its `drafted_pars` equal a `ParCurve` built
  inline in the test from the same `typed_picks`/`bindex` (the guard against the
  standalone setup drifting from the metric's construction).
- Replacement filler is fully neutralized: assert the constructed filler has a
  distinct `"Replacement (...)"` name, keeps the sent player's positions, and its
  ROS **and** full-season lines equal the replacement lines, not the sent
  player's.
- Scenario construction: user roster loses the sent player and gains exactly one
  filler (size unchanged); partner roster gains the intact sent player and drops
  its worst-projected player of that type (size unchanged).
- Monotonic sanity (small synthetic league, common random numbers): removing a
  positive-value hitter from the user and giving him to a rival does not raise
  the user's win% by more than 1.0 pt above baseline (a fixed-seed tolerance
  band; the expected direction is a decrease).

Then the full end-of-effort gate: `pytest` (relevant subset -- name which),
`ruff check .`, `ruff format --check .`, `vulture`, and `mypy` if any touched
file is in `[tool.mypy].files`.

## Phasing

Single phase (one PR). Ordered tasks:
1. `draft_value.projected_par_curve` standalone helper (`run_draft_value` left
   unchanged) + cross-check test.
2. `analysis/trade_pick.py`: scenario construction + replacement filler; test.
3. `analysis/trade_pick.py`: two-run MC + delta extraction; test (sanity).
4. `analysis/trade_pick.py`: `pick_value` / next-year lookup; test.
5. `scripts/trade_pick_calc.py`: CLI + ASCII report.
6. End-of-effort verification gate.
