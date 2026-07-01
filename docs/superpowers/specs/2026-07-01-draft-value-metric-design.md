# Draft-Value Metric — Design Spec

**Date:** 2026-07-01
**Status:** Approved design, ready for implementation planning
**TODO source:** "Draft-value metric: realized contribution vs. draft-slot expectation (per player + per team)"

## Purpose

Measure how much each drafted / kept / waiver-acquired player has actually
contributed relative to what a player at their draft slot was *expected* to
contribute, then roll it up per team to answer **"who drafted well."**

This produces real decision-support numbers (evaluating draft strategy year over
year), so correctness matters: a plausible-but-wrong leaderboard misleads future
draft strategy. Every value must trace back to the actual board VAR, actual draft
slot, and actual stats.

## Value currency: VAR (SGP above positional replacement)

All values are in **VAR** — SGP minus the positional replacement floor, the same
metric that orders the draft board (`build_draft_board`). VAR is chosen over raw
SGP because:

- It is how the draft is actually ordered, so a par curve built from it reflects
  real pick value.
- It handles positional scarcity: a catcher/SS taken early is not wrongly flagged
  a bust against a raw-SGP par.
- **Replacement level is the natural zero**, which unifies the waiver-add rule
  (a waiver pickup's value = its VAR, i.e. value above a replacement-level
  streamer) with the drafted/kept rules.

Realized and estimate VAR **must** use the *same* positional replacement floors
as the preseason board (`sgp/replacement.py` `position_aware_replacement_levels`),
or the "expected" and "realized" numbers will not live on one scale.

## Core metric

For every credited player, in VAR units:

```
value = VAR(estimate) - par(slot)
skill = VAR_preseason(player) - par(slot)      # knowable on draft day ("reach value")
luck  = VAR(estimate) - VAR_preseason(player)  # in-season over/under-performance
```

`value = skill + luck` by construction. Reporting the decomposition separates
draft *skill* (getting a player projected above the slot's par) from in-season
*luck* (the player beating or missing his own projection).

## Two time-horizon estimates (both computed, side-by-side)

Mid-season, realized stats are partial while projections are full-season. We
report two horizons:

- **YTD value** = `VAR(actual-to-date) - par_to_date(slot)`.
  The expected side is scaled to the elapsed season fraction: counting stats
  `* fraction`, rate stats unchanged, re-run through `calculate_player_sgp`;
  replacement floors scaled the same way. Answers "what has been delivered vs
  expected so far."
- **Projected full-season value** = `VAR(actual + ROS) - par_full(slot)`.
  Full-season estimate = actual-to-date counting stats + rest-of-season projected
  counting stats (rates recomputed from the combined line), reusing the ROS
  projections the refresh already produces. Answers "projected final draft value."

At true season end the two converge (ROS -> 0). YTD is the honest delivered
scorecard; projected is the best current estimate of final value.

### Season fraction (YTD scaling)

v1 uses a single league-wide fraction (league games played / full schedule, or an
equivalent date-based fraction). Per-player fractions (each player's team games
played) are a possible later refinement. The exact fraction source is settled in
the implementation plan.

## The par curve and keeper par

- **Drafted par curve:** take the ~200 actually-drafted (non-keeper) players,
  sort by **preseason VAR** descending; `par(i-th drafted pick)` = the i-th
  highest preseason VAR. This is "the value that should have been available at
  that point in the draft if everyone drafted optimally by projection." A pick's
  slot is its ordinal position in draft order among the drafted (non-keeper)
  picks.
- **Keeper par:** a single flat value = **mean preseason VAR of all 30 keepers**.
  Keepers carry no recorded cost/round in this league (just name + team, 3 per
  team), so per-keeper slots are not derivable. A flat keeper-pool mean answers
  "within the elite kept pool, did you keep the better-than-average players, and
  did they deliver." Keeper value = `VAR(estimate) - mean(keeper VAR)`.

## Attribution — per-team roll-up

Iterate over each team's **current** roster. Credit each rostered player by **how
it was acquired**:

| Acquisition (still on current roster) | Baseline (par) | Rationale |
|---|---|---|
| Drafted by this team | `par(draft slot)` | the pick decision |
| Kept by this team | mean keeper VAR | the keep decision |
| Added off waivers by this team | **0** (replacement) -> value = its VAR | finding a gem off waivers = big value |
| Trade-acquired | *excluded* | not a draft decision |
| Dropped / traded away (not on current roster) | *excluded* | — |

Acquisition mode is derived from the league-wide typed transaction feed
(add / drop / trade, per team, from the `analysis/transactions.py` pipeline)
combined with the 2026 draft results and the `league.yaml` keepers list. The
signal is the player's **most recent acquisition transaction by the current
team**: a trade -> exclude; an add (waiver/FA) -> replacement baseline; no
transaction + present in this team's draft/keeper set -> drafted/kept.

## Team roll-up — two numbers

- **Sum** of credited player values (raw total).
- **Per-player average** over the credited set — the **headline** number. A
  churn-heavy team accumulating many small waiver values would otherwise be judged
  on a different scale than a stand-pat team, so the average normalizes for
  transaction volume.

**Open / iterative:** the per-player average has a known tension — heavy waiver
churn with low-but-positive-VAR adds can *dilute* a strong drafted core's average.
This is accepted for v1; we will refine the roll-up after seeing real output.

## Data sources (all existing)

- **Preseason board:** `build_draft_board(conn)` -> per-player VAR, `total_sgp`,
  positions, ids, and the replacement floors to reuse.
- **2026 pick-by-pick (team + overall slot):** reconstructed from
  `data/draft_state.json` (drafted order + teams dict) plus
  `config/draft_order.json` (snake order + pick trades) plus `config/league.yaml`
  keepers. **Note:** 2026 results are not yet in `data/history/draft_results.json`;
  a reconstruction step is required.
- **Actual-to-date stats:** `_load_game_log_totals()` (game-log totals cache) ->
  `calculate_player_sgp`.
- **ROS projected stat lines:** from the refresh's ROS projection output
  (`data/ros_pipeline.py` and the cached ROS projections) — stat lines, not just
  ranks.
- **Current rosters:** the latest `weekly_rosters` snapshot (team -> players).
- **Transactions:** the league-wide typed transaction feed.

## Structure

- **Library module** (e.g. `src/fantasy_baseball/analysis/draft_value.py`) built
  from small, independently testable units:
  - par-curve builder (drafted curve + keeper mean),
  - per-player value calculator (skill / luck / value, for both horizons),
  - acquisition classifier (drafted / kept / waiver / trade / dropped),
  - team roll-up (sum + per-player average).
- **CLI script** `scripts/draft_value.py`: prints a per-player table (slot, par,
  preseason VAR, estimate VAR, skill, luck, value) and a per-team leaderboard,
  with YTD and projected columns, and writes a markdown report artifact.
- **Cross-source joins** use `name::player_type` / normalized names with VAR
  tie-break, per repo convention.
- Output surface for v1 is the CLI + markdown; a season-dashboard page is
  deferred to a later pass.

## Known risks to resolve in the implementation plan

1. **2026 draft-slot reconstruction** (team + overall pick) from
   `draft_state.json` / `draft_order.json` / `league.yaml` — the biggest unknown;
   the drafted-order-to-team-and-slot mapping must be verified against known picks.
2. **YTD to-date scaling** of rate-stat SGP and replacement floors — the fiddliest
   math; scaling counting stats and volume (AB/IP) by fraction while keeping rates,
   then recomputing marginal rate SGP.
3. **Cross-source name joins** (board <-> game logs <-> rosters <-> transactions),
   where case/accent/namesake mismatches are a known source of silent data loss.

## Out of scope for v1

- deltaRoto (roster-fit) as a secondary lens — SGP/VAR is the primary and only
  axis for v1.
- Season-dashboard UI page.
- Per-player season fractions (league-wide fraction only for v1).
- Ownership-period-weighted attribution (we use current-roster + acquisition-mode
  crediting, not day-by-day ownership splits).

## Non-goals / correctness guardrails

- ASCII-only in all code, report renderers, and print strings (Windows cp1252
  stdout).
- Do not use `x or default` for numeric defaults (VAR/SGP can be 0.0 or negative);
  use explicit `is not None` checks, especially in sort keys and the par-curve
  index lookups.
- Reuse existing SGP / VAR / replacement / board / transaction machinery; do not
  reimplement SGP or replacement math.
