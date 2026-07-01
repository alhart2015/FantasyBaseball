# Draft-Value Metric — Design Spec

**Date:** 2026-07-01
**Status:** Approved design, hardened via spec-review; ready for implementation planning
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
  streamer) with the drafted/kept rules. VAR floors are empirical waiver lines
  (`REPLACEMENT_BY_POSITION`), so VAR = 0 genuinely is the waiver-streamer level.

### Single-scale requirement (CRITICAL — do not skip)

Preseason VAR, realized VAR, and estimate VAR **must all be computed on one
scale**, or `value`, `skill`, and `luck` are meaningless.

**Preseason VAR + par curve come from the FROZEN draft-day board, not a live
rebuild.** `VAR_preseason(player)` and the par curve MUST read the frozen
draft-day `var` column in `data/draft_state_board.json` (verified: a list of
~3669 rows, each `{name, player_id=fg_id::player_type, positions, var, adp,
player_type}`). Re-running `build_draft_board(conn)` mid-season would read a
**mutated** in-season positions table (players gain eligibility) and a possibly
newer projections snapshot, producing different VARs than the ones the draft was
actually made against — silently drifting the par curve, `skill`, and every slot
expectation. The frozen JSON is the authoritative draft-day vintage; use it.

**Realized/estimate VAR must reuse the board's exact scale inputs.**
`build_draft_board` computes SGP twice — the second pass uses **pool-derived rate
baselines** (`calculate_replacement_rates(pool, starters)` in
`draft/board.py:61-72`) for AVG/ERA/WHIP, plus `team_ab`/`team_ip` and the
`sgp_denominators`, and derives VAR against `position_aware_replacement_levels`
built on *those same* rates (`board.py:74`). The frozen JSON stores only `var`
(not the floors, rates, denominators, or team volumes), so the module MUST obtain,
and reuse for every realized/estimate line:

1. the same `sgp_denominators` the board used (note: `get_sgp_denominators()`
   returns code defaults, NOT `league.yaml` — pin whichever the board build used),
2. the same pool-derived replacement **rates** (`replacement_avg`,
   `replacement_era`, `replacement_whip`) from `calculate_replacement_rates`,
3. the same `team_ab` / `team_ip`,
4. the same per-position VAR floors from `position_aware_replacement_levels`.

**Mechanism (preferred): extend `build_draft_board` to also return these four
inputs**, and run it against the **draft-day projection + positions vintage**
(preseason `data/projections` files; not the live in-season tables). A
hand-rebuild "from the same pool + config" is discouraged because it must
replicate the board's **two-pass ordering** exactly (first pass with default rates
sets `total_sgp`, which selects the replacement band at `replacement.py:112-114`;
the second pass overwrites `total_sgp` with pool-derived rates — `board.py:58-72`).
A single-pass rebuild picks a different band and silently drifts the scale.

**Validation gate (single-scale oracle):** the reproduced board's `var` MUST
reproduce the frozen `draft_state_board.json` `var` column to within float
tolerance for every joined player. This both proves the reused floors/rates are
draft-day-consistent and catches any projection/positions vintage mismatch loudly.
Calling `calculate_player_sgp` with module defaults (0.250 / 4.50 / 1.35) instead
of the reused inputs is a silent scale bug and is prohibited.

## Core metric

For a player **with a preseason board VAR** (drafted or kept — i.e. on the
preseason board), on the **projected full-season horizon**:

```
value = VAR(estimate) - par(slot)
skill = VAR_preseason(player) - par(slot)      # knowable on draft day ("reach value")
luck  = VAR(estimate) - VAR_preseason(player)  # in-season over/under-performance
```

`value = skill + luck` by construction. Reporting the decomposition separates
draft *skill* (getting a player projected above the slot's par) from in-season
*luck* (the player beating or missing his own projection).

**Scope limits of the decomposition (do not violate):**

- **Projected horizon only.** `skill`/`luck` are defined **only** for the
  projected full-season value. The YTD horizon (below) reports **`value` only** —
  there is no coherent skill/luck split against a fraction-scaled par, so YTD does
  not attempt one.
- **Requires a preseason VAR.** For any player with no preseason board entry
  (undrafted waiver gems; also drafted "fliers" below the board's projection
  thresholds, see Par curve), `VAR_preseason` is undefined, so `skill` and `luck`
  are reported as **N/A** and only `value` is computed. This is exactly the
  waiver-gem population, so the report must render N/A cleanly (never NaN, never a
  dropped row).

## Two time-horizon estimates (both computed, side-by-side)

Mid-season, realized stats are partial while projections are full-season. We
report two horizons:

- **Projected full-season value (primary)** =
  `VAR(actual + ROS) - par_full(slot)`.
  The full-season estimate line (actual-to-date counting stats + rest-of-season
  projected counting stats, with rates recomputed from the combined line)
  **already exists** as `derive_full_season` -> `CacheKey.FULL_SEASON_PROJECTIONS`
  (`data/ros_pipeline.py`, `derive_full_season` ~line 140). Reuse that cached line (MLBAM-keyed);
  do not re-derive it. Answers "projected final draft value." This is the headline
  horizon because it is not distorted by the availability bias below.
- **YTD value (secondary)** = `VAR(actual-to-date) - par_to_date(slot)`.
  Answers "what has been delivered vs expected so far." See scaling rules below.

At true season end the two converge (ROS -> 0). The convergence is an explicit
test oracle (see Validation).

### YTD to-date scaling — exact rules (the fiddliest math)

Let `f` = season fraction elapsed (see below). **One rule applied consistently to
every line — player, floor, and par — computed through the same to-date SGP
path:** multiply the **counting stats `calculate_player_sgp` actually consumes**
by `f` (`r, hr, rbi, sb` for hitters; `w, k, sv` for pitchers; plus the volume
`ab`, `ip`) and leave **rate stats unchanged** (`avg, era, whip` are NOT scaled —
a 0.125 AVG is nonsense). Note `calculate_player_sgp` reads `avg/era/whip`
directly and does not consume `h/er/bb/h_allowed`, so only the counting stats
above and the volumes need scaling. Then:

- `team_ab` / `team_ip` passed to `calculate_player_sgp` **must also be scaled by
  `f`** so the marginal rate-SGP (which is `player_ab/team_ab`-weighted,
  `player_value.py:21-45`) stays proportionally correct. Under this scaling a
  player's rate-SGP is `f`-invariant (numerator and denominator both carry `f`),
  while counting-SGP scales by `f`.
- **Position VAR floors for YTD are recomputed through the SAME to-date path** —
  scale the empirical replacement line's counting stats and its `team_ab`/`team_ip`
  by `f`, hold its rates, and run `calculate_player_sgp`. Do **NOT** use
  `f * floor_full`: floor SGP is **not** linear in `f`, because its rate component
  (ERA/WHIP/AVG marginal) is `f`-invariant while only its counting component
  scales — `f * floor_full` wrongly shrinks the rate component and mis-scales the
  floor by up to ~0.3-0.4 SGP for pitchers at `f=0.5`. This requires the module to
  reach the empirical replacement *lines* (from `REPLACEMENT_BY_POSITION` /
  `position_aware_replacement_levels`), not just the returned floor SGP; the plan
  must expose them. Do **not** fraction-scale the internal replacement *rates*
  (0.250 etc.).
- Both the **actual-to-date** side and the **expected-to-date** side use the
  **same scaled `team_ab`/`team_ip` (`f * full`)** and the same to-date floors, so
  a player's actual and its slot's par live on one scale. The actual side scales
  its `team_ab`/`team_ip` by `f` too (it is not "unscaled" — only the player's real
  accumulated counting stats are used as-is; the team denominators still scale).
- `par_to_date(slot)` = the drafted par curve rebuilt from **expected-to-date**
  VAR (each drafted player's frozen-board preseason line scaled by `f` through the
  path above, VAR recomputed with to-date floors), sorted descending.

Rate-category proration is inherently approximate (SGP rate value is defined
against a full-season team volume), which is a further reason YTD is the
**secondary** horizon. The plan must add an `f<1` oracle (see Validation) since
convergence at `f=1` cannot catch a partial-season floor/volume error.

### Season fraction `f` and its known bias

`f` = league games played / full schedule (a single league-wide fraction for v1;
exact source — standings snapshot game count vs 162, or a date-based fraction —
pinned in the plan). **Known bias, stated explicitly:** scaling the *expected*
side by a league-wide `f` while the *actual* side reflects a player's real
accumulation means an injured/part-time player who played 20% of games while the
league is at 50% is scored against a 50%-scaled par, so availability loss reads as
underperformance. This conflates availability with "delivery." It is why YTD is
**secondary** to the projected horizon (which absorbs availability via ROS).
Per-player fractions are a named future refinement, out of scope for v1.

## The par curve and keeper par

- **Drafted par curve:** take the drafted (non-keeper) players **that have a
  preseason board VAR**, sort by preseason VAR descending; `par(i-th on-board
  drafted pick)` = the i-th highest preseason VAR. This is "the value that should
  have been available at that point in the draft if everyone drafted optimally by
  projection." A pick's **slot** is its ordinal position in draft order **among
  on-board drafted picks**.
  - Drafted "fliers" below the board's projection thresholds
    (`ab < 50` / `ip < 10`, `board.py:46-48`) or otherwise absent from the board
    have **no preseason VAR**: they are excluded from the sorted par curve (so the
    curve has fewer than the ~200 total non-keeper picks) and, if they later have
    an estimate, are credited **value-only** (skill/luck N/A). The curve length is
    "on-board drafted picks," not a fixed 200.
- **Keeper par:** a single flat value = **mean preseason VAR of all 30 keepers**,
  a **fixed preseason reference** computed once and independent of any later moves.
  Keepers carry no recorded cost/round in this league (just name + team, 3 per
  team, verified in `config/league.yaml`), so per-keeper slots are not derivable.
  Keeper value = `VAR(estimate) - mean(keeper VAR)`. A keeper later traded away is
  simply uncredited (see Attribution) but **remains in the fixed 30-keeper mean**;
  the reference does not shift, which keeps every keeper measured against the same
  bar. Consequence: a team that traded a keeper is evaluated on fewer than 3
  keepers — accepted for v1.

## Attribution — per-team roll-up (elimination model)

There is **no trade transaction feed** in this codebase: `fetch_all_transactions`
queries Yahoo with a hardcoded `"add,drop"` type (`yahoo_roster.py:438`), so trade
acquisitions are never ingested. We therefore classify by **elimination** using
the league-wide add/drop feed (which carries per-add `destination_team_name`), the
2026 draft results, and the `league.yaml` keepers list.

Iterate over each team's **current** roster. For each rostered player, classify by
this precedence (first match wins):

1. **Drafted or kept by this team** (present in this team's 2026 draft/keeper set)
   -> credit at `par(draft slot)` (drafted) or `mean keeper VAR` (kept).
   **This takes precedence over any later same-team waiver re-add**, so a
   drafted -> dropped -> re-added-by-same-team player is still judged against his
   draft slot, not baseline 0.
2. **Has an "add" transaction by this team** (waiver/FA pickup) and is NOT in this
   team's draft/keep set -> credit at **0** (replacement) -> value = its VAR.
3. **On the roster, not in this team's draft/keep set, and no "add" txn by this
   team** -> **trade-acquired -> excluded** (a trade leaves no add record, so
   "rostered but otherwise unexplained" is the trade signal).

Players **not on any current roster** (dropped, or traded away) are **excluded**
entirely. Every current-roster player must fall into exactly one of the three
cases above — the classifier must assert this and log any unclassifiable player
rather than silently dropping it. **The classifier must also report the count of
case-3 (trade-excluded) players per team**: because case 3 is a residual bucket, a
transaction-name-join regression (a real waiver add whose name fails to match the
add feed) silently lands in case 3, and a case-3 count that is implausibly high is
the eyeball signal for that regression.

### Known classifier false-positives / edge cases (bounded, stated)

- **Totality is not correctness.** The exactly-one-bucket assertion guarantees a
  player lands *somewhere*, not in the *right* bucket. A normalized-name miss on
  the add feed pushes a genuine waiver add into case 3 (silently excluded). Hence
  the case-3 count safeguard above.
- **Stale add + later trade-reacquire (same team).** A player added off waivers by
  a team, dropped, then re-acquired by that team **via trade** still carries the
  old "add" txn, so case 2 credits waiver value for a stint that was actually a
  trade. Distinguishing this needs ownership-period tracking (out of scope), so it
  is an accepted bounded false-positive.
- **Commissioner / non-add-drop moves.** The feed is `add,drop` only; a
  commissioner-forced roster placement (Yahoo `commish` type) yields a
  currently-rostered player with no add txn -> case 3 -> excluded. Rare in this
  league; surfaced by the case-3 count if it happens.
- **Trade round-trip back to the drafter** (drafted by A -> traded to B -> traded
  back to A): case-1 precedence credits A at the draft slot. Defensible (A did
  draft him and currently rosters him); noted for completeness.

### Accepted attribution limitations (stated, not hidden)

- **Dropped busts are forgiven.** Because crediting is over the *current* roster
  (the user's explicit choice: dropped players excluded), a bad pick that a team
  dropped is charged to nobody — its slot par is not booked as a loss. This
  under-penalizes exactly the bad drafting the metric measures. Accepted for v1;
  a future variant could book dropped-pick par against the drafter.
- **Drafted-then-traded-away value vanishes.** A player a team drafted well and
  then traded is credited to no one (excluded on both sides). Accepted consequence
  of the current-roster model.

## Team roll-up — two numbers

- **Sum** of credited player values (raw total).
- **Per-player average** over the credited set — the **headline** number, plus the
  **credited-player count** displayed alongside it for transparency.

**Stated caveats (open / iterative for v1):**

- *Mixed baselines.* Drafted/kept players are scored against a (often positive)
  par; waiver adds are scored against 0. The average blends `(VAR - par)` and
  `(VAR - 0)` terms — two different zero-points in one headline number.
- *Small-set instability.* A trade-heavy team may have very few *credited* players
  (trades excluded on both sides), so its per-player average is computed over a
  small, high-variance set and is not directly comparable to a stand-pat team's
  average over ~23 players. Displaying the credited count makes this visible.

These are known limitations; the roll-up will be refined after seeing real output.

## Data sources (existing — mostly KV-store blobs, not `data/` files)

Most inputs are KV-store blobs (Upstash on Render, SQLite locally), **not** flat
`data/` files, so a local CLI run needs a live or freshly-synced store. Per repo
memory, `run_season_dashboard.py` can clobber local state via Upstash sync — the
plan must specify how the CLI obtains a consistent snapshot (e.g. `--no-sync` or
an explicit read path).

- **Preseason VAR + par curve:** the **frozen** `data/draft_state_board.json`
  (`var`, `player_id=fg_id::player_type`, `positions`) — the draft-day vintage.
- **Reused scale inputs (floors/rates/denoms/team volumes):** a `build_draft_board`
  run against the **draft-day projection + positions vintage**, validated to
  reproduce the frozen `var` (see Single-scale requirement) — NOT a live in-season
  `build_draft_board(conn)`.
- **2026 pick-by-pick (team + overall slot):** reconstructed from
  `data/draft_state.json` `drafted_players` + `config/draft_order.json`
  (snake order + `trades`) + `config/league.yaml` keepers. See invariant below.
  (2026 results are not yet in `data/history/draft_results.json`.)
- **Actual-to-date stats:** `_load_game_log_totals()` (`game_log_totals:*` cache).
- **Full-season estimate (actual + ROS):** `cache:full_season_projections` via
  `derive_full_season` (reuse, do not recompute).
- **Current rosters:** latest `weekly_rosters` snapshot (KV / `weekly_rosters`
  table). The plan pins the accessor and the "latest snapshot" selection.
- **Transactions:** league-wide add/drop feed (`analysis/transactions.py` /
  `fetch_all_transactions`), per-team via `destination_team_name`.

### Draft-slot reconstruction invariant (must be validated at runtime)

Reconstruction depends on an unstated, brittle invariant that the plan must
document AND assert:

- `data/draft_state.json` `drafted_players` is a flat list of **bare name
  strings**, length 230, `current_pick = 231` (draft complete).
- Indices **0-29** are the 30 keepers in `league.yaml` grouping order; indices
  **30-229** are the 200 real picks in true snake draft order, with pick trades
  from `draft_order.json` applied to get the team-at-each-slot mapping.
- **Validation gate:** after reconstruction, assert (a) exactly 30 keepers match
  `league.yaml` 30/30 after `normalize_name`, (b) the reconstructed per-team
  rosters match a known ground-truth roster (e.g. the user's own team) exactly,
  and (c) every drafted name resolves to a board row or is explicitly logged as an
  off-board flier. Any autopick/undo/out-of-order entry or a keeper block that is
  not exactly the first 30 will misalign slots and MUST fail this gate loudly.

## Cross-source identity joins (name-normalization + type resolution, not id)

The id spaces do **not** line up, so joins are lossy fuzzy matches, not id
matches:

- Projections + game-log totals share a robust **MLBAM id**
  (`ros_pipeline.py`), and `full_season_projections` is MLBAM-keyed.
- The board's `player_id` is `fg_id::player_type` when fg_ids are present, else
  `name::player_type` (`board.py:87-90`).
- **`weekly_rosters` and the transaction feed carry only a Yahoo display name**
  (plus a `yahoo_id` in a *different* id space that is useless as a cross key) and
  Yahoo names may carry `" (Batter)/(Pitcher)"` suffixes that must be stripped.
  These have **no `player_type`**.

Join strategy: strip Yahoo suffixes, `normalize_name` (accent/case), resolve
hitter-vs-pitcher type (from position/eligibility or the game-log hitter/pitcher
split), and key on `name::player_type` with **VAR tie-break on normalized-name
collisions** (repo convention). The namesake collision path is the known
silent-data-loss risk and must be logged, not swallowed.

### Two-way players (Ohtani)

`league.yaml` marks Ohtani "batter only." Game-log totals split hitter/pitcher by
name. The plan must specify that Ohtani is resolved to the **hitter** line for
this league and that the keeper "batter only" note is honored, avoiding a
hitter/pitcher type collision on the normalized name.

## Structure

- **Library module** (e.g. `src/fantasy_baseball/analysis/draft_value.py`) built
  from small, independently testable units:
  - par-curve builder (drafted curve + keeper mean),
  - per-player value calculator (value for both horizons; skill/luck for the
    projected horizon where a preseason VAR exists),
  - acquisition classifier (drafted / kept / waiver / trade-excluded / dropped),
  - team roll-up (sum + per-player average + credited count).
- **CLI script** `scripts/draft_value.py`: prints a per-player table (slot, par,
  preseason VAR, estimate VAR, skill, luck, value) and a per-team leaderboard,
  with YTD and projected columns, and writes a markdown report artifact.
- Output surface for v1 is the CLI + markdown; a season-dashboard page is deferred.

## Validation / acceptance criteria (test oracles)

The metric is correctness-critical and "who drafted well" has no external ground
truth, so the plan MUST include these testable oracles:

1. **Frozen-board single-scale check:** the reproduced draft-day board's `var`
   reproduces the frozen `data/draft_state_board.json` `var` column within float
   tolerance for every joined player. Proves the reused floors/rates/denoms/volumes
   are draft-day-consistent AND that `VAR_preseason` matches the frozen source.
2. **Estimate-scale check:** for an on-board player with actual + ROS == preseason
   projection (synthetic fixture), `VAR(estimate)` reproduces `VAR_preseason`
   within tolerance. Guards the realized/estimate scale bug on the projected path.
3. **Convergence (f=1):** with `f` -> 1 and ROS -> 0 (synthetic end-of-season
   fixture), YTD value and projected value converge for every player.
4. **YTD partial-season correctness (`f<1`):** with a synthetic *healthy* player
   whose actual-to-date == `f *` full projection exactly (linear accumulation,
   rates == projected) at `f=0.5`, YTD `VAR(actual-to-date)` and `par_to_date`
   reproduce the `f`-consistent VAR — specifically it catches a `f * floor_full`
   floor that would mis-scale the rate component. Convergence at `f=1` cannot see
   this, so this oracle is mandatory, not optional.
5. **Decomposition identity:** for on-board players on the projected horizon,
   `skill + luck == value` exactly (float tolerance).
6. **Slot-reconstruction gate:** the three assertions in the invariant section
   (30/30 keeper match, known-team roster match, all drafted names resolved/logged)
   pass on the real 2026 data.
7. **Classifier totality + case-3 sanity:** every current-roster player classifies
   into exactly one bucket; no player is silently dropped; unclassifiable players
   raise/log; and the per-team case-3 (trade-excluded) count is within a plausible
   band (a spike flags a transaction-join regression).
8. **Known-pick sanity:** at least one hand-verified player (e.g. a specific
   keeper) has its VAR, par, and value checked against a manual computation.

## Out of scope for v1

- deltaRoto (roster-fit) secondary lens — SGP/VAR is the only axis for v1.
- Season-dashboard UI page.
- Per-player season fractions (league-wide fraction only).
- Ownership-period-weighted attribution (current-roster + acquisition-mode
  crediting only).
- Booking dropped-pick par against the drafter (a possible future variant).
- Ingesting trade transactions (elimination model avoids needing them).

## Non-goals / correctness guardrails

- ASCII-only in all code, report renderers, and print strings (Windows cp1252
  stdout).
- Do not use `x or default` for numeric defaults (VAR/SGP can be 0.0 or negative);
  use explicit `is not None` checks, especially in sort keys and the par-curve
  index lookups.
- Reuse existing SGP / VAR / replacement / board / full-season / transaction
  machinery; do not reimplement SGP, replacement, or full-season-blend math.
