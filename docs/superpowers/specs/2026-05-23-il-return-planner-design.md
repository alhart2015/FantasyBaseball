# IL Return Planner -- Design

- Date: 2026-05-23
- Status: Approved (pending spec review)
- Surface: "IL Returns" section on the season dashboard `/roster-audit` page
- New module: `src/fantasy_baseball/lineup/il_return_planner.py`

## Problem

When IL players come off the IL, the roster temporarily exceeds the body-count
cap, forcing a drop plus a re-shuffle of active vs. bench. The current tooling
excludes IL players from the optimizer and the roster audit entirely (they
appear as inert `slot="IL"` rows), so it gives no help with the actual decision.

Concrete case driving this: Webb (P) is on the bench with IL status; Hader (P)
is in an IL slot. Both return next week. To get both into active P slots the
manager must drop one player (hitter or pitcher) and bench one pitcher. The tool
must answer: **who to drop, and who to bench.**

## Goals

- Let the manager pick which IL players are "coming back" (checkboxes), and
  compute the resulting optimal legal roster.
- Output a concrete transaction plan: who to drop, who to start, who to bench --
  rendered as a move list (`{name, from_slot, to_slot}`).
- Show the **top 5** plans ranked by deltaRoto, each with deltaRoto, standard
  deviation, and P(helps), consistent with the rest of the dashboard (PR #95).
- Reuse the existing optimizer, displacement model, and band primitives; avoid
  duplicating standings/scoring logic.

## Non-goals

- No free-agent pickups in the plan (rejected "move plan + FA sweep" option).
  The moves are internal to owned + returning players only.
- No return-date modeling. The plan is a "assume these checked players are
  active now" snapshot; the manager decides when to execute. "Next week" is just
  the manager's cue to open the section.
- No bench-insurance / injury-replacement value pricing for the dropped player.

## Key insight: standings already price IL returns (avoid double-count)

`ProjectedStandings.from_rosters` and `build_standings_breakdown_payload` build
the team's projected row from `project_team_stats(roster, displacement=True,
league_context=...)` over the **full roster including IL players**. The pool-slot
pitcher model (`scoring.py`) keeps the top-N arms by team-ROTO (N = active P
slots) and zeroes the excess, *regardless of IL status* -- the docstring states
"Hader returning bumps the worst arm overall."

Consequence: if Webb/Hader are top-9 arms, the dashboard's projected standings
**already assume they are effectively pitching**, with the worst current arms
already zeroed. The "gain" from activating them is largely already banked.

Therefore:

- We must NOT score plans as a naive `before = current active` /
  `after = +Webb +Hader` band -- that double-counts the activation gain the
  standings already include. (This is the documented double-count trap class
  the project has been bitten by before.)
- The real cost of the transaction is the **forced DROP** (a body leaves the
  roster entirely) plus realizing a legal lineup. The objective is therefore
  "least-damaging forced drop."
- The active-vs-bench split is mostly already computed by the displacement model
  (`sf=0` pitchers are exactly "who to bench"). `compute_roster_breakdown()`
  tags each player `ACTIVE` / `DISPLACED` / `IL_FULL`, so the bench side of the
  move list can be read off the post-drop roster rather than re-derived.

The genuinely new logic is narrow: enforce the body cap, rank the forced drop,
and render the result as a transaction list.

## Body-count cap is slot-based, not status-based

Yahoo exempts only players in true IL slots from the active-roster size limit.
A player on the bench with IL status (Webb) still counts; a player in an IL slot
(Hader) does not. So the capacity math keys on `IL_SLOTS` occupancy, NOT on
`Player.is_on_il()` (which also returns True for BN+IL-status players and would
miscount Webb).

- `capacity` = sum of `roster_slots[s]` for every slot `s` not in `IL_SLOTS`.
  For the current league config (C,1B,2B,3B,SS,IF,OF*4,UTIL*2 = 12 hitter slots;
  P*9; BN*2; IL*2 excluded) this is **23**.
- `counted_bodies` = roster players whose `selected_position` is not in
  `IL_SLOTS` (Webb counts via BN; Hader does not).
- Activating Hader moves him out of his IL slot, so he now counts:
  `counted = counted_bodies UNION {checked players currently in IL slots}`.
- `overflow = len(counted) - capacity` = number of forced drops. In the
  Webb/Hader case, overflow = 1.

## Architecture & data flow

- New module `lineup/il_return_planner.py`, entry point:

  ```
  plan_il_returns(
      roster: list[Player],
      activating_il: list[Player],          # the checked IL players
      roster_slots: dict[str, int],
      *,
      projected_standings: ProjectedStandings,
      team_name: str,
      fraction_remaining: float,
      team_sds: Mapping[str, Mapping[Category, float]] | None,
      max_plans: int = 5,
  ) -> IlReturnPlanResult
  ```

- On-demand JSON route (e.g. `GET /il-return-plan?activate=<player_ids>`) on the
  season app. It reloads the same cached inputs the trade-builder route already
  reconstructs (roster Players, ProjectedStandings, team_sds, fraction_remaining,
  roster_slots), calls `plan_il_returns`, and returns the result as JSON. The
  `/roster-audit` page calls it on checkbox change. This keeps the optimizer off
  the refresh pipeline's critical path and mirrors the existing
  `_optimize_one_side` route precedent, so the inputs are known-reconstructable.
- The `/roster-audit` template gains an "IL Returns" section: lists the team's
  IL players with checkboxes (pre-checked by status heuristic is out of scope for
  v1 -- start unchecked or all-checked; see open questions), and a results panel
  rendering the ranked plans.

## Algorithm

1. **Normalize activating players.** For each checked IL player, clear IL signals
   so the optimizer/displacement treat them as active-eligible:
   `dataclasses.replace(p, status="", selected_position=None)`. Otherwise
   `is_on_il()` and the optimizers would keep excluding them.
2. **Build the pool.** `pool` = (roster players not in `IL_SLOTS`, with any BN+IL
   status cleared on the checked ones) + (normalized checked players currently in
   IL slots). Unchecked IL players are dropped from consideration (stay parked).
3. **Compute overflow** (slot-based, per above). If `overflow <= 0`, there is no
   forced drop -- emit a single plan that is just the activation + bench
   reshuffle (still run the optimizer to assign slots).
4. **Enumerate drop-sets** of size `overflow` from `pool`. For `overflow == 1`
   this is `len(pool)` candidates (~24); for `overflow == 2`, `C(len(pool), 2)`
   (~250). If `overflow >= 3` (rare), pre-filter to the bottom-K bodies by raw
   SGP before enumerating to bound the cost.
5. **Score each surviving roster (displacement-aware).** For each drop-set:
   - `survivors = pool - dropset`.
   - Skip if infeasible: the survivors cannot fill every mandatory hitter slot
     (use `can_cover_slots` / optimizer returns empty).
   - Re-run `optimize_hitter_lineup` + `optimize_pitcher_lineup` over survivors to
     get the active assignment and bench. (Perf: a pitcher-only drop leaves the
     hitter solve unchanged and vice-versa -- cache the unaffected side.)
   - Build the resulting hypothetical roster (active slots set, bench on BN,
     dropped players removed) and score it through the SAME pipeline the
     standings use: `project_team_stats(..., displacement=True,
     league_context=...)` -> `score_roto_dict`, against the frozen field
     (`projected_standings.field_stats(team_name)`).
   - deltaRoto = scored team roto minus the current roster's team roto (current
     scored through the identical pipeline). Standard deviation and P(helps) come
     from `compute_delta_roto_band(before, after, ...)` where `before`/`after` are
     the current vs. post-plan contributing sets, so the band reflects the
     marginal effect of the forced drop and stays visually identical to the audit
     rows.
6. **Rank** drop-sets by deltaRoto descending; take the top `max_plans` (5).
   Tie-break by retaining higher raw SGP (i.e., prefer dropping the lower-SGP
   body) for determinism.
7. **Derive the move list** for each plan from `compute_roster_breakdown()` on the
   post-drop roster: `ACTIVE` -> active slot, `DISPLACED`/bench -> `BN`, removed
   players -> `DROP`. Each move is `{name, from_slot, to_slot}`.

Note (implementation risk -- resolve in research/planning): the band's mean
mechanism subtracts the dropped player's full ROS, while the displacement model
may have already scaled an excess player's contribution toward zero. The
deltaRoto used for ranking should come from the displacement-aware roster score
(step 5), and the band should be reconciled so its mean matches that deltaRoto
rather than a full-ROS subtraction. The plan phase must pin down exactly how to
make the band's `mean` agree with the displacement-aware deltaRoto.

## Data model

```
@dataclass
class Move:
    name: str
    player_type: str
    from_slot: str        # current slot: active slot label, "BN", or "IL"
    to_slot: str          # "P"/"OF1"/...  or "BN" or "DROP"

@dataclass
class MovePlan:
    drops: list[str]              # player names dropped (len == overflow)
    moves: list[Move]             # full transaction list
    delta_roto: float             # vs current roster, displacement-aware
    band: dict                    # {mean, sd, p_positive, verdict} from DeltaRotoBand.to_dict()

@dataclass
class IlReturnPlanResult:
    activating: list[str]         # names of IL players being brought back
    capacity: int
    overflow: int                 # forced drops (0 => no drop needed)
    plans: list[MovePlan]         # top 5 by delta_roto desc
    warning: str | None = None    # e.g. all forced drops infeasible
```

## UI (season/roster_audit.html)

- New "IL Returns" section above or below the existing audit table.
- Left: the team's IL players (name, slot, status) each with a checkbox.
- Right: results panel. On checkbox change, fetch `/il-return-plan?activate=...`
  and render up to 5 plans. Each plan card shows: the move list (e.g.
  `Webb  IL  -> P`, `Hader  IL -> P`, `<Pitcher>  P -> BN`, `<Drop>  BN -> DROP`),
  and deltaRoto / std dev / P(helps) using the same band styling/`band_class`
  verdict the audit candidate rows already use.
- Empty state when no IL players exist or none are checked.
- ASCII-only labels in any server-rendered strings (player names may carry
  non-ASCII from data; the season app already reconfigures stdout where needed).

## Edge cases

- No IL players, or none checked: empty state, no computation.
- `overflow <= 0` (an open bench/roster spot already exists): no drop; emit the
  activation + bench reshuffle as a single plan.
- Infeasible drop (e.g. dropping the only catcher leaves a hitter slot
  unfillable): skip that drop-set. If every forced drop is infeasible, return
  `plans=[]` with a `warning`.
- Activating a hitter (not a pitcher): the algorithm is symmetric. "Bench must be
  a pitcher" is an artifact of the current roster, not a rule.
- Unchecked IL player sitting on BN with IL status: stays where it is and keeps
  counting against the cap (it is not in an IL slot). Documented; not auto-moved.
- Multi-position / two-way players: handled by existing optimizer eligibility.

## Testing

- Synthetic Webb/Hader roster (12 hitters filling hitter slots, 9 active P, Webb
  BN+IL-status, Hader IL-slot): activating both => `overflow == 1`, top plan drops
  the lowest-value body, both returnees land in P, one pitcher moves to BN.
- `overflow == 0` case (open bench spot): no drop, activation-only plan.
- Infeasible-drop case (only catcher): that drop-set skipped; warning when all
  infeasible.
- Activating a hitter: symmetric move list.
- No-IL / none-checked: empty result.
- Determinism / tie-break: equal-deltaRoto plans order by dropping lower SGP.
- Capacity is slot-based: a BN+IL-status player counts; an IL-slot player does
  not (regression guard for the Webb-counts/Hader-doesn't distinction).
- One route test for `/il-return-plan` (mirrors `tests/test_web/test_season_routes.py`).

## Risks / open questions

1. **Band/deltaRoto reconciliation** (above) -- the #1 implementation risk. The
   research/plan phase must confirm how to make the band's `mean` agree with the
   displacement-aware deltaRoto, given the displacement model may zero excess
   players the band would otherwise subtract at full ROS.
2. **Checkbox default** -- v1 may start all-checked or all-unchecked. A status
   heuristic (DTD/IL10 pre-checked, IL60 unchecked) is a nice-to-have, not in
   scope unless trivial.
3. **Route input reconstruction** -- confirm during planning that every input
   (`projected_standings`, `team_sds`, `fraction_remaining`, roster Players with
   ROS projections, `roster_slots`) is reachable from cache in a Flask route, as
   the trade-builder route suggests.
4. **Perf** -- overflow is realistically 1-2, so ~24-250 optimizer runs per
   toggle. Confirm this is acceptable interactively; the unaffected-side caching
   optimization should keep it well under a second.
