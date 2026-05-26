# IL Stash Value -- Design

- Date: 2026-05-25
- Status: Approved -- ready for planning
- Surface: a ranked **Stash Board** ("Grab & Stash") section on the season
  dashboard, plus a stash-value column on owned IL players (`/roster-audit`
  page).
- New module: `src/fantasy_baseball/lineup/stash_value.py`

## Framing (refined with user)

The deliverable is a **triage / ranking tool: who deserves your 2 IL slots.**
It is NOT a "projected-standings impact" view -- the displacement model in
`project_team_stats` already prices the on-field effect of a player once he is
active, and the user explicitly does not want that re-surfaced. The ranking
*value* is leverage-aware (deltaRoto, value to the user's team), but the
*output* is a ranked list of injured players plus the IL-slot allocation
verdict, not a standings table.

## Problem

Every player is ranked by SGP -> VAR on rest-of-season counting stats. Two
mechanics make this systematically undervalue injured-but-elite players as
grab-and-stash targets:

1. **Volume-weighted rate SGP.** `calculate_pitching_rate_sgp`
   (`sgp/player_value.py`) computes `marginal = (replacement_rate -
   player_rate) * player_ip / divisor`. A pitcher projected for ~90 ROS IP gets
   his elite ERA/WHIP edge multiplied by ~90, while a healthy 180-IP arm with
   the *same* edge gets multiplied by ~180. The injured arm is penalized twice:
   halved K/W counting totals AND halved rate credit.
2. **A flat active-replacement bar.** `calculate_var` (`sgp/var.py`) subtracts
   the ~90th-best pitcher's SGP from everyone. That is the correct bar for a
   player who occupies one of the 21 active slots all season. It is the wrong
   bar for a stash, whose true cost is one of the **2 IL slots** -- a slot that
   cannot hold a healthy contributor anyway.

Concrete case driving this: Blake Snell projects for ~90 ROS IP with excellent
rates. Today he ranks below a healthy Aaron Nola, because Nola accrues more
counting stats over more innings. But Snell can sit on an IL slot at ~zero cost
to the active roster until he returns, then contribute elite innings on top of a
full lineup. The current value metric is blind to that asymmetry.

Worse, `roster_audit.audit_roster` explicitly removes injured free agents
(`active_fas = [fa for fa in free_agents if not fa.is_on_il()]`), so the waiver
tool will never even surface Snell as a pickup.

## Goals

- A single **stash-value** metric that answers both:
  - **Grab:** which injured free agents are worth acquiring and stashing now.
  - **Hold-vs-drop:** whether an injured player already on the roster is worth
    keeping vs. dropping for active production.
- Express it in **deltaRoto** (the dashboard's existing currency) so it is
  leverage-aware: a stash's rate edge is worth a lot when that category is a
  standings dogfight and ~nothing when it is already locked.
- Reuse the existing displacement model, deltaRoto primitives, and forced-drop
  logic. Add only the narrow new pieces.
- Surface a ranked "Grab & Stash" list plus a stash-value column on owned IL
  players.

## Non-goals

- **No projected-standings-impact view.** The displacement model already prices
  the on-field effect once a player is active; this feature is a ranking/triage
  for IL-slot allocation, not a "here's your new standings table" widget.
- **No risk / return-date discount** (user decision). FanGraphs ROS already
  bakes the injury into the projection (the deflated IP/PA); take it at face
  value. No IL-tier haircut, no return-date parsing, no re-injury model.
- **No cap logic.** League 5652 is uncapped (no IP/GP/acquisition cap, confirmed
  by user + absent from config/cache). The metric assumes more production is
  always better; a cap-aware efficiency premium is explicitly out of scope.
- **No rewrite of VAR/SGP.** The existing per-active-slot value stays as-is for
  draft and active decisions. Stash value is a parallel metric, not a
  replacement.
- **No time-stepped Monte Carlo** with sampled return dates (Approach C, ruled
  out by the no-risk-discount decision).

## League context

- 10 teams. Roster slots: C,1B,2B,3B,SS,IF,OF*4,UTIL*2 (12 hitter slots), P*9,
  BN*2, IL*2.
- Body-count cap (active + bench, IL exempt) = **23** (`roster_capacity`,
  `il_return_planner.py`). Up to 2 additional bodies may sit in IL slots.
- Uncapped: no innings/games/acquisition limit.

## Key insight 1: standings already price OWNED IL players (double-count trap)

`ProjectedStandings.from_rosters` builds each team's row from
`project_team_stats(roster, displacement=True, projection_source=
"full_season_projection")` over the **full roster including IL players**. The
displacement model (`scoring.py`, `_compute_displacement_factors`) keeps the
top-N arms by team-ROTO (N = active P slots) and scales the excess down,
*regardless of IL status* -- a returning IL arm bumps the worst arm overall.

Consequences:

- An IL player **already on your roster is already counted** in your projected
  standings. Computing "Gain of adding him" would double-count.
  Therefore **hold-vs-drop value = the deltaRoto you would LOSE by dropping
  him** (a drop-cost), not an add-gain. This is the same forced-drop
  computation `il_return_planner` already performs.
- A **free-agent** injured player is NOT in your projection, so for him the add
  IS marginal and `compute_delta_roto(add=FA, ...)` is the correct Gain.

This is the #1 implementation risk and mirrors the documented double-count trap
class the project has been bitten by before (see
`2026-05-23-il-return-planner-design.md`). The band's `mean` must be reconciled
with the displacement-aware deltaRoto rather than a naive full-ROS subtraction.

## Key insight 2: Gain is displacement-aware, and the downside is floored at zero

The intuition "a stash is free additive production on an IL slot" is true only
*during* the injury. On return, the player competes for the 9 P (or active
hitter) slots. Two cases, both handled by the pool-slot displacement model:

- **He returns better than your worst arm** -> he bumps that arm down, and his
  marginal Gain = his production MINUS the body he displaces (not gross SGP). A
  real, positive upgrade.
- **He returns worse than your staff** -> he lands in the surplus and gets
  `sf = 0` (`factor = max(0.0, active_pt - il_pt) / active_pt`; the bottom
  `pool_size - active_slots` arms are zeroed). He contributes ~0, displaces no
  one, drags nothing -- you simply do not start him. **Gain ~= 0, never
  negative.**

This is the "no harm, no foul" floor, and it falls straight out of the math:
a free-slot stash can only help or be neutral, so its stash value is **floored
at ~0** and the metric never punishes a mediocre stash for being mediocre -- it
just shows ~0.

So the stash-value number will be smaller than "gross ROS SGP," and that is
correct -- reporting gross SGP would overstate stashes.

(Known, pre-existing limitation worth noting: the pool-slot model zeroes arms
beyond the 9th, which is conservative for pitching -- in reality a rotation
cycles >9 arms through 9 slots across off-days, so deep staffs capture more than
the model credits. This under-credits all surplus arms, stash or not; it is not
stash-specific and is out of scope to fix here. Likewise, the small option value
of a stash as injury insurance is an acknowledged, un-modeled upside, consistent
with the no-risk-discount stance.)

## Key insight 3: the binding cost is IL-slot allocation, not displacement

Because the downside is floored (insight 2), the real cost of a stash is NOT who
he displaces on return -- it is whether he is occupying one of your **2 scarce
IL slots** that a more valuable injured player could use. "He comes back as my
worst pitcher and never starts" is harmless *unless* he was holding a stash spot
away from someone better.

This is the feature's core value proposition: **allocate your 2 IL slots to the
highest-upside injured players.** It is exactly what the Cost branch measures --
open slot -> Cost 0 (floored, can only help); slots full -> Cost = the Gain of
the weakest stash he would displace, so `StashValue = Gain(new) -
Gain(weakest current stash)` goes negative only when your current stashes are
better (correctly: "don't bother") and positive when he is a genuine upgrade
("drop the weak stash for this guy").

## The metric

```
StashValue(player) = Gain - Cost
```

- **Gain**
  - FA (not rostered): `compute_delta_roto` of adding the player's ROS to the
    roster -- displacement-aware, leverage-aware, ROS as-is, no risk discount.
  - Owned IL player: the drop-cost (deltaRoto lost if dropped); his Gain is
    already in the projection (insight 1).
- **Cost** = the deltaRoto sacrificed to make roster room:
  - **Open IL slot** -> Cost = 0. The injured player is added straight to an IL
    slot (IL slots are exempt from the 23-body cap). This is the "free stash."
  - **IL slots full** -> a new injured player must displace a body. The
    drop-candidate pool **includes the existing IL-slotted stashes**, and the
    resolver drops the **lowest-Gain body across {owned IL stashes} UNION {worst
    active/bench bodies}**. Because replacing an IL stash costs zero current
    production (neither contributes now), an IL-for-IL swap will essentially
    always be the lowest-cost option; an active/bench drop wins only when its
    marginal value is genuinely tiny -- the explicit "really good reason." The
    output names the recommended drop (e.g. "Stash Snell -> drop <weakest stash>
    from IL").

### Mental model the dashboard exposes

Rank all injured players (owned + available FAs) by stash Gain. The top
**N = open-or-occupied IL slots (2)** are worth holding. Anything below the
cutline is not worth a stash unless you would bench an active body for a really
good reason. Owned IL players below the cutline that are out-Gained by an
available FA stash are explicit drop-for-stash upgrades.

## Architecture & data flow

- New pure-scoring module `lineup/stash_value.py`. Entry point:

  ```
  score_stash_candidates(
      roster: list[Player],
      free_agents: list[Player],
      projected_standings: ProjectedStandings,
      roster_slots: dict[str, int],
      team_name: str,
      *,
      team_sds: Mapping[str, Mapping[Category, float]] | None,
      fraction_remaining: float,
      max_candidates: int = 25,
  ) -> StashResult
  ```

  Depends only on existing primitives: `compute_delta_roto` /
  `compute_delta_roto_band` (`delta_roto.py`), the displacement-aware
  `project_team_stats` pipeline (`scoring.py`), and the slot/capacity helpers
  (`roster_capacity`, `_counts_against_cap`, `IL_SLOTS`) shared with
  `il_return_planner`. No standings/scoring logic is duplicated.

- Computed on the **refresh pipeline** (FAs are already fetched there for
  roster_audit / buy_low) and cached under a new `CacheKey.STASH = "stash"`.
  `season_data` reads it; `season_routes` serves it to the `/roster-audit`
  template. (Cached, not on-demand: the candidate set is bounded and the list is
  not interactive, unlike the IL-return checkboxes.)

## Gaps to close (the only genuinely new logic)

1. **Stop excluding injured free agents.** Add a stash-candidate path that keeps
   `is_on_il()` FAs (the active roster_audit swap path keeps excluding them --
   unchanged). Confirm the Yahoo FA fetch (`fetch_free_agents`, status 'A' =
   all available) returns injured-but-available players with their IL status;
   `waivers.fetch_and_match_free_agents` already preserves `status`.
2. **Add-only deltaRoto.** `compute_delta_roto` is one-for-one (requires a
   `drop_name`). Open-IL-slot stashes add with no drop. Add a thin add-only
   path (phantom/None drop) or a small wrapper that scores `roster` vs
   `roster + player` through the same `project_team_stats` -> `score_roto_dict`
   pipeline, reconciled with the band mean per insight 1.

## Data model

```
@dataclass
class StashScore:
    name: str
    player_type: str
    status: str                 # IL10 / IL15 / IL60 / DTD / ...
    owned: bool                 # already on the user's roster
    gain: float                 # deltaRoto of the player's ROS (add-gain for FA,
                                # drop-cost for owned)
    cost: float                 # deltaRoto sacrificed to roster him (0 if open IL slot)
    stash_value: float          # gain - cost
    band: dict                  # {mean, sd, p_positive, verdict}
    recommended_drop: str | None  # who to drop to make room (None if free IL slot)

@dataclass
class StashResult:
    open_il_slots: int
    cutline_rank: int           # = total IL slots (2); top-N are "hold"
    candidates: list[StashScore]  # owned + FA, ranked by stash_value desc
    warning: str | None = None
```

## UI (season/roster_audit.html)

- New "Grab & Stash" section listing ranked injured FAs: name, status,
  stash value, band (mean / sd / P(helps) reusing `band_class`), and the
  recommended drop when IL slots are full.
- Stash-value column / badge on the team's own IL players, with an explicit
  "below cutline -- droppable for <FA>" hint when an available stash out-Gains
  them.
- Empty state when there are no injured FAs and no owned IL players.
- ASCII-only server-rendered strings (player names may carry non-ASCII from
  data; the season app reconfigures stdout where needed).

## Edge cases

- No open IL slot and IL slots hold higher-Gain stashes: candidate reported
  with the IL-for-IL drop and a (likely negative) net stash value -> not worth
  it. Correct, not an error.
- DTD / not-yet-IL-eligible player: cannot occupy an IL slot, so Cost always
  uses a bench/active body, never a free IL slot.
- More stash-worthy injured players than IL slots: ranking handles it; an IL
  slot's opportunity cost is the next-best stash forgone (emergent from the
  drop-pool comparison).
- Owned IL player double-count: Gain for owned players is a drop-cost computed
  against the projection that already includes them (insight 1).
- Two-way / multi-position players: handled by existing optimizer eligibility.
- Hitter stash (not a pitcher): symmetric; displacement targets the worst
  active hitter.

## Testing

- Snell-vs-Nola fixture: an injured elite-rate, low-IP arm out-ranks a healthy
  higher-volume arm on stash value when an IL slot is open (the headline case).
- Open IL slot -> Cost == 0; stash value == add-gain.
- IL slots full -> recommended drop is the lowest-Gain owned IL stash, not an
  active/bench body, unless an active body's marginal value is near zero
  (explicit "really good reason" regression guard).
- Owned IL player hold-vs-drop uses drop-cost, not add-gain (double-count
  regression guard): scoring an owned IL player must not add his ROS on top of a
  projection that already contains it.
- DTD player never consumes a free IL slot in the Cost branch.
- Leverage-awareness: the same stash yields higher stash value when its strong
  category is a tight standings race than when that category is locked.
- Band mean reconciles with the displacement-aware deltaRoto (matches the
  il-return-planner reconciliation).
- One route test for the cached `stash` payload (mirrors
  `tests/test_web/test_season_routes.py`).
- `mypy` clean (`stash_value.py` should be added to `[tool.mypy].files`).

## Risks / open questions

1. **Band/deltaRoto reconciliation** (insight 1) -- the #1 risk. The plan phase
   must pin down how the add-only / drop-cost band `mean` agrees with the
   displacement-aware deltaRoto, given displacement may already scale the
   bumped body toward zero. Reuse the resolution from il_return_planner.
2. **FA fetch includes injured players** -- confirm status 'A' returns
   injured-but-available FAs with IL status intact through
   `fetch_and_match_free_agents`.
3. **Refresh inputs reachable** -- confirm `projected_standings`, `team_sds`,
   `fraction_remaining`, roster Players with ROS, and `roster_slots` are all on
   hand at the point in `refresh_pipeline` where the stash payload is built
   (they are for roster_audit, which runs there).
4. **Pool-slot conservatism** (insight 2) -- the displacement model under-credits
   surplus arms generally; documented, not fixed here. Flag if it makes stash
   values feel too low in practice.
5. **Cache cost** -- bounded candidate set (~owned IL + injured FAs), one
   deltaRoto per candidate; should be well within the existing refresh budget.
