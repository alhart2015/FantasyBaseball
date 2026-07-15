# IL Return Planner — two-scenario (injury-volume vs healthy-volume) view

**Date:** 2026-07-14
**Status:** Approved design, ready for planning
**Topic:** `il-return-two-scenario`

## Problem

The IL Return Planner ranks drop/reshuffle plans by rest-of-season (ROS) deltaRoto.
For a player returning from the IL, the ROS projection bakes in expected missed
time as a reduced plate-appearance (PA) / innings (IP) volume. Because roto
counting stats reward volume, a returning star with a genuinely higher per-game
rate can score *below* a healthy, higher-volume teammate on a pure ROS basis.

Concrete case (live 2026-07-14 projections): activating Oneil Cruz, the planner's
top plan is **DROP Oneil Cruz** (deltaRoto +0.34), because his injury-shortened
ROS line (175 PA, SGP 4.52) lands just under bench outfielder Trent Grisham
(245 PA, SGP 4.70). Restoring Cruz to a healthy remaining volume (~223 PA,
SGP 5.75) flips the recommendation: "drop Cruz" falls to last and the best plan
becomes "keep Cruz, drop a spare arm." (Both numbers verified against the live
cache during design.)

The manager makes the start/sit/drop decision *before* the projection catches up
to the player's actual return. At that moment the tool silently commits to the
pessimistic (still-injured) volume, and the manager cannot see how sensitive the
decision is to when the player actually returns.

## Goals

- When the IL Return Planner evaluates activating a volume-suppressed returning
  player, present the drop/reshuffle decision under **two** rest-of-season
  volume assumptions side by side:
  - **As projected** — the current ROS line (injury-reduced volume).
  - **If healthy** — the same player valued at a healthy remaining volume,
    derived from the player's own pre-injury (preseason) pace.
- Make the headline signal obvious: whether the top plan is the **same** under
  both assumptions (decision is robust — just do it) or **differs** (the call
  hinges on the player's return; the manager applies judgment).
- Require **no manual input** (no return-date guess) and **no new data source** —
  use fields already on every `Player`.

## Non-goals

- No change to any other surface: the in-season optimizer, roster audit start/sit,
  projected standings, or Monte Carlo. Scope is the IL Return Planner only.
- No change to the core ROS -> SGP -> deltaRoto pipeline or to how plans are
  ranked *within* a single scenario. The healthy scenario is produced by feeding
  the existing planner a roster variant, not by changing the planner's math.
- No manual return-date / games-remaining input. The two-scenario bracket is the
  deliberate answer to "I will not know the exact return date at decision time."
- No exclusion of the activating player from the drop candidates. "Drop the
  returning player" remains a listed plan — it is genuinely optimal if the player
  stays limited, and the two columns now contextualize it (it correctly sinks to
  the bottom of the healthy list). This was an explicit design decision, not an
  oversight.

## Chosen approach

### 1. Healthy-volume transform

A pure function that, given a `Player` and `fraction_remaining`, returns a copy
with its ROS counting stats scaled up to a healthy remaining volume — or `None`
when no adjustment applies.

```
healthy_rest_of_season(player, fraction_remaining) -> Player | None
```

- **Healthy remaining volume:**
  - Hitters: `preseason.pa * fraction_remaining`
  - Pitchers: `preseason.ip * fraction_remaining`
  The player's *pre-injury* full-season expectation (`preseason`), prorated to
  the games left. `fraction_remaining` is already threaded through the planner
  and route.
- **Scale factor:** `healthy_vol / current_vol`, applied to the **current** ROS
  line. The transform touches exactly these fields (full `HitterStats` /
  `PitcherStats` field sets, so nothing is left implicit):
  - **Hitters — scale by the factor:** `pa, ab, h, r, hr, rbi, sb, g`.
    **Preserve unchanged:** `avg` (rate — its components `h`/`ab` scale together,
    so `avg` stays correct).
  - **Pitchers — scale by the factor:** `ip, w, k, sv, er, bb, h_allowed, g, gs`.
    **Preserve unchanged:** `era, whip` (rates).
  - **Both — clear the cached `sgp` field to `None`** so any downstream read
    recomputes it from the scaled line rather than reusing the stale injury-volume
    SGP. (The planner already recomputes via `calculate_player_sgp(ros, denoms)`;
    clearing `sgp` guards any path that trusts the cached value.)
  Games (`g`, and pitcher `gs`) scale with volume deliberately: leaving them at
  the injury value while `pa`/`ip` inflate would produce an internally
  inconsistent stat line. Only volume is restored; the current talent/form read
  (rates) is kept.
- **Only ever inflates.** Returns `None` when:
  - `preseason` is absent, or
  - the relevant `current_vol` is falsy/zero, or
  - `healthy_vol <= current_vol` (the player is not volume-suppressed — e.g. a
    near-full-time returnee, or one already outpacing preseason).
  A `None` result means "this player contributes no healthy/limited difference,"
  so the caller shows a single list rather than a spurious "both ways."

### 2. Two-scenario wrapper

A new orchestration function in `il_return_planner.py`:

```
plan_il_returns_scenarios(roster, activating_il, roster_slots, *, ...same kwargs
    as plan_il_returns...) -> IlReturnScenarios
```

- Computes, for each activating player, its `healthy_rest_of_season(...)`.
- Runs the **existing** `plan_il_returns` twice, unchanged:
  - `as_projected`: on the roster and `activating_il` exactly as-is.
  - `if_healthy`: on a roster where each *adjusted* activating player's
    `rest_of_season` is swapped to its healthy version (players with a `None`
    adjustment, and all non-activating players, are untouched) **AND** on an
    `activating_il` list carrying those same healthy player objects.
- **The healthy swap must reach both the roster copy AND the `activating_il`
  list.** This is load-bearing: `il_return_planner._build_pool` sources a returnee
  who sits in a *true IL slot* (the primary case — e.g. Cruz) from the passed
  `activating_il` list, not from the roster's counted bodies (`extra = [p for p in
  activating_il if p.player_key not in counted_keys]`). If only the roster copy is
  swapped, `_build_pool` pulls the original injury-volume object from
  `activating_il` and the healthy scenario becomes a silent no-op (`if_healthy` ==
  `as_projected`, `tops_differ` always false). Concretely: build the healthy
  roster first, then re-derive the healthy `activating_il` from it by `player_key`
  (so both the roster body and the activating body are the same healthy object).
- If **no** activating player was adjusted, `if_healthy` is `None` (the caller
  falls back to the single-list view).
- Returns a small result object exposing both `IlReturnPlanResult`s, the list of
  adjusted players with their projected-vs-healthy volume, and a `tops_differ`
  boolean. **Top plan** = the highest-ranked plan, `plans[0]`, of each scenario.
  `tops_differ` is `true` only when **both** scenarios have a top plan and those
  two top plans' drop sets differ, compared as **order-independent sets** of
  dropped player names (the planner sorts `drops` by name, so an order-sensitive
  compare would be a latent bug). When either scenario has no plans (empty or a
  warning), `tops_differ` is `false` (nothing to compare).

`plan_il_returns` itself is **not modified** — this preserves its existing test
coverage and keeps the band/ranking code untouched. The healthy scenario is a
roster transform applied above it.

### 3. Route + template

- `/api/il-return-plan` calls `plan_il_returns_scenarios` and serializes:
  ```
  {
    "as_projected": <IlReturnPlanResult dict>,
    "if_healthy":   <IlReturnPlanResult dict | null>,
    "adjusted":     [{"name", "player_type",
                      "vol_unit": "PA" | "IP",
                      "vol_projected": <float>, "vol_healthy": <float>}...],
    "tops_differ":  <bool>
  }
  ```
  Volume fields are **unit-generic**: `vol_unit` is `"PA"` for hitters and `"IP"`
  for pitchers, and `vol_projected` / `vol_healthy` carry that unit's value. The
  template labels each number with `vol_unit` (e.g. "175 PA -> 223 PA",
  "43 IP -> 59 IP"). There are no PA-named fields, so a pitcher returnee is
  represented correctly.
- Template (`roster_audit.html` IL Returns section): when `if_healthy` is
  non-null, render two ranked lists side by side, each headed by its assumption
  and the returning player(s)' volume (value + `vol_unit`) under it, with a
  one-line headline driven by `tops_differ`. The headline names **every** adjusted
  returnee (the `adjusted[].name` set): `tops_differ == false` -> "robust — same
  top plan whether <names> return healthy or stay limited"; `tops_differ == true`
  -> "the call depends on <names>' return." When `if_healthy` is null, render
  today's single list unchanged.

## Data flow

```
roster (cache) --+--> plan_il_returns(roster, activating_il  as-is) --> as_projected
                 |
                 +--> for each activating player:
                 |       healthy_rest_of_season(p, fraction_remaining)
                 |       -> swap into a roster copy (if not None)
                 |    then re-derive healthy activating_il from that roster copy
                 +--> plan_il_returns(healthy_roster, healthy_activating_il) -->
                 |                                                    if_healthy
                                                                (null if no adj.)
```

The cached `projected_standings` / `team_sds` / Monte-Carlo blobs are read but
never written or altered; the healthy swap exists only in the in-memory roster
copy passed to the second planner run.

## Edge cases / failure modes

- **No preseason on the returning player:** `healthy_rest_of_season` -> `None`;
  that player is not adjusted. If it's the only activating player, `if_healthy`
  is null and the UI shows one list.
- **Returning player not volume-suppressed** (`healthy_vol <= current_vol`, e.g.
  Buxton returning near full pace, or a player outpacing preseason): `None`, no
  adjustment — avoids a fake "both ways" where the two lists would be identical.
- **Multiple returnees (e.g. Cruz + Snell):** each is adjusted independently; the
  healthy scenario restores volume for every adjusted returnee at once. If some
  adjust and others don't, `if_healthy` still renders (at least one adjusted).
- **Pitcher (IP-based) returnee (Snell):** scaling keys on `player_type`; IP and
  pitcher counting stats scale, `era`/`whip` preserved.
- **Two-way player:** the transform operates on a single `Player` row; the
  wrapper keys the healthy swap on `player_key` (matching the planner's existing
  two-way handling) so a two-way player's hitter and pitcher rows are adjusted
  independently and never collide by bare name.
- **`fraction_remaining` at season end (-> 0):** healthy_vol -> 0 <= current_vol
  for a returning player, so `None` (no adjustment). Degenerate but safe.
- **`plan_il_returns` returns a warning / no plans in one scenario:** the wrapper
  passes each `IlReturnPlanResult` through as-is; `tops_differ` is false when
  either scenario lacks a top plan (nothing to compare), and the template renders
  whatever each scenario returned (including its warning).

## Testing expectations

Unit tests for the new pure function and wrapper (no live cache, synthetic
`Player`s and `ProjectedStandings`, mirroring `tests/test_lineup/test_il_return_planner.py`):

- `healthy_rest_of_season`:
  - Hitter: scales `pa, ab, h, r, hr, rbi, sb, g` by `preseason.pa *
    fraction_remaining / ros.pa`, preserves `avg`, clears cached `sgp` to `None`,
    returns a new object (original untouched).
  - Pitcher: scales `ip, w, k, sv, er, bb, h_allowed, g, gs` by the IP-based
    factor, preserves `era`/`whip`, clears cached `sgp`.
  - Returns `None` when preseason absent, when `current_vol` is 0, and when
    `healthy_vol <= current_vol`.
  - Only inflates (never returns a smaller volume); `g`/`gs` scale with volume
    (not left at the injury value).
- `plan_il_returns_scenarios`:
  - When a suppressed returnee is activated, `if_healthy` is non-null and the
    healthy roster's returnee has the inflated ROS.
  - **Returnee in a true IL slot** (not a BN+IL-status body): the healthy
    scenario actually takes effect — `if_healthy` differs from `as_projected`
    (regression guard for the `_build_pool` `extra`-path swap; a wrapper that
    swapped only the roster copy would make these identical).
  - When no returnee is adjusted, `if_healthy` is `None`.
  - `tops_differ` is true in a constructed case where the healthy scenario's top
    drop set differs from the projected scenario's (mirrors the verified Cruz
    flip: projected -> drop the returnee; healthy -> keep the returnee).
  - `as_projected` reproduces the existing `plan_il_returns` result exactly
    (regression guard that the wrapper doesn't perturb the as-is path).
  - `adjusted` carries one entry per adjusted returnee with `vol_unit` +
    `vol_projected`/`vol_healthy` matching the player type (PA for hitters,
    IP for pitchers).
- Route test (`tests/test_web/test_season_routes.py` style): the JSON envelope
  carries `as_projected`, `if_healthy`, `adjusted`, `tops_differ`, with
  `if_healthy` null when no adjustment applies.
- Template verification (rendering/smoke test in the `tests/test_web/` style used
  for the existing roster-audit template): render the IL Returns section with a
  context where (a) `if_healthy` is null -> exactly one ranked list, no
  scenario headers; and (b) `if_healthy` is non-null with `tops_differ` true ->
  two labeled lists ("as projected" / "if healthy") plus the "depends on ...
  return" headline; and true vs false `tops_differ` selects the differ vs robust
  headline text. This guards the single-list fallback branch and the headline
  logic, not just the JSON contract.

## Phasing

Small enough for a single implementation plan:

1. `healthy_rest_of_season` pure function + unit tests.
2. `plan_il_returns_scenarios` wrapper + `IlReturnScenarios` result + unit tests.
3. Route wiring (`/api/il-return-plan`) + route test.
4. Template: side-by-side rendering + headline, single-list fallback.

End-of-effort verification per repo rules: `pytest` (relevant subset +
`test_lineup`, `test_web`, `test_models`), `ruff check`, `ruff format --check`,
`vulture` (no new findings), `mypy` (models/ and any covered touched file).
