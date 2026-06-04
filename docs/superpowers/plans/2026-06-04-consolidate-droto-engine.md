# Consolidate dRoto to a single engine (E)

**Date:** 2026-06-04
**Status:** Phase 0 complete (landed, uncommitted on `main`); Phases 1-4 not started.
**Related memory:** `project_optimizer_league_context_fix` (the bug that started this).

---

## TL;DR / Decision

The codebase computes "what does a roster change do to my roto points" two different
ways. We are consolidating onto **one** engine, **E (the exact recompute)**, used
everywhere. No more two-implementation drift.

This decision is final. Do not re-litigate "E vs M" (see "Why E, settled" below).

---

## The two engines

### E -- exact recompute (the one we keep)
- Canonical impl: `team_roto_total` in `src/fantasy_baseball/lineup/optimizer.py` (~line 170).
- Mechanism: `score_roto_dict(team_end_of_season(YTD_components, project_ros_components(roster, displacement=True, league_context)))`.
  - Counting stats: sum across active roster.
  - Rate stats (AVG/ERA/WHIP): recombined from components (h/ab, er/ip, (bb+h)/ip).
  - Displacement / IL-return handled by the pair-swap pool model (`league_context`).
- **This is identical to how `ProjectedStandings.from_rosters` builds the standings**
  (`src/fantasy_baseball/models/standings.py` ~line 398). So "consolidate to E" ==
  "every dRoto is computed the same way the standings the user looks at are built."

### M -- marginal approximation (the one we retire)
- Canonical impl: `apply_swap_delta` in `src/fantasy_baseball/trades/evaluate.py` (~line 130).
- Mechanism: start from the team's projected standings row, do `current - loses_ros + gains_ros`;
  rate stats recombined from components using a fixed `team_ab`/`team_ip` baseline.
- M is *currently the de-facto shared engine* -- used by draft, trades, waivers,
  transactions analysis, stash, IL planner, AND the lineup band. E is the lone
  outlier (only the optimizer's selection objective).

### Why they diverge (only these three; none are bugs in clean cases)
M is exact in the simple case (no IL, matched baseline). They split on:
1. **Displacement (IL pitchers).** E scales swapped players' contribution via the pool
   model; M uses raw ROS. This is the dominant real-world gap (drove Hader 0.91 vs 0.37).
2. **Anchor.** M perturbs the *current* standings row by one swap; E rebuilds each
   lineup from scratch. Differ when current lineup != the lineup being scored (scoring
   is nonlinear, so deltas differ by operating point).
3. **YTD baseline volume.** Both use team-level, set-in-stone YTD (no bug in either --
   see below). E rebuilds the ROS half from the roster + adds fixed YTD; M uses the
   standings entry's `total_ab`/`total_ip` (YTD+ROS) and swaps the ROS part.

---

## Why E, settled

Two hard requirements from the owner:
1. **Start/sit must assume IL players return** (don't over-index on today's situation).
   -> Only E models this (pool model activates IL pitchers at full ROS, displaces an
   active arm). M cannot do this without becoming E.
2. **YTD accrued stats are set in stone, not a function of the current roster.**
   -> E already satisfies this. `ytd_components()` (`standings.py` ~line 177) reads the
   team's accrued IP/AB straight off the Yahoo standings extras; only the ROS half
   depends on the roster. There is **no YTD bug in E** (an earlier claim that "E
   rebuilds YTD from the active roster" was wrong -- it rebuilds only ROS).

Both requirements point to E. Decision: **consolidate Up to E. Retire M.**

---

## Current state (Phase 0 -- already landed, uncommitted on `main`)

These changes fixed the original closer bug and the review follow-ups. They are on the
right engine (E) and should be committed (suggest a branch + PR before Phase 1):

- `lineup/optimizer.py`:
  - `league_context` now built ONLY in `optimize_pitcher_lineup` (not hitter). Helper
    `_build_league_context` (~line 235); field on `_TeamContext`.
  - `compute_bands: bool = True` added to both optimizers; band gated on
    `compute_bands and fraction_remaining is not None`.
- `lineup/stash_value.py`, `lineup/il_return_planner.py`, `lineup/roster_audit.py`:
  thread the REAL `fraction_remaining` + `compute_bands=False` into their pitcher
  optimizer calls (was `fraction_remaining=None`, which mis-sized IL displacement).
  `_solve_active` / `_solve_lineup` now take `fraction_remaining` as a REQUIRED positional.
- Tests: `tests/test_lineup/test_optimizer.py` (closer regression +
  `test_compute_bands_false_suppresses_band_with_real_fraction`),
  `test_stash_value.py`, `test_il_return_planner.py` (helper-signature call-site updates).

Verified: full suite 2020 passed; ruff/format/vulture/mypy clean; real-roster replay
keeps Hader started, Soriano benched.

---

## The surface to migrate (every M caller)

Run `grep -rn "apply_swap_delta\|compute_delta_roto\|compute_one_for_one_band\|score_swap\|build_swap_standings\|_ev_delta_and_stats" src/ --include=*.py` to refresh this list. As of 2026-06-04:

Band/dRoto primitives (all in `src/fantasy_baseball/lineup/delta_roto.py`):
- `compute_delta_roto_band` (~402) -> `_ev_delta_and_stats` (~210, uses `apply_swap_delta`) for mean; sd via `_swap_category_variance` (~270) + `_category_delta_variance` (~343).
- `compute_delta_roto` (~86) -> `score_swap` (~60).
- `compute_one_for_one_band` (~465).

Callers:
- `lineup/optimizer.py` -- band in both optimizers (~390, ~503).
- `lineup/il_return_planner.py:294` (compute_delta_roto_band).
- `lineup/stash_value.py:271` (compute_delta_roto_band).
- `lineup/roster_audit.py` -- `compute_one_for_one_band` (~371), `score_swap` (~359), `apply_swap_delta` (~351).
- `trades/multi_trade.py` (~339, ~353) -- scores BOTH user and opponent rows (needs E with opponent roster + their YTD).
- `trades/evaluate.py` -- `build_swap_standings` (~215), `apply_swap_delta` (~238, ~306).
- `web/season_data.py` (~1078 build_swap_standings, ~1091 score_swap, ~1094 compute_one_for_one_band).
- `web/season_routes.py` (~903 compute_one_for_one_band, ~1793 compute_delta_roto).
- `draft/eroto_recs.py` (~47 compute_delta_roto, ~109 apply_swap_delta, ~117 score_swap) -- BULK eval, perf-sensitive.
- `analysis/transactions.py:349` (apply_swap_delta) -- post-hoc transaction analysis.

---

## Plan (phased; each phase <= 5 files, suite green throughout)

### Phase 1 -- build the unified primitive (no migration yet)
In `delta_roto.py`, add a single function, e.g.:

```
delta_roto_band(before_roster, after_roster, *, ytd_by_team, projected_standings,
                team_name, team_sds, fraction_remaining, league_context=...)
    -> DeltaRotoBand
```

- `mean` = E recompute: `score_roto_dict(team_end_of_season(YTD, project_ros_components(after_roster, displacement=True, league_context)))[team]["total"]` minus the same for `before_roster`. (Reuse `team_roto_total`'s body -- consider extracting a shared `team_eos_row(roster, ctx)` helper used by both `optimizer.team_roto_total` and this.)
- `sd` / `p_positive`: KEEP the existing machinery (`_swap_category_variance`,
  `_category_delta_variance`, Gauss-Hermite nodes). Re-anchor `before_cs`/`after_cs`
  on E's recomputed CategoryStats rows (from `team_end_of_season`) instead of
  `apply_swap_delta`'s output, so mean and sd come from one operating point (the
  "A-full / coherent" choice the owner approved).
- Add ALONGSIDE the old functions. Migrate nothing.
- Tests: parity on a clean no-IL swap (should ~match old M), the closer+IL case
  (`tests/test_lineup/test_optimizer.py::...elite_closer...` analog), and a toy case.

### Phase 2 -- migrate the lineup (delivers the owner's immediate ask)
- In `optimizer.py`, have the band call the new primitive. Because both selection
  (`roto_delta`) and the band now use E on the same rosters, **`band.mean` == `roto_delta`
  by construction** -- the "two numbers" problem is gone. Consider dropping the separate
  `roto_delta` computation and sourcing it from the band's mean (single code path).
- Verify on real data: `band.mean` and `roto_delta` identical per starter.

### Phase 3 -- migrate the rest, caller by caller
Order: stash -> IL planner -> roster_audit/waivers -> trades (multi_trade, evaluate)
-> draft (eroto_recs) -> transactions analysis -> season_data/season_routes display.
- For BULK paths (draft, trades, waiver audit over many candidates): compute the
  IL-displacement factors ONCE per evaluation context, not per candidate. (This is the
  perf trap from the code-review -- the pool model does a `score_roto` per IL x active
  candidate.) Precompute `_compute_pitcher_pool_factors` / displaced ROS once, reuse.
- multi_trade scores the OPPONENT's row too: E needs the opponent roster + the
  opponent's YTD components (available via opp_rosters + standings). Thread them.
- Perf-check each migration (time the bulk path before/after).

### Phase 4 -- delete M
- Remove `apply_swap_delta`, `build_swap_standings`, `compute_delta_roto`,
  `compute_one_for_one_band`, `score_swap`, `_ev_delta_and_stats`, `team_baseline_volumes`
  once nothing references them. `vulture` will confirm. Update docstrings.
- Also delete `combined_team_roto` in `optimizer.py` (dead code, no callers) or fold it in.

---

## Invariants / gotchas (do not violate)

- **YTD stays team-level.** Always source YTD from `ytd_components()` (standings extras),
  never sum YTD from the current roster. Only ROS depends on the roster.
- **Preserve the uncertainty machinery.** `sd` / `p_positive` / verdict bands are
  load-bearing for the owner's decisions. Keep `_swap_category_variance` +
  `_category_delta_variance` + Gauss-Hermite. Do not drop to point estimates.
- **Displacement once per context** on bulk paths (perf). The pool model is the
  expensive part when IL pitchers exist.
- **fraction_remaining is required for displacement sizing.** Pass the REAL value
  (drives `swap_window_ip`); never let it default to 1.0 mid-season. Band gate is the
  separate `compute_bands` flag.
- **ASCII-only** in source/log/format strings (Windows cp1252; CLAUDE.md). No unicode
  minus/sigma/arrows.
- **Tests are the guardrail.** Do not loosen/skip a failing test to go green. Signature
  changes legitimately require updating *call sites* in tests (not assertions).
- **Player IDs are `name::player_type`**; name-normalize for joins (CLAUDE.md).

## Known pre-existing issues (out of scope, note but do not chase)
- `_swap_category_variance` rate path uses `project_team_sds(displacement=False)` and does
  NOT track a rate-only mean shift -- band WIDTH can under-count for rate-only swaps
  (documented in `delta_roto.py`). Separate fix if ever wanted.

---

## Verification (run before claiming any phase done)
- `pytest -n auto` -- all green (state subset if scoped).
- `ruff check .` ; `ruff format --check .` ; `vulture` (no NEW findings) ;
  `mypy` (delta_roto.py, optimizer.py, stash_value.py, roster_audit.py, trades/* are covered).
- Real-data replay (read-only, prod Upstash via `build_explicit_upstash_kv`): run
  `optimize_pitcher_lineup` on cached `PROJECTIONS`+`ROSTER`+`STANDINGS`; assert Hader
  started, Soriano benched, and `band.mean == roto_delta` per starter.
  (See `project_optimizer_league_context_fix` memory for the read-only replay recipe;
  prior scratch scripts were `scripts/debug_hader_soriano.py` / `verify_hader_fix.py`,
  deleted -- recreate as needed, they read prod and should not be committed.)

## Quick orientation for a fresh session
1. Read this file + the `project_optimizer_league_context_fix` memory.
2. `git status` / `git diff` to see Phase 0 (commit it first if not done).
3. Read `team_roto_total` + `_build_league_context` (optimizer.py), `team_end_of_season`
   + `project_ros_components` + `_apply_displacement` + `_compute_pitcher_pool_factors`
   (scoring.py), `compute_delta_roto_band` + `_ev_delta_and_stats` (delta_roto.py),
   `from_rosters` (standings.py), `apply_swap_delta` (trades/evaluate.py).
4. Start Phase 1.
