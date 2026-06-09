# Draft subsystem

The draft assistant and simulator. Entry points: `scripts/run_draft.py` (interactive CLI + dashboard), `scripts/simulate_draft.py`, `scripts/compare_strategies.py`.

## Data pipeline

FanGraphs CSVs (`data/projections/{season_year}/`) -> `data.projections` blends systems -> `sgp.player_value` calculates per-category SGP -> `sgp.replacement` computes position-specific replacement levels -> `sgp.var` assigns VAR per player -> `draft.board` assembles the ranked board.

**Replacement floors are empirical waiver lines** (`REPLACEMENT_BY_POSITION` per hitter position + SP/RP, via `position_aware_replacement_levels`) -- static per draft, not per-pick. They reflect what's actually free on this league's waiver wire, so VAR measures value over the waiver alternative.

## Scoring modes

Four scoring modes, selected via `draft.scoring_mode` in `config/league.yaml`:

- **var** (Value Above Replacement) -- static: `player SGP - replacement level at position`.
- **vona** (Value Over Next Available) -- dynamic: `player SGP - best remaining in same bucket after opponents' next N picks`. Uses 3 buckets (hitter / SP / closer). Position-level VONA was tested and regressed badly -- do not reintroduce it without new evidence.
- **deltaroto_immediate** -- ERoto marginal delta, scored as the roto-point gain from drafting the player right now. The PR #127 verdict winner; the validated default (`config/league.yaml` uses this + `default` strategy).
- **deltaroto_vopn** -- ERoto delta discounted by picks until next turn (Value of Picking Now). Penalizes players whose scarcity window is short relative to queue length.

`deltaroto_immediate` + `default` is the validated default (PR #127 verdict winner) and is what `league.yaml` currently uses.

## Unified recommend() seam

**All pick sources route through `recommend(ctx, strategy=...)` in `draft/recommend.py`.**

- The dashboard `/api/recs` endpoint calls `recommend()`.
- The simulator (`simulate_draft.run_simulation`) calls `recommend()`.
- `compare_strategies.py` calls `recommend()`.

The call chain:

1. Caller constructs a `RecommendContext` (scoring_mode, team_name, picks_until_next, plus mode-specific fields: `inputs` for deltaRoto, `board`+`config` for var/vona).
2. `rank_for_mode(ctx)` dispatches to `_rank_deltaroto` or `_rank_var_vona`, returning `list[RankedPick]`.
3. `RankedPick` is the uniform pick type. `score` is the active mode's primary metric; `metrics` carries all mode-native metrics (both `immediate_delta` and `value_of_picking_now` for deltaRoto modes so the dashboard can toggle).
4. The overlay (strategy) receives the ranked list and returns a `RankedPick` override or `None` to defer to the slot-gated greedy pick.

## Strategy system

Strategies are orthogonal overlays in the `OVERLAYS` registry (`STRATEGIES` is aliased to it for backward compatibility). An overlay is a function with signature:

```python
def overlay_foo(ranked: list[RankedPick], *, roster_state=None, config=None, **kwargs) -> RankedPick | None:
```

It returns a `RankedPick` to force that pick, or `None` to defer to the slot-gated greedy selection from the ranked list. The key is a plain string in `OVERLAYS` -- if you rename a strategy, grep for the string literal in config files, simulation scripts, and tests (not just the function name).

Selected via `draft.strategy` in `league.yaml`. Strategies are orthogonal to scoring mode: any strategy can be combined with any scoring mode.

### Strategy port status

**Fully ported (faithful behavior):**
- `default` -- no override; always defers. Slot-gated greedy on the ranked list.
- `nonzero_sv`, `two_closers`, `three_closers`, `four_closers` -- closer scheduling by deadline/count.
- `balanced` -- forces hitter or pitcher when positional skew exceeds `BALANCED_MAX_SKEW` (2).
- `no_punt_opp` -- available as a valid strategy key but is a documented fallback (see below).

**Partial ports (deadline/cap logic ported; AVG floor deferred):**
- `no_punt_stagger` -- staggered closer deadlines `[13, 17, 20]` ported faithfully. Missing: team H/AB for AVG floor filter and full dynamic SV-danger check (needs all-team rosters).
- `no_punt_cap3` -- same staggered deadlines + hard cap at 3 closers ported. Missing: AVG floor filter and post-cap closer exclusion from recs (requires raw board IP/SV stats not on `RankedPick`).
- `closers_avg` -- three_closers closer gate ported faithfully. Missing: avg_anchor fallback (needs candidate's absolute projected AVG, not available in `per_category` which carries marginal roto-point deltas).

**Documented fallbacks (FIVE strategies defer entirely to default behavior in the unified engine):**

The following five strategies always return `None` and are equivalent to `default` in the current unified engine. They are listed here so future readers know they are NOT faithfully simulated -- the legacy faithful logic was removed when the pick_* registry was replaced by overlays.

1. `no_punt` -- missing: team H/AB totals for AVG floor filter; team_rosters for dynamic SV danger.
2. `no_punt_opp` -- missing: team_rosters for opponent-relative SV check; absolute AVG stats; ADP context for opportunistic grabs.
3. `avg_hedge` -- missing: team accumulated H/AB from `balance.get_avg_components()` for AVG floor.
4. `avg_anchor` -- missing: candidate's absolute projected AVG (`.285+`); `per_category` carries marginal roto-delta, not batting average.
5. `anti_fragile` -- missing: candidate's absolute projected IP; per_category has no IP (not a roto category).

Threading these signals (team H/AB totals, all-team rosters, absolute AVG/IP projections) into the overlay contract requires changes to every overlay call-site and the sim loop. That work was scoped out; these strategies are placeholders until someone threads those signals.

## Draft dashboard (web-only)

All pick entry happens in `src/fantasy_baseball/web/app.py`. Launch with
`python scripts/run_draft_dashboard.py` (default port 5050, `--debug`
for auto-reload during development). Before a fresh draft -- or any time
the projection CSVs have changed -- pass `--rebuild-board` once to
regenerate `data/draft_state_board.json` from the SQLite projections
pipeline. A legacy CLI fallback lives at `scripts/run_draft.py`.

- `draft_state.json` -- snapshot written atomically by the Flask writer
  endpoints on every pick. Reader (the browser JS) polls
  `/api/state?since=<version>` every 500ms using the delta protocol.
- `draft_state_board.json` -- the ranked board (written once per session
  on `/api/new-draft`).
- `draft_state_delta.json` -- per-version deltas.

State shape: `keepers`, `picks`, `on_the_clock`,
`undo_stack`, `projected_standings_cache`. The legacy `recommendations`
and `balance` fields are still tolerated by readers so the simulator
(unchanged) keeps working.

Writer endpoints in `web/app.py`:
- `POST /api/new-draft` -- seed keepers + set on-the-clock.
- `POST /api/pick` -- record a pick, advance snake order.
- `POST /api/undo` -- pop the most recent live pick.
- `POST /api/on-the-clock` -- manual override.
- `GET /api/recs?team=<name>` -- top 10 ERoto-delta candidates (returns
  503 if the cached board is missing -- run `--rebuild-board` once).
- `GET /api/roster?team=<name>` -- slots + replacement placeholders.
- `GET /api/standings` -- fractional ERoto per team with uncertainty SDs,
  read from `projected_standings_cache`. The cache is refreshed on every
  writer endpoint via `_attach_standings_cache` (best-effort: swallows
  errors so a pick can never fail on standings wiring).

If you change the state shape, both the writer
(`draft/draft_controller.py` + `draft/state.py`) and the reader
(`web/app.py` + `web/static/draft.js`) must be updated together.
