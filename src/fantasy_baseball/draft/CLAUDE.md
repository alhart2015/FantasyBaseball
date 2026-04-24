# Draft subsystem

The draft assistant and simulator. Entry points: `scripts/run_draft.py` (interactive CLI + dashboard), `scripts/simulate_draft.py`, `scripts/compare_strategies.py`.

## Data pipeline

FanGraphs CSVs (`data/projections/{season_year}/`) → `data.projections` blends systems → `sgp.player_value` calculates per-category SGP → `sgp.replacement` computes position-specific replacement levels → `sgp.var` assigns VAR per player → `draft.board` assembles the ranked board.

**Replacement levels recalculate per pick** from the available pool, not the original board, to reflect live positional scarcity during the draft.

## Scoring modes

Two ranking modes, selected via `draft.scoring_mode` in `config/league.yaml`:

- **VAR** (Value Above Replacement) — static: `player SGP - replacement level at position`.
- **VONA** (Value Over Next Available) — dynamic: `player SGP - best remaining in same bucket after opponents' next N picks`. Uses 3 buckets (hitter / SP / closer). **Position-level VONA was tested and regressed badly — do not reintroduce it without new evidence.**

In VONA mode, `recommender.py` blends VONA with leverage weights based on category gaps.

## Strategy system

Each strategy is a `pick_*()` function in `draft/strategy.py` registered in the `STRATEGIES` dict. The key is a plain string — if you rename a strategy, grep for the string literal (config files, simulation scripts, tests) and not just the function name. Strategies layer constraints on top of the recommender: closer timing, AVG floors, category-protection rules. Selected via `draft.strategy` in `league.yaml`.

## Draft dashboard (web-only)

All pick entry happens in `src/fantasy_baseball/web/app.py`. Launch with
`python scripts/run_draft_dashboard.py` (default port 5050, `--debug`
for auto-reload during development). Before a fresh draft — or any time
the projection CSVs have changed — pass `--rebuild-board` once to
regenerate `data/draft_state_board.json` from the SQLite projections
pipeline. The legacy CLI (`scripts/run_draft.py`) is still available as
a fallback.

- `draft_state.json` — snapshot written atomically by the Flask writer
  endpoints on every pick. Reader (the browser JS) polls
  `/api/state?since=<version>` every 500ms using the delta protocol.
- `draft_state_board.json` — the ranked board (written once per session
  on `/api/new-draft`).
- `draft_state_delta.json` — per-version deltas.

State shape (new in 2026-04-24): `keepers`, `picks`, `on_the_clock`,
`undo_stack`, `projected_standings_cache`. The legacy `recommendations`
and `balance` fields are still tolerated by readers so the simulator
(unchanged) keeps working.

Writer endpoints in `web/app.py`:
- `POST /api/new-draft` — seed keepers + set on-the-clock.
- `POST /api/pick` — record a pick, advance snake order.
- `POST /api/undo` — pop the most recent live pick.
- `POST /api/on-the-clock` — manual override.
- `POST /api/reset` — delete state files (requires `{"confirm": "RESET"}`).
- `GET /api/recs?team=<name>` — top 10 ERoto-delta candidates (currently
  returns 501 until real board loading lands — see the post-rework TODOs).
- `GET /api/roster?team=<name>` — slots + replacement placeholders.
- `GET /api/standings` — fractional ERoto per team with uncertainty SDs
  (currently empty until `projected_standings_cache` is populated by
  `apply_pick` — see the post-rework TODOs).

If you change the state shape, both the writer
(`draft/draft_controller.py` + `draft/state.py`) and the reader
(`web/app.py` + `web/static/draft.js`) must be updated together.
