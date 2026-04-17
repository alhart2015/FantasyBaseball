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

## Draft CLI ↔ dashboard

`run_draft.py` writes JSON state files **atomically** (tempfile + rename) after each pick. The Flask app in `web/app.py` polls `/api/state?since=<version>` every 2s using a delta protocol — the full board is sent once, then only changed fields on subsequent polls. Three files:

- `draft_state.json` — snapshot (teams, picks, current state)
- `draft_state_board.json` — the ranked board (sent once per session)
- `draft_state_delta.json` — per-version deltas

If you change the state shape, both the writer (`run_draft.py` + `draft/state.py`) and the reader (`web/app.py` + the JS in `web/static/`) must be updated together.
