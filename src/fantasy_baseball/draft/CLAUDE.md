# Draft subsystem

The draft assistant and simulator. Entry points: `scripts/run_draft_dashboard.py` (primary), `scripts/simulate_draft.py`, `scripts/compare_strategies.py`; `scripts/run_draft.py` is the legacy CLI fallback.

## Rules and invariants

- **Replacement floors are empirical waiver lines** (`position_aware_replacement_levels`) -- static per draft, not per-pick. They reflect what's actually free on this league's waiver wire, so VAR measures value over the waiver alternative. Don't replace them with theoretical positional baselines without evidence.
- **Scoring modes** are selected via `draft.scoring_mode` in `config/league.yaml`; the implementations and their docstrings are the reference. **Position-level VONA was tested and regressed badly -- do not reintroduce it without new evidence.** (Bucket-level VONA -- hitter / SP / closer -- is the surviving design.)
- **All pick sources route through `recommend(ctx, strategy=...)` in `draft/recommend.py`** -- the dashboard, the simulator, and `compare_strategies.py` alike. Never add a pick path that bypasses this seam; extend `RecommendContext` / `RankedPick` instead.
- **Strategies are orthogonal overlays** registered in `OVERLAYS`, keyed by plain string, combinable with any scoring mode. If you rename a strategy, grep for the string literal in config files, simulation scripts, and tests -- not just the function name.
- **Dashboard state changes touch both sides.** The writer (`draft/draft_controller.py` + `draft/state.py`) and the reader (`web/app.py` + `web/static/draft.js`) must be updated together whenever the state shape changes.
- **After projection CSVs change**, run the dashboard once with `--rebuild-board` to regenerate the cached board before drafting.
