# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"          # Install package + dev deps (editable mode)
pytest -v                        # Run all tests
pytest tests/test_sgp/ -v        # Run one test directory
pytest tests/test_draft/test_board.py::test_apply_keepers -v  # Single test
python scripts/run_draft.py      # Interactive draft CLI + web dashboard
python scripts/run_draft.py --mock --position 8 --teams 10   # Mock draft
python scripts/run_lineup.py     # In-season lineup optimizer (requires Yahoo auth)
python scripts/simulate_draft.py -s two_closers --scoring-mode vona  # Simulate
python scripts/compare_strategies.py  # Compare all strategies (slow, ~10min)
python scripts/fetch_positions.py     # Cache Yahoo position data (run before draft)
```

## Architecture

**Yahoo 5x5 roto keeper league toolkit** — draft assistant, strategy simulator, and in-season optimizer.

### Data pipeline (draft)

FanGraphs CSVs (`data/projections/`) → `data/projections.py` blends systems → `sgp/player_value.py` calculates per-category SGP → `sgp/replacement.py` computes position-specific replacement levels → `sgp/var.py` assigns VAR per player → `draft/board.py` assembles the ranked board.

### Two scoring modes in the recommender

- **VAR** (Value Above Replacement): static ranking, `player SGP - replacement level at position`
- **VONA** (Value Over Next Available): dynamic, `player SGP - best remaining in same bucket after opponents' next N picks`. Uses 3 buckets (hitter/SP/closer) — position-level VONA was tested and regressed badly.

When scoring_mode is "vona", the recommender blends VONA with leverage weights based on category gaps.

### Draft CLI ↔ Web dashboard

`run_draft.py` writes JSON state files atomically (tempfile + rename) after each pick. Flask dashboard polls `/api/state?since=<version>` every 2s using a delta protocol — board sent once, only changed fields after that. Three files: `draft_state.json`, `draft_state_board.json`, `draft_state_delta.json`.

### Strategy system

Each strategy is a `pick_*()` function in `draft/strategy.py` registered in the `STRATEGIES` dict. Strategies control closer timing, AVG floors, and category protection rules. The recommender handles player ranking; strategies add constraints on top. Config-driven via `league.yaml` fields `strategy` and `scoring_mode`.

### In-season optimizer

Connects to Yahoo API → pulls roster + standings → `lineup/leverage.py` identifies which categories are closest to gaining/losing a standings point → `lineup/optimizer.py` uses Hungarian algorithm to assign hitters to slots maximizing leverage-weighted SGP → `lineup/waivers.py` evaluates add/drop swaps.

## Key design decisions

- **Replacement levels recalculate per pick** from the available pool, not the original board. This reflects live positional scarcity during the draft.
- **Keeper removal uses normalized names** (accent-stripped, lowercased) with highest-VAR tie-breaking when multiple board entries share a name (e.g., two Juan Sotos).
- **Player IDs are `name::player_type`** (e.g., `"Juan Soto::hitter"`) to disambiguate same-name players.
- **Position collisions** between MLB players and prospects with the same normalized name are resolved by keeping the entry with more eligible positions.
- **Scripts inject `src/` into sys.path** rather than relying solely on the editable install. Both work, but scripts do it explicitly for robustness.
- **Projection files** must be named `{system}-hitters.csv` / `{system}-pitchers.csv`. The blend function validates and gives actionable FanGraphs download links on error.

## Config

All league settings live in `config/league.yaml`. Key fields: `draft.strategy`, `draft.scoring_mode`, `keepers`, `roster_slots`, `sgp_denominators`. See `config/league.yaml.example` for the template. OAuth credentials go in `config/oauth.json` (gitignored).
