# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Correctness over speed

This project makes real decisions (draft picks, lineup choices, trade evaluations) based on the numbers it produces. A wrong answer that looks plausible is worse than no answer — it will propagate into strategy recommendations, simulations, and memory files that mislead future sessions. Verify claims against the actual code and config before stating them. When summarizing simulation results or strategy conclusions, check that what you're saying matches what the code actually does and what the config actually says. If you're unsure, read the source — don't guess from memory.

## Tests are the guardrail — don't modify failing tests without justification

Tests are the primary regression-prevention mechanism in this repo. A failing test means the code is broken, not the test. Do not loosen an assertion, change expected values, skip, or delete a test to make it pass — fix the code instead. If you genuinely believe a test is wrong (the requirement changed, the test asserts incidental behavior, the fixture is stale), state the reason explicitly and get confirmation from the user before editing. A silently "fixed" test is worse than a broken build: it removes a guardrail without anyone noticing.

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

## Reuse before writing

Before writing new logic, check whether the codebase already solves the problem. This project has been built incrementally and many patterns exist in scripts or library modules that handle edge cases you'll miss if you rewrite from scratch. Specifically:

- **Search `src/` and `scripts/` for existing functions** before implementing computation logic. If `summary.py` or `run_lineup.py` already does what you need, extract a shared function into the appropriate library module rather than duplicating the code.
- **Check how existing scripts handle the same data.** Yahoo API data has quirks (case mismatches like `"Util"` vs `"UTIL"`, missing stats in early season, inconsistent stat ID mappings). The existing scripts have already solved these — read them before writing new Yahoo integration code.
- **Projection paths include the season year** (`data/projections/2026/`, not `data/projections/`). Config fields like `season_year` exist for this reason.
- **`simulation.py` is the shared Monte Carlo module.** Use `run_monte_carlo()` and `apply_management_adjustment()` — don't rewrite MC loops.

When building a new feature that orchestrates existing modules (like the season dashboard refresh pipeline), read the scripts that already do similar orchestration (`summary.py`, `run_lineup.py`) and follow their patterns for data loading, projection matching, and edge case handling.

## Config

All league settings live in `config/league.yaml`. Key fields: `draft.strategy`, `draft.scoring_mode`, `keepers`, `roster_slots`, `sgp_denominators`. See `config/league.yaml.example` for the template. OAuth credentials go in `config/oauth.json` (gitignored).
