# In-season lineup subsystem

The in-season optimizer and waiver evaluator. Entry points: `scripts/run_lineup.py`, `scripts/run_season_dashboard.py`, and the Flask app in `web/season_app.py`.

## Pipeline

Yahoo API → pull roster + standings → `lineup.leverage` identifies which categories are closest to gaining or losing a standings point → `lineup.optimizer` uses the Hungarian algorithm to assign hitters to slots, maximizing ΔRoto → `lineup.roster_audit` evaluates drop-for-FA swaps against the projected standings.

## Key modules

- `leverage.py` — category-gap analysis that drives everything downstream.
- `optimizer.py` — Hungarian assignment for hitter slotting (ΔRoto objective).
- `roster_audit.py` — drop/FA swap evaluation using ΔRoto.
- `delta_roto.py` — ΔRoto primitives used by the optimizer and trade evaluator.
- `waivers.py` — free-agent fetch + projection matching helper (used by refresh pipeline).
- `matchups.py`, `yahoo_roster.py` — supporting logic and Yahoo-side glue.

## Monte Carlo

Use `simulation.run_monte_carlo()` and `simulation.apply_management_adjustment()` — do not rewrite MC loops here. The season dashboard's projection refresh (`web/refresh_pipeline.py`) leans on the same module.

## Yahoo API quirks

Case mismatches like `"Util"` vs `"UTIL"`, missing stats early in the season, inconsistent stat-ID mappings — the existing scripts already handle these. Read `run_lineup.py` before writing new Yahoo integration code.
