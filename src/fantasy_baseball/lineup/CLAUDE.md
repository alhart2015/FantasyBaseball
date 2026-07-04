# In-season lineup subsystem

The in-season optimizer and waiver evaluator. Entry points: `scripts/run_lineup.py`, `scripts/run_season_dashboard.py`, and the Flask app in `web/season_app.py`.

## Rules

- **Use `simulation.run_monte_carlo()` / `simulation.run_ros_monte_carlo()`** -- do not rewrite MC loops here. The season dashboard's refresh pipeline (`web/refresh_pipeline.py`) leans on the same module.
- **Yahoo API quirks**: case mismatches like `"Util"` vs `"UTIL"`, missing stats early in the season, inconsistent stat-ID mappings -- the existing scripts already handle these. Read `run_lineup.py` before writing new Yahoo integration code.
