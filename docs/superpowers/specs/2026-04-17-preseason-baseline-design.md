# Preseason baseline: frozen Opening-Day MC

## Problem

`RefreshRun._run_monte_carlo` in `src/fantasy_baseball/web/refresh_pipeline.py`
(lines 748-770) runs two 1000-iteration Monte Carlo simulations on every
refresh. Progress messages label them "Pre-season" MCs, but they actually
simulate against `self.matched` + `self.opp_rosters`, which during the
season are matched against ROS projections (see lines 361-362). So these
MCs are neither preseason nor frozen — they re-compute against live
roster + ROS-projection state every refresh.

Two problems fall out:

1. **Wasted work.** Every refresh spends ~2000 MC iterations recomputing
   against inputs that are better served by the ROS MCs (which already
   account for year-to-date actuals + ROS projections). The existing
   `_run_ros_monte_carlo` does the meaningful live simulation.
2. **Missing baseline.** There is no "what we expected on Opening Day"
   view anywhere in the product. The standings page shows actual standings
   and ROS-projected standings, but you cannot see how far reality has
   drifted from the preseason forecast.

## Solution

Replace `_run_monte_carlo` with a frozen preseason baseline: compute the
two MCs once per season using Opening-Day rosters + preseason projections,
store the result in Redis, and read it on every refresh.

### Storage

A new Redis key per season: `preseason_baseline:{season_year}` (e.g.
`preseason_baseline:2026`). Payload:

```json
{
  "base": { ... },                // run_monte_carlo output, use_management=False
  "with_management": { ... },     // run_monte_carlo output, use_management=True
  "meta": {
    "frozen_at": "2026-04-18T12:34:56Z",
    "season_year": 2026,
    "roster_date": "2026-03-27",
    "projections_source": "blended"
  }
}
```

`base` and `with_management` keep the exact shape of the current
`monte_carlo` cache inner dicts, so `season_routes.py:250-256` (which
already reads `raw_mc["base"]` / `raw_mc["with_management"]`) needs no
changes.

### Generation script

New file: `scripts/freeze_preseason_baseline.py`.

Flow:

1. Load config via `load_config("config/league.yaml")`; read
   `season_start`, `season_year`, `team_name`, `roster_slots`.
2. Yahoo auth + `get_league()` — reuse the pattern from
   `RefreshRun._authenticate`.
3. For every team in `league.teams()`, call
   `fetch_roster(league, team_key, day=config.season_start)`. This returns
   the roster as it stood on Opening Day (Yahoo supports historical
   `day=` fetches for the current season).
4. Load preseason projections from Redis via
   `get_blended_projections(client, "hitters")` and
   `get_blended_projections(client, "pitchers")`.
5. Match each team's roster to preseason projections with
   `match_roster_to_projections` (reuse the function used by
   `_match_roster_to_projections` in the refresh pipeline).
6. Compute `h_slots` / `p_slots` the same way `_run_monte_carlo` does
   (lines 751-753).
7. Run the two MCs:
   ```python
   base = run_monte_carlo(
       all_team_rosters, h_slots, p_slots, config.team_name,
       n_iterations=1000, use_management=False,
   )
   with_mgmt = run_monte_carlo(
       all_team_rosters, h_slots, p_slots, config.team_name,
       n_iterations=1000, use_management=True,
   )
   ```
8. Write `{"base": base, "with_management": with_mgmt, "meta": {...}}`
   to Redis under `preseason_baseline:{season_year}`.

CLI args:

- `--season-year <int>` — default `config.season_year`.
- `--force` — overwrite an existing baseline. If the key exists and
  `--force` is not set, the script prints the existing `frozen_at` and
  exits nonzero so the user explicitly confirms a re-freeze.

Helpers reused verbatim: `fetch_roster`, `match_roster_to_projections`,
`run_monte_carlo`, `get_blended_projections`. No new math.

### Redis store helpers

Add to `src/fantasy_baseball/data/redis_store.py`:

- `get_preseason_baseline(client, season_year) -> dict | None`
- `set_preseason_baseline(client, season_year, payload: dict) -> None`

These wrap the JSON serialization and key format
(`preseason_baseline:{season_year}`). Follow the same pattern as
`get_blended_projections` / the existing getters in that module.

### Refresh pipeline changes

In `src/fantasy_baseball/web/refresh_pipeline.py`:

1. Delete `_run_monte_carlo` (lines 748-770).
2. Delete its call at line 191.
3. Delete `self.base_mc` / `self.mgmt_mc` attribute inits at lines
   149-150. The `_run_ros_monte_carlo` method no longer reads
   `self.base_mc` / `self.mgmt_mc`; it reads the frozen baseline from
   Redis instead.
4. In `_run_ros_monte_carlo`, replace the `write_cache("monte_carlo", ...)`
   block (lines 829-834) with:
   ```python
   from fantasy_baseball.data.redis_store import get_preseason_baseline

   baseline = get_preseason_baseline(_redis_client, self.config.season_year) or {}
   if not baseline:
       self._progress(
           "Preseason baseline missing — run scripts/freeze_preseason_baseline.py"
       )

   write_cache("monte_carlo", {
       "base": baseline.get("base"),
       "with_management": baseline.get("with_management"),
       "rest_of_season": self.rest_of_season_mc,
       "rest_of_season_with_management": self.rest_of_season_mgmt_mc,
   }, self.cache_dir)
   ```

When the baseline is missing, `base` and `with_management` land in the
cache as `None`. `season_routes.py:250-264` already guards with truthy /
membership checks for the ROS keys; the same pattern applied to `base`
collapses the baseline columns to hidden when missing (see UI changes
below).

### UI changes

`src/fantasy_baseball/web/season_routes.py`:

- In the `/standings` handler (around lines 248-256), the existing code
  assigns `mc_data = format_monte_carlo_for_display(raw_mc.get("base", raw_mc), ...)`
  unconditionally. Change to only populate `mc_data` / `mc_mgmt_data`
  when the value is truthy, matching the pattern already used for the
  ROS keys.

`src/fantasy_baseball/web/templates/season/standings.html`:

- Relabel the `mc` / `mc_mgmt` section headings from implicit "Preseason"
  wording to "Opening Day baseline".
- Add a tooltip/subtext showing `meta.roster_date` so the user can see
  exactly when the baseline was captured.
- Continue hiding the section when its data is falsy (existing template
  conditional).

To surface the baseline metadata in the template, the refresh pipeline
lifts `baseline["meta"]` into a top-level `baseline_meta` field of the
`monte_carlo` cache dict. `season_routes.py` passes that field through
to the template as a new `baseline_meta` context variable, and
`standings.html` reads `baseline_meta.roster_date` for the tooltip. The
refresh writer is therefore:

```python
write_cache("monte_carlo", {
    "base": baseline.get("base"),
    "with_management": baseline.get("with_management"),
    "baseline_meta": baseline.get("meta"),
    "rest_of_season": self.rest_of_season_mc,
    "rest_of_season_with_management": self.rest_of_season_mgmt_mc,
}, self.cache_dir)
```

### Tests

- `tests/test_web/test_refresh_steps.py`: the `_run_monte_carlo` test is
  removed. Add a test covering `_run_ros_monte_carlo`'s cache-write step
  that (a) reads a mocked preseason baseline from Redis and propagates
  `base` / `with_management` into the `monte_carlo` cache, and (b)
  writes `None` for both when the baseline is missing.
- `tests/test_web/_refresh_fixture.py`: the mock-MC plumbing that today
  stubs `run_monte_carlo` twice (base + mgmt) can drop the preseason
  half; only the ROS MC still runs in-pipeline.
- New `tests/test_scripts/test_freeze_preseason_baseline.py`: exercises
  the script end-to-end with mocked Yahoo roster fetch, mocked Redis,
  and a small stubbed `run_monte_carlo`. Verifies the Redis write shape
  and the `--force` guard.

### 2026 bootstrap

Once this ships, run:

```
python scripts/freeze_preseason_baseline.py
```

The script fetches all teams' rosters at `2026-03-27`, matches to the
preseason projections already in Redis, runs the two MCs, and writes
`preseason_baseline:2026`. From that point on, every refresh reads this
artifact and skips the ~2000-iteration live MC.

## Risks / edge cases

- **Yahoo historical roster fetch on-season.** `team.roster(day=<past>)`
  must return Opening-Day rosters three weeks after the fact. Yahoo
  supports this for the current season. Verify during implementation by
  running the script against the live 2026 league; if a date too far in
  the past is rejected, fall back to reconstructing via transaction
  history (out of scope here — log an error and exit).
- **Traded players.** `team.roster(day=<date>)` returns the roster as it
  was on that date, so players traded after Opening Day appear on their
  original team in the baseline. Correct behavior for the "what we
  expected" view.
- **Players without preseason projections.** A mid-April waiver pickup
  of a player who was not in the preseason projection pool does not
  apply here — the baseline uses Opening-Day rosters, and every player
  rostered at that point had a preseason projection (or was missing in
  projections already, which `match_roster_to_projections` handles with
  a null row, identical to today).
- **`scripts/freeze_preseason_baseline.py` rerun semantics.** Running
  without `--force` when a baseline already exists must exit nonzero
  with a clear message showing the existing `frozen_at`. This prevents
  an accidental re-freeze (e.g., after the script has been updated to
  fix a bug, but before the new artifact is actually intended).

## Related work

A broader `initialize_season()` helper is queued as a TODO item. It
would bundle all preseason-to-season prep (freeze baseline, archive
prior-season artifacts, warm caches, validate `season_start` /
`season_year` drift) into a single entry point. `freeze_preseason_baseline.py`
becomes one callable subroutine of that helper; this spec does not
block on it.

## Out of scope

- Reconstructing a 2025 (or earlier) baseline retroactively.
- Displaying the baseline anywhere beyond the standings page.
- Comparing per-team drift between baseline and current (a natural
  follow-up once the baseline exists).
- The `initialize_season()` orchestrator itself (tracked separately in
  `TODO.md`).
