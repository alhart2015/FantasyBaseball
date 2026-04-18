# Player Comparison ROS Regression — Root Cause & Fix

## Symptom

The player-comparison tool was showing **preseason** saves totals for
pitchers whose role changed during the season. Test case: **Jakob Junis**
was displayed at ~4 SV (his preseason projection) even though he had
assumed the TEX closer role and real ROS projections had him around 19.

The same bug affected downstream surfaces that read
`cache:ros_projections` — roster audit, ΔRoto calculations, and the
"before/after swap" deltas on the comparison page.

## How the bug behaved

1. The scheduled **admin ROS fetch** job (`_run_rest_of_season_fetch`,
   triggered daily at 05:00 UTC on Render) did its job correctly:
   downloaded fresh FanGraphs CSVs to
   `data/projections/{year}/rest_of_season/{today}/`, blended them in
   memory, and wrote the result to `cache:ros_projections` in Upstash
   Redis. Today's job log confirmed: *"Persisted 4332 ROS hitters +
   5734 ROS pitchers to Redis"*.
2. One hour later, the **refresh** job ran (`run_full_refresh`, 06:00
   UTC). It called `blend_and_cache_ros()` a second time from
   `refresh_pipeline._load_projections()`.
3. Render's filesystem is ephemeral *per instance* — the admin fetch
   and the refresh frequently run on **different Render instances**.
   So the CSVs today's fetch dropped to disk weren't visible to the
   refresh's instance. The only CSV snapshot the refresh's instance
   had was `data/projections/2026/rest_of_season/2026-03-30/` — the
   stale snapshot committed to git.
4. `blend_and_cache_ros()` scans for the latest dated dir, picks
   `2026-03-30`, blends those stale CSVs, and **overwrites**
   `cache:ros_projections` in Redis. Today's refresh log captured this
   in plain sight: *"Loading steamer from 2026-03-30"*.
5. Every reader downstream (player comparison, roster audit, ΔRoto)
   then served values consistent with preseason projections, because
   that's what the overwritten Redis cache now held.

## Why the earlier fix didn't stick

Commit `2a11c1e` (Apr 14) established the invariant:

> Redis is the authoritative source for ROS projections; disk CSVs are
> only a fallback.

Commit `9592b63` (Apr 15, "feat(redis): drop SQLite staging from ROS
projections pipeline") refactored both the admin fetch and the refresh
to share `blend_and_cache_ros`. In the refresh path it wrapped the call
in `try / except FileNotFoundError` with a comment that assumed Render
had no local CSVs. But the assumption was wrong — the `2026-03-30/`
snapshot is committed to git and ships with every Render deploy, so
the `FileNotFoundError` branch never fired and Redis kept getting
overwritten.

## Evidence collected (Upstash, 2026-04-18)

| Source | Junis SV |
|---|---|
| `blended_projections:pitchers` (preseason) | **0.64** |
| `cache:ros_projections` (post-overwrite) | **3.74** |
| Expected fresh ROS (FanGraphs 2026-04-18) | ~19 |

Refresh log 2026-04-18 06:05:35 → *"Loading steamer from 2026-03-30"*
— direct confirmation the refresh was blending the stale snapshot.

## Fix

**Scope:** `src/fantasy_baseball/web/refresh_pipeline.py::_load_projections`

Remove the `blend_and_cache_ros()` call from the refresh. The refresh
now **only reads** `cache:ros_projections` from Redis. The admin fetch
endpoint is the sole authoritative writer for that key, matching the
design intent established in `2a11c1e`.

Also removed:

- The patched `_ros_pipeline_blend` side-effect in
  `tests/test_web/_refresh_fixture.py` (the fixture now seeds
  `cache:ros_projections` directly, same as every other Redis seed).

Added regression guards in `tests/test_web/test_refresh_pipeline.py`:

- `test_refresh_does_not_call_blend_and_cache_ros` — asserts the
  refresh never invokes `blend_and_cache_ros`. If a future refactor
  re-introduces the write, this test blows up loudly.
- `test_refresh_preserves_existing_ros_projections` — snapshots
  `cache:ros_projections` before/after the refresh and asserts
  byte-equality. If the refresh ever writes to that key again (for
  any reason), this fails.

## Operational notes

- On Render, the admin fetch runs 60 minutes before the refresh, so
  `cache:ros_projections` is always fresh by the time refresh needs
  it.
- If the admin fetch fails for a day, the refresh reads yesterday's
  Redis cache — still far better than silently serving the committed
  March 30 snapshot.
- For local dev: populate `cache:ros_projections` by hitting the
  `/admin/fetch_ros_projections` endpoint once. The refresh will then
  read it from Upstash for all subsequent runs.

## Verification

- Full test suite: **1123 passed** (`pytest`).
- Regression tests pass with fix and fail without it (verified by
  temporarily reverting the `refresh_pipeline.py` diff).
- No new ruff, format, or vulture findings introduced.
