# Redis / KV Cache Map

Snapshot of how the season dashboard's Redis/KV state is written and read, to
support cleanup. Backend is Upstash on Render and a file-backed SQLite KV
locally (`data/kv_store.py`); both implement the same `KVStore` subset.

## Two key namespaces

1. **`cache:<name>`** — ephemeral dashboard payloads. Names enumerated in
   `data/cache_keys.py::CacheKey`; written via `write_cache(CacheKey.X, ...)`,
   read via `read_cache`/`read_cache_dict`. Overwritten every refresh.
2. **Bare keys** — durable "data layer" structures in `data/redis_store.py`:
   history hashes, projection blobs, game logs, the preseason blend/baseline.
   Hand-rolled `get_*`/`set_*`/`write_*` accessors, no `cache:` prefix.

## Writer jobs (four independent schedules)

| Job | Entry point | Writes |
|---|---|---|
| **Dashboard refresh** | `web/refresh_pipeline.py::run_full_refresh` (1447) | almost every `cache:*` key + the three history hashes; **also syncs game logs** (`_fetch_game_logs`, step 4 of `RefreshRun.run`) → `game_logs:*`, `game_log_totals:*`, `season_progress` |
| **ROS fetch** (FanGraphs) | `web/season_routes.py::_run_rest_of_season_fetch` (196), async thread, admin/QStash-triggered | `cache:ros_projections`, `cache:full_season_projections`; **also syncs game logs first** (season_routes:216) because ROS blending needs fresh `game_log_totals` |
| **Preseason one-offs** | `scripts/build_db.py` (116-117), `scripts/freeze_preseason_baseline.py` (137) | `blended_projections:*`, `preseason_baseline:{year}` |

Game-log sync (`mlb_game_logs.sync_game_logs`, incremental via a watermark) is
**not** a standalone job — it runs as step 4 of the dashboard refresh (before
`_build_projected_standings`, which depends on it) and again inside the ROS
fetch. The genuinely-separate input the refresh *consumes* is the **ROS fetch**
(`cache:ros_projections` / `cache:full_season_projections`): it runs on its own
admin/QStash schedule, so a refresh can run against stale-or-fresh ROS depending
on timing — that's the real cross-job skew, not game logs.

## `cache:*` keys (CacheKey)

| Key | Written by | Read by (surface) |
|---|---|---|
| `standings` | refresh:471 (+ `standings_history` snapshot 649) | season_data:598; routes:399/543/694/1793/1816 (standings table) |
| `projections` | refresh:807 (+ `projected_standings_history` 820) | routes:419/617/714/869/996/1545/1649 (projected column, deltaRoto) |
| `standings_breakdown` | refresh:823 | routes:438 (ERA/cat breakdown **modal**) |
| `monte_carlo` | refresh:1255 | routes:459 |
| `ros_projections` | **ros_pipeline** via `write_cache` | refresh:540; routes (many) |
| `full_season_projections` | **ros_pipeline** via `write_cache` | refresh:557 |
| `roster` | refresh:958 | routes |
| `opp_rosters` | refresh:698 | routes (trade/opponent views) |
| `lineup_optimal` | refresh:1038 | season_data:805; routes:525 |
| `leverage` | refresh:1175 | routes:706 |
| `rankings` | refresh:943 | season_data:411; routes:710/1313 |
| `roster_audit` | refresh:1119 | routes |
| `stash` | refresh:1145 | routes:593 |
| `spoe` | refresh:1294 | routes:1707 |
| `transactions` | refresh:1372 | refresh:1317 (prior-run diff only) |
| `transaction_analyzer` | refresh:1374 | routes:1745 |
| `streak_scores` | refresh:1424 | routes:500/528 |
| `probable_starters` | refresh:1076 | routes |
| `pending_moves` | refresh:487 | routes |
| `positions` (→ `cache:positions`) | refresh:1101 | routes:1314/1668 |
| `meta` | refresh:1444 | season_data:119 |

## Bare data-layer keys (redis_store)

| Key | Written by | Read by |
|---|---|---|
| `weekly_rosters_history` (hash) | refresh:640 `write_roster_snapshot` | `League.from_redis`, `get_latest_weekly_rosters` |
| `standings_history` (hash) | refresh:649 `write_standings_snapshot` | `League.from_redis`, `get_latest_standings` |
| `projected_standings_history` (hash) | refresh:820 `write_projected_standings_snapshot` | `get_projected_standings_history` |
| `blended_projections:{hitters,pitchers}` | build_db:116-117 | refresh:505; routes:166-167; transactions:560-561; spoe |
| `preseason_baseline:{year}` | freeze_preseason_baseline:137 | refresh:1250 |
| `game_logs:{season}:{id}:{group}` | sync_game_logs | `build_hitter_ytd_game_logs` (#111), `get_player_game_log` |
| `game_log_totals:{hitters,pitchers}` | sync_game_logs:246 | `build_hitter_ytd_game_logs`, `get_game_log_totals` |
| `game_logs:{season}:{dates,player_pos,fetched_through_utc}` | sync_game_logs | sync + AB attribution |
| `season_progress` | sync_game_logs:269 | dashboard header |
| `refresh:lock` | refresh job bodies (`acquire_refresh_lock`, SET NX EX 1800) | the durable cross-instance refresh lock; released in `finally` |

## Inconsistencies & cleanup targets

1. **RESOLVED (PR #112).** `positions` was written twice (`cache:positions`
   plus a bare `positions` via `set_positions` that nothing read). The bare
   write and the `get/set_positions` accessor pair are gone; refresh writes
   positions once via `write_cache(CacheKey.POSITIONS)`.

2. **RESOLVED (PR #112).** ROS/full-season projections were double-written
   via `write_cache` **and** a local `set_ros_projections`. `ros_pipeline`
   now writes both blobs through the single `write_cache` path; the
   `get_/set_ros_projections` + `get_/set_full_season_projections` accessor
   pairs are gone.

3. **PARTIALLY RESOLVED (this PR).** The headline YTD-vintage skew is fixed:
   the refresh now re-derives full-season (ROS + YTD) from the ROS-remaining
   blob + its own freshly-synced game logs (`_load_projections`), instead of
   reading the ROS-fetch job's frozen `cache:full_season_projections`, so the
   per-player full-season lines and the team-YTD overlay share one vintage.
   Still loosely coupled: standings/projected-column/breakdown remain separate
   `cache:*` keys consuming `cache:ros_projections` from the separate job, but
   they no longer mix YTD vintages. `cache:full_season_projections` is now
   effectively redundant (the refresh derives its own); removing the ROS job's
   write of it is a small follow-up.

4. **RESOLVED (this PR).** The breakdown modal's silent
   `contribution_stats = raw_stats * scale_factor` fallback (full_season x
   factor = the pre-#110 double-count) is removed: `PlayerContribution.from_dict`
   no longer fabricates it; an absent value stays empty (renders as visible
   zero, not plausible-wrong). The `season_routes` round-trip is kept only to
   normalize types and default a missing `team_ytd` to `{}`.

5. **RESOLVED (PR #112).** Provenance/version skew is now detectable: every
   `cache:*` payload carries a `{_job, _written_at}` envelope
   (`serialize_cache_payload`).

## Concurrency / durability

- The daily refresh and the ROS fetch are mutually exclusive across Render
  instances and QStash at-least-once redelivery via a durable KV lock
  (`refresh:lock`, SET NX EX 1800), not just the in-process slot. This closes
  the game-log rollup read-modify-write race (concurrent jobs could drop
  players from `game_log_totals`). Residual: a job exceeding the 1800s TTL
  could let a redelivery in — see follow-ups.
- `write_cache` now raises on a configured-backend write failure (it used to
  swallow), so a partial refresh can't report success; the job fails and
  QStash redelivers. `META` is written last as a de-facto commit marker.

## Open follow-ups

- **Lock release is now atomic** (`KVStore.compare_delete`: Lua CAS on
  Upstash, locked `DELETE ... WHERE value=?` on SQLite) -- the get-then-delete
  TOCTOU is closed. (Done.)
- **Remove the now-redundant `cache:full_season_projections` write.** The
  refresh derives full-season itself (#3 above); the ROS job's write of this
  key has no remaining reader. Left in place here only because 4 ros_pipeline
  tests assert it; collapse both in a focused follow-up.
- **Rollup as a hash:** move `game_log_totals:*` from one JSON blob to
  per-player hash fields so concurrent writers touch disjoint fields —
  defense-in-depth even if the durable lock's TTL is exceeded.
- **mlbam<->Yahoo id bridge:** team-YTD AB attribution joins on normalized
  name and now *excludes* same-name collisions (was: silently merged). A real
  id bridge would attribute the colliding name to the correct player instead
  of dropping it.
