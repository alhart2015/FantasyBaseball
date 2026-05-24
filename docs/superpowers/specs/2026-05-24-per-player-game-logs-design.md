# Per-Player Game Logs (Incremental, Box-Score Driven) -- Design

- Date: 2026-05-24
- Status: Approved (pending spec review)
- Surface: refresh pipeline game-log step (`src/fantasy_baseball/web/refresh_pipeline.py::_fetch_game_logs`)
- New/changed modules: `src/fantasy_baseball/data/mlb_game_logs.py` (rewrite), `src/fantasy_baseball/data/redis_store.py` (new helpers), `src/fantasy_baseball/analysis/game_logs.py` (parser refactor + box-score parsers)

## Problem

`fetch_game_log_totals` does a **full rebuild every refresh**: it enumerates all 30
team rosters (~1,500 players), fetches each player's full-season gameLog, sums the
per-game rows into season totals, and overwrites two Redis blobs
(`game_log_totals:hitters` / `:pitchers`). Two consequences:

1. **The per-game rows are discarded.** Only season sums survive, so nothing in the
   app can answer per-game questions (recent form windows, game-by-game history)
   without re-fetching the entire season from the MLB API.
2. **Two-way players are wrong.** Each player is classified as a single
   `PlayerType` from roster position (`pos_type == "Pitcher"`), so Ohtani is counted
   on exactly one side -- his other half is silently dropped from the rollup.

We want to (a) persist the raw per-player per-game logs keyed by player id, (b) keep
the rolled-up totals as the thing that powers existing calcs, and (c) stop
re-pulling the whole season every cycle -- pull only what is new or changed since the
last refresh.

## Goals

- Persist per-player per-game logs keyed by player id, retained across refreshes.
- Keep `game_log_totals:hitters` / `:pitchers` as the calc-powering rollup, now
  **derived** from the per-player store (no downstream consumer changes).
- Incremental fetch: each refresh pulls only games that are new or corrected since a
  precise high-water mark, not the full season.
- **Handle Ohtani (and any two-way player) correctly** -- both his hitting and
  pitching land in the store and in the respective rollups.
- Filter out position-player mop-up pitching from the pitcher side, without
  mis-filtering genuine two-way players.

## Non-goals

- No per-game data surfaced in the UI yet (this is a data-layer change; consumers
  come later).
- No change to the separate `fetch_all_game_logs` JSON-file cache used by the
  streaks/backtest path -- it is untouched.
- No postseason or spring-training ingestion. Regular season (`gameType == "R"`) only.
- No growth of the `KVStore` abstraction (no sorted sets / pipelines). We stay
  within the existing `get/set/delete/keys/mget` + hash subset.

## Verified facts (MLB Stats API probes, 2026-05-24)

These were confirmed empirically against `statsapi.mlb.com/api/v1`; the design
depends on them.

1. **gameLog honors a date filter server-side.** `stats=gameLog&startDate&endDate`
   (ISO `YYYY-MM-DD` or `MM/DD/YYYY`) returns only games in range. (Used for the
   one-time backfill enumeration alternative; primary path uses box scores.)
2. **`game/changes?updatedSince=<ISO-UTC>` is usable once scoped.** Unscoped it is a
   firehose across all levels/years (junk gamePks were 2008 Mexican League, 2010
   exhibition, 2021 DSL). Scoped `+ sportId=1 & season=<year>` it returns a clean,
   complete, non-truncated set (163 for an 8-month-accumulated off-season window;
   ~15-30 per cycle in-season). It filters by **change time**, so it catches new
   games AND corrections to older games in one call.
3. **Box scores carry both stat blocks under one player id.** `game/{gamePk}/boxscore`
   -> `teams.{home,away}.players.ID{mlbam}.stats.{batting,pitching}`. Ohtani's
   2025-09-23 game (gamePk 776213) entry `ID660271` has batting (PA=4, AB=3) AND
   pitching (IP=6.0, K=8) populated. `teams.{side}.pitchers` lists who pitched.
4. **Box-score stat keys match gameLog stat keys** (`plateAppearances`, `atBats`,
   `hits`, `runs`, `homeRuns`, `rbi`, `stolenBases`, `baseOnBalls`;
   `inningsPitched`, `strikeOuts`, `earnedRuns`, `wins`, `saves`, `gamesStarted`,
   `gamesPlayed`). One extraction routine serves both.
5. **`primaryPosition.code` is the only safe two-way discriminator.** Ohtani =
   `'Y'` (Two-Way Player) from BOTH the people endpoint and the team_roster entry;
   his **in-game** box-score position is `'10'` (DH). Cole = `'1'` (P), Betts =
   `'6'` (SS), Judge = `'9'` (RF). Filtering on in-game position would drop Ohtani's
   pitching -- the trap this design must avoid.

## Redis schema

All keys go through `redis_store` (the module owns the schema; no inline `kv.get`
elsewhere). `{season}` is `config.season_year`.

| Key | Type | Contents |
|---|---|---|
| `game_logs:{season}:{mlbam_id}:hitting` | string (JSON) | `{"name": str, "games": [GameRow, ...]}` |
| `game_logs:{season}:{mlbam_id}:pitching` | string (JSON) | `{"name": str, "games": [GameRow, ...]}` |
| `game_logs:{season}:fetched_through_utc` | string | ISO-8601 UTC high-water mark |
| `game_logs:{season}:player_pos` | string (JSON) | `{mlbam_id: primaryPosition_code}` cache |
| `game_logs:{season}:dates` | string (JSON) | sorted list of distinct `"YYYY-MM-DD"` ingested |
| `game_log_totals:hitters` / `:pitchers` | string (JSON) | **unchanged** rollup; now derived |
| `season_progress` | string (JSON) | **unchanged**; `games_elapsed = len(dates)` |

`GameRow` (hitting): `{gamePk, gameNumber, date, pa, ab, h, r, hr, rbi, sb}`.
`GameRow` (pitching): `{gamePk, gameNumber, date, ip, k, er, bb, h_allowed, w, sv}`.

- Per-player keys are keyed by **player id and group**. A two-way player has both a
  `:hitting` and a `:pitching` key. There is no `type` guess anywhere.
- `games` is kept sorted by `date`; the **upsert key is `gamePk`** (globally unique),
  so a doubleheader stores two rows that share a date but differ by gamePk
  (`gameNumber` retained as metadata).
- We considered one Redis hash per group (`hset` field = mlbam_id). Per-player string
  keys were chosen instead: `keys()` + `mget()` (both in the abstraction) cover
  full rebuild, and the rollup is maintained incrementally so we rarely read the
  whole store. Either fits under the limits; per-key strings keep each value tiny.

### Sizing vs free-tier limits (live values from the `fantasybaseball` DB)

Full season raw store ~6-8 MB (verified: 69 bytes/hitter row pretty, ~80k rows).
DB caps: disk 256 MB (currently 12.8 MB used), max entry size 100 MB, **max request
size 10 MB**, 500k commands/month (using ~600-3,500/day). Per-player keys are a few
KB each -- far under the 10 MB request cap that a single mega-blob would flirt with
by September. Adding ~6-8 MB of raw logs is trivial against 256 MB.

## High-water mark (no margin, no timezone fudge)

The watermark is a precise UTC instant, captured at the **start** of the fetch step,
persisted only on a fully successful pass:

```
t_start = datetime.now(timezone.utc)            # captured before any MLB read
since   = get_game_logs_watermark(season)       # previous t_start, or None
... fetch + upsert all targeted games ...
if all_games_processed_ok:
    set_game_logs_watermark(season, t_start.isoformat())   # advance only on success
```

Rationale (this replaces the hand-wavy "minus 1 day for clock skew" from
brainstorming, which was wrong):

- **Timezones never enter.** `datetime.now(timezone.utc)` is an unambiguous instant
  and `updatedSince` takes UTC ISO. We do not reuse the human-facing `last_refresh`
  (local, minute-precision) display string for this; that conflation is what created
  a fake need for a margin. The UI keeps showing `last_refresh` unchanged; the feed
  uses this dedicated key.
- **Capture-before-read closes the only real gap.** Anything that changes during or
  after our fetch has change-time `>= t_start`, so the next run (`updatedSince =
  t_start`) re-catches it. The gamePk-keyed upsert is idempotent, so the tiny
  re-overlap is harmless.
- **Advance only on success.** If any targeted box score fails to fetch, the
  watermark is left unchanged and the whole window is re-attempted next run
  (idempotent). Guarantees no permanent loss from a transient error. (A box score
  that fails persistently is logged loudly; v1 accepts re-attempting it each run.)

## Fetch flow

Player discovery is now **game-centric** -- box scores tell us which players played,
so the 30-team roster enumeration (and its `player_type` guess) is removed from this
step.

### Backfill (watermark missing -> first run / new season)

1. Enumerate season gamePks: `schedule?sportId=1&season&gameType=R`, status Final.
2. Fetch each box score (15-worker pool, as today), parse, upsert per-player rows.
3. Build the rollup from the full store, write `game_log_totals:*`, `dates`,
   `season_progress`.
4. Set the watermark to `t_start`.

~2,400 box-score calls, once. Re-runs if it fails before step 4 (idempotent).

### Incremental (watermark present -> normal refresh)

1. `t_start = now(UTC)`; `since = watermark`.
2. `game/changes?updatedSince=since&sportId=1&season={season}` -> changed gamePks;
   keep `gameType == "R"` and status Final.
3. Fetch those box scores, parse, upsert per-player rows (by gamePk).
4. For each affected `(mlbam_id, group)`: recompute that player's season total from
   its merged rows; update only those entries in the rollup blobs; union new dates
   into `dates`; rewrite `season_progress`.
5. On full success, set watermark `= t_start`.

~15-30 box scores per cycle vs ~1,500 gameLog calls today.

## Ohtani / position-player pitching filter

For each box score, for each player with a non-empty stat block:

- **Batting block present** -> always record a hitting `GameRow` under
  `:{mlbam_id}:hitting`. (Pitchers who bat are negligible under the universal DH and
  harmless; not filtered.)
- **Pitching block present** -> record a pitching `GameRow` under `:pitching`
  **only if** `primaryPosition.code in {"1", "Y"}` (Pitcher or Two-Way). A position
  player (codes 2-10, "O", etc.) appearing in `teams.{side}.pitchers` has their
  pitching line dropped.

`primaryPosition.code` is read from the `game_logs:{season}:player_pos` cache.
Cache miss -> batch-fetch via `people?personIds=...` for all unseen ids in the
current games (one call), update the cache. Box-score `player.person` is shallow
(id/name only), so position cannot come from the box score itself.

**Ohtani safety is explicit and tested:** code `'Y'` keeps both blocks; a regression
test uses his real two-way box score (gamePk 776213) and asserts both a hitting row
(AB=3) and a pitching row (IP=6, K=8) are stored under id 660271 and that he appears
in both `game_log_totals:hitters` and `:pitchers`.

## Rollup derivation

`game_log_totals:hitters` / `:pitchers` keep their existing key names, JSON shape
(`{mlbam_id: {"name", <counting stats>}}`), and `set_game_log_totals` /
`get_game_log_totals` helpers -- downstream (`compute_rankings_from_game_logs`,
`_load_game_log_totals`) is unchanged. The rollup is now derived: a player's totals =
sum over the `games` in their per-player key. Ohtani contributes to **both** rollups.

Behavior change to note: because the per-player store is retained, a player **dropped
from rosters mid-season keeps contributing** to the rollup (today they vanish on the
next rebuild). This is more correct and harmless to id/name lookups; flagged because
it differs from current output.

## Parsing and code reuse

`analysis/game_logs.py` currently parses the gameLog **split** shape
(`split["stat"]` -> stat dict). Refactor the field extraction into
`_hitter_stats_from(stat: dict)` / `_pitcher_stats_from(stat: dict)` (keeping the
`"6.1" -> 6+1/3` innings conversion) and have both the existing gameLog parsers and
the new box-score parsers call them. New box-score parsers read
`player["stats"]["batting"]` / `["pitching"]` and attach `gamePk`, `gameNumber`,
`date` from the game context. No duplicated stat vocabulary.

## Error handling

- Per box-score fetch failure: log, skip that game, mark the pass as not-fully-ok ->
  watermark not advanced (window re-attempted next run).
- `game/changes` failure: abort the incremental step, leave watermark unchanged, let
  the rest of the refresh proceed (matches the current step's resilience posture).
- `people?personIds` failure (or an id still unknown after it): a player whose
  position is unknown **and who has a pitching block** is undecidable, so mark the
  pass not-fully-ok (watermark not advanced) and re-attempt next run; the batting row
  is still upserted (idempotent), and unknown positions for non-pitching players are
  irrelevant. This keeps the "advance only on success / no permanent loss" guarantee
  -- we never silently drop a real pitcher because a lookup blipped. Cached
  two-way/pitcher ids are unaffected.
- All `redis_store` helpers no-op on `client is None` (matches existing pattern).

## Testing

Tests inject a fresh `SqliteKVStore` at `tmp_path` (never prod Upstash -- see the
known test-safety trap). MLB responses are recorded JSON fixtures; no live calls.

- Box-score parsers: hitting + pitching extraction matches expected `GameRow`s.
- **Ohtani fixture (gamePk 776213):** both rows stored under 660271; both rollups
  include him.
- Position-player pitching: a player with `primaryPosition.code == "6"` who appears
  in `pitchers` has the pitching row dropped, batting row kept.
- Doubleheader: two rows same date, distinct gamePk, both retained.
- Upsert idempotency / correction: re-ingesting a gamePk with revised stats
  overwrites by gamePk (no duplicate, totals reflect the revision).
- Watermark: advances to `t_start` on success; left unchanged when a box-score fetch
  raises; `updatedSince` is the prior `t_start`.
- Rollup derivation: summed totals equal direct sums of stored rows.
- `games_elapsed == len(dates)`.

## Rollout / phasing

Per the repo's phased-execution rule (<= 5 files/phase, verify between phases):

- **Phase 1 -- schema + parsers:** `redis_store` helpers (per-player get/set, watermark,
  player_pos cache, dates), `analysis/game_logs.py` extraction refactor + box-score
  parsers, fixtures, unit tests.
- **Phase 2 -- sync engine:** rewrite `data/mlb_game_logs.py` with backfill +
  incremental + watermark + filter + rollup derivation; tests.
- **Phase 3 -- wire-in:** point `refresh_pipeline._fetch_game_logs` at the new engine,
  retire the roster-enumeration full rebuild, verify `season_progress`; integration
  test on the refresh fixture.

First production refresh after deploy sees a missing watermark and runs the one-time
backfill; the derived rollup overwrites the existing `game_log_totals:*` keys with no
shape change, so consumers keep working.

## Risks

- **`game/changes` completeness:** assumes a game going Final and a stat correction
  both register as a change. Mitigated by capture-before-read + idempotent re-pull;
  if ever observed to miss, the backfill path is a full re-sync.
- **Backfill cost:** ~2,400 box scores. One-time; re-runs only if interrupted before
  the watermark is set.
- **`keys()` on Upstash** for full rebuild scans the (small, ~3k-key) namespace;
  steady-state incremental never calls it.
