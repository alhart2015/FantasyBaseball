# SQLite Database — Design Spec

Centralized SQLite database for fantasy baseball data. Replaces ad-hoc CSV parsing and JSON file reading with structured queries. Stores projections (raw and blended), draft history, weekly rosters, and standings.

## Database File

- **Path:** `data/fantasy.db` (gitignored)
- **Rebuild:** `python scripts/build_db.py` drops all tables and reloads from source files
- **Live appends:** Dashboard refresh appends roster snapshots and standings via library functions

## Schema

### `raw_projections`

Direct import of every FanGraphs CSV row. All columns preserved.

```sql
CREATE TABLE raw_projections (
    year        INTEGER NOT NULL,
    system      TEXT NOT NULL,           -- steamer, zips, atc, the-bat-x, oopsy
    player_type TEXT NOT NULL,           -- hitter, pitcher
    name        TEXT NOT NULL,
    team        TEXT,
    fg_id       TEXT,                    -- FanGraphs PlayerId
    mlbam_id    INTEGER,                 -- MLB.com MLBAMID

    -- Hitter stats (NULL for pitchers)
    pa  REAL, ab  REAL, h   REAL, r   REAL, hr  REAL,
    rbi REAL, sb  REAL, cs  REAL, bb  REAL, so  REAL,
    avg REAL, obp REAL, slg REAL, ops REAL, iso REAL,
    babip REAL, woba REAL, wrc_plus REAL, war REAL,

    -- Pitcher stats (NULL for hitters)
    w   REAL, l   REAL, sv  REAL, ip  REAL, er  REAL,
    k   REAL, bb_p REAL, h_allowed REAL,
    era REAL, whip REAL, fip REAL, k9  REAL, bb9 REAL,
    hr_p REAL, war_p REAL,

    -- Shared
    adp REAL,
    g   REAL,

    UNIQUE (year, system, player_type, fg_id)
);
CREATE INDEX idx_raw_name ON raw_projections(year, name);
```

No PRIMARY KEY constraint — use SQLite's implicit `rowid` instead. The UNIQUE constraint enforces dedup but allows NULL `fg_id` (SQLite treats each NULL as distinct in UNIQUE). When `fg_id` is missing, the row is still inserted and queryable by name.

### `blended_projections`

Computed from `raw_projections` using `blend_projections()`. Fantasy-relevant columns only.

```sql
CREATE TABLE blended_projections (
    year        INTEGER NOT NULL,
    fg_id       TEXT NOT NULL,           -- canonical player ID
    name        TEXT NOT NULL,
    team        TEXT,
    player_type TEXT NOT NULL,           -- hitter, pitcher

    -- Hitter counting stats
    pa  REAL, ab  REAL, h   REAL,
    r   REAL, hr  REAL, rbi REAL, sb  REAL,
    avg REAL,                            -- computed from h/ab

    -- Pitcher counting stats
    w   REAL, k   REAL, sv  REAL,
    ip  REAL, er  REAL, bb  REAL, h_allowed REAL,
    era REAL, whip REAL,                 -- computed from components

    adp REAL,

    PRIMARY KEY (year, fg_id)
);
```

### `draft_results`

From `data/historical_drafts_resolved.json`.

```sql
CREATE TABLE draft_results (
    year    INTEGER NOT NULL,
    pick    INTEGER NOT NULL,
    round   INTEGER NOT NULL,
    team    TEXT NOT NULL,
    player  TEXT NOT NULL,
    fg_id   TEXT,                        -- resolved via name match to raw_projections

    PRIMARY KEY (year, pick)
);
```

`fg_id` is resolved by joining draft player names to `raw_projections` by normalized name. Unmatched players (retired, etc.) get NULL.

### `weekly_rosters`

From `data/rosters/*.json` files + appended during dashboard refresh.

Slot keys come from the roster JSON as-is (e.g., `OF`, `OF2`, `OF3`, `P`, `P2`). No normalization needed — they're already unique per snapshot.

No `fg_id` on this table. Joining to `blended_projections` requires a name match at query time. This is intentional — roster data comes from Yahoo which doesn't carry FanGraphs IDs, and name resolution is lossy enough that baking it in at load time would create false matches.

```sql
CREATE TABLE weekly_rosters (
    snapshot_date TEXT NOT NULL,          -- YYYY-MM-DD (Monday of scoring week)
    week_num     INTEGER,
    team         TEXT NOT NULL,
    slot         TEXT NOT NULL,           -- C, 1B, OF, P, BN, IL, etc.
    player_name  TEXT NOT NULL,
    positions    TEXT,                    -- comma-separated eligible positions

    PRIMARY KEY (snapshot_date, team, slot)
);
```

### `standings`

From `data/historical_standings.json` + appended during dashboard refresh.

```sql
CREATE TABLE standings (
    year          INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,          -- 'final' for end-of-season, YYYY-MM-DD for live
    team          TEXT NOT NULL,
    rank          INTEGER,
    r    REAL, hr   REAL, rbi  REAL, sb   REAL, avg  REAL,
    w    REAL, k    REAL, sv   REAL, era  REAL, whip REAL,

    PRIMARY KEY (year, snapshot_date, team)
);
```

## Population Strategy

### Rebuild from files (`build_db.py`)

1. Drop all tables, recreate schema
2. **raw_projections:** Scan `data/projections/{year}/` for all CSV files. Parse system name and player type from filename. Filenames vary: `steamer-hitters.csv`, `steamer-hitters-2025.csv`, `zips-hitters-2028-proj-from-2026-03-25.csv`. Use the existing `_find_file()` logic in `data/fangraphs.py` for flex-matching, or split on `-hitters`/`-pitchers` to extract system name (everything before) and player type. Import all rows with year, system, player_type columns added. Map FanGraphs column names to DB column names (e.g., `SO` → `so`, `PlayerId` → `fg_id`, `MLBAMID` → `mlbam_id`).
3. **blended_projections:** For each year that has projection CSVs, call `blend_projections()` with the configured systems/weights. Insert the resulting DataFrames. Rate stats (AVG, ERA, WHIP) are recomputed from blended counting stats, not directly blended.
4. **draft_results:** Load `historical_drafts_resolved.json`. For each pick, attempt to resolve `fg_id` by matching normalized player name to `raw_projections` for that year. Strip "(Batter)"/"(Pitcher)" suffixes before matching.
5. **weekly_rosters:** Scan `data/rosters/*.json`. Flatten each roster dict into one row per player (snapshot_date, week_num, team, slot, player_name, positions).
6. **standings:** Load `historical_standings.json`. Insert each team's stats with `snapshot_date = 'final'`.

### Live appends (dashboard refresh)

Two functions in `db.py` called from `season_data.py` during refresh:

- **`append_roster_snapshot(db_path, roster, snapshot_date, week_num, team)`** — Inserts one row per player. Skips if `snapshot_date + team` already exists (INSERT OR IGNORE).
- **`append_standings_snapshot(db_path, standings, year, snapshot_date)`** — Inserts one row per team. Skips if `year + snapshot_date + team` already exists.

## Module

**`src/fantasy_baseball/data/db.py`** — All database functions:
- `create_tables(conn)` — DDL
- `load_raw_projections(conn, projections_dir)` — CSV import
- `load_blended_projections(conn, projections_dir, systems, weights)` — blend + insert
- `load_draft_results(conn, drafts_path)` — JSON import with fg_id resolution
- `load_weekly_rosters(conn, rosters_dir)` — JSON import
- `load_standings(conn, standings_path)` — JSON import
- `append_roster_snapshot(conn, roster, snapshot_date, week_num, team)` — live append
- `append_standings_snapshot(conn, standings, year, snapshot_date)` — live append
- `get_db_path()` — returns `data/fantasy.db` path
- `get_connection(db_path=None)` — returns sqlite3 connection

**`scripts/build_db.py`** — CLI entry point. Calls all `load_*` functions.

## File Changes

```
src/fantasy_baseball/data/db.py        # New: all DB functions
scripts/build_db.py                    # New: rebuild script
.gitignore                             # Add: data/fantasy.db
```

The existing `blend_projections()` function is reused as-is — `load_blended_projections` calls it and inserts the result.

## Example Queries

```sql
-- James Wood across all projection years
SELECT year, r, hr, rbi, sb, avg FROM blended_projections
WHERE name = 'James Wood' ORDER BY year;

-- All OFs projected for 25+ HR in 2026
SELECT name, team, hr, rbi, sb, avg FROM blended_projections
WHERE year = 2026 AND player_type = 'hitter' AND hr >= 25
ORDER BY hr DESC;

-- Draft history for a player
SELECT year, round, pick, team FROM draft_results
WHERE player LIKE '%Soto%' ORDER BY year;

-- Standings trajectory across seasons
SELECT year, team, r, hr, rbi, avg, era, whip
FROM standings WHERE snapshot_date = 'final'
ORDER BY year, rank;

-- Compare systems for a player
SELECT system, r, hr, rbi, sb, avg FROM raw_projections
WHERE year = 2026 AND name = 'James Wood';
```
