# SQLite Database Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SQLite database that centralizes projection, draft, roster, and standings data for easy ad-hoc queries.

**Architecture:** Single module `data/db.py` with table creation and loading functions. A `build_db.py` script rebuilds from source files. The dashboard refresh appends live data. Reuses existing `blend_projections()` and `load_projection_set()` — no new data logic.

**Tech Stack:** sqlite3 (stdlib), pandas (for CSV/DataFrame handling), existing `data/projections.py` and `data/fangraphs.py`

**Spec:** `docs/superpowers/specs/2026-03-26-sqlite-database-design.md`

---

## File Structure

```
src/fantasy_baseball/data/db.py     # New: all DB functions
scripts/build_db.py                 # New: rebuild script
tests/test_data/test_db.py          # New: DB tests
.gitignore                          # Modify: add data/fantasy.db
```

**Key responsibility:** `db.py` owns schema creation, CSV/JSON loading, and append functions. It imports from existing modules (`projections.py`, `fangraphs.py`, `name_utils.py`) but doesn't duplicate their logic.

---

## Task 1: Schema + Connection Helpers

**Files:**
- Create: `src/fantasy_baseball/data/db.py`
- Create: `tests/test_data/test_db.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add `data/fantasy.db` and `data/*.db` to `.gitignore`**

- [ ] **Step 2: Write test for `create_tables` and `get_connection`**

```python
import sqlite3
from fantasy_baseball.data.db import create_tables, get_connection, DB_PATH


def test_create_tables_creates_all_five(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor]
    assert "raw_projections" in tables
    assert "blended_projections" in tables
    assert "draft_results" in tables
    assert "weekly_rosters" in tables
    assert "standings" in tables
    conn.close()


def test_create_tables_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    create_tables(conn)  # should not raise
    conn.close()


def test_get_connection_returns_connection():
    conn = get_connection(":memory:")
    assert isinstance(conn, sqlite3.Connection)
    conn.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_data/test_db.py -v`

- [ ] **Step 4: Implement `db.py` with schema and connection helpers**

```python
"""SQLite database for fantasy baseball data."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "fantasy.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_projections (
    year        INTEGER NOT NULL,
    system      TEXT NOT NULL,
    player_type TEXT NOT NULL,
    name        TEXT NOT NULL,
    team        TEXT,
    fg_id       TEXT,
    mlbam_id    INTEGER,
    pa REAL, ab REAL, h REAL, r REAL, hr REAL,
    rbi REAL, sb REAL, cs REAL, bb REAL, so REAL,
    avg REAL, obp REAL, slg REAL, ops REAL, iso REAL,
    babip REAL, woba REAL, wrc_plus REAL, war REAL,
    w REAL, l REAL, sv REAL, ip REAL, er REAL,
    k REAL, bb_p REAL, h_allowed REAL,
    era REAL, whip REAL, fip REAL, k9 REAL, bb9 REAL,
    hr_p REAL, war_p REAL,
    adp REAL, g REAL,
    UNIQUE (year, system, player_type, fg_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_name ON raw_projections(year, name);

CREATE TABLE IF NOT EXISTS blended_projections (
    year        INTEGER NOT NULL,
    fg_id       TEXT NOT NULL,
    name        TEXT NOT NULL,
    team        TEXT,
    player_type TEXT NOT NULL,
    pa REAL, ab REAL, h REAL,
    r REAL, hr REAL, rbi REAL, sb REAL,
    avg REAL,
    w REAL, k REAL, sv REAL,
    ip REAL, er REAL, bb REAL, h_allowed REAL,
    era REAL, whip REAL,
    adp REAL,
    PRIMARY KEY (year, fg_id)
);

CREATE TABLE IF NOT EXISTS draft_results (
    year    INTEGER NOT NULL,
    pick    INTEGER NOT NULL,
    round   INTEGER NOT NULL,
    team    TEXT NOT NULL,
    player  TEXT NOT NULL,
    fg_id   TEXT,
    PRIMARY KEY (year, pick)
);

CREATE TABLE IF NOT EXISTS weekly_rosters (
    snapshot_date TEXT NOT NULL,
    week_num     INTEGER,
    team         TEXT NOT NULL,
    slot         TEXT NOT NULL,
    player_name  TEXT NOT NULL,
    positions    TEXT,
    PRIMARY KEY (snapshot_date, team, slot)
);

CREATE TABLE IF NOT EXISTS standings (
    year          INTEGER NOT NULL,
    snapshot_date TEXT NOT NULL,
    team          TEXT NOT NULL,
    rank          INTEGER,
    r REAL, hr REAL, rbi REAL, sb REAL, avg REAL,
    w REAL, k REAL, sv REAL, era REAL, whip REAL,
    PRIMARY KEY (year, snapshot_date, team)
);
"""


def get_connection(db_path=None):
    """Return a sqlite3 connection. Defaults to DB_PATH."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn):
    """Create all tables (idempotent via IF NOT EXISTS)."""
    conn.executescript(SCHEMA)
    conn.commit()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data/test_db.py -v`

- [ ] **Step 6: Commit**

```bash
git add .gitignore src/fantasy_baseball/data/db.py tests/test_data/test_db.py
git commit -m "feat: SQLite schema and connection helpers"
```

---

## Task 2: Load Raw Projections

**Files:**
- Modify: `src/fantasy_baseball/data/db.py`
- Modify: `tests/test_data/test_db.py`

**Context:** Raw projections come from FanGraphs CSV files in `data/projections/{year}/`. Each file is named `{system}-{hitters|pitchers}.csv` (with variations). The existing `load_projection_set()` in `data/fangraphs.py` handles finding and parsing these files. We need to map FanGraphs column names to our DB column names.

- [ ] **Step 1: Write test for `load_raw_projections`**

```python
import pandas as pd
from fantasy_baseball.data.db import create_tables, load_raw_projections


def test_load_raw_projections(tmp_path):
    # Create a minimal hitter CSV
    csv_dir = tmp_path / "2026"
    csv_dir.mkdir()
    hitter_csv = csv_dir / "steamer-hitters.csv"
    hitter_csv.write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,CS,BB,SO,AVG,OBP,SLG,OPS,ISO,BABIP,wOBA,wRC+,WAR,ADP,G,PlayerId,MLBAMID\n'
        '"James Wood","WSN",600,520,140,85,26,80,15,5,70,150,0.269,0.350,0.480,0.830,0.211,0.320,0.370,130,4.0,50.0,145,"29518",695578\n'
    )
    pitcher_csv = csv_dir / "steamer-pitchers.csv"
    pitcher_csv.write_text(
        'Name,Team,W,L,SV,ERA,G,GS,IP,H,R,ER,HR,BB,SO,K/9,BB/9,K/BB,HR/9,WHIP,FIP,WAR,ADP,PlayerId,MLBAMID\n'
        '"Corbin Burnes","BAL",14,7,0,3.20,32,32,200,170,75,71,20,50,220,9.9,2.3,4.4,0.9,1.10,3.10,5.0,15.0,"19361",669203\n'
    )

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    load_raw_projections(conn, tmp_path)

    rows = conn.execute("SELECT * FROM raw_projections WHERE year=2026").fetchall()
    assert len(rows) == 2

    hitter = conn.execute(
        "SELECT name, hr, sb, avg, fg_id FROM raw_projections WHERE name='James Wood'"
    ).fetchone()
    assert hitter is not None
    assert hitter["hr"] == 26
    assert hitter["fg_id"] == "29518"

    pitcher = conn.execute(
        "SELECT name, w, k, era, fg_id FROM raw_projections WHERE name='Corbin Burnes'"
    ).fetchone()
    assert pitcher is not None
    assert pitcher["w"] == 14
    assert pitcher["k"] == 220

    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `load_raw_projections`**

The function should:
1. Scan `projections_dir` for year subdirectories
2. For each year dir, find all CSV files
3. Parse system name and player type from filename (split on `-hitters`/`-pitchers`)
4. Read CSV with pandas
5. Map FanGraphs column names to DB columns. Key mappings:
   - `Name` → `name`, `Team` → `team`, `PlayerId` → `fg_id`, `MLBAMID` → `mlbam_id`
   - Hitters: `PA`→`pa`, `AB`→`ab`, `H`→`h`, `R`→`r`, `HR`→`hr`, `RBI`→`rbi`, `SB`→`sb`, `CS`→`cs`, `BB`→`bb`, `SO`→`so`, `AVG`→`avg`, `OBP`→`obp`, `SLG`→`slg`, `OPS`→`ops`, `ISO`→`iso`, `BABIP`→`babip`, `wOBA`→`woba`, `wRC+`→`wrc_plus`, `WAR`→`war`, `ADP`→`adp`, `G`→`g`
   - Pitchers: `W`→`w`, `L`→`l`, `SV`→`sv`, `IP`→`ip`, `ER`→`er`, `SO`→`k`, `BB`→`bb_p`, `H`→`h_allowed`, `ERA`→`era`, `WHIP`→`whip`, `FIP`→`fip`, `K/9`→`k9`, `BB/9`→`bb9`, `HR`→`hr_p`, `WAR`→`war_p` (note: pitcher HR and H are different columns than hitter HR/H)
6. Add `year`, `system`, `player_type` columns
7. Insert into DB using `INSERT OR IGNORE`

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add -u && git add tests/
git commit -m "feat: load raw projections from FanGraphs CSVs into SQLite"
```

---

## Task 3: Load Blended Projections

**Files:**
- Modify: `src/fantasy_baseball/data/db.py`
- Modify: `tests/test_data/test_db.py`

**Context:** Reuse existing `blend_projections()` from `data/projections.py`. Call it for each year, insert the resulting DataFrames. The blend function returns `(hitters_df, pitchers_df)` with columns like `r`, `hr`, `name`, `fg_id`, `player_type`, etc.

- [ ] **Step 1: Write test for `load_blended_projections`**

Use the same tmp CSV files from Task 2. With only one system, blending is a passthrough.

```python
from fantasy_baseball.data.db import load_blended_projections


def test_load_blended_projections(tmp_path):
    # Same CSV setup as raw test...
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)

    systems = ["steamer"]
    weights = {"steamer": 1.0}
    load_blended_projections(conn, tmp_path, systems, weights)

    hitter = conn.execute(
        "SELECT name, hr, avg, fg_id FROM blended_projections WHERE name='James Wood' AND year=2026"
    ).fetchone()
    assert hitter is not None
    assert hitter["fg_id"] == "29518"

    pitcher = conn.execute(
        "SELECT name, w, era FROM blended_projections WHERE name='Corbin Burnes' AND year=2026"
    ).fetchone()
    assert pitcher is not None

    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `load_blended_projections`**

```python
def load_blended_projections(conn, projections_dir, systems, weights):
    """Blend projections for each year and insert into blended_projections table."""
    from fantasy_baseball.data.projections import blend_projections

    projections_dir = Path(projections_dir)
    for year_dir in sorted(projections_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)

        # Check which systems are available for this year
        available = [s for s in systems if any(year_dir.glob(f"*{s}*"))]
        if not available:
            continue

        year_weights = {s: weights[s] for s in available}
        try:
            hitters_df, pitchers_df = blend_projections(year_dir, available, year_weights)
        except (FileNotFoundError, ValueError):
            continue

        for df in [hitters_df, pitchers_df]:
            if df.empty:
                continue
            df = df.copy()
            df["year"] = year
            # Ensure fg_id exists
            if "fg_id" not in df.columns:
                df["fg_id"] = df["name"]
            cols = [c for c in df.columns if c in BLENDED_COLUMNS]
            df[cols].to_sql("blended_projections", conn, if_exists="append", index=False)

        conn.commit()
```

Define `BLENDED_COLUMNS` as the list of columns matching the table schema.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat: load blended projections into SQLite"
```

---

## Task 4: Load Draft Results + Standings + Weekly Rosters

**Files:**
- Modify: `src/fantasy_baseball/data/db.py`
- Modify: `tests/test_data/test_db.py`

**Context:** Three JSON loaders. Draft results need fg_id resolution via name matching to raw_projections. Standings and rosters are straightforward inserts.

- [ ] **Step 1: Write tests**

```python
import json


def test_load_draft_results(tmp_path):
    drafts = {
        "2025": [
            {"pick": 1, "round": 1, "team": "Hart of the Order", "player": "Juan Soto"},
            {"pick": 2, "round": 1, "team": "SkeleThor", "player": "Shohei Ohtani (Batter)"},
        ]
    }
    drafts_path = tmp_path / "drafts.json"
    drafts_path.write_text(json.dumps(drafts))

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_draft_results(conn, drafts_path)

    rows = conn.execute("SELECT * FROM draft_results ORDER BY pick").fetchall()
    assert len(rows) == 2
    assert rows[0]["player"] == "Juan Soto"
    assert rows[1]["player"] == "Shohei Ohtani"  # suffix stripped
    conn.close()


def test_load_standings(tmp_path):
    standings = {
        "2023": {
            "standings": [
                {"name": "Hart of the Order", "team_key": "k1", "rank": 1,
                 "stats": {"R": 900, "HR": 250, "RBI": 880, "SB": 150, "AVG": 0.260,
                           "W": 80, "K": 1400, "SV": 90, "ERA": 3.60, "WHIP": 1.20}},
            ]
        }
    }
    path = tmp_path / "standings.json"
    path.write_text(json.dumps(standings))

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_standings(conn, path)

    row = conn.execute("SELECT * FROM standings").fetchone()
    assert row["year"] == 2023
    assert row["snapshot_date"] == "final"
    assert row["r"] == 900
    conn.close()


def test_load_weekly_rosters(tmp_path):
    roster_dir = tmp_path / "rosters"
    roster_dir.mkdir()
    roster = {
        "snapshot_date": "2026-03-23",
        "week_num": 1,
        "team": "Hart of the Order",
        "league": 5652,
        "roster": {
            "C": {"name": "Ivan Herrera", "positions": ["C", "Util"]},
            "OF": {"name": "Juan Soto", "positions": ["OF", "Util"]},
        }
    }
    (roster_dir / "2026-03-23_hart_roster.json").write_text(json.dumps(roster))

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_weekly_rosters(conn, roster_dir)

    rows = conn.execute("SELECT * FROM weekly_rosters ORDER BY slot").fetchall()
    assert len(rows) == 2
    assert rows[0]["player_name"] == "Ivan Herrera"
    assert rows[0]["positions"] == "C, Util"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement all three loaders**

`load_draft_results(conn, drafts_path)`:
- Load JSON, iterate years and picks
- Strip " (Batter)"/" (Pitcher)" from player names
- Attempt fg_id resolution: query `raw_projections` by normalized name for that year
- INSERT OR IGNORE

`load_standings(conn, standings_path)`:
- Load JSON, iterate years
- Insert each team with `snapshot_date='final'`

`load_weekly_rosters(conn, rosters_dir)`:
- Glob `*.json` files in rosters_dir
- For each file, flatten the roster dict: one row per slot
- Join positions list with ", "
- INSERT OR IGNORE

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_data/test_db.py -v`

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat: load draft results, standings, and weekly rosters into SQLite"
```

---

## Task 5: Live Append Functions

**Files:**
- Modify: `src/fantasy_baseball/data/db.py`
- Modify: `tests/test_data/test_db.py`

**Context:** Called from the dashboard refresh to add current-week roster snapshots and standings. Must be idempotent (skip if snapshot already exists).

- [ ] **Step 1: Write tests**

```python
def test_append_roster_snapshot(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)

    roster = [
        {"name": "Juan Soto", "selected_position": "OF", "positions": ["OF", "Util"]},
        {"name": "Corbin Burnes", "selected_position": "P", "positions": ["SP"]},
    ]
    append_roster_snapshot(conn, roster, "2026-03-24", 1, "Hart of the Order")

    rows = conn.execute("SELECT * FROM weekly_rosters").fetchall()
    assert len(rows) == 2

    # Idempotent: second call should not duplicate
    append_roster_snapshot(conn, roster, "2026-03-24", 1, "Hart of the Order")
    rows = conn.execute("SELECT * FROM weekly_rosters").fetchall()
    assert len(rows) == 2
    conn.close()


def test_append_standings_snapshot(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)

    standings = [
        {"name": "Hart of the Order", "rank": 1,
         "stats": {"R": 100, "HR": 30, "RBI": 95, "SB": 20, "AVG": 0.265,
                   "W": 10, "K": 200, "SV": 15, "ERA": 3.50, "WHIP": 1.18}},
    ]
    append_standings_snapshot(conn, standings, 2026, "2026-03-24")

    row = conn.execute("SELECT * FROM standings").fetchone()
    assert row["year"] == 2026
    assert row["snapshot_date"] == "2026-03-24"

    # Idempotent
    append_standings_snapshot(conn, standings, 2026, "2026-03-24")
    assert conn.execute("SELECT COUNT(*) FROM standings").fetchone()[0] == 1
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement append functions**

```python
def append_roster_snapshot(conn, roster, snapshot_date, week_num, team):
    """Append a roster snapshot from dashboard refresh. Idempotent."""
    for player in roster:
        positions = ", ".join(player.get("positions", []))
        conn.execute(
            "INSERT OR IGNORE INTO weekly_rosters "
            "(snapshot_date, week_num, team, slot, player_name, positions) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (snapshot_date, week_num, team,
             player.get("selected_position", "BN"),
             player["name"], positions),
        )
    conn.commit()


def append_standings_snapshot(conn, standings, year, snapshot_date):
    """Append a standings snapshot from dashboard refresh. Idempotent."""
    for team in standings:
        stats = team.get("stats", {})
        conn.execute(
            "INSERT OR IGNORE INTO standings "
            "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (year, snapshot_date, team["name"], team.get("rank"),
             stats.get("R", 0), stats.get("HR", 0), stats.get("RBI", 0),
             stats.get("SB", 0), stats.get("AVG", 0),
             stats.get("W", 0), stats.get("K", 0), stats.get("SV", 0),
             stats.get("ERA", 0), stats.get("WHIP", 0)),
        )
    conn.commit()
```

- [ ] **Step 4: Run all tests**

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat: idempotent live append functions for roster and standings snapshots"
```

---

## Task 6: Build Script + Integration Test

**Files:**
- Create: `scripts/build_db.py`
- Modify: `tests/test_data/test_db.py`

- [ ] **Step 1: Create `build_db.py`**

```python
#!/usr/bin/env python3
"""Rebuild the SQLite database from source files."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import (
    DB_PATH,
    create_tables,
    get_connection,
    load_blended_projections,
    load_draft_results,
    load_raw_projections,
    load_standings,
    load_weekly_rosters,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
DRAFTS_PATH = PROJECT_ROOT / "data" / "historical_drafts_resolved.json"
STANDINGS_PATH = PROJECT_ROOT / "data" / "historical_standings.json"
ROSTERS_DIR = PROJECT_ROOT / "data" / "rosters"


def main():
    config = load_config(CONFIG_PATH)
    db_path = DB_PATH
    print(f"Building database: {db_path}")

    # Delete existing DB for clean rebuild
    if db_path.exists():
        db_path.unlink()
        print("  Deleted existing database")

    conn = get_connection(db_path)
    create_tables(conn)
    print("  Created tables")

    load_raw_projections(conn, PROJECTIONS_DIR)
    raw_count = conn.execute("SELECT COUNT(*) FROM raw_projections").fetchone()[0]
    print(f"  Loaded {raw_count} raw projection rows")

    load_blended_projections(
        conn, PROJECTIONS_DIR,
        config.projection_systems, config.projection_weights,
    )
    blended_count = conn.execute("SELECT COUNT(*) FROM blended_projections").fetchone()[0]
    print(f"  Loaded {blended_count} blended projection rows")

    if DRAFTS_PATH.exists():
        load_draft_results(conn, DRAFTS_PATH)
        draft_count = conn.execute("SELECT COUNT(*) FROM draft_results").fetchone()[0]
        print(f"  Loaded {draft_count} draft picks")

    if STANDINGS_PATH.exists():
        load_standings(conn, STANDINGS_PATH)
        standings_count = conn.execute("SELECT COUNT(*) FROM standings").fetchone()[0]
        print(f"  Loaded {standings_count} standings rows")

    if ROSTERS_DIR.exists():
        load_weekly_rosters(conn, ROSTERS_DIR)
        roster_count = conn.execute("SELECT COUNT(*) FROM weekly_rosters").fetchone()[0]
        print(f"  Loaded {roster_count} roster entries")

    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write integration test**

```python
def test_build_db_end_to_end(tmp_path):
    """Integration test: build a DB from minimal test data and query it."""
    # Setup: create projection CSVs, draft JSON, standings JSON, roster JSON
    proj_dir = tmp_path / "projections" / "2026"
    proj_dir.mkdir(parents=True)
    (proj_dir / "steamer-hitters.csv").write_text(
        'Name,Team,PA,AB,H,R,HR,RBI,SB,AVG,G,PlayerId,MLBAMID\n'
        '"James Wood","WSN",600,520,140,85,26,80,15,0.269,145,"29518",695578\n'
    )
    (proj_dir / "steamer-pitchers.csv").write_text(
        'Name,Team,W,L,SV,ERA,IP,ER,BB,SO,H,HR,WHIP,G,GS,PlayerId,MLBAMID\n'
        '"Corbin Burnes","BAL",14,7,0,3.20,200,71,50,220,170,20,1.10,32,32,"19361",669203\n'
    )

    drafts_path = tmp_path / "drafts.json"
    drafts_path.write_text(json.dumps({
        "2025": [{"pick": 1, "round": 1, "team": "Hart", "player": "James Wood"}]
    }))

    standings_path = tmp_path / "standings.json"
    standings_path.write_text(json.dumps({
        "2025": {"standings": [
            {"name": "Hart", "team_key": "k1", "rank": 1,
             "stats": {"R": 900, "HR": 250, "RBI": 880, "SB": 150, "AVG": 0.260,
                       "W": 80, "K": 1400, "SV": 90, "ERA": 3.60, "WHIP": 1.20}}
        ]}
    }))

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    create_tables(conn)
    load_raw_projections(conn, tmp_path / "projections")
    load_blended_projections(conn, tmp_path / "projections", ["steamer"], {"steamer": 1.0})
    load_draft_results(conn, drafts_path)
    load_standings(conn, standings_path)

    # Verify queries work
    wood = conn.execute(
        "SELECT year, hr, avg FROM blended_projections WHERE name='James Wood'"
    ).fetchone()
    assert wood is not None
    assert wood["hr"] == 26

    systems = conn.execute(
        "SELECT system, hr FROM raw_projections WHERE name='James Wood'"
    ).fetchall()
    assert len(systems) == 1
    assert systems[0]["system"] == "steamer"

    draft = conn.execute("SELECT * FROM draft_results WHERE year=2025").fetchone()
    assert draft["player"] == "James Wood"

    standings = conn.execute("SELECT * FROM standings WHERE year=2025").fetchone()
    assert standings["r"] == 900

    conn.close()
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/test_data/test_db.py -v`

- [ ] **Step 4: Run `build_db.py` against real data**

Run: `python scripts/build_db.py`
Expected: Prints row counts for each table, creates `data/fantasy.db`

- [ ] **Step 5: Verify with a query**

Run: `python -c "import sqlite3; conn=sqlite3.connect('data/fantasy.db'); conn.row_factory=sqlite3.Row; r=conn.execute(\"SELECT year, r, hr, rbi, sb, avg FROM blended_projections WHERE name='James Wood' ORDER BY year\").fetchall(); [print(dict(row)) for row in r]"`

- [ ] **Step 6: Commit**

```bash
git add scripts/build_db.py -u
git commit -m "feat: build_db.py script and integration test"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Schema + connection helpers | db.py, .gitignore |
| 2 | Load raw projections from CSVs | db.py (load_raw_projections) |
| 3 | Load blended projections | db.py (load_blended_projections) |
| 4 | Load drafts, standings, rosters from JSON | db.py (3 loaders) |
| 5 | Live append functions | db.py (append_roster_snapshot, append_standings_snapshot) |
| 6 | Build script + integration test | build_db.py |
