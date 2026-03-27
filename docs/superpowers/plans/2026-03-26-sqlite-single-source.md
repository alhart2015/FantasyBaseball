# SQLite Single Source of Truth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SQLite the single read path for projections and positions — all downstream code reads from `fantasy.db` instead of CSV files or JSON caches.

**Architecture:** Add two read functions to `db.py` (`get_blended_projections`, `get_positions`), add a `positions` table, change `build_draft_board()` to accept a `conn` parameter instead of file paths, and update all callers (scripts + season_data.py) to open a connection and pass it through. The CSV/JSON loading code stays intact for `build_db.py` (the write path). Tests that call `build_draft_board` switch to using an in-memory SQLite database populated from the existing test fixture CSVs.

**Tech Stack:** Python, SQLite, pandas

---

### Task 1: Add `positions` table to schema and write function

**Files:**
- Modify: `src/fantasy_baseball/data/db.py`
- Test: `tests/test_data/test_db.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_data/test_db.py`, add:

```python
def test_load_positions(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    positions = {
        "Aaron Judge": ["OF", "DH"],
        "Gerrit Cole": ["SP"],
        "Shohei Ohtani": ["Util"],
    }
    load_positions(conn, positions)

    rows = conn.execute("SELECT * FROM positions ORDER BY name").fetchall()
    assert len(rows) == 3
    judge = [r for r in rows if r["name"] == "Aaron Judge"][0]
    assert judge["positions"] == "OF, DH"
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_db.py::test_load_positions -v`
Expected: FAIL — `ImportError: cannot import name 'load_positions'`

- [ ] **Step 3: Add positions table to SCHEMA and write `load_positions`**

In `src/fantasy_baseball/data/db.py`, add to the SCHEMA string (after the `game_logs` table):

```sql
CREATE TABLE IF NOT EXISTS positions (
    name       TEXT NOT NULL PRIMARY KEY,
    positions  TEXT NOT NULL
);
```

Then add the function:

```python
def load_positions(conn, positions: dict[str, list[str]]) -> None:
    """Load position eligibility into the positions table.

    ``positions`` is a dict mapping player name to a list of position strings.
    Uses INSERT OR REPLACE so repeated calls are idempotent.
    """
    rows = [
        (name, ", ".join(pos_list))
        for name, pos_list in positions.items()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO positions (name, positions) VALUES (?, ?)",
        rows,
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_db.py::test_load_positions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/db.py tests/test_data/test_db.py
git commit -m "feat: add positions table and load_positions to db.py"
```

---

### Task 2: Add `get_positions` read function

**Files:**
- Modify: `src/fantasy_baseball/data/db.py`
- Test: `tests/test_data/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
def test_get_positions(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    positions = {
        "Aaron Judge": ["OF", "DH"],
        "Gerrit Cole": ["SP"],
    }
    load_positions(conn, positions)
    result = get_positions(conn)

    assert result == {"Aaron Judge": ["OF", "DH"], "Gerrit Cole": ["SP"]}
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_db.py::test_get_positions -v`
Expected: FAIL — `ImportError: cannot import name 'get_positions'`

- [ ] **Step 3: Implement `get_positions`**

```python
def get_positions(conn) -> dict[str, list[str]]:
    """Read position eligibility from the database.

    Returns a dict mapping player name to list of position strings,
    matching the format of ``load_positions_cache()``.
    """
    rows = conn.execute("SELECT name, positions FROM positions").fetchall()
    return {
        row["name"]: [p.strip() for p in row["positions"].split(",")]
        for row in rows
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_db.py::test_get_positions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/db.py tests/test_data/test_db.py
git commit -m "feat: add get_positions read function to db.py"
```

---

### Task 3: Add `get_blended_projections` read function

**Files:**
- Modify: `src/fantasy_baseball/data/db.py`
- Test: `tests/test_data/test_db.py`

- [ ] **Step 1: Write the failing test**

```python
def test_get_blended_projections(tmp_path):
    """Round-trip: load blended projections, then read them back."""
    import shutil

    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    # load_blended_projections expects year subdirectories, so create one
    fixtures = Path(__file__).parent.parent / "fixtures"
    year_dir = tmp_path / "projections" / "2026"
    year_dir.mkdir(parents=True)
    for csv in fixtures.glob("*.csv"):
        shutil.copy(csv, year_dir / csv.name)

    load_blended_projections(conn, tmp_path / "projections", ["steamer"], None)

    hitters, pitchers = get_blended_projections(conn, year=2026)

    # Fixture has 5 hitters and 2 pitchers
    assert len(hitters) == 5
    assert len(pitchers) == 2

    # player_type must be preserved (downstream code requires it)
    assert "player_type" in hitters.columns
    assert (hitters["player_type"] == "hitter").all()
    assert (pitchers["player_type"] == "pitcher").all()

    # year column should be dropped (not part of blend_projections output)
    assert "year" not in hitters.columns

    # Check required columns exist
    for col in ("name", "fg_id", "ab", "h", "r", "hr", "rbi", "sb", "avg", "adp"):
        assert col in hitters.columns, f"Missing hitter column: {col}"
    for col in ("name", "fg_id", "w", "k", "sv", "ip", "er", "bb", "h_allowed", "era", "whip", "adp"):
        assert col in pitchers.columns, f"Missing pitcher column: {col}"

    # Verify a specific player
    judge = hitters[hitters["name"] == "Aaron Judge"]
    assert len(judge) == 1
    assert judge.iloc[0]["hr"] > 0
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_db.py::test_get_blended_projections -v`
Expected: FAIL — `ImportError: cannot import name 'get_blended_projections'`

- [ ] **Step 3: Implement `get_blended_projections`**

In `db.py`:

```python
def get_blended_projections(
    conn, year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read blended projections from the database.

    Returns (hitters_df, pitchers_df) matching the format produced by
    ``blend_projections()`` in projections.py.

    If *year* is None, uses the maximum year in the table (current season).
    """
    if year is None:
        row = conn.execute(
            "SELECT MAX(year) as y FROM blended_projections"
        ).fetchone()
        year = row["y"] if row and row["y"] is not None else 0

    hitters = pd.read_sql_query(
        "SELECT * FROM blended_projections WHERE year = ? AND player_type = 'hitter'",
        conn, params=(year,),
    )
    pitchers = pd.read_sql_query(
        "SELECT * FROM blended_projections WHERE year = ? AND player_type = 'pitcher'",
        conn, params=(year,),
    )

    # Drop the year column (not part of blend_projections output).
    # Keep player_type — downstream code (backfill, SGP, player_id) requires it.
    for df in (hitters, pitchers):
        if "year" in df.columns:
            df.drop(columns=["year"], inplace=True)

    return hitters, pitchers
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_db.py::test_get_blended_projections -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/db.py tests/test_data/test_db.py
git commit -m "feat: add get_blended_projections read function to db.py"
```

---

### Task 4: Update `build_db.py` to ingest positions

**Files:**
- Modify: `scripts/build_db.py`

- [ ] **Step 1: Update build_db.py to load positions**

After the existing data loads, add:

```python
from fantasy_baseball.data.db import load_positions
from fantasy_baseball.data.yahoo_players import load_positions_cache

POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"

# ... inside main(), after load_weekly_rosters:

if POSITIONS_PATH.exists():
    positions = load_positions_cache(POSITIONS_PATH)
    load_positions(conn, positions)
    pos_count = conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
    print(f"  Loaded {pos_count} player positions")
```

- [ ] **Step 2: Run build_db.py to verify it works**

Run: `python scripts/build_db.py`
Expected: Output includes "Loaded N player positions" line

- [ ] **Step 3: Verify positions table has data**

Run: `python -c "import sqlite3; conn = sqlite3.connect('data/fantasy.db'); print(conn.execute('SELECT COUNT(*) FROM positions').fetchone()[0])"`
Expected: Number > 0 (should match number of entries in player_positions.json)

- [ ] **Step 4: Commit**

```bash
git add scripts/build_db.py
git commit -m "feat: build_db.py ingests positions into SQLite"
```

---

### Task 5: Change `build_draft_board()` to read from SQLite

**Files:**
- Modify: `src/fantasy_baseball/draft/board.py`
- Test: `tests/test_draft/test_board.py`

This is the core change. Replace `projections_dir`/`positions_path`/`systems`/`weights` params with a single `conn` param.

- [ ] **Step 1: Update the test fixture to use SQLite**

Replace the `position_cache` fixture and update `TestBuildDraftBoard` in `tests/test_draft/test_board.py`:

```python
import shutil
from fantasy_baseball.data.db import (
    get_connection, create_tables,
    load_blended_projections, load_positions,
)


@pytest.fixture
def board_conn(tmp_path, fixtures_dir):
    """Build a SQLite DB from test fixture CSVs + positions."""
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    create_tables(conn)

    # load_blended_projections expects year subdirectories
    year_dir = tmp_path / "projections" / "2026"
    year_dir.mkdir(parents=True)
    for csv in fixtures_dir.glob("*.csv"):
        shutil.copy(csv, year_dir / csv.name)

    load_blended_projections(conn, tmp_path / "projections", ["steamer"], None)

    # Load position data
    positions = {
        "Aaron Judge": ["OF", "DH"],
        "Mookie Betts": ["OF", "SS"],
        "Adley Rutschman": ["C"],
        "Marcus Semien": ["2B", "SS"],
        "Gerrit Cole": ["SP"],
        "Emmanuel Clase": ["RP"],
        "Corbin Burnes": ["SP"],
    }
    load_positions(conn, positions)
    yield conn
    conn.close()
```

Update all test methods to use `board_conn` instead of `fixtures_dir` and `position_cache`:

```python
class TestBuildDraftBoard:
    def test_returns_dataframe_with_required_columns(self, board_conn):
        board = build_draft_board(conn=board_conn)
        assert "name" in board.columns
        assert "positions" in board.columns
        assert "total_sgp" in board.columns
        assert "var" in board.columns
        assert "best_position" in board.columns

    def test_players_ranked_by_var_descending(self, board_conn):
        board = build_draft_board(conn=board_conn)
        vars_list = board["var"].tolist()
        assert vars_list == sorted(vars_list, reverse=True)

    def test_all_fixture_players_present(self, board_conn):
        board = build_draft_board(conn=board_conn)
        assert len(board) == 7

    def test_positions_from_cache(self, board_conn):
        board = build_draft_board(conn=board_conn)
        judge = board[board["name"] == "Aaron Judge"].iloc[0]
        assert "OF" in judge["positions"]


class TestApplyKeepers:
    def test_removes_keepers_from_board(self, board_conn):
        board = build_draft_board(conn=board_conn)
        keepers = [{"name": "Aaron Judge", "team": "Spacemen"}]
        filtered = apply_keepers(board, keepers)
        assert "Aaron Judge" not in filtered["name"].values
        assert len(filtered) == len(board) - 1

    def test_keeper_not_in_projections_is_ignored(self, board_conn):
        board = build_draft_board(conn=board_conn)
        keepers = [{"name": "Nonexistent Player", "team": "Nobody"}]
        filtered = apply_keepers(board, keepers)
        assert len(filtered) == len(board)
```

- [ ] **Step 2: Run the tests to verify they fail (old signature)**

Run: `pytest tests/test_draft/test_board.py::TestBuildDraftBoard -v`
Expected: FAIL — `build_draft_board()` doesn't accept `conn` param yet

- [ ] **Step 3: Update `build_draft_board()` signature and implementation**

In `src/fantasy_baseball/draft/board.py`:

Replace the imports at the top:
```python
from fantasy_baseball.data.db import get_blended_projections, get_positions
```

Remove the old imports:
```python
# Remove these:
# from fantasy_baseball.data.projections import blend_projections
# from fantasy_baseball.data.yahoo_players import load_positions_cache
```

Replace the function signature and first two lines:

```python
def build_draft_board(
    conn,
    sgp_overrides: dict[str, float] | None = None,
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
) -> pd.DataFrame:
    """Build a ranked draft board from projections and position data in SQLite."""
    hitters, pitchers = get_blended_projections(conn)
    positions = get_positions(conn)
```

The rest of the function body (lines 116-181) stays exactly the same — it operates on the DataFrames and dict regardless of where they came from.

- [ ] **Step 4: Run the board tests to verify they pass**

Run: `pytest tests/test_draft/test_board.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/board.py tests/test_draft/test_board.py
git commit -m "refactor: build_draft_board reads from SQLite instead of CSV/JSON"
```

---

### Task 6: Update integration tests

**Files:**
- Modify: `tests/test_integration/test_draft_integration.py`
- Modify: `tests/test_integration/test_sgp_pipeline.py`

- [ ] **Step 1: Update `test_draft_integration.py`**

Replace the `full_board` fixture to use SQLite:

```python
from fantasy_baseball.data.db import (
    get_connection, create_tables,
    load_blended_projections, load_positions,
)
from fantasy_baseball.data.yahoo_players import load_positions_cache

@pytest.fixture(scope="module")
def full_board(config: LeagueConfig) -> pd.DataFrame:
    """Build a draft board from real projections via SQLite."""
    conn = get_connection(":memory:")
    create_tables(conn)

    load_blended_projections(conn, _PROJECTIONS_DIR, config.projection_systems, config.projection_weights)

    if _POSITIONS_PATH.exists():
        positions = load_positions_cache(_POSITIONS_PATH)
        load_positions(conn, positions)

    board = build_draft_board(
        conn=conn,
        sgp_overrides=config.sgp_overrides,
        roster_slots=config.roster_slots,
        num_teams=config.num_teams,
    )
    conn.close()
    return board
```

- [ ] **Step 2: Update `test_sgp_pipeline.py`**

The `blend_projections` calls in this file test the CSV blending logic directly, which is still valid (it's the write-path code). These tests do NOT need to change — they test `projections.py`, not `board.py`. Verify they still pass.

- [ ] **Step 3: Run all integration tests**

Run: `pytest tests/test_integration/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration/test_draft_integration.py
git commit -m "test: update draft integration tests to use SQLite"
```

---

### Task 7: Update `run_draft.py`

**Files:**
- Modify: `scripts/run_draft.py`

- [ ] **Step 1: Update imports and board construction**

Replace the path constants and board building:

```python
# Remove these lines:
# POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
# PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"

# Add:
from fantasy_baseball.data.db import get_connection, DB_PATH
```

In `main()`, replace the `build_draft_board()` call:

```python
    # Build draft board from SQLite
    print("Building draft board...")
    conn = get_connection()
    full_board = build_draft_board(
        conn=conn,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=num_teams,
    )
    conn.close()
```

- [ ] **Step 2: Verify the script still works**

Run: `python scripts/run_draft.py --mock --position 8 --teams 10 < /dev/null`
Expected: Builds board and shows top 25 available players before exiting

- [ ] **Step 3: Commit**

```bash
git add scripts/run_draft.py
git commit -m "refactor: run_draft.py reads from SQLite"
```

---

### Task 8: Update `simulate_draft.py`

**Files:**
- Modify: `scripts/simulate_draft.py`

- [ ] **Step 1: Update `build_board_and_context()`**

Replace the board building section (~lines 231-239):

```python
from fantasy_baseball.data.db import get_connection

# In build_board_and_context():
    conn = get_connection()
    full_board = build_draft_board(
        conn=conn,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
    )
    conn.close()
```

Remove `POSITIONS_PATH` and `PROJECTIONS_DIR` constants if no longer used elsewhere in the file.

- [ ] **Step 2: Run a quick simulation to verify**

Run: `python scripts/simulate_draft.py -s two_closers --scoring-mode vona -n 1 --seed 42`
Expected: Completes one simulation without errors

- [ ] **Step 3: Commit**

```bash
git add scripts/simulate_draft.py
git commit -m "refactor: simulate_draft.py reads from SQLite"
```

---

### Task 9: Update in-season scripts (`run_lineup.py`, `summary.py`, `run_trades.py`, `recommend_players.py`)

**Files:**
- Modify: `scripts/run_lineup.py`
- Modify: `scripts/summary.py`
- Modify: `scripts/run_trades.py`
- Modify: `scripts/recommend_players.py`

These scripts call `blend_projections()` directly (not `build_draft_board`). They need to switch to `get_blended_projections()`.

- [ ] **Step 1: Update `run_lineup.py`**

Replace:
```python
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
```
With:
```python
from fantasy_baseball.data.db import get_connection, get_blended_projections, get_positions
```

Replace the projection loading (~lines 257-267):
```python
    print("Loading projections...")
    conn = get_connection()
    hitters_proj, pitchers_proj = get_blended_projections(conn)
    # ... existing _name_norm code stays ...
    positions_cache = get_positions(conn)
    conn.close()
    norm_positions = {normalize_name(k): v for k, v in positions_cache.items()}
```

- [ ] **Step 2: Update `summary.py`**

Replace the import and projection loading (~lines 21, 88-90):
```python
from fantasy_baseball.data.db import get_connection, get_blended_projections
```

```python
    print("Loading projections...")
    conn = get_connection()
    hitters_proj, pitchers_proj = get_blended_projections(conn)
    conn.close()
```

- [ ] **Step 3: Update `run_trades.py`**

Replace the import and projection loading (~lines 16, 103-105):
```python
from fantasy_baseball.data.db import get_connection, get_blended_projections
```

```python
    print("Loading projections...")
    conn = get_connection()
    hitters_proj, pitchers_proj = get_blended_projections(conn)
    conn.close()
```

- [ ] **Step 4: Update `recommend_players.py`**

Replace the import and projection loading (~lines 16, 64-66):
```python
from fantasy_baseball.data.db import get_connection, get_blended_projections
```

```python
    print("Loading projections...")
    conn = get_connection()
    hitters_proj, pitchers_proj = get_blended_projections(conn)
    conn.close()
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_lineup.py scripts/summary.py scripts/run_trades.py scripts/recommend_players.py
git commit -m "refactor: in-season scripts read projections from SQLite"
```

---

### Task 10: Update `season_data.py`

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

- [ ] **Step 1: Replace the blend_projections import and call**

In `run_full_refresh()`, replace the lazy import and call (~lines 327, 371-378):

Replace:
```python
        from fantasy_baseball.data.projections import blend_projections
```
With:
```python
        from fantasy_baseball.data.db import get_connection, get_blended_projections
```

Replace the projection blending block:
```python
        # --- Step 4: Read projections from SQLite ---
        _set_refresh_progress("Loading projections...")
        db_conn = get_connection()
        hitters_proj, pitchers_proj = get_blended_projections(db_conn)
        db_conn.close()
        hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)
```

- [ ] **Step 2: Run the season dashboard tests**

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: All PASS (or skip if they require Yahoo auth)

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "refactor: season_data.py reads projections from SQLite"
```

---

### Task 11: Update `backtest_2025.py` and `analyze_mock.py`

**Files:**
- Modify: `scripts/backtest_2025.py`
- Modify: `scripts/analyze_mock.py`

- [ ] **Step 1: Update `backtest_2025.py`**

The backtest uses 2025 data with hardcoded systems/weights (steamer+zips 50/50). It must build its own in-memory DB to preserve this explicit control:

```python
from fantasy_baseball.data.db import get_connection, create_tables, load_blended_projections, load_positions
from fantasy_baseball.data.yahoo_players import load_positions_cache
```

Replace the `build_draft_board` call (~line 336-342):
```python
    # Build a temporary DB with 2025 projections and specific systems/weights
    conn = get_connection(":memory:")
    create_tables(conn)
    load_blended_projections(
        conn, PROJECTIONS_DIR, ["steamer", "zips"],
        {"steamer": 0.50, "zips": 0.50},
    )
    if POSITIONS_PATH.exists():
        load_positions(conn, load_positions_cache(POSITIONS_PATH))

    board = build_draft_board(
        conn=conn,
        num_teams=10,
    )
    conn.close()
```

Keep `PROJECTIONS_DIR` and `POSITIONS_PATH` constants — the backtest still needs them for its explicit 2025 data loading.

- [ ] **Step 2: Update `analyze_mock.py`**

```python
from fantasy_baseball.data.db import get_connection
```

Replace the `build_draft_board` call (~lines 32-40):
```python
    conn = get_connection()
    board = build_draft_board(
        conn=conn,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=10,
    )
    conn.close()
```

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest_2025.py scripts/analyze_mock.py
git commit -m "refactor: backtest and analyze scripts read from SQLite"
```

---

### Task 12: Run full test suite and clean up

**Files:**
- Possibly modify: any file with residual issues

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: All 558+ tests PASS

- [ ] **Step 2: Fix any failures**

Address any test failures caused by the signature change. Common issues:
- Tests that import `blend_projections` from `board.py` (it no longer re-exports it)
- Tests that pass `projections_dir`/`positions_path` to `build_draft_board`

- [ ] **Step 3: Remove unused imports from `board.py`**

Verify `blend_projections` and `load_positions_cache` are no longer imported in `board.py`.

- [ ] **Step 4: Run tests once more**

Run: `pytest -v`
Expected: All PASS

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: clean up residual imports after SQLite migration"
```

---

### Task 13: Update TODO.md

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Mark the TODO item as done**

Change `- [ ]` to `- [x]` for the "SQLite as single source of truth" item.

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs: mark SQLite single-source-of-truth TODO as complete"
```
