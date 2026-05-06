# Hot Streaks — Phase 1 (Data Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the data fetch + DuckDB storage layer for hot/cold streak analysis. After this phase: a single CLI command pulls all qualified hitter game logs and per-PA Statcast data for a season into `data/streaks/streaks.duckdb`, idempotently.

**Architecture:** New `src/fantasy_baseball/streaks/` package, hard-isolated from the production pipeline (no imports to/from `web/`, `lineup/`, `data/redis_store.py`). DuckDB local store at `data/streaks/streaks.duckdb` (gitignored). Game logs come from the MLB Stats API (extending parsing of the existing `analysis/game_logs.py` helpers); per-PA Statcast comes from `pybaseball.statcast()` against Baseball Savant in week-long chunks. All upserts are idempotent so re-runs skip already-fetched player-seasons and date ranges.

**Tech Stack:** DuckDB (analytical SQL), pybaseball (Statcast client), MLB Stats API (game logs via `requests`), pytest (TDD throughout), pandas (Statcast → row-dict conversion).

**Spec:** `docs/superpowers/specs/2026-05-06-hot-streaks-design.md`

---

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/streaks/__init__.py` | Create | Package marker |
| `src/fantasy_baseball/streaks/data/__init__.py` | Create | Package marker |
| `src/fantasy_baseball/streaks/data/schema.py` | Create | `get_connection`, `init_schema` — DuckDB DDL |
| `src/fantasy_baseball/streaks/data/load.py` | Create | Idempotent upsert fns + existence queries |
| `src/fantasy_baseball/streaks/data/qualified_hitters.py` | Create | Fetch list of ≥150 PA hitters for a season |
| `src/fantasy_baseball/streaks/data/game_logs.py` | Create | Streaks-specific hitter game log parser + fetcher (includes bb, k, name, team, season) |
| `src/fantasy_baseball/streaks/data/statcast.py` | Create | `pybaseball.statcast` wrapper, weekly chunks, terminal-PA filter |
| `src/fantasy_baseball/streaks/data/fetch_history.py` | Create | Orchestrator: qualified hitters → game logs → Statcast → DuckDB |
| `scripts/streaks/__init__.py` | Create | Package marker |
| `scripts/streaks/fetch_history.py` | Create | CLI entry point with argparse |
| `tests/test_streaks/__init__.py` | Create | Package marker |
| `tests/test_streaks/test_schema.py` | Create | Schema init idempotency |
| `tests/test_streaks/test_load.py` | Create | Upsert idempotency + existence queries |
| `tests/test_streaks/test_qualified_hitters.py` | Create | MLB Stats API leaderboard parsing (mocked) |
| `tests/test_streaks/test_game_logs.py` | Create | Per-game parsing + fetcher (mocked HTTP) |
| `tests/test_streaks/test_statcast.py` | Create | Statcast filtering + chunking (mocked pybaseball) |
| `tests/test_streaks/test_fetch_history.py` | Create | Orchestrator (mocked everything) |
| `pyproject.toml` | Modify | Add `duckdb`, `pybaseball` dev deps; add streaks/ to mypy + strict overrides |

---

## Task 1: Add new dev dependencies

**Files:**
- Modify: `pyproject.toml:25-34`

- [ ] **Step 1.1: Edit `pyproject.toml`** — add `duckdb` and `pybaseball` to dev deps.

Replace the existing `dev = [...]` block (lines 25-34) with:

```toml
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.10",
    "fakeredis>=2.21",
    "ruff>=0.6",
    "vulture>=2.11",
    "types-PyYAML>=6.0",
    "types-requests>=2.31",
    "duckdb>=1.0",
    "pybaseball>=2.2",
]
```

- [ ] **Step 1.2: Install the new deps**

Run:
```
pip install -e ".[dev]"
```

- [ ] **Step 1.3: Verify imports work**

Run:
```
python -c "import duckdb; import pybaseball; print(duckdb.__version__, pybaseball.__version__)"
```

Expected: prints two version strings, no traceback.

- [ ] **Step 1.4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(streaks): add duckdb + pybaseball dev deps"
```

---

## Task 2: Create the streaks package skeleton + DuckDB schema

**Files:**
- Create: `src/fantasy_baseball/streaks/__init__.py`
- Create: `src/fantasy_baseball/streaks/data/__init__.py`
- Create: `src/fantasy_baseball/streaks/data/schema.py`
- Create: `tests/test_streaks/__init__.py`
- Create: `tests/test_streaks/test_schema.py`

- [ ] **Step 2.1: Create empty package markers**

Create `src/fantasy_baseball/streaks/__init__.py` with content:
```python
"""Hot/cold streak analysis (research). Isolated from production pipeline."""
```

Create `src/fantasy_baseball/streaks/data/__init__.py` with content:
```python
```
(empty file — just the marker)

Create `tests/test_streaks/__init__.py` with content:
```python
```
(empty file)

- [ ] **Step 2.2: Write the failing test for `init_schema`**

Create `tests/test_streaks/test_schema.py` with:

```python
"""Tests for streaks DuckDB schema initialization."""

import duckdb

from fantasy_baseball.streaks.data.schema import init_schema


def test_init_schema_creates_all_tables():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert tables == {
        "hitter_games",
        "hitter_statcast_pa",
        "hitter_windows",
        "thresholds",
        "hitter_streak_labels",
    }


def test_init_schema_is_idempotent():
    conn = duckdb.connect(":memory:")
    init_schema(conn)
    init_schema(conn)  # should not raise
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert len(tables) == 5
```

- [ ] **Step 2.3: Run the test to verify it fails**

Run:
```
pytest tests/test_streaks/test_schema.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'fantasy_baseball.streaks.data.schema'`

- [ ] **Step 2.4: Implement `schema.py`**

Create `src/fantasy_baseball/streaks/data/schema.py` with:

```python
"""DuckDB schema for the streaks analysis project.

All DDL is `CREATE TABLE IF NOT EXISTS` so init_schema is idempotent and
safe to call on every connection open.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

DEFAULT_DB_PATH = Path("data/streaks/streaks.duckdb")

_SCHEMA_DDL = [
    """
    CREATE TABLE IF NOT EXISTS hitter_games (
        player_id INTEGER NOT NULL,
        name VARCHAR NOT NULL,
        team VARCHAR,
        season INTEGER NOT NULL,
        date DATE NOT NULL,
        pa INTEGER NOT NULL,
        ab INTEGER NOT NULL,
        h INTEGER NOT NULL,
        hr INTEGER NOT NULL,
        r INTEGER NOT NULL,
        rbi INTEGER NOT NULL,
        sb INTEGER NOT NULL,
        bb INTEGER NOT NULL,
        k INTEGER NOT NULL,
        PRIMARY KEY (player_id, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_statcast_pa (
        player_id INTEGER NOT NULL,
        date DATE NOT NULL,
        pa_index INTEGER NOT NULL,
        event VARCHAR,
        launch_speed DOUBLE,
        launch_angle DOUBLE,
        estimated_woba_using_speedangle DOUBLE,
        barrel BOOLEAN,
        PRIMARY KEY (player_id, date, pa_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_windows (
        player_id INTEGER NOT NULL,
        window_end DATE NOT NULL,
        window_days INTEGER NOT NULL,
        pa INTEGER NOT NULL,
        hr INTEGER NOT NULL,
        r INTEGER NOT NULL,
        rbi INTEGER NOT NULL,
        sb INTEGER NOT NULL,
        avg DOUBLE,
        babip DOUBLE,
        k_pct DOUBLE,
        bb_pct DOUBLE,
        iso DOUBLE,
        ev_avg DOUBLE,
        barrel_pct DOUBLE,
        xwoba_avg DOUBLE,
        pt_bucket VARCHAR NOT NULL,
        PRIMARY KEY (player_id, window_end, window_days)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS thresholds (
        season_set VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        window_days INTEGER NOT NULL,
        pt_bucket VARCHAR NOT NULL,
        p10 DOUBLE NOT NULL,
        p90 DOUBLE NOT NULL,
        PRIMARY KEY (season_set, category, window_days, pt_bucket)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_streak_labels (
        player_id INTEGER NOT NULL,
        window_end DATE NOT NULL,
        window_days INTEGER NOT NULL,
        category VARCHAR NOT NULL,
        label VARCHAR NOT NULL,
        PRIMARY KEY (player_id, window_end, window_days, category)
    )
    """,
]


def get_connection(path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Open (or create) the streaks DuckDB at *path* and return the connection.

    Parent directory is created if missing. Schema is initialized on every open.
    """
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    init_schema(conn)
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all streaks tables if they don't already exist."""
    for ddl in _SCHEMA_DDL:
        conn.execute(ddl)
```

- [ ] **Step 2.5: Run the test, verify it passes**

Run:
```
pytest tests/test_streaks/test_schema.py -v
```

Expected: 2 PASSED.

- [ ] **Step 2.6: Commit**

```bash
git add src/fantasy_baseball/streaks/ tests/test_streaks/__init__.py tests/test_streaks/test_schema.py
git commit -m "feat(streaks): DuckDB schema + idempotent init_schema"
```

---

## Task 3: Idempotent upsert for `hitter_games`

**Files:**
- Create: `src/fantasy_baseball/streaks/data/load.py`
- Create: `tests/test_streaks/test_load.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_streaks/test_load.py` with:

```python
"""Tests for streaks DuckDB upserts and existence queries."""

from datetime import date

import duckdb
import pytest

from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.schema import init_schema


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _row(player_id=660271, dt=date(2024, 4, 1), hr=1):
    return {
        "player_id": player_id,
        "name": "Mike Trout",
        "team": "LAA",
        "season": 2024,
        "date": dt,
        "pa": 4,
        "ab": 3,
        "h": 1,
        "hr": hr,
        "r": 1,
        "rbi": 2,
        "sb": 0,
        "bb": 1,
        "k": 1,
    }


def test_upsert_hitter_games_inserts_rows(conn):
    upsert_hitter_games(conn, [_row(), _row(dt=date(2024, 4, 2), hr=0)])
    count = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    assert count == 2


def test_upsert_hitter_games_is_idempotent(conn):
    upsert_hitter_games(conn, [_row(), _row(dt=date(2024, 4, 2))])
    upsert_hitter_games(conn, [_row(), _row(dt=date(2024, 4, 2))])  # same rows
    count = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    assert count == 2


def test_upsert_hitter_games_updates_on_pk_collision(conn):
    upsert_hitter_games(conn, [_row(hr=1)])
    upsert_hitter_games(conn, [_row(hr=2)])  # same (player_id, date), new hr value
    hr = conn.execute("SELECT hr FROM hitter_games").fetchone()[0]
    assert hr == 2


def test_upsert_hitter_games_empty_list_is_noop(conn):
    upsert_hitter_games(conn, [])
    count = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    assert count == 0
```

- [ ] **Step 3.2: Run, verify it fails**

Run:
```
pytest tests/test_streaks/test_load.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'fantasy_baseball.streaks.data.load'`.

- [ ] **Step 3.3: Implement `load.py`**

Create `src/fantasy_baseball/streaks/data/load.py` with:

```python
"""Idempotent loaders for the streaks DuckDB tables."""

from __future__ import annotations

from typing import Any

import duckdb

_HITTER_GAME_COLS = (
    "player_id",
    "name",
    "team",
    "season",
    "date",
    "pa",
    "ab",
    "h",
    "hr",
    "r",
    "rbi",
    "sb",
    "bb",
    "k",
)


def upsert_hitter_games(
    conn: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]
) -> None:
    """Insert or replace rows in `hitter_games` keyed by (player_id, date).

    Empty input is a no-op. DuckDB's `INSERT OR REPLACE` handles PK
    collisions atomically.
    """
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_HITTER_GAME_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_games ({', '.join(_HITTER_GAME_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [tuple(r[c] for c in _HITTER_GAME_COLS) for r in rows])
```

- [ ] **Step 3.4: Run, verify it passes**

Run:
```
pytest tests/test_streaks/test_load.py -v
```

Expected: 4 PASSED.

- [ ] **Step 3.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/load.py tests/test_streaks/test_load.py
git commit -m "feat(streaks): idempotent upsert_hitter_games"
```

---

## Task 4: Idempotent upsert for `hitter_statcast_pa`

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/load.py`
- Modify: `tests/test_streaks/test_load.py`

- [ ] **Step 4.1: Add the failing test**

Append to `tests/test_streaks/test_load.py`:

```python
from fantasy_baseball.streaks.data.load import upsert_statcast_pa


def _statcast_row(player_id=660271, dt=date(2024, 4, 1), pa_index=1, event="single"):
    return {
        "player_id": player_id,
        "date": dt,
        "pa_index": pa_index,
        "event": event,
        "launch_speed": 95.5,
        "launch_angle": 12.0,
        "estimated_woba_using_speedangle": 0.45,
        "barrel": False,
    }


def test_upsert_statcast_pa_inserts(conn):
    upsert_statcast_pa(conn, [_statcast_row(pa_index=1), _statcast_row(pa_index=2)])
    count = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert count == 2


def test_upsert_statcast_pa_is_idempotent(conn):
    upsert_statcast_pa(conn, [_statcast_row(pa_index=1)])
    upsert_statcast_pa(conn, [_statcast_row(pa_index=1)])
    count = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert count == 1


def test_upsert_statcast_pa_handles_null_event(conn):
    row = _statcast_row()
    row["event"] = None
    upsert_statcast_pa(conn, [row])
    out = conn.execute("SELECT event FROM hitter_statcast_pa").fetchone()
    assert out[0] is None


def test_upsert_statcast_pa_empty_noop(conn):
    upsert_statcast_pa(conn, [])
    count = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert count == 0
```

- [ ] **Step 4.2: Run, verify it fails**

Run:
```
pytest tests/test_streaks/test_load.py::test_upsert_statcast_pa_inserts -v
```

Expected: FAIL with `ImportError: cannot import name 'upsert_statcast_pa'`.

- [ ] **Step 4.3: Implement `upsert_statcast_pa`**

Append to `src/fantasy_baseball/streaks/data/load.py`:

```python
_STATCAST_COLS = (
    "player_id",
    "date",
    "pa_index",
    "event",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "barrel",
)


def upsert_statcast_pa(
    conn: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]
) -> None:
    """Insert or replace rows in `hitter_statcast_pa` keyed by (player_id, date, pa_index)."""
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_STATCAST_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_statcast_pa ({', '.join(_STATCAST_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [tuple(r[c] for c in _STATCAST_COLS) for r in rows])
```

- [ ] **Step 4.4: Run, verify all `test_load.py` tests pass**

Run:
```
pytest tests/test_streaks/test_load.py -v
```

Expected: 8 PASSED.

- [ ] **Step 4.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/load.py tests/test_streaks/test_load.py
git commit -m "feat(streaks): idempotent upsert_statcast_pa"
```

---

## Task 5: Existence queries for skip-already-fetched logic

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/load.py`
- Modify: `tests/test_streaks/test_load.py`

- [ ] **Step 5.1: Add the failing test**

Append to `tests/test_streaks/test_load.py`:

```python
from fantasy_baseball.streaks.data.load import (
    existing_player_seasons,
    existing_statcast_dates,
)


def test_existing_player_seasons_empty(conn):
    assert existing_player_seasons(conn) == set()


def test_existing_player_seasons_returns_distinct_pairs(conn):
    upsert_hitter_games(
        conn,
        [
            _row(player_id=660271, dt=date(2024, 4, 1)),
            _row(player_id=660271, dt=date(2024, 4, 2)),  # same (player, season)
            _row(player_id=545361, dt=date(2024, 4, 1)),
        ],
    )
    pairs = existing_player_seasons(conn)
    assert pairs == {(660271, 2024), (545361, 2024)}


def test_existing_statcast_dates_returns_distinct_dates(conn):
    upsert_statcast_pa(
        conn,
        [
            _statcast_row(pa_index=1, dt=date(2024, 4, 1)),
            _statcast_row(pa_index=2, dt=date(2024, 4, 1)),  # same date
            _statcast_row(pa_index=1, dt=date(2024, 4, 2)),
        ],
    )
    dates = existing_statcast_dates(conn)
    assert dates == {date(2024, 4, 1), date(2024, 4, 2)}
```

- [ ] **Step 5.2: Run, verify it fails**

Run:
```
pytest tests/test_streaks/test_load.py::test_existing_player_seasons_empty -v
```

Expected: FAIL with `ImportError: cannot import name 'existing_player_seasons'`.

- [ ] **Step 5.3: Implement existence queries**

Append to `src/fantasy_baseball/streaks/data/load.py`:

```python
from datetime import date


def existing_player_seasons(
    conn: duckdb.DuckDBPyConnection,
) -> set[tuple[int, int]]:
    """Return distinct (player_id, season) pairs already loaded in hitter_games.

    Used by fetch orchestration to skip player-seasons we've already pulled.
    """
    rows = conn.execute(
        "SELECT DISTINCT player_id, season FROM hitter_games"
    ).fetchall()
    return {(int(r[0]), int(r[1])) for r in rows}


def existing_statcast_dates(conn: duckdb.DuckDBPyConnection) -> set[date]:
    """Return distinct calendar dates already loaded in hitter_statcast_pa.

    Used by Statcast fetch to skip date ranges we've already pulled.
    """
    rows = conn.execute("SELECT DISTINCT date FROM hitter_statcast_pa").fetchall()
    return {r[0] for r in rows}
```

Move the `from datetime import date` to the top of the file (it's used in the `existing_statcast_dates` return type annotation). The top of `load.py` should now read:

```python
"""Idempotent loaders for the streaks DuckDB tables."""

from __future__ import annotations

from datetime import date
from typing import Any

import duckdb
```

- [ ] **Step 5.4: Run, verify all load tests pass**

Run:
```
pytest tests/test_streaks/test_load.py -v
```

Expected: 11 PASSED.

- [ ] **Step 5.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/load.py tests/test_streaks/test_load.py
git commit -m "feat(streaks): existence queries for skip-already-fetched logic"
```

---

## Task 6: Fetch qualified hitters (≥150 PA) for a season

**Files:**
- Create: `src/fantasy_baseball/streaks/data/qualified_hitters.py`
- Create: `tests/test_streaks/test_qualified_hitters.py`

The MLB Stats API exposes `/stats/leaders` (or `/stats?stats=season&group=hitting`) returning leaderboard entries with `plateAppearances` per player. Easiest is `statsapi.league_leader_data("plateAppearances", season=year, statGroup="hitting", limit=1000)` from the `MLB-StatsAPI` package already in dependencies.

- [ ] **Step 6.1: Write the failing test**

Create `tests/test_streaks/test_qualified_hitters.py` with:

```python
"""Tests for the ≥150 PA qualified hitters fetch."""

from unittest.mock import patch

from fantasy_baseball.streaks.data.qualified_hitters import (
    fetch_qualified_hitters,
    parse_leader_row,
)


def test_parse_leader_row_extracts_id_name_team_pa():
    row = {
        "person": {"id": 660271, "fullName": "Mike Trout"},
        "team": {"abbreviation": "LAA"},
        "value": "162",
    }
    parsed = parse_leader_row(row)
    assert parsed == {
        "player_id": 660271,
        "name": "Mike Trout",
        "team": "LAA",
        "pa": 162,
    }


def test_parse_leader_row_handles_missing_team():
    row = {
        "person": {"id": 545361, "fullName": "Free Agent"},
        "team": {},
        "value": "150",
    }
    parsed = parse_leader_row(row)
    assert parsed["team"] is None


def test_fetch_qualified_hitters_filters_below_min_pa():
    fake_response = {
        "leagueLeaders": [
            {
                "leaders": [
                    {
                        "person": {"id": 1, "fullName": "Above Cutoff"},
                        "team": {"abbreviation": "NYY"},
                        "value": "151",
                    },
                    {
                        "person": {"id": 2, "fullName": "Right At Cutoff"},
                        "team": {"abbreviation": "BOS"},
                        "value": "150",
                    },
                    {
                        "person": {"id": 3, "fullName": "Below Cutoff"},
                        "team": {"abbreviation": "TBR"},
                        "value": "149",
                    },
                ]
            }
        ]
    }
    with patch(
        "fantasy_baseball.streaks.data.qualified_hitters.statsapi.get",
        return_value=fake_response,
    ):
        result = fetch_qualified_hitters(season=2024, min_pa=150)
    ids = {r["player_id"] for r in result}
    assert ids == {1, 2}  # 3 is below cutoff


def test_fetch_qualified_hitters_passes_correct_params():
    with patch(
        "fantasy_baseball.streaks.data.qualified_hitters.statsapi.get",
        return_value={"leagueLeaders": [{"leaders": []}]},
    ) as mock:
        fetch_qualified_hitters(season=2024)
    args, kwargs = mock.call_args
    assert args[0] == "stats_leaders"
    assert kwargs["params"]["season"] == 2024
    assert kwargs["params"]["leaderCategories"] == "plateAppearances"
    assert kwargs["params"]["statGroup"] == "hitting"
```

- [ ] **Step 6.2: Run, verify it fails**

Run:
```
pytest tests/test_streaks/test_qualified_hitters.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 6.3: Implement `qualified_hitters.py`**

Create `src/fantasy_baseball/streaks/data/qualified_hitters.py` with:

```python
"""Fetch the list of hitters with ≥min_pa PA in a given season.

Wraps the MLB Stats API ``/stats/leaders?leaderCategories=plateAppearances``
endpoint via the ``statsapi`` package (MLB-StatsAPI). Returns one row per
qualifying hitter with player_id, name, team, and PA.
"""

from __future__ import annotations

from typing import Any

import statsapi


def parse_leader_row(row: dict[str, Any]) -> dict[str, Any]:
    """Extract player_id, name, team, pa from one /stats/leaders entry."""
    person = row.get("person", {})
    team = row.get("team", {})
    return {
        "player_id": int(person["id"]),
        "name": person["fullName"],
        "team": team.get("abbreviation"),
        "pa": int(row["value"]),
    }


def fetch_qualified_hitters(
    season: int, min_pa: int = 150, limit: int = 1000
) -> list[dict[str, Any]]:
    """Return all hitters with PA >= min_pa for the given season.

    Each result dict has keys: player_id, name, team, pa.
    """
    response = statsapi.get(
        "stats_leaders",
        params={
            "leaderCategories": "plateAppearances",
            "season": season,
            "statGroup": "hitting",
            "limit": limit,
        },
    )
    leaders_groups = response.get("leagueLeaders", [])
    rows: list[dict[str, Any]] = []
    for group in leaders_groups:
        for leader in group.get("leaders", []):
            parsed = parse_leader_row(leader)
            if parsed["pa"] >= min_pa:
                rows.append(parsed)
    return rows
```

- [ ] **Step 6.4: Run, verify passes**

Run:
```
pytest tests/test_streaks/test_qualified_hitters.py -v
```

Expected: 4 PASSED.

- [ ] **Step 6.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/qualified_hitters.py tests/test_streaks/test_qualified_hitters.py
git commit -m "feat(streaks): fetch_qualified_hitters from MLB Stats API leaders endpoint"
```

---

## Task 7: Streaks-specific game log parser + per-season fetcher

The existing `analysis/game_logs.py::parse_hitter_game_log` returns only `{date, pa, ab, h, hr, r, rbi, sb}` — missing BB, K, name, team, and season needed for `hitter_games`. We'll write a streaks-specific parser that captures everything we need. Reusing the URL/HTTP shape from existing code; only the parsing changes.

**Files:**
- Create: `src/fantasy_baseball/streaks/data/game_logs.py`
- Create: `tests/test_streaks/test_game_logs.py`

- [ ] **Step 7.1: Write the failing test**

Create `tests/test_streaks/test_game_logs.py` with:

```python
"""Tests for streaks-specific game log parsing and per-season fetch."""

from unittest.mock import Mock, patch

from fantasy_baseball.streaks.data.game_logs import (
    fetch_hitter_season_game_logs,
    parse_hitter_game_log_full,
)


def _split(date="2024-04-01", **stat_overrides):
    stat = {
        "plateAppearances": 4,
        "atBats": 3,
        "hits": 1,
        "homeRuns": 1,
        "runs": 1,
        "rbi": 2,
        "stolenBases": 0,
        "baseOnBalls": 1,
        "strikeOuts": 1,
    }
    stat.update(stat_overrides)
    return {"date": date, "stat": stat}


def test_parse_hitter_game_log_full_extracts_all_columns():
    row = parse_hitter_game_log_full(
        _split(),
        player_id=660271,
        name="Mike Trout",
        team="LAA",
        season=2024,
    )
    assert row == {
        "player_id": 660271,
        "name": "Mike Trout",
        "team": "LAA",
        "season": 2024,
        "date": "2024-04-01",
        "pa": 4,
        "ab": 3,
        "h": 1,
        "hr": 1,
        "r": 1,
        "rbi": 2,
        "sb": 0,
        "bb": 1,
        "k": 1,
    }


def test_parse_hitter_game_log_full_defaults_missing_stats_to_zero():
    row = parse_hitter_game_log_full(
        {"date": "2024-04-01", "stat": {}},
        player_id=1,
        name="X",
        team=None,
        season=2024,
    )
    assert row["pa"] == 0
    assert row["bb"] == 0


def test_fetch_hitter_season_game_logs_returns_one_row_per_split():
    fake_resp = Mock()
    fake_resp.raise_for_status = Mock()
    fake_resp.json = Mock(
        return_value={
            "stats": [
                {
                    "splits": [
                        _split(date="2024-04-01"),
                        _split(date="2024-04-02", homeRuns=0),
                    ]
                }
            ]
        }
    )
    with patch(
        "fantasy_baseball.streaks.data.game_logs.requests.get", return_value=fake_resp
    ):
        rows = fetch_hitter_season_game_logs(
            player_id=660271, name="Mike Trout", team="LAA", season=2024
        )
    assert len(rows) == 2
    assert rows[0]["date"] == "2024-04-01"
    assert rows[0]["hr"] == 1
    assert rows[1]["hr"] == 0
    assert all(r["player_id"] == 660271 for r in rows)
    assert all(r["season"] == 2024 for r in rows)


def test_fetch_hitter_season_game_logs_handles_empty_splits():
    fake_resp = Mock()
    fake_resp.raise_for_status = Mock()
    fake_resp.json = Mock(return_value={"stats": [{"splits": []}]})
    with patch(
        "fantasy_baseball.streaks.data.game_logs.requests.get", return_value=fake_resp
    ):
        rows = fetch_hitter_season_game_logs(
            player_id=1, name="X", team=None, season=2024
        )
    assert rows == []
```

- [ ] **Step 7.2: Run, verify it fails**

Run:
```
pytest tests/test_streaks/test_game_logs.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 7.3: Implement `game_logs.py`**

Create `src/fantasy_baseball/streaks/data/game_logs.py` with:

```python
"""Hitter game log fetch for the streaks project.

This is a streaks-specific parser that captures every column the
`hitter_games` table needs (player_id, name, team, season, plus bb/k that
the existing analysis/game_logs.py omits). The HTTP shape is identical;
only the parsing differs.
"""

from __future__ import annotations

from typing import Any

import requests

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def parse_hitter_game_log_full(
    split: dict[str, Any],
    *,
    player_id: int,
    name: str,
    team: str | None,
    season: int,
) -> dict[str, Any]:
    """Parse one /people/{id}/stats?stats=gameLog split into a hitter_games row."""
    stat = split.get("stat", {})
    return {
        "player_id": player_id,
        "name": name,
        "team": team,
        "season": season,
        "date": split["date"],
        "pa": int(stat.get("plateAppearances", 0)),
        "ab": int(stat.get("atBats", 0)),
        "h": int(stat.get("hits", 0)),
        "hr": int(stat.get("homeRuns", 0)),
        "r": int(stat.get("runs", 0)),
        "rbi": int(stat.get("rbi", 0)),
        "sb": int(stat.get("stolenBases", 0)),
        "bb": int(stat.get("baseOnBalls", 0)),
        "k": int(stat.get("strikeOuts", 0)),
    }


def fetch_hitter_season_game_logs(
    player_id: int, name: str, team: str | None, season: int, timeout: float = 15.0
) -> list[dict[str, Any]]:
    """Fetch one season of game logs for one hitter as upsert-ready dicts.

    Returns one dict per game played. Empty list if the player has no logs.
    """
    url = f"{MLB_API_BASE}/people/{player_id}/stats"
    params: dict[str, str | int] = {
        "stats": "gameLog",
        "group": "hitting",
        "season": season,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    return [
        parse_hitter_game_log_full(
            s, player_id=player_id, name=name, team=team, season=season
        )
        for s in splits
    ]
```

- [ ] **Step 7.4: Run, verify passes**

Run:
```
pytest tests/test_streaks/test_game_logs.py -v
```

Expected: 4 PASSED.

- [ ] **Step 7.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/game_logs.py tests/test_streaks/test_game_logs.py
git commit -m "feat(streaks): per-season hitter game log fetch with full stat columns"
```

---

## Task 8: Per-PA Statcast fetcher (pybaseball wrapper)

`pybaseball.statcast(start_dt, end_dt)` returns a pitch-level DataFrame for the entire MLB on those dates. Terminal-PA rows have a non-null `events` column. We chunk the season into 7-day requests (pybaseball's recommended chunk size to avoid Savant timeouts) and convert each chunk's terminal rows into upsert-ready dicts.

Barrel computation: pybaseball's returned DataFrame typically contains an integer `barrel` field per Statcast (1 / 0). If the column is missing in a chunk, treat as `None`.

**Files:**
- Create: `src/fantasy_baseball/streaks/data/statcast.py`
- Create: `tests/test_streaks/test_statcast.py`

- [ ] **Step 8.1: Write the failing test**

Create `tests/test_streaks/test_statcast.py` with:

```python
"""Tests for the per-PA Statcast fetcher."""

from datetime import date
from unittest.mock import patch

import pandas as pd

from fantasy_baseball.streaks.data.statcast import (
    chunk_date_range,
    filter_terminal_pa,
    pitches_to_pa_rows,
)


def test_chunk_date_range_produces_seven_day_chunks():
    chunks = list(chunk_date_range(date(2024, 4, 1), date(2024, 4, 21), days=7))
    assert chunks == [
        (date(2024, 4, 1), date(2024, 4, 7)),
        (date(2024, 4, 8), date(2024, 4, 14)),
        (date(2024, 4, 15), date(2024, 4, 21)),
    ]


def test_chunk_date_range_handles_partial_final_chunk():
    chunks = list(chunk_date_range(date(2024, 4, 1), date(2024, 4, 10), days=7))
    assert chunks == [
        (date(2024, 4, 1), date(2024, 4, 7)),
        (date(2024, 4, 8), date(2024, 4, 10)),
    ]


def test_filter_terminal_pa_keeps_only_rows_with_events():
    df = pd.DataFrame(
        {
            "events": [None, "single", None, "strikeout"],
            "batter": [1, 1, 1, 1],
            "game_date": ["2024-04-01"] * 4,
        }
    )
    out = filter_terminal_pa(df)
    assert list(out["events"]) == ["single", "strikeout"]


def test_pitches_to_pa_rows_assigns_pa_index_per_player_per_date():
    df = pd.DataFrame(
        {
            "events": ["single", "double", "strikeout", "home_run"],
            "batter": [660271, 660271, 545361, 660271],
            "game_date": ["2024-04-01", "2024-04-01", "2024-04-01", "2024-04-02"],
            "launch_speed": [95.0, 102.0, None, 110.0],
            "launch_angle": [10.0, 25.0, None, 28.0],
            "estimated_woba_using_speedangle": [0.4, 0.7, 0.0, 0.95],
            "barrel": [0, 1, 0, 1],
        }
    )
    rows = pitches_to_pa_rows(df)
    rows.sort(key=lambda r: (r["player_id"], r["date"], r["pa_index"]))

    # Trout 4/1: 2 PAs, indices 1 and 2
    assert rows[0]["player_id"] == 660271 and rows[0]["date"] == date(2024, 4, 1)
    assert rows[0]["pa_index"] == 1
    assert rows[0]["event"] == "single"
    assert rows[1]["pa_index"] == 2
    assert rows[1]["event"] == "double"
    assert rows[1]["barrel"] is True
    # Trout 4/2: 1 PA, index 1
    assert rows[2]["date"] == date(2024, 4, 2)
    assert rows[2]["pa_index"] == 1
    # Other player 4/1: 1 PA, index 1
    assert rows[3]["player_id"] == 545361
    assert rows[3]["pa_index"] == 1
    assert rows[3]["launch_speed"] is None  # NaN → None


def test_pitches_to_pa_rows_handles_missing_barrel_column():
    df = pd.DataFrame(
        {
            "events": ["single"],
            "batter": [660271],
            "game_date": ["2024-04-01"],
            "launch_speed": [95.0],
            "launch_angle": [10.0],
            "estimated_woba_using_speedangle": [0.4],
        }
    )
    rows = pitches_to_pa_rows(df)
    assert rows[0]["barrel"] is None


def test_fetch_statcast_pa_for_date_range_concatenates_chunks():
    from fantasy_baseball.streaks.data.statcast import fetch_statcast_pa_for_date_range

    chunk_a = pd.DataFrame(
        {
            "events": ["single"],
            "batter": [660271],
            "game_date": ["2024-04-01"],
            "launch_speed": [95.0],
            "launch_angle": [10.0],
            "estimated_woba_using_speedangle": [0.4],
            "barrel": [0],
        }
    )
    chunk_b = pd.DataFrame(
        {
            "events": ["home_run"],
            "batter": [660271],
            "game_date": ["2024-04-08"],
            "launch_speed": [110.0],
            "launch_angle": [28.0],
            "estimated_woba_using_speedangle": [0.95],
            "barrel": [1],
        }
    )
    with patch(
        "fantasy_baseball.streaks.data.statcast.statcast",
        side_effect=[chunk_a, chunk_b],
    ):
        rows = fetch_statcast_pa_for_date_range(date(2024, 4, 1), date(2024, 4, 14))
    assert len(rows) == 2
    assert {r["event"] for r in rows} == {"single", "home_run"}
```

- [ ] **Step 8.2: Run, verify it fails**

Run:
```
pytest tests/test_streaks/test_statcast.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 8.3: Implement `statcast.py`**

Create `src/fantasy_baseball/streaks/data/statcast.py` with:

```python
"""Per-PA Statcast fetch via pybaseball.

Pulls pitch-level data in 7-day chunks (pybaseball's recommended size to
avoid Baseball Savant timeouts), filters to terminal-PA rows (where
``events`` is non-null), and assigns a per-(player, date) PA index for
the (player_id, date, pa_index) primary key.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any

import pandas as pd
from pybaseball import statcast


def chunk_date_range(
    start: date, end: date, days: int = 7
) -> Iterator[tuple[date, date]]:
    """Yield (chunk_start, chunk_end) tuples covering [start, end] in *days*-long chunks.

    Final chunk is shorter if the range doesn't divide evenly.
    """
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=days - 1), end)
        yield (current, chunk_end)
        current = chunk_end + timedelta(days=1)


def filter_terminal_pa(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where the pitch ended a plate appearance (events non-null)."""
    return df[df["events"].notna()].reset_index(drop=True)


def _val_or_none(v: Any) -> Any:
    """Convert pandas/numpy NaN to None; pass everything else through unchanged."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def pitches_to_pa_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a Statcast pitch DataFrame to upsert-ready PA rows.

    Filters to terminal PAs, assigns pa_index per (batter, game_date), and
    converts NaN values to None.
    """
    df = filter_terminal_pa(df)
    if df.empty:
        return []
    df = df.sort_values(["batter", "game_date"]).reset_index(drop=True)
    df["pa_index"] = df.groupby(["batter", "game_date"]).cumcount() + 1

    rows: list[dict[str, Any]] = []
    has_barrel = "barrel" in df.columns
    for r in df.itertuples(index=False):
        rows.append(
            {
                "player_id": int(r.batter),
                "date": pd.to_datetime(r.game_date).date(),
                "pa_index": int(r.pa_index),
                "event": _val_or_none(r.events),
                "launch_speed": _val_or_none(getattr(r, "launch_speed", None)),
                "launch_angle": _val_or_none(getattr(r, "launch_angle", None)),
                "estimated_woba_using_speedangle": _val_or_none(
                    getattr(r, "estimated_woba_using_speedangle", None)
                ),
                "barrel": (
                    bool(r.barrel) if has_barrel and not pd.isna(r.barrel) else None
                ),
            }
        )
    return rows


def fetch_statcast_pa_for_date_range(
    start: date, end: date, chunk_days: int = 7
) -> list[dict[str, Any]]:
    """Fetch and parse all per-PA Statcast rows in [start, end].

    Chunks the date range to avoid Baseball Savant timeouts. Returns
    upsert-ready dicts keyed for `hitter_statcast_pa`.
    """
    all_rows: list[dict[str, Any]] = []
    for chunk_start, chunk_end in chunk_date_range(start, end, chunk_days):
        df = statcast(start_dt=chunk_start.isoformat(), end_dt=chunk_end.isoformat())
        all_rows.extend(pitches_to_pa_rows(df))
    return all_rows
```

- [ ] **Step 8.4: Run, verify passes**

Run:
```
pytest tests/test_streaks/test_statcast.py -v
```

Expected: 6 PASSED.

- [ ] **Step 8.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/statcast.py tests/test_streaks/test_statcast.py
git commit -m "feat(streaks): per-PA Statcast fetch with 7-day chunking + NaN handling"
```

---

## Task 9: Orchestrator — `fetch_season(season, conn)`

Pulls everything together:
1. Get qualified hitters for the season.
2. Filter out (player_id, season) pairs already in `hitter_games`.
3. For each remaining player, fetch their game logs and upsert.
4. Determine season's date range from existing game logs in DB.
5. Filter Statcast date range to skip dates already in `hitter_statcast_pa`.
6. Pull Statcast PA data and upsert.
7. Return a summary dict (rows_inserted, players_fetched, etc.).

**Files:**
- Create: `src/fantasy_baseball/streaks/data/fetch_history.py`
- Create: `tests/test_streaks/test_fetch_history.py`

- [ ] **Step 9.1: Write the failing test**

Create `tests/test_streaks/test_fetch_history.py` with:

```python
"""Tests for the fetch_season orchestrator."""

from datetime import date
from unittest.mock import patch

import duckdb
import pytest

from fantasy_baseball.streaks.data.fetch_history import fetch_season
from fantasy_baseball.streaks.data.schema import init_schema


@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    init_schema(c)
    yield c
    c.close()


def _stub_qualified():
    return [
        {"player_id": 660271, "name": "Mike Trout", "team": "LAA", "pa": 162},
        {"player_id": 545361, "name": "Other Hitter", "team": "BOS", "pa": 200},
    ]


def _stub_game_logs(player_id, name, team, season, **_kwargs):
    return [
        {
            "player_id": player_id,
            "name": name,
            "team": team,
            "season": season,
            "date": "2024-04-01",
            "pa": 4,
            "ab": 3,
            "h": 1,
            "hr": 1,
            "r": 1,
            "rbi": 2,
            "sb": 0,
            "bb": 1,
            "k": 1,
        }
    ]


def _stub_statcast(start, end, **_kwargs):
    return [
        {
            "player_id": 660271,
            "date": date(2024, 4, 1),
            "pa_index": 1,
            "event": "single",
            "launch_speed": 95.0,
            "launch_angle": 10.0,
            "estimated_woba_using_speedangle": 0.4,
            "barrel": False,
        }
    ]


def test_fetch_season_loads_game_logs_and_statcast(conn):
    with (
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_qualified_hitters",
            return_value=_stub_qualified(),
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_hitter_season_game_logs",
            side_effect=_stub_game_logs,
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_statcast_pa_for_date_range",
            side_effect=_stub_statcast,
        ),
    ):
        summary = fetch_season(season=2024, conn=conn)

    games = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    statcast = conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0]
    assert games == 2  # one row each for Trout and Other Hitter
    assert statcast == 1
    assert summary["players_fetched"] == 2
    assert summary["game_log_rows"] == 2
    assert summary["statcast_rows"] == 1


def test_fetch_season_skips_already_loaded_players(conn):
    # Pre-populate Trout
    conn.execute(
        """
        INSERT INTO hitter_games VALUES
        (660271, 'Mike Trout', 'LAA', 2024, '2024-03-28', 4, 3, 1, 1, 1, 2, 0, 1, 1)
        """
    )

    fetch_calls: list[int] = []

    def _record_calls(player_id, name, team, season, **_kwargs):
        fetch_calls.append(player_id)
        return _stub_game_logs(player_id, name, team, season)

    with (
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_qualified_hitters",
            return_value=_stub_qualified(),
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_hitter_season_game_logs",
            side_effect=_record_calls,
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_statcast_pa_for_date_range",
            side_effect=_stub_statcast,
        ),
    ):
        fetch_season(season=2024, conn=conn)

    # Only the second hitter should have been fetched (660271 was already loaded)
    assert fetch_calls == [545361]


def test_fetch_season_uses_correct_date_range_for_statcast(conn):
    captured: dict[str, date] = {}

    def _capture_dates(start, end, **_kwargs):
        captured["start"] = start
        captured["end"] = end
        return []

    with (
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_qualified_hitters",
            return_value=_stub_qualified(),
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_hitter_season_game_logs",
            side_effect=_stub_game_logs,
        ),
        patch(
            "fantasy_baseball.streaks.data.fetch_history.fetch_statcast_pa_for_date_range",
            side_effect=_capture_dates,
        ),
    ):
        fetch_season(season=2024, conn=conn)

    # Statcast range should span 3/15 .. 11/15 of the season year (covers all of MLB regular season + playoffs)
    assert captured["start"] == date(2024, 3, 15)
    assert captured["end"] == date(2024, 11, 15)
```

- [ ] **Step 9.2: Run, verify it fails**

Run:
```
pytest tests/test_streaks/test_fetch_history.py -v
```

Expected: FAIL — module not found.

- [ ] **Step 9.3: Implement `fetch_history.py`**

Create `src/fantasy_baseball/streaks/data/fetch_history.py` with:

```python
"""Orchestrator: fetch one season of game logs + Statcast PA data into DuckDB.

Idempotent: skips player-seasons already present in `hitter_games`. Skips
dates already present in `hitter_statcast_pa` by adjusting the requested
Statcast window.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import duckdb

from fantasy_baseball.streaks.data.game_logs import fetch_hitter_season_game_logs
from fantasy_baseball.streaks.data.load import (
    existing_player_seasons,
    existing_statcast_dates,
    upsert_hitter_games,
    upsert_statcast_pa,
)
from fantasy_baseball.streaks.data.qualified_hitters import fetch_qualified_hitters
from fantasy_baseball.streaks.data.statcast import fetch_statcast_pa_for_date_range

logger = logging.getLogger(__name__)

# MLB regular season + early postseason; safe envelope for Statcast pulls.
_SEASON_START_MMDD = (3, 15)
_SEASON_END_MMDD = (11, 15)


def fetch_season(
    season: int, conn: duckdb.DuckDBPyConnection, min_pa: int = 150
) -> dict[str, Any]:
    """Fetch and load one season of game logs + Statcast PA data.

    Returns a summary dict with row counts.
    """
    qualified = fetch_qualified_hitters(season=season, min_pa=min_pa)
    logger.info("Season %s: %d qualified hitters", season, len(qualified))

    already = existing_player_seasons(conn)
    to_fetch = [q for q in qualified if (q["player_id"], season) not in already]
    logger.info("Season %s: %d new players to fetch", season, len(to_fetch))

    game_log_rows = 0
    for i, player in enumerate(to_fetch):
        try:
            rows = fetch_hitter_season_game_logs(
                player_id=player["player_id"],
                name=player["name"],
                team=player["team"],
                season=season,
            )
            upsert_hitter_games(conn, rows)
            game_log_rows += len(rows)
        except Exception as e:  # noqa: BLE001 — log and continue on per-player error
            logger.warning(
                "Game log fetch failed for %s (%s): %s",
                player["name"],
                player["player_id"],
                e,
            )
        if (i + 1) % 25 == 0:
            logger.info("  fetched %d/%d game logs", i + 1, len(to_fetch))

    start = date(season, *_SEASON_START_MMDD)
    end = date(season, *_SEASON_END_MMDD)
    loaded_dates = existing_statcast_dates(conn)
    # If we've already loaded any dates in this season, skip Statcast (a partial
    # load is rare; we treat it as an all-or-nothing per-season pull for simplicity).
    statcast_rows = 0
    season_dates_loaded = {d for d in loaded_dates if d.year == season}
    if not season_dates_loaded:
        rows = fetch_statcast_pa_for_date_range(start, end)
        upsert_statcast_pa(conn, rows)
        statcast_rows = len(rows)
    else:
        logger.info(
            "Season %s: %d Statcast dates already loaded, skipping Statcast pull",
            season,
            len(season_dates_loaded),
        )

    return {
        "season": season,
        "players_fetched": len(to_fetch),
        "game_log_rows": game_log_rows,
        "statcast_rows": statcast_rows,
    }
```

- [ ] **Step 9.4: Run, verify passes**

Run:
```
pytest tests/test_streaks/test_fetch_history.py -v
```

Expected: 3 PASSED.

- [ ] **Step 9.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/fetch_history.py tests/test_streaks/test_fetch_history.py
git commit -m "feat(streaks): fetch_season orchestrator with idempotent skip logic"
```

---

## Task 10: CLI script `scripts/streaks/fetch_history.py`

**Files:**
- Create: `scripts/streaks/__init__.py`
- Create: `scripts/streaks/fetch_history.py`

- [ ] **Step 10.1: Create the package marker**

Create `scripts/streaks/__init__.py` with content:
```python
```
(empty)

- [ ] **Step 10.2: Implement the CLI**

Create `scripts/streaks/fetch_history.py` with:

```python
"""CLI: fetch one season of game logs + Statcast PA data into the streaks DuckDB.

Usage:
    python scripts/streaks/fetch_history.py --season 2024
    python scripts/streaks/fetch_history.py --season 2025 --min-pa 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.fetch_history import fetch_season
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--min-pa", type=int, default=150)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.db_path)
    try:
        summary = fetch_season(season=args.season, conn=conn, min_pa=args.min_pa)
    finally:
        conn.close()

    print(f"Done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10.3: Smoke-test the CLI's `--help`**

Run:
```
python scripts/streaks/fetch_history.py --help
```

Expected: argparse help text printed; exit 0; no traceback.

- [ ] **Step 10.4: Commit**

```bash
git add scripts/streaks/__init__.py scripts/streaks/fetch_history.py
git commit -m "feat(streaks): CLI entry point for one-season historical fetch"
```

---

## Task 11: Type checking, lint, format, and final verification

**Files:**
- Modify: `pyproject.toml` — add streaks/ to mypy files + strict overrides

- [ ] **Step 11.1: Add streaks/ paths to mypy in `pyproject.toml`**

Insert into the `files = [...]` list at line 80 (immediately after `"src/fantasy_baseball/simulation.py",`):

```
    "src/fantasy_baseball/streaks/",
```

(`streaks` sorts after `sgp` and after `simulation.py` alphabetically — `s-t-r` follows `s-i-m` which follows `s-g-p`.)

Then in `[[tool.mypy.overrides]] module = [...]`, add `"fantasy_baseball.streaks.*"` so the strict subset applies.

The override block becomes:

```toml
[[tool.mypy.overrides]]
module = [
    "fantasy_baseball.analysis.*",
    "fantasy_baseball.draft.*",
    "fantasy_baseball.lineup.*",
    "fantasy_baseball.models.*",
    "fantasy_baseball.sgp.*",
    "fantasy_baseball.streaks.*",
    "fantasy_baseball.trades.*",
    "fantasy_baseball.utils.*",
]
disallow_any_generics = true
no_implicit_optional = true
strict_equality = true
```

- [ ] **Step 11.2: Run mypy on streaks/**

Run:
```
mypy src/fantasy_baseball/streaks/
```

Expected: `Success: no issues found in N source files`.

If errors: fix them in source. Common issues to expect:
- `pybaseball` and `statsapi` likely lack stubs; the project's `ignore_missing_imports = true` should handle this. If a specific unknown-module error appears, add a `# type: ignore[import-untyped]` to the import line.

- [ ] **Step 11.3: Run ruff check + format**

Run:
```
ruff check .
ruff format --check .
```

Expected: zero violations on both. If format drift, run `ruff format .` and re-check.

- [ ] **Step 11.4: Run vulture**

Run:
```
vulture
```

Expected: no NEW dead-code findings introduced by streaks/. Pre-existing findings in other modules are fine — call them out if any.

- [ ] **Step 11.5: Run the full streaks test suite**

Run:
```
pytest tests/test_streaks/ -v
```

Expected: all PASSED. Count should match: schema (2) + load (11) + qualified_hitters (4) + game_logs (4) + statcast (6) + fetch_history (3) = **30 PASSED**.

- [ ] **Step 11.6: Run the full project test suite to confirm no regressions**

Run:
```
pytest -v
```

Expected: full suite passes — the new streaks/ code should not affect anything else (it's isolated and nothing imports from it).

- [ ] **Step 11.7: Commit the pyproject changes**

```bash
git add pyproject.toml
git commit -m "chore(streaks): enable mypy + strict overrides for streaks package"
```

- [ ] **Step 11.8: Phase 1 wrap-up — append milestone to spec Progress Log**

Edit `docs/superpowers/specs/2026-05-06-hot-streaks-design.md` and append to the Progress Log section:

```markdown
### 2026-05-06 — Phase 1 (data layer) implemented

- DuckDB schema, idempotent loaders, qualified-hitter fetch, per-season game log fetch, per-PA Statcast fetch, and a fetch_history CLI all landed.
- 30 unit tests covering schema, upserts, existence queries, parsing, chunking, and orchestration. Full project suite passes.
- Streaks package added to mypy strict overrides.
- Next: Phase 1 acceptance — actually run `python scripts/streaks/fetch_history.py --season 2023/2024/2025` and validate the row counts make sense (~150-200 qualified hitters/season, ~25K-30K game logs/season, ~150K-200K Statcast PAs/season).
- After acceptance, write the Phase 2 plan (window aggregation + threshold calibration).
```

Then commit:

```bash
git add -f docs/superpowers/specs/2026-05-06-hot-streaks-design.md
git commit -m "docs(streaks): log Phase 1 completion in spec progress"
```

- [ ] **Step 11.9: Manual acceptance run** (post-merge)

This is *not* part of the automated plan but should happen before Phase 2 is planned. Run:

```
python scripts/streaks/fetch_history.py --season 2025
```

Expected behavior:
- ~150-200 qualified hitters reported.
- Game log fetches take 5-15 minutes (one HTTP request per player).
- Statcast pulls take another 5-15 minutes (~30 weekly chunks).
- Final summary printed: `{'season': 2025, 'players_fetched': N, 'game_log_rows': ~25K, 'statcast_rows': ~180K}`.
- `data/streaks/streaks.duckdb` is ~50-150 MB.
- Re-running the same command is fast (skips already-loaded data).

If the row counts are way off (e.g. 0 game logs, or 1M Statcast rows), pause and investigate before doing 2023 and 2024.

---

## Self-Review Notes

**Spec coverage check:**
- Spec §3 (Data layer / Sources): covered by Tasks 6, 7, 8.
- Spec §3 (DuckDB schema): covered by Task 2.
- Spec §3 (Idempotency): covered by Tasks 3, 4, 5, 9.
- Spec §3 (Historical scope: 2023-2025): the orchestrator handles one season per invocation; running it three times completes the historical scope. Acceptance step 11.9 covers it.
- Spec §3 (Player qualification: ≥150 PA): covered by Task 6 (`min_pa` parameter, default 150).
- Spec §2 (Architecture: hard isolation): no streaks/ file imports from `web/`, `lineup/`, `data/redis_store.py`. Verify by grep before merging Phase 1.

**Type consistency check:**
- `upsert_hitter_games` / `upsert_statcast_pa` accept `list[dict[str, Any]]` and require the keys defined in `_HITTER_GAME_COLS` / `_STATCAST_COLS`. Producers (`fetch_hitter_season_game_logs`, `pitches_to_pa_rows`) emit dicts matching those keys exactly.
- `existing_player_seasons` returns `set[tuple[int, int]]` as `(player_id, season)`; `fetch_season` filters on `(player_id, season)` — types match.
- `existing_statcast_dates` returns `set[date]`; orchestrator filters loaded dates by `d.year == season` — types match.
- Date types: `parse_hitter_game_log_full` returns date as a string `"2024-04-01"`; DuckDB DATE column accepts ISO-format strings on insert. Verify by running test_load tests before final commit.

**Things explicitly NOT in this plan (future phases):**
- Window aggregation (`hitter_windows` table population) — Phase 2.
- Threshold calibration (`thresholds` table population) — Phase 2.
- Streak labeling (`hitter_streak_labels` table population) — Phase 2.
- Continuation analysis — Phase 3.
- Predictive model — Phase 4.
- Weekly report — Phase 5.
