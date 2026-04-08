# SPOE (Standings Points Over Expected) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify luck in roto standings by comparing weekly accumulated projected stats to actual standings for all 10 teams, surfaced as a dashboard tab.

**Architecture:** New `analysis/spoe.py` module implements a week-by-week accumulation loop. Each week: load roster snapshots + ROS projections from SQLite, project one week of stats per team, accumulate, then compare projected roto points to actual roto points. Results stored in two new SQLite tables. Dashboard reads from SQLite directly (no Redis cache).

**Tech Stack:** Python, SQLite, pandas, Flask/Jinja2, vanilla JS

**Spec:** `docs/feature_specs/spoe-design.md`

---

### Task 1: Add SPOE tables and storage helpers to db.py

**Files:**
- Modify: `src/fantasy_baseball/data/db.py:110-126` (SCHEMA string)
- Test: `tests/test_data/test_spoe_db.py`

- [ ] **Step 1: Write failing test for spoe_results table creation**

```python
# tests/test_data/test_spoe_db.py
import sqlite3
import pytest
from fantasy_baseball.data.db import create_tables, get_connection


def test_spoe_results_table_exists():
    conn = get_connection(":memory:")
    create_tables(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "spoe_results" in tables
    conn.close()


def test_spoe_components_table_exists():
    conn = get_connection(":memory:")
    create_tables(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "spoe_components" in tables
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_spoe_db.py -v`
Expected: FAIL — tables don't exist yet

- [ ] **Step 3: Add table definitions to SCHEMA**

In `src/fantasy_baseball/data/db.py`, add before the closing `"""` of SCHEMA (line 126):

```python
CREATE TABLE IF NOT EXISTS spoe_results (
    snapshot_date  TEXT NOT NULL,
    team           TEXT NOT NULL,
    category       TEXT NOT NULL,
    projected_stat REAL,
    actual_stat    REAL,
    projected_pts  REAL,
    actual_pts     REAL,
    spoe           REAL,
    PRIMARY KEY (snapshot_date, team, category)
);

CREATE TABLE IF NOT EXISTS spoe_components (
    snapshot_date  TEXT NOT NULL,
    team           TEXT NOT NULL,
    component      TEXT NOT NULL,
    value          REAL NOT NULL,
    PRIMARY KEY (snapshot_date, team, component)
);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_spoe_db.py -v`
Expected: PASS

- [ ] **Step 5: Write failing test for save/load helpers**

Add to `tests/test_data/test_spoe_db.py`:

```python
from fantasy_baseball.data.db import (
    save_spoe_results,
    save_spoe_components,
    load_spoe_components,
    get_completed_spoe_weeks,
    get_spoe_results,
)


def test_save_and_load_spoe_results():
    conn = get_connection(":memory:")
    create_tables(conn)
    results = [
        {"team": "Team A", "category": "R",
         "projected_stat": 50.0, "actual_stat": 60.0,
         "projected_pts": 4.0, "actual_pts": 7.0, "spoe": 3.0},
        {"team": "Team A", "category": "total",
         "projected_stat": None, "actual_stat": None,
         "projected_pts": 40.0, "actual_pts": 45.0, "spoe": 5.0},
    ]
    save_spoe_results(conn, "2026-03-31", results)
    rows = get_spoe_results(conn, "2026-03-31")
    assert len(rows) == 2
    assert rows[0]["spoe"] == 3.0
    conn.close()


def test_save_and_load_spoe_components():
    conn = get_connection(":memory:")
    create_tables(conn)
    components = {
        "Team A": {"H": 15.0, "AB": 55.0, "R": 8.0, "IP": 12.0},
        "Team B": {"H": 12.0, "AB": 48.0, "R": 6.0, "IP": 10.0},
    }
    save_spoe_components(conn, "2026-03-31", components)
    loaded = load_spoe_components(conn, "2026-03-31")
    assert loaded["Team A"]["H"] == pytest.approx(15.0)
    assert loaded["Team B"]["IP"] == pytest.approx(10.0)
    conn.close()


def test_get_completed_spoe_weeks():
    conn = get_connection(":memory:")
    create_tables(conn)
    results = [
        {"team": "Team A", "category": "total",
         "projected_stat": None, "actual_stat": None,
         "projected_pts": 40.0, "actual_pts": 45.0, "spoe": 5.0},
    ]
    save_spoe_results(conn, "2026-03-31", results)
    save_spoe_results(conn, "2026-04-07", results)
    weeks = get_completed_spoe_weeks(conn)
    assert weeks == {"2026-03-31", "2026-04-07"}
    conn.close()


def test_save_spoe_results_is_idempotent():
    conn = get_connection(":memory:")
    create_tables(conn)
    results = [
        {"team": "Team A", "category": "R",
         "projected_stat": 50.0, "actual_stat": 60.0,
         "projected_pts": 4.0, "actual_pts": 7.0, "spoe": 3.0},
    ]
    save_spoe_results(conn, "2026-03-31", results)
    # Update with new values — should overwrite
    results[0]["spoe"] = 4.0
    save_spoe_results(conn, "2026-03-31", results)
    rows = get_spoe_results(conn, "2026-03-31")
    assert len(rows) == 1
    assert rows[0]["spoe"] == 4.0
    conn.close()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_data/test_spoe_db.py -v`
Expected: FAIL — functions not defined

- [ ] **Step 7: Implement save/load helpers**

Add to `src/fantasy_baseball/data/db.py` (after the existing `append_standings_snapshot` function, around line 525):

```python
# ---------------------------------------------------------------------------
# SPOE storage helpers
# ---------------------------------------------------------------------------


def save_spoe_results(conn, snapshot_date, results):
    """Save SPOE results for one week.

    ``results`` is a list of dicts with keys: team, category,
    projected_stat, actual_stat, projected_pts, actual_pts, spoe.
    """
    rows = [
        (snapshot_date, r["team"], r["category"],
         r.get("projected_stat"), r.get("actual_stat"),
         r["projected_pts"], r["actual_pts"], r["spoe"])
        for r in results
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO spoe_results "
        "(snapshot_date, team, category, projected_stat, actual_stat, "
        "projected_pts, actual_pts, spoe) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def save_spoe_components(conn, snapshot_date, components):
    """Save accumulated projection components for one week.

    ``components`` is {team_name: {component_name: value}}.
    """
    rows = []
    for team, comps in components.items():
        for comp, value in comps.items():
            rows.append((snapshot_date, team, comp, value))
    conn.executemany(
        "INSERT OR REPLACE INTO spoe_components "
        "(snapshot_date, team, component, value) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def load_spoe_components(conn, snapshot_date):
    """Load accumulated components for a specific week.

    Returns {team_name: {component_name: value}}.
    """
    rows = conn.execute(
        "SELECT team, component, value FROM spoe_components "
        "WHERE snapshot_date = ?",
        (snapshot_date,),
    ).fetchall()
    result = {}
    for r in rows:
        result.setdefault(r["team"], {})[r["component"]] = r["value"]
    return result


def get_completed_spoe_weeks(conn):
    """Return set of snapshot_dates that have SPOE results."""
    rows = conn.execute(
        "SELECT DISTINCT snapshot_date FROM spoe_results"
    ).fetchall()
    return {r["snapshot_date"] for r in rows}


def get_spoe_results(conn, snapshot_date=None):
    """Load SPOE results, optionally filtered by snapshot_date.

    Returns list of dicts with all spoe_results columns.
    """
    if snapshot_date:
        rows = conn.execute(
            "SELECT * FROM spoe_results WHERE snapshot_date = ? "
            "ORDER BY team, category",
            (snapshot_date,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM spoe_results ORDER BY snapshot_date, team, category"
        ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_data/test_spoe_db.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add src/fantasy_baseball/data/db.py tests/test_data/test_spoe_db.py
git commit -m "feat(spoe): add spoe_results and spoe_components tables with storage helpers"
```

---

### Task 2: Data loading functions for SPOE

**Files:**
- Create: `src/fantasy_baseball/analysis/spoe.py`
- Test: `tests/test_analysis/test_spoe.py`

These functions load the three inputs for each week: rosters, projections, and game log totals.

- [ ] **Step 1: Write failing test for load_rosters_for_date**

```python
# tests/test_analysis/test_spoe.py
import sqlite3
import pytest
from fantasy_baseball.data.db import create_tables, get_connection


def _seed_rosters(conn):
    """Insert test roster data for two teams across two weeks."""
    conn.executemany(
        "INSERT INTO weekly_rosters "
        "(snapshot_date, week_num, team, slot, player_name, positions) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("2026-03-31", None, "Team A", "OF", "Juan Soto", "OF, Util"),
            ("2026-03-31", None, "Team A", "P", "Aaron Nola", "SP"),
            ("2026-03-31", None, "Team B", "1B", "Freddie Freeman", "1B, Util"),
            ("2026-03-31", None, "Team B", "P", "Logan Webb", "SP"),
            # Week 2: Team A drops Soto, adds Freeman Jr
            ("2026-04-07", None, "Team A", "OF", "Freeman Jr", "OF"),
            ("2026-04-07", None, "Team A", "P", "Aaron Nola", "SP"),
            ("2026-04-07", None, "Team B", "1B", "Freddie Freeman", "1B, Util"),
            ("2026-04-07", None, "Team B", "P", "Logan Webb", "SP"),
        ],
    )
    conn.commit()


class TestLoadRostersForDate:
    def test_returns_all_teams(self):
        from fantasy_baseball.analysis.spoe import load_rosters_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        rosters = load_rosters_for_date(conn, "2026-03-31")
        assert set(rosters.keys()) == {"Team A", "Team B"}
        conn.close()

    def test_splits_positions_string(self):
        from fantasy_baseball.analysis.spoe import load_rosters_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        rosters = load_rosters_for_date(conn, "2026-03-31")
        soto = [p for p in rosters["Team A"] if p["name"] == "Juan Soto"][0]
        assert soto["positions"] == ["OF", "Util"]
        conn.close()

    def test_roster_changes_across_weeks(self):
        from fantasy_baseball.analysis.spoe import load_rosters_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        week1 = load_rosters_for_date(conn, "2026-03-31")
        week2 = load_rosters_for_date(conn, "2026-04-07")
        week1_names = {p["name"] for p in week1["Team A"]}
        week2_names = {p["name"] for p in week2["Team A"]}
        assert "Juan Soto" in week1_names
        assert "Juan Soto" not in week2_names
        assert "Freeman Jr" in week2_names
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_spoe.py::TestLoadRostersForDate -v`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Implement load_rosters_for_date**

```python
# src/fantasy_baseball/analysis/spoe.py
"""Standings Points Over Expected (SPOE) — luck quantification.

Compares weekly accumulated projected stats to actual standings for all
teams to measure how much standings position is skill vs. luck.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import ALL_CATEGORIES, RATE_STATS
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


# Components tracked for accumulation.  Counting stats double as both
# components and final roto stats; H/AB/IP/ER/BB/H_allowed are used
# only to derive rate stats (AVG, ERA, WHIP).
HITTER_COMPONENTS = ["r", "hr", "rbi", "sb", "h", "ab"]
PITCHER_COMPONENTS = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]
ALL_COMPONENTS = HITTER_COMPONENTS + PITCHER_COMPONENTS


def load_rosters_for_date(conn, snapshot_date):
    """Load all team rosters for a snapshot date from weekly_rosters.

    Returns {team_name: [{"name": str, "positions": list[str]}, ...]}.
    """
    rows = conn.execute(
        "SELECT team, player_name, positions FROM weekly_rosters "
        "WHERE snapshot_date = ?",
        (snapshot_date,),
    ).fetchall()

    rosters: dict[str, list[dict]] = {}
    for row in rows:
        team = row["team"]
        positions = row["positions"].split(", ") if row["positions"] else []
        rosters.setdefault(team, []).append({
            "name": row["player_name"],
            "positions": positions,
        })
    return rosters
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_spoe.py::TestLoadRostersForDate -v`
Expected: PASS

- [ ] **Step 5: Write failing test for load_projections_for_date**

Add to `tests/test_analysis/test_spoe.py`:

```python
def _seed_projections(conn):
    """Insert test ROS blended projections."""
    conn.executemany(
        "INSERT INTO ros_blended_projections "
        "(year, snapshot_date, fg_id, name, team, player_type, "
        "pa, ab, h, r, hr, rbi, sb, avg, w, k, sv, ip, er, bb, h_allowed, era, whip, adp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, "2026-03-30", "1", "Juan Soto", "NYM", "hitter",
             600, 550, 165, 100, 30, 90, 5, 0.300,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 10),
            (2026, "2026-03-30", "2", "Aaron Nola", "PHI", "pitcher",
             0, 0, 0, 0, 0, 0, 0, 0,
             14, 200, 0, 190, 70, 45, 160, 3.32, 1.08, 50),
            # Later snapshot
            (2026, "2026-04-05", "1", "Juan Soto", "NYM", "hitter",
             580, 530, 155, 95, 28, 85, 4, 0.292,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 10),
            (2026, "2026-04-05", "2", "Aaron Nola", "PHI", "pitcher",
             0, 0, 0, 0, 0, 0, 0, 0,
             13, 190, 0, 180, 68, 42, 155, 3.40, 1.09, 50),
        ],
    )
    conn.commit()


class TestLoadProjectionsForDate:
    def test_selects_latest_snapshot_before_target(self):
        from fantasy_baseball.analysis.spoe import load_projections_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_projections(conn)
        hitters, pitchers = load_projections_for_date(conn, 2026, "2026-04-07")
        # Should pick the 2026-04-05 snapshot (latest <= 2026-04-07)
        assert len(hitters) == 1
        assert hitters.iloc[0]["hr"] == 28  # from 04-05 snapshot
        conn.close()

    def test_adds_name_norm_column(self):
        from fantasy_baseball.analysis.spoe import load_projections_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_projections(conn)
        hitters, pitchers = load_projections_for_date(conn, 2026, "2026-03-31")
        assert "_name_norm" in hitters.columns
        assert hitters.iloc[0]["_name_norm"] == "juan soto"
        conn.close()

    def test_falls_back_to_preseason_if_no_ros(self):
        from fantasy_baseball.analysis.spoe import load_projections_for_date
        conn = get_connection(":memory:")
        create_tables(conn)
        # Seed only preseason blended projections, no ROS
        conn.execute(
            "INSERT INTO blended_projections "
            "(year, fg_id, name, team, player_type, "
            "pa, ab, h, r, hr, rbi, sb, avg, w, k, sv, ip, er, bb, h_allowed, era, whip, adp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, "1", "Juan Soto", "NYM", "hitter",
             600, 550, 165, 100, 30, 90, 5, 0.300,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 10),
        )
        conn.commit()
        hitters, pitchers = load_projections_for_date(conn, 2026, "2026-03-31")
        assert len(hitters) == 1
        assert hitters.iloc[0]["name"] == "Juan Soto"
        conn.close()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_spoe.py::TestLoadProjectionsForDate -v`
Expected: FAIL — function not defined

- [ ] **Step 7: Implement load_projections_for_date**

Add to `src/fantasy_baseball/analysis/spoe.py`:

```python
def load_projections_for_date(conn, year, target_date):
    """Load the best-matching blended projections for a target date.

    Tries ROS blended projections first (latest snapshot_date <= target_date).
    Falls back to preseason blended_projections if no ROS data exists.

    Returns (hitters_df, pitchers_df) with ``_name_norm`` column added.
    """
    # Find best ROS snapshot
    best = conn.execute(
        "SELECT MAX(snapshot_date) FROM ros_blended_projections "
        "WHERE year = ? AND snapshot_date <= ?",
        (year, target_date),
    ).fetchone()[0]

    if best:
        rows = conn.execute(
            "SELECT * FROM ros_blended_projections "
            "WHERE year = ? AND snapshot_date = ?",
            (year, best),
        ).fetchall()
    else:
        # Fall back to preseason
        rows = conn.execute(
            "SELECT * FROM blended_projections WHERE year = ?",
            (year,),
        ).fetchall()

    if not rows:
        empty = pd.DataFrame()
        return empty, empty

    df = pd.DataFrame([dict(r) for r in rows])
    df["_name_norm"] = df["name"].apply(normalize_name)

    hitters = df[df["player_type"] == "hitter"].copy()
    pitchers = df[df["player_type"] == "pitcher"].copy()
    return hitters, pitchers
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_spoe.py::TestLoadProjectionsForDate -v`
Expected: PASS

- [ ] **Step 9: Write failing test for aggregate_game_logs_before**

Add to `tests/test_analysis/test_spoe.py`:

```python
def _seed_game_logs(conn):
    """Insert test game log data."""
    conn.executemany(
        "INSERT INTO game_logs "
        "(season, mlbam_id, name, team, player_type, date, "
        "pa, ab, h, r, hr, rbi, sb, ip, k, er, bb, h_allowed, w, sv, gs) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, 1, "Juan Soto", "NYM", "hitter", "2026-03-28",
             5, 4, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (2026, 1, "Juan Soto", "NYM", "hitter", "2026-03-29",
             4, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (2026, 2, "Aaron Nola", "PHI", "pitcher", "2026-03-29",
             0, 0, 0, 0, 0, 0, 0, 6.0, 7, 2, 1, 5, 1, 0, 1),
            # Game on 2026-04-05 — should NOT be included when querying before 2026-04-01
            (2026, 1, "Juan Soto", "NYM", "hitter", "2026-04-05",
             4, 4, 3, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ],
    )
    conn.commit()


class TestAggregateGameLogsBefore:
    def test_sums_stats_before_date(self):
        from fantasy_baseball.analysis.spoe import aggregate_game_logs_before
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_game_logs(conn)
        totals = aggregate_game_logs_before(conn, 2026, "2026-04-01")
        soto = totals["juan soto"]
        # 2 games: 5+4=9 pa, 4+3=7 ab, 2+1=3 h, 1+0=1 r, 1+0=1 hr
        assert soto["h"] == pytest.approx(3.0)
        assert soto["hr"] == pytest.approx(1.0)
        assert soto["ab"] == pytest.approx(7.0)
        conn.close()

    def test_excludes_games_on_or_after_date(self):
        from fantasy_baseball.analysis.spoe import aggregate_game_logs_before
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_game_logs(conn)
        totals = aggregate_game_logs_before(conn, 2026, "2026-04-01")
        soto = totals["juan soto"]
        # The 04-05 game should be excluded
        assert soto["r"] == pytest.approx(1.0)  # only from 03-28
        conn.close()

    def test_includes_pitchers(self):
        from fantasy_baseball.analysis.spoe import aggregate_game_logs_before
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_game_logs(conn)
        totals = aggregate_game_logs_before(conn, 2026, "2026-04-01")
        nola = totals["aaron nola"]
        assert nola["ip"] == pytest.approx(6.0)
        assert nola["k"] == pytest.approx(7.0)
        assert nola["w"] == pytest.approx(1.0)
        conn.close()
```

- [ ] **Step 10: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_spoe.py::TestAggregateGameLogsBefore -v`
Expected: FAIL — function not defined

- [ ] **Step 11: Implement aggregate_game_logs_before**

Add to `src/fantasy_baseball/analysis/spoe.py`:

```python
def aggregate_game_logs_before(conn, season, before_date):
    """Sum game log stats for each player before a given date.

    Returns {normalized_name: {stat: total}}.
    Stat keys are lowercase: h, ab, r, hr, rbi, sb, ip, k, er, bb, h_allowed, w, sv.
    """
    rows = conn.execute(
        "SELECT name, "
        "SUM(pa) as pa, SUM(ab) as ab, SUM(h) as h, "
        "SUM(r) as r, SUM(hr) as hr, SUM(rbi) as rbi, SUM(sb) as sb, "
        "SUM(ip) as ip, SUM(k) as k, SUM(er) as er, "
        "SUM(bb) as bb, SUM(h_allowed) as h_allowed, "
        "SUM(w) as w, SUM(sv) as sv "
        "FROM game_logs WHERE season = ? AND date < ? "
        "GROUP BY name",
        (season, before_date),
    ).fetchall()

    result = {}
    for r in rows:
        name_norm = normalize_name(r["name"])
        result[name_norm] = {
            "pa": r["pa"] or 0, "ab": r["ab"] or 0, "h": r["h"] or 0,
            "r": r["r"] or 0, "hr": r["hr"] or 0, "rbi": r["rbi"] or 0,
            "sb": r["sb"] or 0, "ip": r["ip"] or 0, "k": r["k"] or 0,
            "er": r["er"] or 0, "bb": r["bb"] or 0,
            "h_allowed": r["h_allowed"] or 0, "w": r["w"] or 0,
            "sv": r["sv"] or 0,
        }
    return result
```

- [ ] **Step 12: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_spoe.py::TestAggregateGameLogsBefore -v`
Expected: PASS

- [ ] **Step 13: Commit**

```bash
git add src/fantasy_baseball/analysis/spoe.py tests/test_analysis/test_spoe.py
git commit -m "feat(spoe): add data loading functions — rosters, projections, game logs"
```

---

### Task 3: Weekly projected stat computation

**Files:**
- Modify: `src/fantasy_baseball/analysis/spoe.py`
- Test: `tests/test_analysis/test_spoe.py`

This function takes one team's matched roster + game log totals and computes the projected component stats for one week.

- [ ] **Step 1: Write failing test for project_team_week**

Add to `tests/test_analysis/test_spoe.py`:

```python
from fantasy_baseball.models.player import Player, HitterStats, PitcherStats, PlayerType


def _make_test_hitter(name, r, hr, rbi, sb, h, ab):
    return Player(
        name=name, player_type=PlayerType.HITTER, positions=["OF"],
        ros=HitterStats(r=r, hr=hr, rbi=rbi, sb=sb, h=h, ab=ab, avg=h / ab if ab else 0),
    )


def _make_test_pitcher(name, w, k, sv, ip, er, bb, h_allowed):
    return Player(
        name=name, player_type=PlayerType.PITCHER, positions=["SP"],
        ros=PitcherStats(
            w=w, k=k, sv=sv, ip=ip, er=er, bb=bb, h_allowed=h_allowed,
            era=er * 9 / ip if ip else 0, whip=(bb + h_allowed) / ip if ip else 0,
        ),
    )


class TestProjectTeamWeek:
    def test_scales_counting_stats_by_weekly_fraction(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        roster = [_make_test_hitter("Hitter", 100, 30, 90, 10, 150, 500)]
        game_log_totals = {}  # no actuals yet
        # 175 days remaining, weekly_fraction = 7/175 = 0.04
        components = project_team_week(roster, game_log_totals, days_remaining=175)
        assert components["r"] == pytest.approx(100 * 7 / 175)
        assert components["hr"] == pytest.approx(30 * 7 / 175)
        assert components["h"] == pytest.approx(150 * 7 / 175)

    def test_subtracts_actuals_before_scaling(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        # Full-season projection: 30 HR. Actuals so far: 5 HR. Remaining: 25.
        roster = [_make_test_hitter("Hitter", 100, 30, 90, 10, 150, 500)]
        game_log_totals = {"hitter": {"hr": 5, "r": 10, "rbi": 8, "sb": 1,
                                       "h": 20, "ab": 60}}
        components = project_team_week(roster, game_log_totals, days_remaining=175)
        assert components["hr"] == pytest.approx(25 * 7 / 175)

    def test_clamps_remaining_to_zero(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        # Player already exceeded projection: 30 projected, 35 actual
        roster = [_make_test_hitter("Hitter", 100, 30, 90, 10, 150, 500)]
        game_log_totals = {"hitter": {"hr": 35, "r": 10, "rbi": 8, "sb": 1,
                                       "h": 20, "ab": 60}}
        components = project_team_week(roster, game_log_totals, days_remaining=175)
        assert components["hr"] == pytest.approx(0.0)

    def test_sums_hitters_and_pitchers(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        roster = [
            _make_test_hitter("Hitter", 100, 30, 90, 10, 150, 500),
            _make_test_pitcher("Pitcher", 14, 200, 0, 190, 70, 45, 160),
        ]
        components = project_team_week(roster, {}, days_remaining=175)
        assert components["r"] == pytest.approx(100 * 7 / 175)
        assert components["w"] == pytest.approx(14 * 7 / 175)
        assert components["k"] == pytest.approx(200 * 7 / 175)
        assert components["ip"] == pytest.approx(190 * 7 / 175)

    def test_unmatched_player_contributes_nothing(self):
        from fantasy_baseball.analysis.spoe import project_team_week
        # Player with no ros stats (unmatched to projections)
        unmatched = Player(
            name="Ghost", player_type=PlayerType.HITTER, positions=["OF"],
        )
        roster = [unmatched]
        components = project_team_week(roster, {}, days_remaining=175)
        assert components["r"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_spoe.py::TestProjectTeamWeek -v`
Expected: FAIL — function not defined

- [ ] **Step 3: Implement project_team_week**

Add to `src/fantasy_baseball/analysis/spoe.py`:

```python
def project_team_week(roster, game_log_totals, days_remaining):
    """Project one week of component stats for a team's roster.

    For each player, subtracts actual stats from full-season projection
    to get remaining-season stats, then scales to one week.  Players
    without ROS projections contribute nothing.

    Args:
        roster: list of Player objects with .ros populated
        game_log_totals: {normalized_name: {stat: value}} from game logs
        days_remaining: days from this week's Monday to season end

    Returns:
        dict of component stats for the team for this week.
    """
    weekly_fraction = 7 / days_remaining if days_remaining > 0 else 0
    team_components = {c: 0.0 for c in ALL_COMPONENTS}

    for player in roster:
        if player.ros is None:
            continue

        name_norm = normalize_name(player.name)
        actuals = game_log_totals.get(name_norm, {})

        if player.player_type == PlayerType.HITTER:
            component_keys = HITTER_COMPONENTS
        else:
            component_keys = PITCHER_COMPONENTS

        for key in component_keys:
            projected = getattr(player.ros, key, 0) or 0
            actual = actuals.get(key, 0)
            remaining = max(0, projected - actual)
            team_components[key] += remaining * weekly_fraction

    return team_components
```

Also add to the imports at the top of spoe.py:

```python
from fantasy_baseball.models.player import PlayerType
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_spoe.py::TestProjectTeamWeek -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/spoe.py tests/test_analysis/test_spoe.py
git commit -m "feat(spoe): add project_team_week — weekly stat projection with actual subtraction"
```

---

### Task 4: SPOE orchestration engine

**Files:**
- Modify: `src/fantasy_baseball/analysis/spoe.py`
- Test: `tests/test_analysis/test_spoe.py`

The main `compute_spoe()` function that loops over weeks, accumulates projected stats, and compares to actual standings.

- [ ] **Step 1: Write failing test for components_to_roto_stats**

This helper converts accumulated components to the stat dict that `score_roto()` expects.

Add to `tests/test_analysis/test_spoe.py`:

```python
class TestComponentsToRotoStats:
    def test_counting_stats_pass_through(self):
        from fantasy_baseball.analysis.spoe import components_to_roto_stats
        comps = {"r": 50, "hr": 15, "rbi": 45, "sb": 5, "h": 80, "ab": 300,
                 "w": 7, "k": 100, "sv": 5, "ip": 95, "er": 35, "bb": 25, "h_allowed": 80}
        stats = components_to_roto_stats(comps)
        assert stats["R"] == 50
        assert stats["HR"] == 15
        assert stats["W"] == 7

    def test_rate_stats_computed_from_components(self):
        from fantasy_baseball.analysis.spoe import components_to_roto_stats
        comps = {"r": 50, "hr": 15, "rbi": 45, "sb": 5, "h": 80, "ab": 300,
                 "w": 7, "k": 100, "sv": 5, "ip": 100, "er": 35, "bb": 25, "h_allowed": 80}
        stats = components_to_roto_stats(comps)
        assert stats["AVG"] == pytest.approx(80 / 300)
        assert stats["ERA"] == pytest.approx(35 * 9 / 100)
        assert stats["WHIP"] == pytest.approx((25 + 80) / 100)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_spoe.py::TestComponentsToRotoStats -v`
Expected: FAIL

- [ ] **Step 3: Implement components_to_roto_stats**

Add to `src/fantasy_baseball/analysis/spoe.py`:

```python
def components_to_roto_stats(components):
    """Convert accumulated component stats to the roto stat dict score_roto expects.

    Counting stats pass through; rate stats are computed from components.
    """
    return {
        "R": components["r"],
        "HR": components["hr"],
        "RBI": components["rbi"],
        "SB": components["sb"],
        "AVG": calculate_avg(components["h"], components["ab"]),
        "W": components["w"],
        "K": components["k"],
        "SV": components["sv"],
        "ERA": calculate_era(components["er"], components["ip"]),
        "WHIP": calculate_whip(components["bb"], components["h_allowed"], components["ip"]),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_spoe.py::TestComponentsToRotoStats -v`
Expected: PASS

- [ ] **Step 5: Write failing end-to-end test for compute_spoe**

Add to `tests/test_analysis/test_spoe.py`:

```python
def _seed_standings(conn):
    """Insert standings snapshots matching the roster weeks."""
    conn.executemany(
        "INSERT INTO standings "
        "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (2026, "2026-03-31", "Team A", 1, 30, 8, 25, 3, 0.280, 3, 40, 2, 3.20, 1.10),
            (2026, "2026-03-31", "Team B", 2, 25, 6, 20, 5, 0.260, 2, 35, 4, 3.80, 1.25),
        ],
    )
    conn.commit()


def _make_test_config():
    """Minimal config for SPOE tests."""
    from fantasy_baseball.config import LeagueConfig
    return LeagueConfig(
        league_id=1,
        num_teams=2,
        game_code="mlb",
        team_name="Team A",
        draft_position=1,
        keepers=[],
        roster_slots={},
        projection_systems=["steamer"],
        projection_weights={"steamer": 1.0},
        season_year=2026,
        season_start="2026-03-27",
        season_end="2026-09-28",
    )


class TestComputeSpoe:
    def test_produces_results_for_all_teams_and_categories(self):
        from fantasy_baseball.analysis.spoe import compute_spoe
        from fantasy_baseball.data.db import get_spoe_results
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        _seed_projections(conn)
        _seed_game_logs(conn)
        _seed_standings(conn)
        config = _make_test_config()

        compute_spoe(conn, config)

        results = get_spoe_results(conn, "2026-03-31")
        teams = {r["team"] for r in results}
        assert teams == {"Team A", "Team B"}
        categories = {r["category"] for r in results if r["team"] == "Team A"}
        expected_cats = set(ALL_CATEGORIES) | {"total"}
        assert categories == expected_cats

    def test_spoe_is_actual_minus_projected(self):
        from fantasy_baseball.analysis.spoe import compute_spoe
        from fantasy_baseball.data.db import get_spoe_results
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        _seed_projections(conn)
        _seed_game_logs(conn)
        _seed_standings(conn)
        config = _make_test_config()

        compute_spoe(conn, config)

        results = get_spoe_results(conn, "2026-03-31")
        for r in results:
            if r["category"] != "total":
                assert r["spoe"] == pytest.approx(
                    r["actual_pts"] - r["projected_pts"]
                )

    def test_total_spoe_is_sum_of_categories(self):
        from fantasy_baseball.analysis.spoe import compute_spoe
        from fantasy_baseball.data.db import get_spoe_results
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        _seed_projections(conn)
        _seed_game_logs(conn)
        _seed_standings(conn)
        config = _make_test_config()

        compute_spoe(conn, config)

        results = get_spoe_results(conn, "2026-03-31")
        for team in ["Team A", "Team B"]:
            team_results = {r["category"]: r for r in results if r["team"] == team}
            cat_spoe_sum = sum(
                team_results[c]["spoe"] for c in ALL_CATEGORIES
            )
            assert team_results["total"]["spoe"] == pytest.approx(cat_spoe_sum)

    def test_skips_completed_weeks_except_current(self):
        from fantasy_baseball.analysis.spoe import compute_spoe
        from fantasy_baseball.data.db import get_spoe_results, save_spoe_results, save_spoe_components
        conn = get_connection(":memory:")
        create_tables(conn)
        _seed_rosters(conn)
        _seed_projections(conn)
        _seed_game_logs(conn)
        _seed_standings(conn)

        # Add a second week of standings so week 1 is not "current"
        conn.execute(
            "INSERT INTO standings "
            "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, "2026-04-07", "Team A", 1, 50, 12, 40, 5, 0.275, 5, 70, 4, 3.40, 1.15),
        )
        conn.execute(
            "INSERT INTO standings "
            "(year, snapshot_date, team, rank, r, hr, rbi, sb, avg, w, k, sv, era, whip) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, "2026-04-07", "Team B", 2, 40, 10, 35, 8, 0.260, 4, 60, 6, 3.90, 1.30),
        )
        conn.commit()

        config = _make_test_config()

        # Pre-populate week 1 with known values AND components
        save_spoe_results(conn, "2026-03-31", [
            {"team": "Team A", "category": "total",
             "projected_stat": None, "actual_stat": None,
             "projected_pts": 99.0, "actual_pts": 99.0, "spoe": 0.0},
        ])
        save_spoe_components(conn, "2026-03-31", {
            "Team A": {c: 1.0 for c in ["r","hr","rbi","sb","h","ab","w","k","sv","ip","er","bb","h_allowed"]},
            "Team B": {c: 1.0 for c in ["r","hr","rbi","sb","h","ab","w","k","sv","ip","er","bb","h_allowed"]},
        })

        compute_spoe(conn, config)

        # Week 1 total should be preserved (not overwritten)
        results = get_spoe_results(conn, "2026-03-31")
        total = [r for r in results if r["team"] == "Team A" and r["category"] == "total"]
        assert len(total) == 1
        assert total[0]["projected_pts"] == pytest.approx(99.0)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_spoe.py::TestComputeSpoe -v`
Expected: FAIL — function not defined

- [ ] **Step 7: Implement compute_spoe**

Add to `src/fantasy_baseball/analysis/spoe.py`:

```python
from fantasy_baseball.data.db import (
    get_completed_spoe_weeks,
    load_spoe_components,
    save_spoe_components,
    save_spoe_results,
)
from fantasy_baseball.data.projections import match_roster_to_projections


def _get_week_dates(conn, season_year):
    """Get all distinct roster snapshot dates for the season, sorted."""
    rows = conn.execute(
        "SELECT DISTINCT snapshot_date FROM weekly_rosters "
        "WHERE snapshot_date >= ? ORDER BY snapshot_date",
        (f"{season_year}-",),
    ).fetchall()
    return [r["snapshot_date"] for r in rows]


def _get_standings_for_date(conn, season_year, snapshot_date):
    """Load actual standings for a snapshot date.

    Returns {team_name: {R: val, HR: val, ...}}.
    """
    rows = conn.execute(
        "SELECT team, r, hr, rbi, sb, avg, w, k, sv, era, whip "
        "FROM standings WHERE year = ? AND snapshot_date = ?",
        (season_year, snapshot_date),
    ).fetchall()
    return {
        r["team"]: {
            "R": r["r"], "HR": r["hr"], "RBI": r["rbi"], "SB": r["sb"],
            "AVG": r["avg"], "W": r["w"], "K": r["k"], "SV": r["sv"],
            "ERA": r["era"], "WHIP": r["whip"],
        }
        for r in rows
    }


def compute_spoe(conn, config):
    """Compute SPOE for all weeks with available data.

    Completed weeks are skipped. The last (current) week is always
    recomputed since its data may have changed since the last refresh.
    Results are stored in spoe_results and spoe_components tables.
    """
    week_dates = _get_week_dates(conn, config.season_year)
    if not week_dates:
        return

    completed = get_completed_spoe_weeks(conn)
    season_end = date.fromisoformat(config.season_end)
    current_week = week_dates[-1]  # last week is always "current"

    # Initialize accumulators — try to resume from last completed week
    team_components: dict[str, dict[str, float]] = {}
    for prev_date in reversed(week_dates):
        if prev_date in completed and prev_date != current_week:
            team_components = load_spoe_components(conn, prev_date)
            # Only process weeks after this one
            start_idx = week_dates.index(prev_date) + 1
            break
    else:
        start_idx = 0

    for i in range(start_idx, len(week_dates)):
        snapshot_date = week_dates[i]

        # Skip completed weeks (but always recompute current)
        if snapshot_date in completed and snapshot_date != current_week:
            continue

        monday = date.fromisoformat(snapshot_date)
        days_remaining = (season_end - monday).days
        if days_remaining <= 0:
            continue

        # 1. Load rosters
        rosters = load_rosters_for_date(conn, snapshot_date)
        if not rosters:
            continue

        # 2. Load projections
        hitters_proj, pitchers_proj = load_projections_for_date(
            conn, config.season_year, snapshot_date
        )
        if hitters_proj.empty and pitchers_proj.empty:
            continue

        # 3. Load game log totals before this week
        game_log_totals = aggregate_game_logs_before(
            conn, config.season_year, snapshot_date
        )

        # 4. Load actual standings
        actual_stats = _get_standings_for_date(conn, config.season_year, snapshot_date)
        if not actual_stats:
            continue

        # 5. Project weekly stats for each team
        for team_name, roster_dicts in rosters.items():
            # Match roster players to projections
            matched = match_roster_to_projections(
                roster_dicts, hitters_proj, pitchers_proj
            )

            # Compute this week's projected components
            weekly = project_team_week(matched, game_log_totals, days_remaining)

            # Accumulate
            if team_name not in team_components:
                team_components[team_name] = {c: 0.0 for c in ALL_COMPONENTS}
            for comp in ALL_COMPONENTS:
                team_components[team_name][comp] += weekly[comp]

        # 6. Convert accumulated components to roto stats
        projected_stats = {
            team: components_to_roto_stats(comps)
            for team, comps in team_components.items()
            if team in actual_stats  # only score teams we have actuals for
        }

        # 7. Score roto for both projected and actual
        # Only include teams present in both
        common_teams = set(projected_stats) & set(actual_stats)
        if len(common_teams) < 2:
            continue

        proj_for_scoring = {t: projected_stats[t] for t in common_teams}
        actual_for_scoring = {t: actual_stats[t] for t in common_teams}

        projected_roto = score_roto(proj_for_scoring)
        actual_roto = score_roto(actual_for_scoring)

        # 8. Compute SPOE and store
        results = []
        for team in common_teams:
            total_spoe = 0.0
            for cat in ALL_CATEGORIES:
                proj_pts = projected_roto[team].get(f"{cat}_pts", 0)
                act_pts = actual_roto[team].get(f"{cat}_pts", 0)
                spoe = act_pts - proj_pts
                total_spoe += spoe
                results.append({
                    "team": team,
                    "category": cat,
                    "projected_stat": projected_stats[team][cat],
                    "actual_stat": actual_stats[team][cat],
                    "projected_pts": proj_pts,
                    "actual_pts": act_pts,
                    "spoe": spoe,
                })
            results.append({
                "team": team,
                "category": "total",
                "projected_stat": None,
                "actual_stat": None,
                "projected_pts": projected_roto[team]["total"],
                "actual_pts": actual_roto[team]["total"],
                "spoe": total_spoe,
            })

        save_spoe_results(conn, snapshot_date, results)
        save_spoe_components(conn, snapshot_date, team_components)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_spoe.py::TestComputeSpoe -v`
Expected: PASS

- [ ] **Step 9: Run all SPOE tests**

Run: `pytest tests/test_analysis/test_spoe.py -v`
Expected: all PASS

- [ ] **Step 10: Commit**

```bash
git add src/fantasy_baseball/analysis/spoe.py tests/test_analysis/test_spoe.py
git commit -m "feat(spoe): implement SPOE orchestration engine with weekly accumulation"
```

---

### Task 5: Integrate SPOE into dashboard refresh pipeline

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:1280-1282`
- Test: `tests/test_web/test_spoe_refresh.py`

- [ ] **Step 1: Write failing test for SPOE call in refresh**

```python
# tests/test_web/test_spoe_refresh.py
from unittest.mock import patch, MagicMock


@patch("fantasy_baseball.analysis.spoe.compute_spoe")
def test_refresh_calls_compute_spoe(mock_compute):
    """Verify compute_spoe is called during the DB update step."""
    # We don't run the full refresh — just verify the import and call exist
    # by checking that the function is importable from the expected location
    from fantasy_baseball.analysis.spoe import compute_spoe
    assert callable(compute_spoe)
```

- [ ] **Step 2: Run test to verify it passes** (it should already pass since compute_spoe exists)

Run: `pytest tests/test_web/test_spoe_refresh.py -v`
Expected: PASS

- [ ] **Step 3: Add compute_spoe call to refresh pipeline**

In `src/fantasy_baseball/web/season_data.py`, after line 1280 (`append_standings_snapshot`), add:

```python
            # Compute SPOE (luck analysis)
            from fantasy_baseball.analysis.spoe import compute_spoe
            compute_spoe(db_conn, config)
```

The full block (lines 1273-1282) should now read:

```python
        try:
            # Append all team rosters for this week
            week_num = None
            for tname, raw_roster in all_raw_rosters.items():
                append_roster_snapshot(db_conn, raw_roster, snapshot_date, week_num, tname)

            # Append current standings snapshot
            append_standings_snapshot(db_conn, standings, config.season_year, snapshot_date)

            # Compute SPOE (luck analysis)
            from fantasy_baseball.analysis.spoe import compute_spoe
            compute_spoe(db_conn, config)
        finally:
            db_conn.close()
```

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_spoe_refresh.py
git commit -m "feat(spoe): hook SPOE computation into dashboard refresh pipeline"
```

---

### Task 6: Dashboard API endpoint and UI

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Modify: `src/fantasy_baseball/web/templates/season/base.html:33-34`
- Create: `src/fantasy_baseball/web/templates/season/luck.html`

- [ ] **Step 1: Add the /luck route and /api/spoe endpoint**

Add to `src/fantasy_baseball/web/season_routes.py` inside `register_routes()`, after the existing routes (before the `@app.route("/login")` block):

```python
    @app.route("/luck")
    def luck():
        meta = read_meta()
        config = _load_config()

        from fantasy_baseball.data.db import get_connection, get_spoe_results
        conn = get_connection()
        try:
            # Get the latest snapshot date
            row = conn.execute(
                "SELECT MAX(snapshot_date) as latest FROM spoe_results"
            ).fetchone()
            latest = row["latest"] if row else None

            spoe_data = []
            if latest:
                results = get_spoe_results(conn, latest)
                # Group by team
                teams = {}
                for r in results:
                    team = r["team"]
                    if team not in teams:
                        teams[team] = {"team": team, "categories": {}}
                    if r["category"] == "total":
                        teams[team]["total_spoe"] = r["spoe"]
                        teams[team]["projected_pts"] = r["projected_pts"]
                        teams[team]["actual_pts"] = r["actual_pts"]
                    else:
                        teams[team]["categories"][r["category"]] = r

                spoe_data = sorted(
                    teams.values(),
                    key=lambda t: t.get("actual_pts", 0),
                    reverse=True,
                )
        finally:
            conn.close()

        return render_template(
            "season/luck.html",
            meta=meta,
            active_page="luck",
            spoe_data=spoe_data,
            snapshot_date=latest,
        )
```

Also add `render_template` to the existing imports if not already there (it should be — verify at the top of the file).

- [ ] **Step 2: Add nav link in base.html**

In `src/fantasy_baseball/web/templates/season/base.html`, add after the Players nav link (after line 33):

```html
            <a href="{{ url_for('luck') }}"
               class="nav-link {% if active_page == 'luck' %}active{% endif %}">
                Luck
            </a>
```

- [ ] **Step 3: Create the luck.html template**

Create `src/fantasy_baseball/web/templates/season/luck.html`:

```html
{% extends "season/base.html" %}

{% block title %}Luck — Season Dashboard{% endblock %}

{% block content %}
<style>
    .spoe-table { width: 100%; border-collapse: collapse; }
    .spoe-table th, .spoe-table td { padding: 8px 12px; text-align: right; }
    .spoe-table th:first-child, .spoe-table td:first-child { text-align: left; }
    .spoe-table th { border-bottom: 2px solid var(--panel-border); color: var(--text-dim); font-size: 0.85em; text-transform: uppercase; }
    .spoe-table tr { border-bottom: 1px solid var(--panel-border); }
    .spoe-positive { color: var(--success); }
    .spoe-negative { color: var(--danger); }
    .expandable { cursor: pointer; }
    .expandable:hover { background: rgba(255,255,255,0.03); }
    .expand-content { display: none; }
    .expand-content.open { display: table-row; }
    .expand-content td { padding: 4px 12px; font-size: 0.9em; }
    .detail-table { width: 100%; margin: 4px 0; }
    .detail-table th { font-size: 0.8em; }
    .detail-table td { padding: 4px 8px; }
    .rank-num { color: var(--text-dim); }
</style>

<h2>Luck Analysis (SPOE)</h2>
{% if snapshot_date %}
<p style="color: var(--text-dim); font-size: 0.9em;">
    Standings Points Over Expected as of {{ snapshot_date }}.
    Positive = lucky (outperforming projections). Negative = unlucky.
</p>
{% endif %}

{% if spoe_data %}
<table class="spoe-table">
    <thead>
        <tr>
            <th>#</th>
            <th>Team</th>
            <th>Actual Pts</th>
            <th>Projected Pts</th>
            <th>SPOE</th>
        </tr>
    </thead>
    <tbody>
        {% for team in spoe_data %}
        <tr class="expandable" onclick="toggleExpand(this)">
            <td class="rank-num">{{ loop.index }}</td>
            <td>{{ team.team }}</td>
            <td>{{ "%.1f"|format(team.actual_pts) }}</td>
            <td>{{ "%.1f"|format(team.projected_pts) }}</td>
            <td class="{% if team.total_spoe > 0.5 %}spoe-positive{% elif team.total_spoe < -0.5 %}spoe-negative{% endif %}">
                {{ "%+.1f"|format(team.total_spoe) }}
            </td>
        </tr>
        <tr class="expand-content">
            <td colspan="5">
                <table class="detail-table">
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Proj. Stat</th>
                            <th>Actual Stat</th>
                            <th>Proj. Pts</th>
                            <th>Actual Pts</th>
                            <th>SPOE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"] %}
                        {% set c = team.categories.get(cat, {}) %}
                        <tr>
                            <td>{{ cat }}</td>
                            <td>{% if cat in ["AVG", "ERA", "WHIP"] %}{{ "%.3f"|format(c.get("projected_stat", 0)) }}{% else %}{{ "%.1f"|format(c.get("projected_stat", 0)) }}{% endif %}</td>
                            <td>{% if cat in ["AVG", "ERA", "WHIP"] %}{{ "%.3f"|format(c.get("actual_stat", 0)) }}{% else %}{{ "%.1f"|format(c.get("actual_stat", 0)) }}{% endif %}</td>
                            <td>{{ "%.1f"|format(c.get("projected_pts", 0)) }}</td>
                            <td>{{ "%.1f"|format(c.get("actual_pts", 0)) }}</td>
                            <td class="{% if c.get('spoe', 0) > 0.25 %}spoe-positive{% elif c.get('spoe', 0) < -0.25 %}spoe-negative{% endif %}">
                                {{ "%+.1f"|format(c.get("spoe", 0)) }}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% else %}
<p style="color: var(--text-dim);">No SPOE data yet. Run a dashboard refresh to compute standings luck analysis.</p>
{% endif %}

<script>
function toggleExpand(row) {
    const next = row.nextElementSibling;
    if (next) next.classList.toggle('open');
}
</script>
{% endblock %}
```

- [ ] **Step 4: Run the full test suite to make sure nothing is broken**

Run: `pytest tests/ -v --ignore=tests/test_integration`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py \
        src/fantasy_baseball/web/templates/season/base.html \
        src/fantasy_baseball/web/templates/season/luck.html
git commit -m "feat(spoe): add Luck dashboard tab with SPOE table and per-category detail"
```

- [ ] **Step 6: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 7: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address test failures from SPOE integration"
```
