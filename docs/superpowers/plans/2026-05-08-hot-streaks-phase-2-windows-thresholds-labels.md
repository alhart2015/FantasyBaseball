# Hot Streaks — Phase 2 (Windows, Thresholds, Labels) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate `hitter_windows`, `thresholds`, and `hitter_streak_labels` so a notebook can produce empirical hot/cold thresholds for hitters across 3-, 7-, and 14-day windows. After this phase: a single CLI rebuilds the full label set on top of the (re-fetched) raw data, and a distribution notebook validates that the calibrated thresholds pass the eyeball test.

**Architecture:** Stage A reshapes the raw layer to capture every box-score and Statcast peripheral the rest of the project will need (one-shot re-fetch, no further reprocessing). Stage B builds rolling-window aggregates in pandas (DuckDB → DataFrame → groupby+rolling → upsert). Stage C derives empirical p10/p90 thresholds in DuckDB SQL (percentiles are SQL-natural) and applies them per row to populate `hitter_streak_labels`. Stage D is the Phase-2 acceptance notebook.

**Tech Stack:** DuckDB (analytical SQL + percentile_cont), pandas (window aggregation, idempotent because we replay against deterministic input), pybaseball (Statcast — re-fetch only), MLB Stats API (game logs — re-fetch only), pytest (TDD throughout), Jupyter (acceptance notebook).

**Spec:** `docs/superpowers/specs/2026-05-06-hot-streaks-design.md` — see "Next milestone" at the bottom for the open questions resolved by this plan.

---

## Design Decisions (resolved before plan-writing)

1. **BABIP/ISO sourcing — Path (a):** Extend `hitter_games` with the missing box-score components (`b2`, `b3`, `sf`, `hbp`) plus everything else plausibly useful in Phase 3-5 (`ibb`, `cs`, `gidp`, `sh`, `ci`, `is_home`). Path (b) — deriving from Statcast event counts — was rejected: ~1-2% PA-count drift vs the box score and event-string parsing complexity.
2. **Statcast peripherals expansion:** Add `at_bat_number`, `bb_type`, `estimated_ba_using_speedangle`, `hit_distance_sc` to `hitter_statcast_pa`. `at_bat_number` is the chronological-stability fix flagged as Open Question 2 — sort by it before assigning `pa_index`.
3. **PA identity check:** Capture `sh` and `ci` so the loader can assert `pa == ab + bb + hbp + sf + sh + ci` per game. Violations get logged + counted in the fetch summary; do not raise.
4. **Window granularity:** One row per `(player_id, calendar_date in [first_played, last_played], window_days ∈ {3, 7, 14})`. Calendar dates (not just played dates) so an analyst querying "last 7 days through today" gets a row even on off-days. Filter PA < 5 windows out before write.
5. **Threshold calibration set:** All `hitter_windows` rows whose player had ≥150 PA in that season, stratified by (category × window_days × pt_bucket). Holds back 2026 and (per the spec's Phase 4 split) eventually 2025 for out-of-sample validation.
6. **Migration path:** ALTER TABLE existing DB to add new columns + DELETE rows + re-run fetch. Cleanest because the local DB is gitignored and we're committing to one re-fetch.

---

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/streaks/models.py` | Modify | Extend `HitterGame` + `HitterStatcastPA` with new fields |
| `src/fantasy_baseball/streaks/data/schema.py` | Modify | Update DDL for `hitter_games` + `hitter_statcast_pa` |
| `src/fantasy_baseball/streaks/data/game_logs.py` | Modify | Parse new gameLog fields; expose `pa_identity_gap` helper |
| `src/fantasy_baseball/streaks/data/statcast.py` | Modify | Parse new Statcast fields; sort by `at_bat_number` before `pa_index` |
| `src/fantasy_baseball/streaks/data/fetch_history.py` | Modify | Surface PA-identity violation count in summary |
| `src/fantasy_baseball/streaks/data/migrate.py` | Create | One-shot ALTER+DELETE migration for existing DBs |
| `scripts/streaks/migrate.py` | Create | CLI entry for the migration |
| `src/fantasy_baseball/streaks/windows.py` | Create | Pandas-backed rolling-window aggregator + idempotent upsert |
| `scripts/streaks/compute_windows.py` | Create | CLI entry to (re)build `hitter_windows` |
| `src/fantasy_baseball/streaks/thresholds.py` | Create | DuckDB-SQL p10/p90 calibration over qualified-hitter rows |
| `src/fantasy_baseball/streaks/labels.py` | Create | Threshold application → `hitter_streak_labels` |
| `scripts/streaks/compute_labels.py` | Create | CLI: chain windows → thresholds → labels |
| `tests/test_streaks/test_models.py` | Create | Verify dataclass field surface (cheap regression guard) |
| `tests/test_streaks/test_schema.py` | Modify | Assert new columns exist with expected types |
| `tests/test_streaks/test_game_logs.py` | Modify | New-field parsing + identity-check tests |
| `tests/test_streaks/test_statcast.py` | Modify | New-field parsing + chronological `pa_index` test |
| `tests/test_streaks/test_fetch_history.py` | Modify | Identity-violation surfacing in summary |
| `tests/test_streaks/test_migrate.py` | Create | Idempotency: skip when columns already exist |
| `tests/test_streaks/test_windows.py` | Create | Rolling sums, rate stats, Statcast peripherals, PT bucket, PA<5 filter |
| `tests/test_streaks/test_thresholds.py` | Create | Percentile calibration over fixture |
| `tests/test_streaks/test_labels.py` | Create | Hot/cold/neutral assignment from fixture thresholds |
| `notebooks/streaks/01_distributions.ipynb` | Create (gitignored) | Distribution plots, threshold-table eyeball check |

---

## Task 1: Extend `HitterGame` and `HitterStatcastPA` dataclasses

`load.py` derives column tuples via `dataclasses.fields()`, so adding fields to the dataclass auto-updates the SQL upserts. Field order (declaration order) must match the DDL column order — append new fields at the end.

**Files:**
- Modify: `src/fantasy_baseball/streaks/models.py:34-78`
- Create: `tests/test_streaks/test_models.py`

- [ ] **Step 1.1: Write failing dataclass-shape test**

Create `tests/test_streaks/test_models.py`:

```python
"""Cheap regression guard: dataclass field surface stays in lockstep with the DDL.

If this fails after a column add, update both `models.py` and `schema.py`
together — they are co-load-bearing for `load.py`'s attrgetter-based upsert.
"""

from __future__ import annotations

from dataclasses import fields

from fantasy_baseball.streaks.models import HitterGame, HitterStatcastPA


def test_hitter_game_fields_in_expected_order() -> None:
    expected = (
        "player_id", "game_pk", "name", "team", "season", "date",
        "pa", "ab", "h", "hr", "r", "rbi", "sb", "bb", "k",
        "b2", "b3", "sf", "hbp", "ibb", "cs", "gidp", "sh", "ci", "is_home",
    )
    assert tuple(f.name for f in fields(HitterGame)) == expected


def test_hitter_statcast_pa_fields_in_expected_order() -> None:
    expected = (
        "player_id", "date", "pa_index", "event",
        "launch_speed", "launch_angle", "estimated_woba_using_speedangle", "barrel",
        "at_bat_number", "bb_type", "estimated_ba_using_speedangle", "hit_distance_sc",
    )
    assert tuple(f.name for f in fields(HitterStatcastPA)) == expected
```

- [ ] **Step 1.2: Run the test, confirm it fails**

```
pytest tests/test_streaks/test_models.py -v
```

Expected: both tests FAIL — current dataclasses are missing the new fields.

- [ ] **Step 1.3: Add the new fields to `HitterGame`**

In `src/fantasy_baseball/streaks/models.py`, replace the `HitterGame` class (lines 34-57) with:

```python
@dataclass(frozen=True, slots=True)
class HitterGame:
    """One game of hitter counting stats. Maps to `hitter_games` row.

    PK is (player_id, game_pk). ``game_pk`` is the MLB Stats API gamePk
    integer — unique per game, so it disambiguates doubleheaders that
    share a date. ``date`` stays as a non-PK column for query convenience.

    Captures every box-score component the streaks project needs for rate
    stats (BABIP/ISO from b2/b3/sf), refined walk rate (uBB% from ibb),
    PA-identity reconciliation (pa = ab + bb + hbp + sf + sh + ci), and
    plausible Phase 3+ context (cs for SB attempts, gidp for luck signal,
    is_home for park splits).
    """

    player_id: int
    game_pk: int
    name: str
    team: str | None
    season: int
    date: date
    pa: int
    ab: int
    h: int
    hr: int
    r: int
    rbi: int
    sb: int
    bb: int
    k: int
    b2: int
    b3: int
    sf: int
    hbp: int
    ibb: int
    cs: int
    gidp: int
    sh: int
    ci: int
    is_home: bool
```

- [ ] **Step 1.4: Add the new fields to `HitterStatcastPA`**

In the same file, replace the `HitterStatcastPA` class (lines 60-77) with:

```python
@dataclass(frozen=True, slots=True)
class HitterStatcastPA:
    """One terminal-PA row from Baseball Savant. Maps to `hitter_statcast_pa` row.

    PK is (player_id, date, pa_index). ``pa_index`` is now derived after
    sorting by ``at_bat_number`` within (batter, game_date), so it is
    chronologically stable across re-fetches.
    """

    player_id: int
    date: date
    pa_index: int
    event: str | None
    launch_speed: float | None
    launch_angle: float | None
    estimated_woba_using_speedangle: float | None
    barrel: bool | None
    at_bat_number: int | None
    bb_type: str | None
    estimated_ba_using_speedangle: float | None
    hit_distance_sc: float | None
```

- [ ] **Step 1.5: Run the test, confirm it passes**

```
pytest tests/test_streaks/test_models.py -v
```

Expected: both tests PASS.

- [ ] **Step 1.6: Commit**

```bash
git add src/fantasy_baseball/streaks/models.py tests/test_streaks/test_models.py
git commit -m "feat(streaks): extend HitterGame and HitterStatcastPA for Phase 2 columns"
```

---

## Task 2: Update DDL for the new columns

The DDL must match the dataclass column order exactly — `load.py` upsert relies on positional binding via attrgetter.

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/schema.py:15-48`
- Modify: `tests/test_streaks/test_schema.py`

- [ ] **Step 2.1: Write failing schema-shape tests**

Append to `tests/test_streaks/test_schema.py`:

```python
def test_hitter_games_has_new_phase_2_columns() -> None:
    conn = get_connection(":memory:")
    rows = conn.execute("PRAGMA table_info('hitter_games')").fetchall()
    cols = {r[1]: r[2] for r in rows}  # name -> type
    assert cols["b2"] == "INTEGER"
    assert cols["b3"] == "INTEGER"
    assert cols["sf"] == "INTEGER"
    assert cols["hbp"] == "INTEGER"
    assert cols["ibb"] == "INTEGER"
    assert cols["cs"] == "INTEGER"
    assert cols["gidp"] == "INTEGER"
    assert cols["sh"] == "INTEGER"
    assert cols["ci"] == "INTEGER"
    assert cols["is_home"] == "BOOLEAN"


def test_hitter_statcast_pa_has_new_phase_2_columns() -> None:
    conn = get_connection(":memory:")
    rows = conn.execute("PRAGMA table_info('hitter_statcast_pa')").fetchall()
    cols = {r[1]: r[2] for r in rows}
    assert cols["at_bat_number"] == "INTEGER"
    assert cols["bb_type"] == "VARCHAR"
    assert cols["estimated_ba_using_speedangle"] == "DOUBLE"
    assert cols["hit_distance_sc"] == "DOUBLE"
```

(The existing test file already imports `get_connection`. If not, add `from fantasy_baseball.streaks.data.schema import get_connection` at the top.)

- [ ] **Step 2.2: Run the new tests, confirm they fail**

```
pytest tests/test_streaks/test_schema.py::test_hitter_games_has_new_phase_2_columns tests/test_streaks/test_schema.py::test_hitter_statcast_pa_has_new_phase_2_columns -v
```

Expected: both FAIL — columns don't exist yet.

- [ ] **Step 2.3: Update the `hitter_games` DDL**

In `src/fantasy_baseball/streaks/data/schema.py`, replace the `hitter_games` DDL block (lines 16-35) with:

```python
    """
    CREATE TABLE IF NOT EXISTS hitter_games (
        player_id INTEGER NOT NULL,
        game_pk INTEGER NOT NULL,
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
        b2 INTEGER NOT NULL,
        b3 INTEGER NOT NULL,
        sf INTEGER NOT NULL,
        hbp INTEGER NOT NULL,
        ibb INTEGER NOT NULL,
        cs INTEGER NOT NULL,
        gidp INTEGER NOT NULL,
        sh INTEGER NOT NULL,
        ci INTEGER NOT NULL,
        is_home BOOLEAN NOT NULL,
        PRIMARY KEY (player_id, game_pk)
    )
    """,
```

- [ ] **Step 2.4: Update the `hitter_statcast_pa` DDL**

In the same file, replace the `hitter_statcast_pa` DDL block (lines 36-48) with:

```python
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
        at_bat_number INTEGER,
        bb_type VARCHAR,
        estimated_ba_using_speedangle DOUBLE,
        hit_distance_sc DOUBLE,
        PRIMARY KEY (player_id, date, pa_index)
    )
    """,
```

- [ ] **Step 2.5: Run all schema tests, confirm pass**

```
pytest tests/test_streaks/test_schema.py -v
```

Expected: all pass.

- [ ] **Step 2.6: Commit**

```bash
git add src/fantasy_baseball/streaks/data/schema.py tests/test_streaks/test_schema.py
git commit -m "feat(streaks): add Phase 2 columns to hitter_games + hitter_statcast_pa DDL"
```

---

## Task 3: Update `game_logs` parser + PA identity check

The MLB Stats API gameLog stat object exposes `doubles`, `triples`, `sacFlies`, `hitByPitch`, `intentionalWalks`, `caughtStealing`, `groundIntoDoublePlay`, `sacBunts`, and `catchersInterference`. The split itself exposes `isHome`. We capture all of them and add a helper that returns the PA-identity gap so the orchestrator can log + count violations.

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/game_logs.py`
- Modify: `tests/test_streaks/test_game_logs.py`

- [ ] **Step 3.1: Write failing tests for new-field parsing and identity helper**

Append to `tests/test_streaks/test_game_logs.py`:

```python
from fantasy_baseball.streaks.data.game_logs import (
    pa_identity_gap,
    parse_hitter_game_log_full,
)
from fantasy_baseball.streaks.models import HitterGame


def _make_split(stat: dict, *, is_home: bool = True, game_pk: int = 1, date: str = "2025-04-01") -> dict:
    return {
        "game": {"gamePk": game_pk},
        "date": date,
        "isHome": is_home,
        "stat": stat,
    }


def test_parse_captures_new_fields() -> None:
    split = _make_split(
        {
            "plateAppearances": 5, "atBats": 4, "hits": 2, "homeRuns": 1,
            "runs": 1, "rbi": 2, "stolenBases": 0, "baseOnBalls": 1, "strikeOuts": 1,
            "doubles": 1, "triples": 0, "sacFlies": 0, "hitByPitch": 0,
            "intentionalWalks": 0, "caughtStealing": 0, "groundIntoDoublePlay": 0,
            "sacBunts": 0, "catchersInterference": 0,
        },
        is_home=False,
    )
    g = parse_hitter_game_log_full(split, player_id=1, name="X", team="ABC", season=2025)
    assert g.b2 == 1
    assert g.b3 == 0
    assert g.sf == 0
    assert g.hbp == 0
    assert g.ibb == 0
    assert g.cs == 0
    assert g.gidp == 0
    assert g.sh == 0
    assert g.ci == 0
    assert g.is_home is False


def test_parse_treats_missing_fields_as_zero() -> None:
    # Older API responses or partial splits may omit columns. Default to 0
    # so the row still loads and the identity check catches genuine drift.
    split = _make_split({"plateAppearances": 1, "atBats": 1, "hits": 0, "homeRuns": 0,
                         "runs": 0, "rbi": 0, "stolenBases": 0, "baseOnBalls": 0,
                         "strikeOuts": 1})
    g = parse_hitter_game_log_full(split, player_id=1, name="X", team=None, season=2025)
    assert g.b2 == 0 and g.b3 == 0 and g.sf == 0 and g.hbp == 0
    assert g.ibb == 0 and g.cs == 0 and g.gidp == 0 and g.sh == 0 and g.ci == 0
    assert g.is_home is True  # default


def test_pa_identity_gap_zero_for_clean_row() -> None:
    g = HitterGame(
        player_id=1, game_pk=1, name="X", team=None, season=2025, date=date(2025, 4, 1),
        pa=5, ab=3, h=1, hr=0, r=0, rbi=0, sb=0, bb=1, k=1,
        b2=0, b3=0, sf=1, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0, is_home=True,
    )
    # 5 == 3 + 1 + 0 + 1 + 0 + 0
    assert pa_identity_gap(g) == 0


def test_pa_identity_gap_detects_drift() -> None:
    g = HitterGame(
        player_id=1, game_pk=1, name="X", team=None, season=2025, date=date(2025, 4, 1),
        pa=5, ab=3, h=1, hr=0, r=0, rbi=0, sb=0, bb=1, k=1,
        b2=0, b3=0, sf=0, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0, is_home=True,
    )
    # PA=5, components sum to 4 -> gap of +1
    assert pa_identity_gap(g) == 1
```

(The test file already imports `from datetime import date`; if not, add it.)

- [ ] **Step 3.2: Run the new tests, confirm they fail**

```
pytest tests/test_streaks/test_game_logs.py -v -k "new_fields or missing_fields or pa_identity"
```

Expected: FAIL — `pa_identity_gap` doesn't exist; parser doesn't read the new fields.

- [ ] **Step 3.3: Update `parse_hitter_game_log_full` and add `pa_identity_gap`**

Replace the entire body of `src/fantasy_baseball/streaks/data/game_logs.py` with:

```python
"""Hitter game log fetch for the streaks project.

This is a streaks-specific parser that captures every column the
`hitter_games` table needs (player_id, name, team, season, plus bb/k that
the existing analysis/game_logs.py omits, plus the Phase 2 box-score
expansion: b2/b3/sf/hbp/ibb/cs/gidp/sh/ci, plus split-level is_home).
The HTTP shape is identical; only the parsing differs.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import requests

from fantasy_baseball.streaks.models import HitterGame

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def parse_hitter_game_log_full(
    split: dict[str, Any],
    *,
    player_id: int,
    name: str,
    team: str | None,
    season: int,
) -> HitterGame:
    """Parse one /people/{id}/stats?stats=gameLog split into a :class:`HitterGame`.

    Uses the split's ``game.gamePk`` (a unique MLB game identifier) for
    the row PK alongside ``player_id`` so doubleheader games on the same
    date don't collide.
    """
    stat = split.get("stat", {})
    return HitterGame(
        player_id=player_id,
        game_pk=int(split["game"]["gamePk"]),
        name=name,
        team=team,
        season=season,
        date=date.fromisoformat(split["date"]),
        pa=int(stat.get("plateAppearances", 0)),
        ab=int(stat.get("atBats", 0)),
        h=int(stat.get("hits", 0)),
        hr=int(stat.get("homeRuns", 0)),
        r=int(stat.get("runs", 0)),
        rbi=int(stat.get("rbi", 0)),
        sb=int(stat.get("stolenBases", 0)),
        bb=int(stat.get("baseOnBalls", 0)),
        k=int(stat.get("strikeOuts", 0)),
        b2=int(stat.get("doubles", 0)),
        b3=int(stat.get("triples", 0)),
        sf=int(stat.get("sacFlies", 0)),
        hbp=int(stat.get("hitByPitch", 0)),
        ibb=int(stat.get("intentionalWalks", 0)),
        cs=int(stat.get("caughtStealing", 0)),
        gidp=int(stat.get("groundIntoDoublePlay", 0)),
        sh=int(stat.get("sacBunts", 0)),
        ci=int(stat.get("catchersInterference", 0)),
        is_home=bool(split.get("isHome", True)),
    )


def pa_identity_gap(g: HitterGame) -> int:
    """Return ``g.pa - (ab + bb + hbp + sf + sh + ci)``.

    Zero means the box-score components sum to PA. A non-zero gap signals
    either an upstream API drift (a new component we don't capture) or a
    parser bug — the orchestrator logs a warning and counts violations
    rather than raising, so a single bad row doesn't kill a multi-hour
    season fetch.
    """
    return g.pa - (g.ab + g.bb + g.hbp + g.sf + g.sh + g.ci)


def fetch_hitter_season_game_logs(
    player_id: int, name: str, team: str | None, season: int, timeout: float = 15.0
) -> list[HitterGame]:
    """Fetch one season of game logs for one hitter.

    Returns one :class:`HitterGame` per game played. Empty list if the player
    has no logs.
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
        parse_hitter_game_log_full(s, player_id=player_id, name=name, team=team, season=season)
        for s in splits
    ]
```

- [ ] **Step 3.4: Run the new tests, confirm they pass**

```
pytest tests/test_streaks/test_game_logs.py -v
```

Expected: all pass (including any pre-existing tests).

- [ ] **Step 3.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/game_logs.py tests/test_streaks/test_game_logs.py
git commit -m "feat(streaks): parse Phase 2 game-log fields + PA-identity helper"
```

---

## Task 4: Update Statcast parser (new fields + chronological `pa_index`)

Sort by `[batter, game_date, at_bat_number]` so `pa_index` becomes deterministic across re-fetches. Capture the four new Statcast columns.

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/statcast.py:52-81`
- Modify: `tests/test_streaks/test_statcast.py`

- [ ] **Step 4.1: Write failing tests for new fields and chronological ordering**

Append to `tests/test_streaks/test_statcast.py`:

```python
def test_pitches_to_pa_rows_captures_new_statcast_fields() -> None:
    df = pd.DataFrame(
        [
            {
                "batter": 1, "game_date": "2025-04-01", "events": "single",
                "launch_speed": 100.0, "launch_angle": 12.0,
                "estimated_woba_using_speedangle": 0.85, "barrel": False,
                "at_bat_number": 1, "bb_type": "line_drive",
                "estimated_ba_using_speedangle": 0.71, "hit_distance_sc": 220.0,
            }
        ]
    )
    rows = pitches_to_pa_rows(df)
    assert len(rows) == 1
    r = rows[0]
    assert r.at_bat_number == 1
    assert r.bb_type == "line_drive"
    assert r.estimated_ba_using_speedangle == 0.71
    assert r.hit_distance_sc == 220.0


def test_pitches_to_pa_rows_assigns_pa_index_in_at_bat_number_order() -> None:
    # Same player, same date, three PAs; pass them in shuffled order with
    # at_bat_numbers 1, 2, 3 — pa_index should track at_bat_number, not
    # input row order.
    df = pd.DataFrame(
        [
            {"batter": 1, "game_date": "2025-04-01", "events": "double",
             "launch_speed": None, "launch_angle": None,
             "estimated_woba_using_speedangle": None, "barrel": None,
             "at_bat_number": 3, "bb_type": None,
             "estimated_ba_using_speedangle": None, "hit_distance_sc": None},
            {"batter": 1, "game_date": "2025-04-01", "events": "strikeout",
             "launch_speed": None, "launch_angle": None,
             "estimated_woba_using_speedangle": None, "barrel": None,
             "at_bat_number": 1, "bb_type": None,
             "estimated_ba_using_speedangle": None, "hit_distance_sc": None},
            {"batter": 1, "game_date": "2025-04-01", "events": "walk",
             "launch_speed": None, "launch_angle": None,
             "estimated_woba_using_speedangle": None, "barrel": None,
             "at_bat_number": 2, "bb_type": None,
             "estimated_ba_using_speedangle": None, "hit_distance_sc": None},
        ]
    )
    rows = pitches_to_pa_rows(df)
    by_pa_index = {r.pa_index: r for r in rows}
    assert by_pa_index[1].event == "strikeout"  # at_bat_number=1
    assert by_pa_index[2].event == "walk"        # at_bat_number=2
    assert by_pa_index[3].event == "double"      # at_bat_number=3


def test_pitches_to_pa_rows_handles_missing_at_bat_number_column() -> None:
    # Older Savant exports may not include at_bat_number. Fall back to
    # row order within (batter, game_date) — matches the Phase 1 behavior.
    df = pd.DataFrame(
        [
            {"batter": 1, "game_date": "2025-04-01", "events": "single",
             "launch_speed": None, "launch_angle": None,
             "estimated_woba_using_speedangle": None, "barrel": None,
             "bb_type": None, "estimated_ba_using_speedangle": None,
             "hit_distance_sc": None},
        ]
    )
    rows = pitches_to_pa_rows(df)
    assert len(rows) == 1
    assert rows[0].pa_index == 1
    assert rows[0].at_bat_number is None
```

- [ ] **Step 4.2: Run the new tests, confirm they fail**

```
pytest tests/test_streaks/test_statcast.py -v -k "new_statcast or at_bat_number_order or missing_at_bat_number"
```

Expected: FAIL — fields not captured; sort key not used.

- [ ] **Step 4.3: Update `pitches_to_pa_rows`**

Replace the body of `pitches_to_pa_rows` (lines 52-81) and the import block in `src/fantasy_baseball/streaks/data/statcast.py` so the function reads:

```python
def pitches_to_pa_rows(df: pd.DataFrame) -> list[HitterStatcastPA]:
    """Convert a Statcast pitch DataFrame to a list of :class:`HitterStatcastPA`.

    Filters to terminal PAs, sorts by ``[batter, game_date, at_bat_number]``
    (so ``pa_index`` is chronologically stable across re-fetches), assigns
    ``pa_index`` per (batter, game_date), and converts NaN/NaT/pd.NA values
    to None.
    """
    df = filter_terminal_pa(df)
    if df.empty:
        return []
    sort_cols = ["batter", "game_date"]
    if "at_bat_number" in df.columns:
        sort_cols.append("at_bat_number")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["pa_index"] = df.groupby(["batter", "game_date"]).cumcount() + 1

    rows: list[HitterStatcastPA] = []
    has_barrel = "barrel" in df.columns
    has_at_bat_number = "at_bat_number" in df.columns
    has_bb_type = "bb_type" in df.columns
    has_xba = "estimated_ba_using_speedangle" in df.columns
    has_distance = "hit_distance_sc" in df.columns
    for r in df.itertuples(index=False):
        rows.append(
            HitterStatcastPA(
                player_id=int(r.batter),
                date=pd.to_datetime(r.game_date).date(),
                pa_index=int(r.pa_index),
                event=_na_to_none(r.events),
                launch_speed=_na_to_none(getattr(r, "launch_speed", None)),
                launch_angle=_na_to_none(getattr(r, "launch_angle", None)),
                estimated_woba_using_speedangle=_na_to_none(
                    getattr(r, "estimated_woba_using_speedangle", None)
                ),
                barrel=(bool(r.barrel) if has_barrel and not pd.isna(r.barrel) else None),
                at_bat_number=(
                    _na_to_none(getattr(r, "at_bat_number", None)) if has_at_bat_number else None
                ),
                bb_type=(_na_to_none(getattr(r, "bb_type", None)) if has_bb_type else None),
                estimated_ba_using_speedangle=(
                    _na_to_none(getattr(r, "estimated_ba_using_speedangle", None))
                    if has_xba
                    else None
                ),
                hit_distance_sc=(
                    _na_to_none(getattr(r, "hit_distance_sc", None)) if has_distance else None
                ),
            )
        )
    return rows
```

- [ ] **Step 4.4: Run all Statcast tests, confirm pass**

```
pytest tests/test_streaks/test_statcast.py -v
```

Expected: all pass.

- [ ] **Step 4.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/statcast.py tests/test_streaks/test_statcast.py
git commit -m "feat(streaks): capture Phase 2 Statcast columns + chronological pa_index"
```

---

## Task 5: Surface PA-identity violations in the fetch summary

`fetch_history.fetch_season` already returns a summary dict with `players_attempted`, `game_log_rows`, `statcast_rows`. Add a `pa_identity_violations` count so the user sees if anything drifted during re-fetch.

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/fetch_history.py`
- Modify: `tests/test_streaks/test_fetch_history.py`

- [ ] **Step 5.1: Write failing test for violation count**

Append to `tests/test_streaks/test_fetch_history.py`:

```python
def test_fetch_season_counts_pa_identity_violations(monkeypatch, tmp_path) -> None:
    """A game with PA != AB+BB+HBP+SF+SH+CI is logged + counted, not raised."""
    from datetime import date

    from fantasy_baseball.streaks.data import fetch_history as fh
    from fantasy_baseball.streaks.data.schema import get_connection
    from fantasy_baseball.streaks.models import HitterGame, QualifiedHitter

    # One player; one good game, one game with a PA gap of +1.
    good = HitterGame(
        player_id=1, game_pk=1, name="X", team=None, season=2025, date=date(2025, 4, 1),
        pa=4, ab=3, h=1, hr=0, r=0, rbi=0, sb=0, bb=1, k=1,
        b2=0, b3=0, sf=0, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0, is_home=True,
    )
    bad = HitterGame(
        player_id=1, game_pk=2, name="X", team=None, season=2025, date=date(2025, 4, 2),
        pa=5, ab=3, h=1, hr=0, r=0, rbi=0, sb=0, bb=1, k=1,
        b2=0, b3=0, sf=0, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0, is_home=True,
    )

    monkeypatch.setattr(
        fh, "fetch_qualified_hitters",
        lambda season, min_pa: [QualifiedHitter(player_id=1, name="X", team=None, pa=200)],
    )
    monkeypatch.setattr(
        fh, "fetch_hitter_season_game_logs",
        lambda player_id, name, team, season: [good, bad],
    )
    monkeypatch.setattr(fh, "fetch_statcast_pa_for_date_range", lambda start, end: [])

    conn = get_connection(tmp_path / "t.duckdb")
    summary = fh.fetch_season(season=2025, conn=conn)

    assert summary["pa_identity_violations"] == 1
    assert summary["game_log_rows"] == 2  # both rows still loaded
```

- [ ] **Step 5.2: Run the new test, confirm it fails**

```
pytest tests/test_streaks/test_fetch_history.py::test_fetch_season_counts_pa_identity_violations -v
```

Expected: FAIL — `pa_identity_violations` not in the summary dict.

- [ ] **Step 5.3: Update `fetch_season`**

In `src/fantasy_baseball/streaks/data/fetch_history.py`, add the import and update the per-player loop + summary. Replace the whole module body with:

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
import requests

from fantasy_baseball.streaks.data.game_logs import (
    fetch_hitter_season_game_logs,
    pa_identity_gap,
)
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


def fetch_season(season: int, conn: duckdb.DuckDBPyConnection, min_pa: int = 150) -> dict[str, Any]:
    """Fetch and load one season of game logs + Statcast PA data.

    Returns a summary dict with row counts and a ``pa_identity_violations``
    count (games where ``pa != ab + bb + hbp + sf + sh + ci`` — logged at
    WARNING but never raised, so a single bad row can't kill a 2-hour fetch).
    """
    qualified = fetch_qualified_hitters(season=season, min_pa=min_pa)
    logger.info("Season %s: %d qualified hitters", season, len(qualified))

    already = existing_player_seasons(conn)
    to_fetch = [q for q in qualified if (q.player_id, season) not in already]
    logger.info("Season %s: %d new players to fetch", season, len(to_fetch))

    game_log_rows = 0
    pa_identity_violations = 0
    for i, player in enumerate(to_fetch):
        try:
            games = fetch_hitter_season_game_logs(
                player_id=player.player_id,
                name=player.name,
                team=player.team,
                season=season,
            )
            for g in games:
                gap = pa_identity_gap(g)
                if gap != 0:
                    pa_identity_violations += 1
                    logger.warning(
                        "PA identity gap of %d for %s (%s) on %s game_pk=%d",
                        gap, g.name, g.player_id, g.date.isoformat(), g.game_pk,
                    )
            upsert_hitter_games(conn, games)
            game_log_rows += len(games)
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning(
                "Game log fetch failed for %s (%s): %s",
                player.name,
                player.player_id,
                e,
            )
        if (i + 1) % 25 == 0:
            logger.info("  fetched %d/%d game logs", i + 1, len(to_fetch))

    start = date(season, *_SEASON_START_MMDD)
    end = date(season, *_SEASON_END_MMDD)
    loaded_dates = existing_statcast_dates(conn)
    statcast_rows = 0
    season_dates_loaded = {d for d in loaded_dates if d.year == season}
    if not season_dates_loaded:
        statcast_pa = fetch_statcast_pa_for_date_range(start, end)
        upsert_statcast_pa(conn, statcast_pa)
        statcast_rows = len(statcast_pa)
    else:
        logger.info(
            "Season %s: %d Statcast dates already loaded, skipping Statcast pull",
            season,
            len(season_dates_loaded),
        )

    return {
        "season": season,
        "players_attempted": len(to_fetch),
        "game_log_rows": game_log_rows,
        "pa_identity_violations": pa_identity_violations,
        "statcast_rows": statcast_rows,
    }
```

- [ ] **Step 5.4: Run all fetch_history tests, confirm pass**

```
pytest tests/test_streaks/test_fetch_history.py -v
```

Expected: all pass (including any pre-existing tests).

- [ ] **Step 5.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/fetch_history.py tests/test_streaks/test_fetch_history.py
git commit -m "feat(streaks): count PA-identity violations during season fetch"
```

---

## Task 6: One-shot migration for existing local DBs

The Phase 1 DB has the old schema. Migration: ALTER TABLE add columns (idempotent — skip if already present), DELETE rows from `hitter_games`, `hitter_statcast_pa`, and the (still-empty) downstream tables. After this, re-running `fetch_history` repopulates with the new columns.

**Files:**
- Create: `src/fantasy_baseball/streaks/data/migrate.py`
- Create: `scripts/streaks/migrate.py`
- Create: `tests/test_streaks/test_migrate.py`

- [ ] **Step 6.1: Write failing migration tests**

Create `tests/test_streaks/test_migrate.py`:

```python
"""Tests for the Phase 2 schema migration on an existing Phase 1 DB."""

from __future__ import annotations

import duckdb
import pytest

from fantasy_baseball.streaks.data.migrate import migrate_to_phase_2


def _phase_1_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Recreate the Phase 1 (pre-migration) DDL for testing."""
    conn.execute(
        """
        CREATE TABLE hitter_games (
            player_id INTEGER NOT NULL,
            game_pk INTEGER NOT NULL,
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
            PRIMARY KEY (player_id, game_pk)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE hitter_statcast_pa (
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
        """
    )
    conn.execute(
        "INSERT INTO hitter_games VALUES (1, 1, 'X', 'ABC', 2025, '2025-04-01', "
        "4, 3, 1, 0, 0, 0, 0, 1, 1)"
    )


def test_migrate_adds_new_columns_to_hitter_games() -> None:
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info('hitter_games')").fetchall()}
    for col in ("b2", "b3", "sf", "hbp", "ibb", "cs", "gidp", "sh", "ci", "is_home"):
        assert col in cols, f"missing column {col}"


def test_migrate_adds_new_columns_to_hitter_statcast_pa() -> None:
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info('hitter_statcast_pa')").fetchall()}
    for col in ("at_bat_number", "bb_type", "estimated_ba_using_speedangle", "hit_distance_sc"):
        assert col in cols, f"missing column {col}"


def test_migrate_deletes_existing_rows_to_force_refetch() -> None:
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    assert conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM hitter_statcast_pa").fetchone()[0] == 0


def test_migrate_is_idempotent() -> None:
    conn = duckdb.connect(":memory:")
    _phase_1_schema(conn)
    migrate_to_phase_2(conn)
    # Second call should not raise (column already exists).
    migrate_to_phase_2(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info('hitter_games')").fetchall()}
    assert "b2" in cols  # still present, didn't blow up
```

- [ ] **Step 6.2: Run the tests, confirm they fail**

```
pytest tests/test_streaks/test_migrate.py -v
```

Expected: ImportError — `migrate.py` doesn't exist.

- [ ] **Step 6.3: Create the migration module**

Create `src/fantasy_baseball/streaks/data/migrate.py`:

```python
"""One-shot Phase 2 schema migration.

Adds the box-score expansion columns to ``hitter_games`` and the Statcast
peripheral columns to ``hitter_statcast_pa`` on a pre-existing Phase 1 DB.
DELETEs all rows from both tables so re-running :mod:`fetch_history`
repopulates with the new columns; downstream tables (``hitter_windows``,
``thresholds``, ``hitter_streak_labels``) are also cleared as a safety
measure (they should be empty in Phase 1 but the DELETE is cheap).

Idempotent: ALTER TABLE ADD COLUMN failures (column already exists) are
caught per column. Safe to re-run.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

_GAMES_NEW_COLUMNS: tuple[tuple[str, str], ...] = (
    ("b2", "INTEGER"),
    ("b3", "INTEGER"),
    ("sf", "INTEGER"),
    ("hbp", "INTEGER"),
    ("ibb", "INTEGER"),
    ("cs", "INTEGER"),
    ("gidp", "INTEGER"),
    ("sh", "INTEGER"),
    ("ci", "INTEGER"),
    ("is_home", "BOOLEAN"),
)
_STATCAST_NEW_COLUMNS: tuple[tuple[str, str], ...] = (
    ("at_bat_number", "INTEGER"),
    ("bb_type", "VARCHAR"),
    ("estimated_ba_using_speedangle", "DOUBLE"),
    ("hit_distance_sc", "DOUBLE"),
)
_TABLES_TO_CLEAR: tuple[str, ...] = (
    "hitter_games",
    "hitter_statcast_pa",
    "hitter_windows",
    "thresholds",
    "hitter_streak_labels",
)


def _column_names(conn: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {r[1] for r in rows}


def _add_column_if_missing(
    conn: duckdb.DuckDBPyConnection, table: str, column: str, sql_type: str
) -> None:
    if column in _column_names(conn, table):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
    logger.info("Added %s.%s (%s)", table, column, sql_type)


def migrate_to_phase_2(conn: duckdb.DuckDBPyConnection) -> None:
    """Add Phase 2 columns and clear stale rows from a Phase 1 DB.

    No-op for already-migrated DBs.
    """
    for col, sql_type in _GAMES_NEW_COLUMNS:
        _add_column_if_missing(conn, "hitter_games", col, sql_type)
    for col, sql_type in _STATCAST_NEW_COLUMNS:
        _add_column_if_missing(conn, "hitter_statcast_pa", col, sql_type)
    for table in _TABLES_TO_CLEAR:
        # Some downstream tables may not exist on a fresh Phase 1 DB; tolerate that.
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except duckdb.CatalogException:
            continue
        if n:
            logger.info("Clearing %d rows from %s", n, table)
            conn.execute(f"DELETE FROM {table}")
```

- [ ] **Step 6.4: Run the tests, confirm they pass**

```
pytest tests/test_streaks/test_migrate.py -v
```

Expected: all pass.

- [ ] **Step 6.5: Create the CLI entry point**

Create `scripts/streaks/migrate.py`:

```python
"""CLI: run the Phase 2 schema migration against the local streaks DuckDB.

Usage:
    python -m scripts.streaks.migrate [--db PATH]

After this, re-run::

    python -m scripts.streaks.fetch_history --season 2023
    python -m scripts.streaks.fetch_history --season 2024
    python -m scripts.streaks.fetch_history --season 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/streaks/migrate.py` without -m.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fantasy_baseball.streaks.data.migrate import migrate_to_phase_2  # noqa: E402
from fantasy_baseball.streaks.data.schema import (  # noqa: E402
    DEFAULT_DB_PATH,
    get_connection,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate streaks DB to Phase 2 schema.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to streaks.duckdb")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db)
    migrate_to_phase_2(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6.6: Verify the CLI parses without crashing**

```
python -c "from scripts.streaks.migrate import main; main(['--help'])"
```

Expected: prints help, exits.

- [ ] **Step 6.7: Commit**

```bash
git add src/fantasy_baseball/streaks/data/migrate.py scripts/streaks/migrate.py tests/test_streaks/test_migrate.py
git commit -m "feat(streaks): add Phase 2 schema migration script"
```

---

## Task 7: Run the migration + re-fetch the corpus

**Runtime task — no code changes.** This is the one-shot reprocessing the user signed off on. Expect ~2 hours wall clock for Statcast pulls; game logs are faster (~10 min for ~1200 player-season HTTP calls).

- [ ] **Step 7.1: Run the migration**

```
python scripts/streaks/migrate.py
```

Expected log output:
```
INFO fantasy_baseball.streaks.data.migrate: Added hitter_games.b2 (INTEGER)
INFO fantasy_baseball.streaks.data.migrate: Added hitter_games.b3 (INTEGER)
... (10 lines for hitter_games)
INFO fantasy_baseball.streaks.data.migrate: Added hitter_statcast_pa.at_bat_number (INTEGER)
... (4 lines for hitter_statcast_pa)
INFO fantasy_baseball.streaks.data.migrate: Clearing 134441 rows from hitter_games
INFO fantasy_baseball.streaks.data.migrate: Clearing 598363 rows from hitter_statcast_pa
```

- [ ] **Step 7.2: Verify the DB now has the new schema and is empty**

```
python -c "
import duckdb
conn = duckdb.connect('data/streaks/streaks.duckdb')
print('games rows:', conn.execute('SELECT COUNT(*) FROM hitter_games').fetchone()[0])
print('statcast rows:', conn.execute('SELECT COUNT(*) FROM hitter_statcast_pa').fetchone()[0])
print('games cols:', [r[1] for r in conn.execute(\"PRAGMA table_info('hitter_games')\").fetchall()])
"
```

Expected: both counts 0; games cols include `is_home` etc.

- [ ] **Step 7.3: Re-fetch 2023**

```
python scripts/streaks/fetch_history.py --season 2023
```

Expected summary at end:
```
{'season': 2023, 'players_attempted': 404, 'game_log_rows': 44707,
 'pa_identity_violations': <small number — single digits to low double digits is normal>,
 'statcast_rows': 202177}
```

If `pa_identity_violations` exceeds ~50, **stop** and investigate — likely either a parser bug or an API field-name drift. Do not proceed to 2024/2025 until the cause is understood.

- [ ] **Step 7.4: Re-fetch 2024**

```
python scripts/streaks/fetch_history.py --season 2024
```

Expected summary: ~410 players, ~45,472 game logs, ~197,983 Statcast rows.

- [ ] **Step 7.5: Re-fetch 2025**

```
python scripts/streaks/fetch_history.py --season 2025
```

Expected summary: ~393 players, ~44,262 game logs, ~198,203 Statcast rows.

- [ ] **Step 7.6: Verify final counts match Phase 1 acceptance**

```
python -c "
import duckdb
conn = duckdb.connect('data/streaks/streaks.duckdb')
for season in (2023, 2024, 2025):
    g = conn.execute('SELECT COUNT(*) FROM hitter_games WHERE season=?', (season,)).fetchone()[0]
    print(f'{season}: {g} game logs')
g = conn.execute('SELECT COUNT(*) FROM hitter_games').fetchone()[0]
s = conn.execute('SELECT COUNT(*) FROM hitter_statcast_pa').fetchone()[0]
print(f'Total: {g} games, {s} statcast rows')
"
```

Expected: 2023=44,707; 2024=45,472; 2025=44,262; total games 134,441; total statcast ~598K.

- [ ] **Step 7.7: Spot-check one new column has real data**

```
python -c "
import duckdb
conn = duckdb.connect('data/streaks/streaks.duckdb')
print(conn.execute('SELECT SUM(b2), SUM(b3), SUM(sf), SUM(hbp), SUM(sh), SUM(ci) FROM hitter_games').fetchone())
print(conn.execute(\"SELECT COUNT(*) FROM hitter_statcast_pa WHERE at_bat_number IS NOT NULL\").fetchone())
"
```

Expected: doubles in the ~7K-8K range per season (~22K-24K total), triples ~700-1000/yr, SF ~900/yr, HBP ~2K/yr, SH ~400-700/yr, CI < 100/yr. Statcast `at_bat_number` non-null count should be ≥98% of total Statcast rows.

- [ ] **Step 7.8: Update the spec's Progress Log**

Append to `docs/superpowers/specs/2026-05-06-hot-streaks-design.md`:

```markdown
### 2026-05-08 — Phase 2 schema migration + re-fetch

Re-pulled all three seasons against the expanded `hitter_games` schema
(added b2/b3/sf/hbp/ibb/cs/gidp/sh/ci/is_home) and the expanded
`hitter_statcast_pa` schema (added at_bat_number/bb_type/
estimated_ba_using_speedangle/hit_distance_sc). PA-identity
(pa = ab + bb + hbp + sf + sh + ci) violation counts were <observed
counts go here>; spot-checks of doubles/triples/SF/HBP totals matched
expected ranges.

Final row counts unchanged from Phase 1 acceptance: 134,441 game logs,
~598K Statcast PAs, 1,207 player-seasons.
```

- [ ] **Step 7.9: Commit the spec update**

```bash
git add docs/superpowers/specs/2026-05-06-hot-streaks-design.md
git commit -m "docs(streaks): record Phase 2 schema migration in progress log"
```

---

## Task 8: Window aggregation — rolling sums per (player, calendar_date)

`windows.py` exposes `compute_windows(conn)` which loads all `hitter_games` rows into a DataFrame, reindexes each player to a continuous calendar over `[first_played, last_played]` (zero-fills off-days), computes 3/7/14-day rolling sums, joins per-window Statcast averages, derives rate stats, assigns PT bucket, drops PA<5 rows, and idempotently upserts into `hitter_windows`.

This task lands the rolling-sum core. Rate-stat derivations (Task 9) and Statcast peripherals + bucket + upsert (Task 10) build on it.

**Files:**
- Create: `src/fantasy_baseball/streaks/windows.py`
- Create: `tests/test_streaks/test_windows.py`

- [ ] **Step 8.1: Write failing rolling-sum test**

Create `tests/test_streaks/test_windows.py`:

```python
"""Tests for the streaks rolling-window aggregator.

Synthetic data only — populates an in-memory DuckDB with a small set of
hitter_games rows and asserts rolling sums match expected values.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.models import HitterGame
from fantasy_baseball.streaks.windows import _compute_rolling_sums


def _g(player_id: int, day: int, **kwargs: int) -> HitterGame:
    """Build a HitterGame with sensible defaults; override fields via kwargs."""
    defaults = dict(
        player_id=player_id, game_pk=player_id * 1000 + day, name="X",
        team="ABC", season=2025, date=date(2025, 4, day),
        pa=4, ab=4, h=1, hr=0, r=0, rbi=0, sb=0, bb=0, k=1,
        b2=0, b3=0, sf=0, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0, is_home=True,
    )
    defaults.update(kwargs)
    return HitterGame(**defaults)


def test_rolling_sums_3_day_window_aggregates_played_and_off_days() -> None:
    """Player plays 3 games on 4/1, 4/2, 4/4. Window ending 4/4 (3-day) covers 4/2, 4/3, 4/4."""
    conn = get_connection(":memory:")
    upsert_hitter_games(conn, [
        _g(1, 1, pa=4, ab=4, h=1, hr=0, k=1),
        _g(1, 2, pa=5, ab=4, h=2, hr=1, k=1),
        _g(1, 4, pa=4, ab=3, h=1, hr=0, bb=1, k=0),
    ])
    df = _compute_rolling_sums(conn, window_days=3)
    row = df[(df["player_id"] == 1) & (df["window_end"] == pd.Timestamp("2025-04-04"))].iloc[0]
    # 4/2 + 4/3 (off, zeros) + 4/4
    assert row["pa"] == 9
    assert row["hr"] == 1
    assert row["h"] == 3


def test_rolling_sums_emits_row_for_off_day_inside_active_range() -> None:
    """Player plays 4/1 and 4/3. Window ending 4/2 (off-day) covers only 4/1 (PA=4)."""
    conn = get_connection(":memory:")
    upsert_hitter_games(conn, [
        _g(2, 1, pa=4),
        _g(2, 3, pa=5),
    ])
    df = _compute_rolling_sums(conn, window_days=3)
    mask = (df["player_id"] == 2) & (df["window_end"] == pd.Timestamp("2025-04-02"))
    assert mask.any()
    assert df[mask].iloc[0]["pa"] == 4


def test_rolling_sums_does_not_emit_before_first_played_or_after_last() -> None:
    conn = get_connection(":memory:")
    upsert_hitter_games(conn, [_g(3, 5, pa=4), _g(3, 7, pa=4)])
    df = _compute_rolling_sums(conn, window_days=3)
    pdates = set(df[df["player_id"] == 3]["window_end"].dt.date)
    assert pdates == {date(2025, 4, 5), date(2025, 4, 6), date(2025, 4, 7)}


def test_rolling_sums_window_days_7_and_14() -> None:
    """Same data, both windows return; 14-day sum == 7-day sum here (only 7 days of data)."""
    conn = get_connection(":memory:")
    upsert_hitter_games(conn, [_g(4, d, pa=4, ab=4, h=1) for d in range(1, 8)])
    df7 = _compute_rolling_sums(conn, window_days=7)
    df14 = _compute_rolling_sums(conn, window_days=14)
    end = pd.Timestamp("2025-04-07")
    p7 = df7[(df7["player_id"] == 4) & (df7["window_end"] == end)].iloc[0]
    p14 = df14[(df14["player_id"] == 4) & (df14["window_end"] == end)].iloc[0]
    assert p7["pa"] == 28
    assert p14["pa"] == 28
```

- [ ] **Step 8.2: Run the new tests, confirm they fail**

```
pytest tests/test_streaks/test_windows.py -v
```

Expected: ImportError — `windows.py` doesn't exist.

- [ ] **Step 8.3: Implement `_compute_rolling_sums`**

Create `src/fantasy_baseball/streaks/windows.py`:

```python
"""Rolling-window aggregator for the streaks project.

Generates one row per (player_id, calendar_date in [first_played, last_played],
window_days in {3, 7, 14}) populated into ``hitter_windows`` — the threshold
calibration and label assignment in :mod:`thresholds` and :mod:`labels`
build on top of these rows.

Implementation: load ``hitter_games`` into pandas, per-player reindex to the
calendar between first and last played (zero-fill off-days), apply pandas
rolling sums, then join per-window Statcast averages computed in DuckDB.
Pandas is the right tool here because per-player reindexing + rolling +
join is awkward in pure SQL; DuckDB handles the percentile work in
:mod:`thresholds`.

Idempotent: the upsert uses ``INSERT OR REPLACE`` keyed by
``(player_id, window_end, window_days)``.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_DAYS: tuple[int, ...] = (3, 7, 14)

# Box-score components we sum across windows. PA stays the canonical PA
# count even though it's also derivable from the components — the loader's
# PA-identity check is upstream so we trust the stored value.
_SUM_COLS: tuple[str, ...] = (
    "pa", "ab", "h", "hr", "r", "rbi", "sb", "bb", "k",
    "b2", "b3", "sf", "hbp",
)


def _compute_rolling_sums(
    conn: duckdb.DuckDBPyConnection, window_days: int
) -> pd.DataFrame:
    """Return a DataFrame of rolling sums for every (player, calendar-date) pair.

    Columns: player_id, window_end (Timestamp), window_days, plus one column
    per name in ``_SUM_COLS``. No PA<5 filter applied here — caller filters.
    """
    games = conn.execute(
        f"SELECT player_id, date, {', '.join(_SUM_COLS)} FROM hitter_games"
    ).df()
    if games.empty:
        return pd.DataFrame(
            columns=["player_id", "window_end", "window_days", *_SUM_COLS]
        )

    games["date"] = pd.to_datetime(games["date"])

    out_frames: list[pd.DataFrame] = []
    for player_id, player_games in games.groupby("player_id", sort=False):
        first_played = player_games["date"].min()
        last_played = player_games["date"].max()
        # Reindex to a continuous daily calendar between first and last played.
        idx = pd.date_range(first_played, last_played, freq="D")
        per_day = (
            player_games.set_index("date")[list(_SUM_COLS)]
            .groupby(level=0)
            .sum()  # collapse doubleheaders into one daily row
            .reindex(idx, fill_value=0)
        )
        rolling = per_day.rolling(window=window_days, min_periods=1).sum().astype(int)
        rolling = rolling.reset_index().rename(columns={"index": "window_end"})
        rolling.insert(0, "player_id", int(player_id))
        rolling["window_days"] = window_days
        out_frames.append(rolling)

    return pd.concat(out_frames, ignore_index=True)
```

- [ ] **Step 8.4: Run the tests, confirm they pass**

```
pytest tests/test_streaks/test_windows.py -v
```

Expected: all four tests pass.

- [ ] **Step 8.5: Commit**

```bash
git add src/fantasy_baseball/streaks/windows.py tests/test_streaks/test_windows.py
git commit -m "feat(streaks): rolling-window box-score sums in windows.py"
```

---

## Task 9: Window rate stats + Statcast peripherals

Derive per-window AVG, BABIP, ISO, K%, BB% from the rolling sums; left-join per-window EV/barrel%/xwOBA averages from `hitter_statcast_pa`.

**Files:**
- Modify: `src/fantasy_baseball/streaks/windows.py`
- Modify: `tests/test_streaks/test_windows.py`

- [ ] **Step 9.1: Write failing tests for rate-stat derivation**

Append to `tests/test_streaks/test_windows.py`:

```python
import math

from fantasy_baseball.streaks.windows import _add_rate_stats, _add_statcast_peripherals
from fantasy_baseball.streaks.models import HitterStatcastPA
from fantasy_baseball.streaks.data.load import upsert_statcast_pa


def test_add_rate_stats_computes_avg_babip_iso_k_bb() -> None:
    sums = pd.DataFrame(
        [{
            "player_id": 1, "window_end": pd.Timestamp("2025-04-07"), "window_days": 7,
            "pa": 30, "ab": 26, "h": 8, "hr": 2, "r": 5, "rbi": 6, "sb": 1, "bb": 3, "k": 6,
            "b2": 2, "b3": 0, "sf": 1, "hbp": 0,
        }]
    )
    out = _add_rate_stats(sums)
    row = out.iloc[0]
    # avg = 8/26
    assert math.isclose(row["avg"], 8 / 26)
    # babip = (h - hr) / (ab - k - hr + sf) = 6 / (26 - 6 - 2 + 1) = 6/19
    assert math.isclose(row["babip"], 6 / 19)
    # iso = (b2 + 2*b3 + 3*hr) / ab = (2 + 0 + 6) / 26
    assert math.isclose(row["iso"], 8 / 26)
    # k_pct, bb_pct
    assert math.isclose(row["k_pct"], 6 / 30)
    assert math.isclose(row["bb_pct"], 3 / 30)


def test_add_rate_stats_handles_zero_denominators() -> None:
    sums = pd.DataFrame(
        [{
            "player_id": 1, "window_end": pd.Timestamp("2025-04-07"), "window_days": 7,
            "pa": 0, "ab": 0, "h": 0, "hr": 0, "r": 0, "rbi": 0, "sb": 0, "bb": 0, "k": 0,
            "b2": 0, "b3": 0, "sf": 0, "hbp": 0,
        }]
    )
    out = _add_rate_stats(sums)
    row = out.iloc[0]
    # All denominators zero -> NaN
    for col in ("avg", "babip", "iso", "k_pct", "bb_pct"):
        assert pd.isna(row[col])


def test_add_statcast_peripherals_aggregates_per_window() -> None:
    """Player has 3 PAs on 4/1, 2 PAs on 4/3. Window ending 4/3 (3-day) averages all 5."""
    conn = get_connection(":memory:")
    upsert_statcast_pa(conn, [
        HitterStatcastPA(
            player_id=1, date=date(2025, 4, 1), pa_index=i, event="single",
            launch_speed=100.0, launch_angle=10.0,
            estimated_woba_using_speedangle=0.8, barrel=False,
            at_bat_number=i, bb_type="line_drive",
            estimated_ba_using_speedangle=0.6, hit_distance_sc=200.0,
        )
        for i in (1, 2, 3)
    ] + [
        HitterStatcastPA(
            player_id=1, date=date(2025, 4, 3), pa_index=i, event="strikeout",
            launch_speed=None, launch_angle=None,
            estimated_woba_using_speedangle=0.0, barrel=True,
            at_bat_number=i, bb_type=None,
            estimated_ba_using_speedangle=0.0, hit_distance_sc=None,
        )
        for i in (1, 2)
    ])
    sums = pd.DataFrame(
        [{"player_id": 1, "window_end": pd.Timestamp("2025-04-03"), "window_days": 3,
          "pa": 5}]
    )
    out = _add_statcast_peripherals(conn, sums)
    row = out.iloc[0]
    # ev_avg averages only the non-null launch_speeds (the 3 from 4/1) -> 100.0
    assert math.isclose(row["ev_avg"], 100.0)
    # barrel_pct: 1 of 5 barrels = 0.2
    assert math.isclose(row["barrel_pct"], 0.2)
    # xwoba_avg: (0.8*3 + 0.0*2) / 5 = 0.48
    assert math.isclose(row["xwoba_avg"], 0.48)


def test_add_statcast_peripherals_returns_nan_for_window_with_no_statcast_data() -> None:
    conn = get_connection(":memory:")
    sums = pd.DataFrame(
        [{"player_id": 99, "window_end": pd.Timestamp("2025-04-03"), "window_days": 3,
          "pa": 5}]
    )
    out = _add_statcast_peripherals(conn, sums)
    row = out.iloc[0]
    assert pd.isna(row["ev_avg"])
    assert pd.isna(row["barrel_pct"])
    assert pd.isna(row["xwoba_avg"])
```

- [ ] **Step 9.2: Run the new tests, confirm they fail**

```
pytest tests/test_streaks/test_windows.py -v -k "rate_stats or statcast_peripherals"
```

Expected: ImportError — helpers don't exist.

- [ ] **Step 9.3: Add `_add_rate_stats` and `_add_statcast_peripherals`**

Append to `src/fantasy_baseball/streaks/windows.py`:

```python
def _add_rate_stats(sums: pd.DataFrame) -> pd.DataFrame:
    """Add avg, babip, iso, k_pct, bb_pct columns to a rolling-sums frame.

    NaN where the denominator is zero (e.g. zero-PA windows from off-day
    rows that haven't been filtered yet).
    """
    out = sums.copy()
    ab = out["ab"].astype("float64")
    pa = out["pa"].astype("float64")
    babip_denom = (out["ab"] - out["k"] - out["hr"] + out["sf"]).astype("float64")
    iso_num = (out["b2"] + 2 * out["b3"] + 3 * out["hr"]).astype("float64")

    # ``.where(denom > 0)`` returns NaN where the denominator is zero, so
    # we never produce inf (numerators are finite ints). No need for the
    # deprecated ``mode.use_inf_as_na`` option context.
    out["avg"] = (out["h"] / ab).where(ab > 0)
    out["babip"] = ((out["h"] - out["hr"]) / babip_denom).where(babip_denom > 0)
    out["iso"] = (iso_num / ab).where(ab > 0)
    out["k_pct"] = (out["k"] / pa).where(pa > 0)
    out["bb_pct"] = (out["bb"] / pa).where(pa > 0)
    return out


def _add_statcast_peripherals(
    conn: duckdb.DuckDBPyConnection, sums: pd.DataFrame
) -> pd.DataFrame:
    """Left-join per-window EV/barrel%/xwOBA averages from hitter_statcast_pa.

    Computed in DuckDB SQL keyed on (player_id, date) so the aggregation
    runs in the database; the result is a small per-(player, day) frame
    that we then sum-then-divide across the window in pandas.
    """
    if sums.empty:
        for col in ("ev_avg", "barrel_pct", "xwoba_avg"):
            sums[col] = pd.NA
        return sums

    daily = conn.execute(
        """
        SELECT
            player_id,
            date,
            SUM(launch_speed) FILTER (WHERE launch_speed IS NOT NULL) AS ls_sum,
            COUNT(launch_speed) FILTER (WHERE launch_speed IS NOT NULL) AS ls_n,
            SUM(CASE WHEN barrel THEN 1 ELSE 0 END) FILTER (WHERE barrel IS NOT NULL) AS barrel_sum,
            COUNT(*) FILTER (WHERE barrel IS NOT NULL) AS barrel_n,
            SUM(estimated_woba_using_speedangle) FILTER (WHERE estimated_woba_using_speedangle IS NOT NULL) AS xwoba_sum,
            COUNT(estimated_woba_using_speedangle) FILTER (WHERE estimated_woba_using_speedangle IS NOT NULL) AS xwoba_n
        FROM hitter_statcast_pa
        GROUP BY player_id, date
        """
    ).df()
    daily["date"] = pd.to_datetime(daily["date"])

    out_rows: list[dict[str, float | int | pd.Timestamp]] = []
    for window_days, window_group in sums.groupby("window_days", sort=False):
        end_dates = window_group[["player_id", "window_end"]].drop_duplicates()
        for _, row in end_dates.iterrows():
            pid = int(row["player_id"])
            end = row["window_end"]
            start = end - pd.Timedelta(days=int(window_days) - 1)
            mask = (daily["player_id"] == pid) & (daily["date"] >= start) & (daily["date"] <= end)
            sub = daily.loc[mask]
            ls_n = int(sub["ls_n"].sum())
            barrel_n = int(sub["barrel_n"].sum())
            xwoba_n = int(sub["xwoba_n"].sum())
            out_rows.append({
                "player_id": pid,
                "window_end": end,
                "window_days": int(window_days),
                "ev_avg": float(sub["ls_sum"].sum()) / ls_n if ls_n else float("nan"),
                "barrel_pct": float(sub["barrel_sum"].sum()) / barrel_n if barrel_n else float("nan"),
                "xwoba_avg": float(sub["xwoba_sum"].sum()) / xwoba_n if xwoba_n else float("nan"),
            })
    peripherals = pd.DataFrame(out_rows)
    return sums.merge(peripherals, on=["player_id", "window_end", "window_days"], how="left")
```

- [ ] **Step 9.4: Run the tests, confirm they pass**

```
pytest tests/test_streaks/test_windows.py -v
```

Expected: all pass.

- [ ] **Step 9.5: Commit**

```bash
git add src/fantasy_baseball/streaks/windows.py tests/test_streaks/test_windows.py
git commit -m "feat(streaks): rate stats + Statcast peripherals for window aggregates"
```

---

## Task 10: PT bucket assignment + idempotent upsert + CLI

Wraps the rolling-sum + rate-stat + Statcast-peripheral pipeline into a `compute_windows(conn)` entrypoint. Filters PA<5, assigns PT bucket per spec (`5-9 = low`, `10-19 = mid`, `≥20 = high`), upserts via `INSERT OR REPLACE` keyed on `(player_id, window_end, window_days)`. The CLI `scripts/streaks/compute_windows.py` shells out to it.

**Files:**
- Modify: `src/fantasy_baseball/streaks/windows.py`
- Modify: `tests/test_streaks/test_windows.py`
- Create: `scripts/streaks/compute_windows.py`

- [ ] **Step 10.1: Write failing tests for bucket assignment + upsert**

Append to `tests/test_streaks/test_windows.py`:

```python
from fantasy_baseball.streaks.windows import _assign_pt_bucket, compute_windows


def test_assign_pt_bucket_boundaries() -> None:
    df = pd.DataFrame({"pa": [4, 5, 9, 10, 19, 20, 50]})
    out = _assign_pt_bucket(df)
    # PA<5 rows are dropped before bucketing; this helper assumes filter already applied.
    assert list(out["pt_bucket"]) == ["low", "low", "mid", "mid", "high", "high"], out
    # Note: the input had PA=4 which should already be filtered upstream;
    # _assign_pt_bucket is called after the PA>=5 filter.


def test_compute_windows_filters_pa_lt_5_and_writes_buckets() -> None:
    conn = get_connection(":memory:")
    # Player 1: 8 PA over 3 days (mid bucket for 3-day window)
    # Player 2: 2 PA over 3 days (filtered out)
    upsert_hitter_games(conn, [
        _g(1, 1, pa=4, ab=4, h=1), _g(1, 2, pa=2, ab=2, h=0), _g(1, 3, pa=2, ab=2, h=1),
        _g(2, 1, pa=1, ab=1, h=0), _g(2, 2, pa=1, ab=1, h=0),
    ])
    n = compute_windows(conn)
    assert n > 0
    rows = conn.execute(
        "SELECT player_id, window_end, window_days, pa, pt_bucket "
        "FROM hitter_windows ORDER BY player_id, window_end, window_days"
    ).fetchall()
    by_key = {(r[0], r[1], r[2]): r for r in rows}
    # 3-day window ending 4/3 for player 1: PA = 4+2+2 = 8 -> 'low'
    key = (1, date(2025, 4, 3), 3)
    assert key in by_key
    assert by_key[key][3] == 8
    assert by_key[key][4] == "low"
    # Player 2 should have no rows (all windows have PA<5)
    assert all(r[0] != 2 for r in rows)


def test_compute_windows_is_idempotent() -> None:
    conn = get_connection(":memory:")
    upsert_hitter_games(conn, [
        _g(1, d, pa=5, ab=4, h=1) for d in range(1, 8)
    ])
    n1 = compute_windows(conn)
    n2 = compute_windows(conn)
    # Same input -> same row count, no PK duplications.
    assert n1 == n2
    total = conn.execute("SELECT COUNT(*) FROM hitter_windows").fetchone()[0]
    assert total == n1
```

- [ ] **Step 10.2: Run the new tests, confirm they fail**

```
pytest tests/test_streaks/test_windows.py -v -k "pt_bucket or compute_windows"
```

Expected: FAIL — `_assign_pt_bucket` and `compute_windows` don't exist.

- [ ] **Step 10.3: Add `_assign_pt_bucket` and `compute_windows`**

Append to `src/fantasy_baseball/streaks/windows.py`:

```python
def _assign_pt_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Assign 'low' (5-9) / 'mid' (10-19) / 'high' (>=20) based on PA.

    Caller must have already filtered PA<5 rows out — this helper does not
    re-filter (the schema requires pt_bucket NOT NULL).
    """
    out = df.copy()
    bins = [4, 9, 19, 10**9]
    labels = ["low", "mid", "high"]
    out["pt_bucket"] = pd.cut(out["pa"], bins=bins, labels=labels).astype("string")
    return out


_HITTER_WINDOWS_COLS: tuple[str, ...] = (
    "player_id", "window_end", "window_days",
    "pa", "hr", "r", "rbi", "sb",
    "avg", "babip", "k_pct", "bb_pct", "iso",
    "ev_avg", "barrel_pct", "xwoba_avg",
    "pt_bucket",
)


def compute_windows(conn: duckdb.DuckDBPyConnection) -> int:
    """Rebuild ``hitter_windows`` from ``hitter_games`` + ``hitter_statcast_pa``.

    Generates rows for every (player, calendar_date in
    [first_played, last_played], window_days in {3, 7, 14}) where the
    window's PA >= 5. Returns the total row count written.

    Idempotent: ``INSERT OR REPLACE`` keyed by (player_id, window_end, window_days).
    """
    all_rows: list[pd.DataFrame] = []
    for window_days in WINDOW_DAYS:
        sums = _compute_rolling_sums(conn, window_days=window_days)
        sums = sums[sums["pa"] >= 5].copy()
        if sums.empty:
            continue
        sums = _add_rate_stats(sums)
        sums = _add_statcast_peripherals(conn, sums)
        sums = _assign_pt_bucket(sums)
        all_rows.append(sums)

    if not all_rows:
        return 0

    out = pd.concat(all_rows, ignore_index=True)[list(_HITTER_WINDOWS_COLS)]
    # DuckDB takes pandas Timestamps natively for DATE columns.
    placeholders = ", ".join(["?"] * len(_HITTER_WINDOWS_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_windows ({', '.join(_HITTER_WINDOWS_COLS)}) "
        f"VALUES ({placeholders})"
    )
    rows = [tuple(None if pd.isna(v) else v for v in r) for r in out.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)
    logger.info("Wrote %d rows to hitter_windows", len(rows))
    return len(rows)
```

- [ ] **Step 10.4: Run all windows tests, confirm pass**

```
pytest tests/test_streaks/test_windows.py -v
```

Expected: all pass.

- [ ] **Step 10.5: Create the CLI**

Create `scripts/streaks/compute_windows.py`:

```python
"""CLI: rebuild ``hitter_windows`` from the current ``hitter_games`` + ``hitter_statcast_pa``.

Usage:
    python -m scripts.streaks.compute_windows [--db PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fantasy_baseball.streaks.data.schema import (  # noqa: E402
    DEFAULT_DB_PATH,
    get_connection,
)
from fantasy_baseball.streaks.windows import compute_windows  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild hitter_windows.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db)
    n = compute_windows(conn)
    print(f"Wrote {n} rows to hitter_windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10.6: Verify the CLI parses**

```
python -c "from scripts.streaks.compute_windows import main; main(['--help'])"
```

Expected: prints help.

- [ ] **Step 10.7: Commit**

```bash
git add src/fantasy_baseball/streaks/windows.py tests/test_streaks/test_windows.py scripts/streaks/compute_windows.py
git commit -m "feat(streaks): compute_windows entry point + CLI"
```

---

## Task 11: Threshold calibration

`thresholds.py` exposes `compute_thresholds(conn, season_set, qualifying_pa=150)`. Selects `hitter_windows` rows whose player had ≥`qualifying_pa` PA in the relevant season, computes p10/p90 per (category × window_days × pt_bucket) using DuckDB's `percentile_cont`, and idempotently writes to `thresholds`. Categories: `hr`, `r`, `rbi`, `sb`, `avg`.

**Files:**
- Create: `src/fantasy_baseball/streaks/thresholds.py`
- Create: `tests/test_streaks/test_thresholds.py`

- [ ] **Step 11.1: Write failing threshold-calibration tests**

Create `tests/test_streaks/test_thresholds.py`:

```python
"""Tests for threshold calibration."""

from __future__ import annotations

from datetime import date

import pytest

from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import HitterGame
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows


def _g(pid: int, day: int, **kwargs: int) -> HitterGame:
    defaults = dict(
        player_id=pid, game_pk=pid * 1000 + day, name="X", team="ABC", season=2025,
        date=date(2025, 4, day),
        pa=4, ab=4, h=1, hr=0, r=0, rbi=0, sb=0, bb=0, k=1,
        b2=0, b3=0, sf=0, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0, is_home=True,
    )
    defaults.update(kwargs)
    return HitterGame(**defaults)


def test_compute_thresholds_writes_p10_p90_per_strata() -> None:
    """Set up two qualifying players, build windows, compute thresholds, verify shape."""
    conn = get_connection(":memory:")
    games: list[HitterGame] = []
    # Player 1: 200 PA over 50 days, 5 HR (hot)
    for d in range(1, 51):
        games.append(_g(1, d, pa=4, ab=4, h=1, hr=(1 if d % 10 == 0 else 0)))
    # Player 2: 200 PA over 50 days, 0 HR (cold)
    for d in range(1, 51):
        games.append(_g(2, d, pa=4, ab=4, h=1, hr=0))
    upsert_hitter_games(conn, games)
    compute_windows(conn)

    n = compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    assert n > 0

    rows = conn.execute(
        "SELECT category, window_days, pt_bucket, p10, p90 "
        "FROM thresholds WHERE season_set = '2025' "
        "ORDER BY category, window_days, pt_bucket"
    ).fetchall()
    # 5 categories x 3 window_days x up to 3 buckets = up to 45 rows; here
    # we should have at least one row per (category, window_days) seen in data.
    cats = {r[0] for r in rows}
    assert cats == {"hr", "r", "rbi", "sb", "avg"}
    for r in rows:
        assert r[3] <= r[4], f"p10 > p90 for {r}"


def test_compute_thresholds_excludes_unqualified_player_seasons() -> None:
    """Player with <150 PA in a season is dropped from calibration entirely."""
    conn = get_connection(":memory:")
    # Player 1: ~28 PA total, season=2025 (won't qualify at 150 cutoff)
    games = [_g(1, d, pa=4, ab=4, h=1) for d in range(1, 8)]
    upsert_hitter_games(conn, games)
    compute_windows(conn)

    n = compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    # No qualifying rows -> no thresholds written.
    assert n == 0


def test_compute_thresholds_is_idempotent() -> None:
    conn = get_connection(":memory:")
    games = [
        _g(pid, d, pa=4, ab=4, h=1, hr=(1 if pid == 1 and d % 5 == 0 else 0))
        for pid in (1, 2, 3)
        for d in range(1, 51)
    ]
    upsert_hitter_games(conn, games)
    compute_windows(conn)
    n1 = compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    n2 = compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    assert n1 == n2
    total = conn.execute(
        "SELECT COUNT(*) FROM thresholds WHERE season_set = '2025'"
    ).fetchone()[0]
    assert total == n1
```

- [ ] **Step 11.2: Run the tests, confirm they fail**

```
pytest tests/test_streaks/test_thresholds.py -v
```

Expected: ImportError — module missing.

- [ ] **Step 11.3: Implement `compute_thresholds`**

Create `src/fantasy_baseball/streaks/thresholds.py`:

```python
"""Empirical threshold calibration for streak labels.

For a given calibration set (e.g. ``"2023-2025"`` or a single season),
computes p10 and p90 per (category × window_days × pt_bucket) using only
windows from player-seasons with >= ``qualifying_pa`` PA. Writes results
to the ``thresholds`` table (idempotent — ``INSERT OR REPLACE`` keyed on
(season_set, category, window_days, pt_bucket)).

The five categories match the project's hitter roto stats: HR, R, RBI,
SB (counting), and AVG (rate). For the counting stats we percentile the
raw window count; for AVG we percentile the rate column from
``hitter_windows.avg``.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = ("hr", "r", "rbi", "sb", "avg")


def compute_thresholds(
    conn: duckdb.DuckDBPyConnection,
    *,
    season_set: str,
    qualifying_pa: int = 150,
) -> int:
    """(Re)build ``thresholds`` rows for the given ``season_set``.

    ``season_set`` is a free-form label (e.g. ``"2025"``, ``"2023-2025"``);
    the SQL filter on which seasons to include is derived from it.
    Currently supports either a single season ``"YYYY"`` or a hyphenated
    range ``"YYYY-YYYY"``.

    Returns the number of rows written.
    """
    seasons = _parse_season_set(season_set)
    season_list_sql = ", ".join(str(s) for s in seasons)

    # Qualifying players: aggregate PA from hitter_games over the seasons
    # in scope, keep player_ids with sum(pa) >= qualifying_pa.
    qualifying_sql = f"""
        SELECT player_id, season FROM hitter_games
        WHERE season IN ({season_list_sql})
        GROUP BY player_id, season
        HAVING SUM(pa) >= {qualifying_pa}
    """

    # Windows from those qualifying player-seasons. We join hitter_windows
    # to a derived qualifying-player-seasons table on (player_id, season),
    # using the calendar year of window_end as the season key.
    windows_sql = f"""
        WITH qualified AS ({qualifying_sql})
        SELECT
            w.player_id,
            EXTRACT(YEAR FROM w.window_end)::INTEGER AS season,
            w.window_end, w.window_days, w.pt_bucket,
            w.hr, w.r, w.rbi, w.sb, w.avg
        FROM hitter_windows w
        JOIN qualified q
          ON q.player_id = w.player_id
         AND q.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
    """

    # Drop any pre-existing rows for this season_set before re-writing.
    conn.execute("DELETE FROM thresholds WHERE season_set = ?", [season_set])

    written = 0
    for category in CATEGORIES:
        # AVG is a rate column; counting cats are int columns.
        col = "avg" if category == "avg" else category
        rows = conn.execute(
            f"""
            WITH src AS ({windows_sql})
            SELECT
                window_days,
                pt_bucket,
                percentile_cont(0.1) WITHIN GROUP (ORDER BY {col}) AS p10,
                percentile_cont(0.9) WITHIN GROUP (ORDER BY {col}) AS p90
            FROM src
            WHERE {col} IS NOT NULL
            GROUP BY window_days, pt_bucket
            HAVING COUNT(*) >= 1
            """
        ).fetchall()
        for window_days, pt_bucket, p10, p90 in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO thresholds
                (season_set, category, window_days, pt_bucket, p10, p90)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [season_set, category, int(window_days), pt_bucket, float(p10), float(p90)],
            )
            written += 1
    logger.info("Wrote %d threshold rows for season_set=%s", written, season_set)
    return written


def _parse_season_set(season_set: str) -> list[int]:
    """Parse ``"YYYY"`` or ``"YYYY-YYYY"`` into an inclusive list of seasons."""
    if "-" in season_set:
        start_str, end_str = season_set.split("-", 1)
        return list(range(int(start_str), int(end_str) + 1))
    return [int(season_set)]
```

- [ ] **Step 11.4: Run the tests, confirm they pass**

```
pytest tests/test_streaks/test_thresholds.py -v
```

Expected: all pass.

- [ ] **Step 11.5: Commit**

```bash
git add src/fantasy_baseball/streaks/thresholds.py tests/test_streaks/test_thresholds.py
git commit -m "feat(streaks): empirical p10/p90 threshold calibration"
```

---

## Task 12: Label application

`labels.py` exposes `apply_labels(conn, season_set)`. For each row in `hitter_windows` that has a matching threshold (same `pt_bucket` and `window_days`, `season_set` parameter), produce one `hitter_streak_labels` row per category: `'hot'` if value ≥ p90, `'cold'` if ≤ p10, else `'neutral'`.

**Files:**
- Create: `src/fantasy_baseball/streaks/labels.py`
- Create: `tests/test_streaks/test_labels.py`

- [ ] **Step 12.1: Write failing label tests**

Create `tests/test_streaks/test_labels.py`:

```python
"""Tests for label application."""

from __future__ import annotations

from datetime import date

import pytest

from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.models import HitterGame
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows


def _seed_population(conn) -> None:
    """Synthetic dataset: spread of HR rates so percentile thresholds are non-degenerate."""
    games: list[HitterGame] = []
    # 5 players, 50 days each, varying HR rates 0-4 per 10 days.
    for pid in range(1, 6):
        for d in range(1, 51):
            hr = 1 if (d % (12 - 2 * pid) == 0) else 0
            games.append(HitterGame(
                player_id=pid, game_pk=pid * 1000 + d, name=f"P{pid}", team="ABC",
                season=2025, date=date(2025, 4, d),
                pa=4, ab=4, h=1, hr=hr, r=0, rbi=0, sb=0, bb=0, k=1,
                b2=0, b3=0, sf=0, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0, is_home=True,
            ))
    upsert_hitter_games(conn, games)
    compute_windows(conn)
    compute_thresholds(conn, season_set="2025", qualifying_pa=150)


def test_apply_labels_writes_rows_per_category() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    n = apply_labels(conn, season_set="2025")
    assert n > 0
    cats = {r[0] for r in conn.execute("SELECT DISTINCT category FROM hitter_streak_labels").fetchall()}
    assert cats == {"hr", "r", "rbi", "sb", "avg"}


def test_apply_labels_classifies_hot_above_p90_cold_below_p10() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    apply_labels(conn, season_set="2025")
    # Pull one threshold row + a window row matching it; confirm the label
    # respects the inequality.
    threshold = conn.execute(
        "SELECT category, window_days, pt_bucket, p10, p90 "
        "FROM thresholds WHERE season_set = '2025' AND category = 'hr' LIMIT 1"
    ).fetchone()
    cat, win, bucket, p10, p90 = threshold
    sample = conn.execute(
        "SELECT player_id, window_end, hr FROM hitter_windows "
        "WHERE window_days = ? AND pt_bucket = ? LIMIT 5",
        [win, bucket],
    ).fetchall()
    for pid, end, hr in sample:
        label = conn.execute(
            "SELECT label FROM hitter_streak_labels "
            "WHERE player_id = ? AND window_end = ? AND window_days = ? AND category = ?",
            [pid, end, win, cat],
        ).fetchone()[0]
        if hr >= p90:
            assert label == "hot", f"hr={hr} >= p90={p90} but label={label}"
        elif hr <= p10:
            assert label == "cold", f"hr={hr} <= p10={p10} but label={label}"
        else:
            assert label == "neutral"


def test_apply_labels_is_idempotent() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    n1 = apply_labels(conn, season_set="2025")
    n2 = apply_labels(conn, season_set="2025")
    assert n1 == n2
    total = conn.execute("SELECT COUNT(*) FROM hitter_streak_labels").fetchone()[0]
    assert total == n1


def test_apply_labels_skips_windows_without_matching_thresholds() -> None:
    """A window in a bucket with no threshold row gets no labels (not an error)."""
    conn = get_connection(":memory:")
    _seed_population(conn)
    # Manually delete one bucket's thresholds and re-apply.
    conn.execute("DELETE FROM thresholds WHERE season_set='2025' AND pt_bucket='high'")
    apply_labels(conn, season_set="2025")
    # No labels should reference a 'high'-bucket window.
    cnt = conn.execute(
        """
        SELECT COUNT(*) FROM hitter_streak_labels l
        JOIN hitter_windows w
          ON w.player_id = l.player_id
         AND w.window_end = l.window_end
         AND w.window_days = l.window_days
        WHERE w.pt_bucket = 'high'
        """
    ).fetchone()[0]
    assert cnt == 0
```

- [ ] **Step 12.2: Run the tests, confirm they fail**

```
pytest tests/test_streaks/test_labels.py -v
```

Expected: ImportError.

- [ ] **Step 12.3: Implement `apply_labels`**

Create `src/fantasy_baseball/streaks/labels.py`:

```python
"""Apply calibrated thresholds to ``hitter_windows`` -> ``hitter_streak_labels``.

For each (window row × category), label is:
  - 'hot'     if value >= p90
  - 'cold'    if value <= p10
  - 'neutral' otherwise

Windows whose (window_days, pt_bucket) combo has no threshold row in the
named ``season_set`` are skipped entirely (no labels written).

Idempotent: rebuilds all rows for the season_set on each call.
"""

from __future__ import annotations

import logging

import duckdb

from fantasy_baseball.streaks.thresholds import CATEGORIES

logger = logging.getLogger(__name__)


def apply_labels(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Rebuild ``hitter_streak_labels`` from ``hitter_windows`` joined to thresholds.

    Returns total rows written across all categories.
    """
    # Wipe labels generated under this season_set's thresholds. Since the
    # labels table doesn't carry season_set, we wipe all rows whose
    # window-row+category have a threshold in this season_set; simpler is
    # to wipe everything and rebuild — labels are tied to the latest
    # thresholds anyway.
    conn.execute("DELETE FROM hitter_streak_labels")

    written = 0
    for category in CATEGORIES:
        col = "avg" if category == "avg" else category
        sql = f"""
            INSERT OR REPLACE INTO hitter_streak_labels
                (player_id, window_end, window_days, category, label)
            SELECT
                w.player_id,
                w.window_end,
                w.window_days,
                ? AS category,
                CASE
                    WHEN w.{col} IS NULL THEN 'neutral'
                    WHEN w.{col} >= t.p90 THEN 'hot'
                    WHEN w.{col} <= t.p10 THEN 'cold'
                    ELSE 'neutral'
                END AS label
            FROM hitter_windows w
            JOIN thresholds t
              ON t.season_set = ?
             AND t.category = ?
             AND t.window_days = w.window_days
             AND t.pt_bucket = w.pt_bucket
        """
        before = conn.execute("SELECT COUNT(*) FROM hitter_streak_labels").fetchone()[0]
        conn.execute(sql, [category, season_set, category])
        after = conn.execute("SELECT COUNT(*) FROM hitter_streak_labels").fetchone()[0]
        written += after - before

    logger.info("Wrote %d label rows for season_set=%s", written, season_set)
    return written
```

- [ ] **Step 12.4: Run the tests, confirm they pass**

```
pytest tests/test_streaks/test_labels.py -v
```

Expected: all pass.

- [ ] **Step 12.5: Commit**

```bash
git add src/fantasy_baseball/streaks/labels.py tests/test_streaks/test_labels.py
git commit -m "feat(streaks): hot/cold/neutral label application"
```

---

## Task 13: `compute_labels.py` CLI — chain windows → thresholds → labels

A single entry point the user can run nightly that rebuilds the full label set.

**Files:**
- Create: `scripts/streaks/compute_labels.py`

- [ ] **Step 13.1: Create the CLI**

Create `scripts/streaks/compute_labels.py`:

```python
"""CLI: rebuild hitter_windows + thresholds + hitter_streak_labels.

Usage:
    python -m scripts.streaks.compute_labels [--db PATH] [--season-set 2023-2025] [--qualifying-pa 150]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fantasy_baseball.streaks.data.schema import (  # noqa: E402
    DEFAULT_DB_PATH,
    get_connection,
)
from fantasy_baseball.streaks.labels import apply_labels  # noqa: E402
from fantasy_baseball.streaks.thresholds import compute_thresholds  # noqa: E402
from fantasy_baseball.streaks.windows import compute_windows  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild streaks windows + thresholds + labels.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--season-set", default="2023-2025")
    parser.add_argument("--qualifying-pa", type=int, default=150)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db)
    n_windows = compute_windows(conn)
    n_thresholds = compute_thresholds(
        conn, season_set=args.season_set, qualifying_pa=args.qualifying_pa
    )
    n_labels = apply_labels(conn, season_set=args.season_set)
    print(
        f"windows: {n_windows} rows; "
        f"thresholds: {n_thresholds} rows for season_set={args.season_set}; "
        f"labels: {n_labels} rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 13.2: Run the full pipeline against the local DB**

```
python scripts/streaks/compute_labels.py --season-set 2023-2025
```

Expected: prints something like:
```
windows: ~600,000 rows; thresholds: ~45 rows for season_set=2023-2025; labels: ~3,000,000 rows.
```

(Exact numbers depend on calendar coverage; what matters is non-zero on all three lines, no exceptions.)

- [ ] **Step 13.3: Spot-check the threshold output makes sense**

```
python -c "
import duckdb
conn = duckdb.connect('data/streaks/streaks.duckdb')
print('-- HR thresholds (counting) --')
for r in conn.execute(
    \"SELECT category, window_days, pt_bucket, p10, p90 FROM thresholds \"
    \"WHERE season_set='2023-2025' AND category='hr' \"
    \"ORDER BY window_days, pt_bucket\"
).fetchall():
    print(r)
print('-- AVG thresholds (rate) --')
for r in conn.execute(
    \"SELECT category, window_days, pt_bucket, p10, p90 FROM thresholds \"
    \"WHERE season_set='2023-2025' AND category='avg' \"
    \"ORDER BY window_days, pt_bucket\"
).fetchall():
    print(r)
"
```

Eyeball expectations:
- `hr` p90 for `7d × high` should be ~3 HR (a tear-of-the-week threshold).
- `hr` p90 for `3d × low` should be ~1 HR (rare: 1 HR in 5-9 PA).
- `avg` p90 for `14d × high` should be in the .375-.420 range.
- `avg` p10 for `14d × high` should be in the .150-.190 range.

If the eyeballs are wildly off, **stop** and inspect — likely a bucket-assignment or qualification bug.

- [ ] **Step 13.4: Commit**

```bash
git add scripts/streaks/compute_labels.py
git commit -m "feat(streaks): compute_labels CLI chains windows -> thresholds -> labels"
```

---

## Task 14: Phase-2 acceptance notebook (`01_distributions.ipynb`)

The Phase-2 gate per the spec: "Notebook 01 produces threshold tables that pass eyeball test." This task creates the notebook structure and content as a `.py` file under `notebooks/streaks/`, then converts to `.ipynb` via `jupytext`. (Notebooks are gitignored; the source `.py` is committed so the notebook is reproducible.)

**Files:**
- Create: `notebooks/streaks/01_distributions.py` (jupytext source — gitignored along with the .ipynb)

> **Note:** Per the spec, `notebooks/streaks/` is gitignored. We still keep the source under `notebooks/streaks/` so the user can regenerate it locally. If the user wants the notebook source committed, the file should move to `docs/streaks/notebooks/01_distributions.py` instead — flag this question to the user during execution.

- [ ] **Step 14.1: Verify `jupytext` is available**

```
python -c "import jupytext; print(jupytext.__version__)"
```

If ImportError: `pip install jupytext` (or add to dev deps in `pyproject.toml`).

- [ ] **Step 14.2: Create the notebook source**

Create `notebooks/streaks/01_distributions.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Hot Streaks Phase 2 — Distributions & Threshold Sanity Check
#
# This is the Phase-2 acceptance notebook. It loads the calibrated
# `thresholds` table and the underlying `hitter_windows` data, and
# confirms that the empirical p10/p90 thresholds make intuitive sense
# for each category × window × playing-time bucket.
#
# **Run after:** `python scripts/streaks/compute_labels.py --season-set 2023-2025`

# %%
import duckdb
import matplotlib.pyplot as plt
import pandas as pd

DB = "data/streaks/streaks.duckdb"
SEASON_SET = "2023-2025"

conn = duckdb.connect(DB)

# %% [markdown]
# ## Threshold table
#
# All p10 / p90 thresholds calibrated from 2023-2025 windows for hitters
# with ≥150 PA in the relevant season.

# %%
thresholds = conn.execute(
    "SELECT category, window_days, pt_bucket, p10, p90 "
    "FROM thresholds WHERE season_set = ? "
    "ORDER BY category, window_days, pt_bucket",
    [SEASON_SET],
).df()
thresholds

# %% [markdown]
# ## Distribution histograms — counting categories
#
# For each (category, window_days, pt_bucket), plot the empirical
# distribution and overlay p10/p90 vertical lines. Bucket order: low / mid / high.

# %%
windows = conn.execute(
    """
    SELECT player_id, window_end, window_days, pt_bucket,
           pa, hr, r, rbi, sb, avg
    FROM hitter_windows
    """
).df()

for cat in ("hr", "r", "rbi", "sb"):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=False)
    for col, wd in enumerate((3, 7, 14)):
        for row, bucket in enumerate(("low", "mid", "high")):
            ax = axes[row, col]
            sub = windows[(windows["window_days"] == wd) & (windows["pt_bucket"] == bucket)]
            ax.hist(sub[cat], bins=range(int(sub[cat].max()) + 2), edgecolor="black")
            t = thresholds[
                (thresholds["category"] == cat)
                & (thresholds["window_days"] == wd)
                & (thresholds["pt_bucket"] == bucket)
            ]
            if not t.empty:
                ax.axvline(t.iloc[0]["p10"], color="blue", linestyle="--", label=f"p10={t.iloc[0]['p10']:.1f}")
                ax.axvline(t.iloc[0]["p90"], color="red", linestyle="--", label=f"p90={t.iloc[0]['p90']:.1f}")
                ax.legend(fontsize=8)
            ax.set_title(f"{cat.upper()} — {wd}d × {bucket}")
    fig.suptitle(f"Distribution + thresholds: {cat.upper()}", fontsize=14)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Distribution — AVG (rate)

# %%
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for col, wd in enumerate((3, 7, 14)):
    for row, bucket in enumerate(("low", "mid", "high")):
        ax = axes[row, col]
        sub = windows[(windows["window_days"] == wd) & (windows["pt_bucket"] == bucket)]
        ax.hist(sub["avg"].dropna(), bins=40, edgecolor="black")
        t = thresholds[
            (thresholds["category"] == "avg")
            & (thresholds["window_days"] == wd)
            & (thresholds["pt_bucket"] == bucket)
        ]
        if not t.empty:
            ax.axvline(t.iloc[0]["p10"], color="blue", linestyle="--", label=f"p10={t.iloc[0]['p10']:.3f}")
            ax.axvline(t.iloc[0]["p90"], color="red", linestyle="--", label=f"p90={t.iloc[0]['p90']:.3f}")
            ax.legend(fontsize=8)
        ax.set_title(f"AVG — {wd}d × {bucket}")
fig.suptitle("Distribution + thresholds: AVG", fontsize=14)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Eyeball checklist
#
# Confirm before signing off:
#
# 1. **HR / 7d / high** p90 ≈ 3 (a "hot HR week" is 3+ HR)
# 2. **AVG / 14d / high** p90 ∈ [.375, .420]
# 3. **AVG / 14d / high** p10 ∈ [.150, .190]
# 4. **SB / 7d / high** p90 ≈ 2 (rare to steal 2+ in a week as a top-PA hitter)
# 5. No category has p10 > p90 (this would be a bug)
# 6. Bucket monotonicity: for counting cats, p90 should rise as bucket
#    moves low → mid → high (more PA = more counts). Visual check.

# %%
print("Notebook done.")
```

- [ ] **Step 14.3: Convert to .ipynb and run end-to-end**

```
jupytext --to notebook notebooks/streaks/01_distributions.py
jupyter nbconvert --to notebook --execute notebooks/streaks/01_distributions.ipynb --output 01_distributions.ipynb
```

Expected: notebook executes without errors. Open it in Jupyter and run through the eyeball checklist.

- [ ] **Step 14.4: Append Phase 2 acceptance entry to the spec**

Append to `docs/superpowers/specs/2026-05-06-hot-streaks-design.md`:

```markdown
### 2026-05-08 — Phase 2 (windows, thresholds, labels) accepted

Notebook 01_distributions ran clean over 2023-2025 with ~150 PA
qualification. Threshold eyeball checklist:

- HR / 7d / high p90 = <fill in>
- AVG / 14d / high p90 = <fill in>; p10 = <fill in>
- SB / 7d / high p90 = <fill in>
- All p10 ≤ p90 across 45 strata: <yes/no>
- Bucket monotonicity holds for counting cats: <yes/no>

Row counts: hitter_windows ≈ <fill in>; hitter_streak_labels ≈ <fill in>.

#### Next milestone

Phase 3 — continuation analysis (the go/no-go gate). For each labeled
(player, window_end, category) row in 2023-2024, compute the next-window
outcome and tabulate persistence rates. Stratify by streak strength,
PT bucket, and player-season skill quartile. Compare to base rates.
2025 reserved as a held-out test set; 2026 is out-of-sample for
production inference.
```

- [ ] **Step 14.5: Commit**

```bash
git add notebooks/streaks/01_distributions.py docs/superpowers/specs/2026-05-06-hot-streaks-design.md
git commit -m "docs(streaks): Phase 2 acceptance notebook + spec progress entry"
```

---

## Final verification

After all 14 tasks land, run the full project verification suite at the repo root.

- [ ] **Step F.1: Run the project test suite**

```
pytest -v
```

Expected: all pass. The streaks tests should be ~50+ tests; the rest of the project should be unaffected.

- [ ] **Step F.2: Lint**

```
ruff check .
ruff format --check .
```

Expected: zero violations. Run `ruff format .` to fix any drift.

- [ ] **Step F.3: Vulture (dead-code check)**

```
vulture
```

Expected: no NEW findings introduced. Pre-existing unrelated findings are acceptable; call them out if you see them.

- [ ] **Step F.4: mypy (strict for streaks)**

```
mypy
```

Expected: zero errors. The streaks package is in `[tool.mypy].files` and the strict overrides list, so all new code must type-check cleanly.

- [ ] **Step F.5: Push the branch and open a PR**

The implementation will likely happen on a feature branch, not the plan branch. Push and open a PR with a body that summarizes which tasks landed, the row counts after re-fetch, and the threshold eyeball-check numbers from Task 14. **Do not auto-merge** — the user explicitly owns the merge gate per the project's "no merge without asking" guidance. Pause and ask once the PR is open.

---

## Spec coverage check

Mapping each Phase 2 spec requirement to a task:

| Spec requirement | Task |
|------------------|------|
| Window aggregation per (player, window_end, window_days∈{3,7,14}) | 8 |
| Sum-based counting cats: HR, R, RBI, SB | 8 |
| Rate stats: AVG, BABIP, ISO, K%, BB% | 9 |
| Statcast peripherals: ev_avg, barrel_pct, xwoba_avg | 9 |
| PT bucket: low (5-9 PA) / mid (10-19 PA) / high (≥20 PA); skip <5 PA | 10 |
| Idempotent upsert into `hitter_windows` | 10 |
| Threshold calibration: percentile_cont(0.1) / percentile_cont(0.9) per (category × window × bucket) | 11 |
| Player qualification: ≥150 PA in season | 11 |
| Categories tracked: HR, R, RBI, SB, AVG | 11, 12 |
| Empirical labels: hot/cold/neutral per (window, category) | 12 |
| Notebook 01 distribution + threshold sanity check | 14 |
| BABIP/ISO sourcing decision (Open Q3) | 1, 2, 3, 7 (Path a) |
| Doubleheader / pa_index stability (Open Q2) | 4 (resolved via at_bat_number sort) |
| All-or-nothing Statcast skip (Open Q1) | Deferred — no Phase 2 task; documented |
| Phase 2 gate: notebook 01 passes eyeball | 14 |

All Phase 2 spec items are covered. Open Q1 is intentionally deferred per the design-decision section above.
