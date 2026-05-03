# Trends Charts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `/trends` page with two interactive Chart.js line graphs — actual standings over time and projected ERoto over time — with per-stat tabs, hover-to-highlight, and click-to-toggle.

**Architecture:** Persist projected standings to a new `projected_standings_history` Redis hash (mirrors the existing `standings_history`). Backfill historical projections via a one-time roster-extrapolation script that uses today's per-player full-season projection against historical roster snapshots. Add `/trends` page route and `/api/trends/series` endpoint that transposes both history hashes into a per-team time series and serves them in a single payload. Render with Chart.js loaded from CDN.

**Tech Stack:** Python (Flask, pytest, fakeredis), vanilla JS (Chart.js v4 from CDN), HTML/CSS in Jinja templates.

**Spec:** `docs/superpowers/specs/2026-05-02-trends-charts-design.md`

---

## File Structure

| File | Role |
|---|---|
| `src/fantasy_baseball/data/redis_store.py` | Add `PROJECTED_STANDINGS_HISTORY_KEY`, `write_projected_standings_snapshot`, `get_projected_standings_day`, `get_projected_standings_history` |
| `src/fantasy_baseball/data/kv_sync.py` | Add new hash to `_HASH_KEYS` so local↔remote sync covers it |
| `src/fantasy_baseball/web/refresh_pipeline.py` | Append snapshot to `projected_standings_history` after `_build_projected_standings` writes the cache |
| `tests/test_data/test_redis_store_projected_standings.py` (new) | Coverage for the four new helpers (mirrors `test_redis_store_standings.py`) |
| `tests/test_web/test_refresh_pipeline.py` | Assert `projected_standings_history` is populated after a refresh |
| `scripts/backfill_projected_standings_history.py` (new) | One-time roster-extrapolation backfill |
| `tests/test_scripts/test_backfill_projected_standings_history.py` (new) | End-to-end test of the backfill against seeded fakeredis |
| `src/fantasy_baseball/web/season_data.py` | New `build_trends_series` builder that reads both history hashes and returns the API payload |
| `src/fantasy_baseball/web/season_routes.py` | New `/trends` page route + `/api/trends/series` JSON endpoint |
| `tests/test_web/test_trends_route.py` (new) | Coverage for the new endpoint |
| `src/fantasy_baseball/web/templates/season/trends.html` (new) | Page template with two `<canvas>` and tab strips |
| `src/fantasy_baseball/web/templates/season/base.html` | Add "Trends" link to the sidebar |
| `src/fantasy_baseball/web/static/season_trends.js` (new) | Fetch + render + tab-switching + hover-highlight |

## Conventions used throughout this plan

- Conventional commit prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).
- Tests committed alongside the code that makes them pass.
- Pre-commit hooks must pass; never bypass with `--no-verify`.
- Player keys are `name::player_type` per CLAUDE.md.
- Use the existing `fake_redis` fixture from `tests/conftest.py` (FakeRedis with `decode_responses=True`).
- No bare-name lookups: keep going through `weekly_rosters_history` (which carries `yahoo_id` and `player_name`).

---

## Phase 0 — Setup

### Task 0: Create feature branch

**Files:** none

- [ ] **Step 1: Confirm clean working tree**

```
git status
git branch --show-current
```

Expected: clean tree.

- [ ] **Step 2: Create and switch to a feature branch off the current branch**

```
git checkout -b feat/trends-charts
```

Note: this work is on top of `cleanup/web-simplify-phase1` (current branch). If main has moved, rebase later or fork from main per user direction. Default: branch from current HEAD.

---

## Phase 1 — Data model: persist projected standings history

### Task 1: Add Redis helpers for projected_standings_history (TDD)

**Files:**
- Create: `tests/test_data/test_redis_store_projected_standings.py`
- Modify: `src/fantasy_baseball/data/redis_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_data/test_redis_store_projected_standings.py`:

```python
"""Tests for projected_standings_history helpers."""

from datetime import date

from fantasy_baseball.data import redis_store
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
)


def _projected(eff: date, teams: list[tuple[str, dict]]) -> ProjectedStandings:
    return ProjectedStandings(
        effective_date=eff,
        entries=[
            ProjectedStandingsEntry(team_name=name, stats=CategoryStats.from_dict(stats))
            for name, stats in teams
        ],
    )


PROJ_DAY_1 = _projected(
    date(2026, 4, 15),
    [
        (
            "Alpha",
            {
                "R": 880,
                "HR": 230,
                "RBI": 820,
                "SB": 110,
                "AVG": 0.265,
                "W": 75,
                "K": 1450,
                "SV": 60,
                "ERA": 3.55,
                "WHIP": 1.20,
            },
        ),
        (
            "Beta",
            {
                "R": 820,
                "HR": 200,
                "RBI": 780,
                "SB": 90,
                "AVG": 0.258,
                "W": 70,
                "K": 1380,
                "SV": 55,
                "ERA": 3.78,
                "WHIP": 1.25,
            },
        ),
    ],
)

PROJ_DAY_2 = _projected(
    date(2026, 4, 22),
    [
        (
            "Alpha",
            {
                "R": 890,
                "HR": 235,
                "RBI": 830,
                "SB": 112,
                "AVG": 0.266,
                "W": 76,
                "K": 1455,
                "SV": 62,
                "ERA": 3.50,
                "WHIP": 1.19,
            },
        ),
    ],
)


def test_write_and_read_single_day(fake_redis):
    redis_store.write_projected_standings_snapshot(fake_redis, PROJ_DAY_1)
    loaded = redis_store.get_projected_standings_day(fake_redis, "2026-04-15")
    assert loaded == PROJ_DAY_1


def test_overwrites_same_date(fake_redis):
    redis_store.write_projected_standings_snapshot(fake_redis, PROJ_DAY_1)
    same_date_new = _projected(
        date(2026, 4, 15),
        [("Alpha", {"R": 999})],
    )
    redis_store.write_projected_standings_snapshot(fake_redis, same_date_new)
    loaded = redis_store.get_projected_standings_day(fake_redis, "2026-04-15")
    assert loaded == same_date_new


def test_get_history_returns_all_dates(fake_redis):
    redis_store.write_projected_standings_snapshot(fake_redis, PROJ_DAY_1)
    redis_store.write_projected_standings_snapshot(fake_redis, PROJ_DAY_2)
    history = redis_store.get_projected_standings_history(fake_redis)
    assert set(history.keys()) == {"2026-04-15", "2026-04-22"}
    assert history["2026-04-22"] == PROJ_DAY_2


def test_get_history_empty(fake_redis):
    assert redis_store.get_projected_standings_history(fake_redis) == {}


def test_write_none_client_noop():
    redis_store.write_projected_standings_snapshot(None, PROJ_DAY_1)


def test_get_day_none_client_returns_none():
    assert redis_store.get_projected_standings_day(None, "2026-04-15") is None


def test_get_history_none_client_returns_empty():
    assert redis_store.get_projected_standings_history(None) == {}


def test_get_day_ignores_corrupt_json(fake_redis):
    fake_redis.hset(
        redis_store.PROJECTED_STANDINGS_HISTORY_KEY, "2026-04-15", "not json {{{"
    )
    assert redis_store.get_projected_standings_day(fake_redis, "2026-04-15") is None
```

- [ ] **Step 2: Run the tests, verify they fail**

```
pytest tests/test_data/test_redis_store_projected_standings.py -v
```

Expected: ImportError or AttributeError on `PROJECTED_STANDINGS_HISTORY_KEY` / `write_projected_standings_snapshot` / etc.

- [ ] **Step 3: Add the helpers in `redis_store.py`**

Open `src/fantasy_baseball/data/redis_store.py`. Add a `ProjectedStandings` import next to the existing `Standings` import:

```python
from fantasy_baseball.models.standings import ProjectedStandings, Standings
```

Then, immediately after the `get_standings_history` function (~line 454), append:

```python
PROJECTED_STANDINGS_HISTORY_KEY = "projected_standings_history"


def write_projected_standings_snapshot(client, projected: ProjectedStandings) -> None:
    """Write a ProjectedStandings snapshot keyed by its effective_date.

    Idempotent overwrite — same-day refreshes replace the previous
    snapshot (last-write-wins). No-op when ``client`` is None.
    """
    if client is None:
        return
    client.hset(
        PROJECTED_STANDINGS_HISTORY_KEY,
        projected.effective_date.isoformat(),
        json.dumps(projected.to_json()),
    )


def get_projected_standings_day(client, snapshot_date: str) -> ProjectedStandings | None:
    """Return the ProjectedStandings for one snapshot date, or None if missing/corrupt."""
    if client is None:
        return None
    raw = client.hget(PROJECTED_STANDINGS_HISTORY_KEY, snapshot_date)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return ProjectedStandings.from_json(data)


def get_projected_standings_history(client) -> dict[str, ProjectedStandings]:
    """Return the entire history as {snapshot_date: ProjectedStandings}.

    Corrupt JSON entries are silently skipped.
    """
    if client is None:
        return {}
    raw_map = client.hgetall(PROJECTED_STANDINGS_HISTORY_KEY)
    if not raw_map:
        return {}
    out: dict[str, ProjectedStandings] = {}
    for d, raw in raw_map.items():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out[d] = ProjectedStandings.from_json(data)
    return out
```

- [ ] **Step 4: Run the tests, verify they pass**

```
pytest tests/test_data/test_redis_store_projected_standings.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```
git add tests/test_data/test_redis_store_projected_standings.py src/fantasy_baseball/data/redis_store.py
git commit -m "feat(redis_store): add projected_standings_history helpers"
```

---

### Task 2: Sync the new hash to local SQLite

**Files:**
- Modify: `src/fantasy_baseball/data/kv_sync.py`
- Modify: `tests/test_data/test_kv_sync.py`

- [ ] **Step 1: Read the existing test file to understand the pattern**

```
cat tests/test_data/test_kv_sync.py | head -80
```

The existing tests verify `_HASH_KEYS` covers `STANDINGS_HISTORY_KEY` and `WEEKLY_ROSTERS_HISTORY_KEY`. The change is to add `PROJECTED_STANDINGS_HISTORY_KEY` to that frozenset.

- [ ] **Step 2: Add a failing test in `tests/test_data/test_kv_sync.py`**

Append:

```python
def test_sync_replicates_projected_standings_history(monkeypatch, tmp_path):
    """Projected standings history is hash-typed and must round-trip
    through sync_remote_to_local."""
    import json
    from fantasy_baseball.data import kv_store, kv_sync, redis_store

    monkeypatch.setenv("RENDER", "")

    remote = kv_store.SqliteKVStore(tmp_path / "remote.db")
    local = kv_store.SqliteKVStore(tmp_path / "local.db")

    remote.hset(
        redis_store.PROJECTED_STANDINGS_HISTORY_KEY,
        "2026-04-15",
        json.dumps({"effective_date": "2026-04-15", "teams": []}),
    )

    kv_sync.sync_remote_to_local(remote=remote, local=local)

    assert (
        local.hget(redis_store.PROJECTED_STANDINGS_HISTORY_KEY, "2026-04-15")
        == '{"effective_date": "2026-04-15", "teams": []}'
    )
```

- [ ] **Step 3: Run the test, verify it fails**

```
pytest tests/test_data/test_kv_sync.py::test_sync_replicates_projected_standings_history -v
```

Expected: FAIL — the field never gets copied because `PROJECTED_STANDINGS_HISTORY_KEY` is not in `_HASH_KEYS`.

- [ ] **Step 4: Add the new key to `_HASH_KEYS`**

In `src/fantasy_baseball/data/kv_sync.py`:

Replace the import block:

```python
from fantasy_baseball.data.redis_store import (
    STANDINGS_HISTORY_KEY,
    WEEKLY_ROSTERS_HISTORY_KEY,
)
```

with:

```python
from fantasy_baseball.data.redis_store import (
    PROJECTED_STANDINGS_HISTORY_KEY,
    STANDINGS_HISTORY_KEY,
    WEEKLY_ROSTERS_HISTORY_KEY,
)
```

Then update the `_HASH_KEYS` line:

```python
_HASH_KEYS: frozenset[str] = frozenset(
    {WEEKLY_ROSTERS_HISTORY_KEY, STANDINGS_HISTORY_KEY, PROJECTED_STANDINGS_HISTORY_KEY}
)
```

Update the docstring on `sync_remote_to_local` so the comment about "exactly two hash-typed keys" stays accurate. Find:

```
- The schema has exactly two hash-typed keys
  (``weekly_rosters_history``, ``standings_history``); everything else
  is a string.
```

Replace with:

```
- The schema has three hash-typed keys
  (``weekly_rosters_history``, ``standings_history``,
  ``projected_standings_history``); everything else is a string.
```

- [ ] **Step 5: Run the test, verify it passes**

```
pytest tests/test_data/test_kv_sync.py -v
```

Expected: all green (new test plus all existing).

- [ ] **Step 6: Commit**

```
git add src/fantasy_baseball/data/kv_sync.py tests/test_data/test_kv_sync.py
git commit -m "feat(kv_sync): replicate projected_standings_history"
```

---

### Task 3: Wire the refresh pipeline to write each snapshot

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`
- Modify: `tests/test_web/test_refresh_pipeline.py`

- [ ] **Step 1: Add a failing test in `tests/test_web/test_refresh_pipeline.py`**

Inside `class TestRefreshShape`, after `test_all_expected_cache_files_written`, add:

```python
def test_projected_standings_history_populated(self, configured_test_env, fake_redis):
    """Each refresh appends a snapshot to projected_standings_history."""
    from fantasy_baseball.data.redis_store import (
        PROJECTED_STANDINGS_HISTORY_KEY,
        get_projected_standings_history,
    )

    with patched_refresh_environment(fake_redis):
        refresh_pipeline.run_full_refresh()

    history = get_projected_standings_history(fake_redis)
    assert len(history) >= 1, "Expected at least one projected standings snapshot"
    snap_date, projected = next(iter(history.items()))
    assert len(projected.entries) == 12
    assert {e.team_name for e in projected.entries} == {f"Team {i:02d}" for i in range(1, 13)}
```

- [ ] **Step 2: Run the test, verify it fails**

```
pytest tests/test_web/test_refresh_pipeline.py::TestRefreshShape::test_projected_standings_history_populated -v
```

Expected: FAIL — `len(history) == 0`.

- [ ] **Step 3: Modify `_build_projected_standings`**

In `src/fantasy_baseball/web/refresh_pipeline.py`, find `_build_projected_standings`. After the `write_cache(CacheKey.PROJECTIONS, ...)` call (~line 597), and before the `write_cache(CacheKey.STANDINGS_BREAKDOWN, ...)` call, insert:

```python
        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import write_projected_standings_snapshot

        write_projected_standings_snapshot(get_kv(), self.projected_standings)
```

(Match the existing style: imports inside the method body — `_build_projected_standings` already does this for `build_team_sds` etc.)

- [ ] **Step 4: Run the test, verify it passes**

```
pytest tests/test_web/test_refresh_pipeline.py -v
```

Expected: new test green; all existing tests still green.

- [ ] **Step 5: Commit**

```
git add src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/test_refresh_pipeline.py
git commit -m "feat(refresh): append to projected_standings_history each run"
```

---

## Phase 2 — Backfill historical projections

### Task 4: Write the backfill script (TDD)

**Files:**
- Create: `tests/test_scripts/test_backfill_projected_standings_history.py`
- Create: `scripts/backfill_projected_standings_history.py`

- [ ] **Step 1: Write a failing integration test**

Create `tests/test_scripts/test_backfill_projected_standings_history.py`:

```python
"""End-to-end test for backfill_projected_standings_history.

Seeds fakeredis with weekly_rosters_history + ros_projections, runs the
script's main entry point, asserts projected_standings_history is
populated with one snapshot per roster date.
"""

import json
from datetime import date
from unittest.mock import patch

import pytest

from fantasy_baseball.data import redis_store


@pytest.fixture
def seeded_redis(fake_redis, monkeypatch):
    """Populate fakeredis with 2 dates × 2 teams of roster data and
    minimal blended/ROS projections."""
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key

    # Two roster dates; both teams keep the same single-player roster
    # so we can predict the resulting projected standings exactly.
    rosters = {
        "2026-04-01": [
            {
                "team": "Alpha",
                "player_name": "Player One",
                "slot": "OF",
                "positions": "OF",
                "status": "",
                "yahoo_id": "p1",
            },
            {
                "team": "Beta",
                "player_name": "Player Two",
                "slot": "SP",
                "positions": "SP",
                "status": "",
                "yahoo_id": "p2",
            },
        ],
        "2026-04-15": [
            {
                "team": "Alpha",
                "player_name": "Player One",
                "slot": "OF",
                "positions": "OF",
                "status": "",
                "yahoo_id": "p1",
            },
            {
                "team": "Beta",
                "player_name": "Player Two",
                "slot": "SP",
                "positions": "SP",
                "status": "",
                "yahoo_id": "p2",
            },
        ],
    }
    for snap_date, entries in rosters.items():
        fake_redis.hset(
            redis_store.WEEKLY_ROSTERS_HISTORY_KEY,
            snap_date,
            json.dumps(entries),
        )

    # Two corresponding standings snapshots (the script doesn't read
    # standings, but League.from_redis joins on team_key, so include
    # them so team_keys are non-empty).
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-01",
        json.dumps(
            {
                "effective_date": "2026-04-01",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {
                            "R": 0,
                            "HR": 0,
                            "RBI": 0,
                            "SB": 0,
                            "AVG": 0,
                            "W": 0,
                            "K": 0,
                            "SV": 0,
                            "ERA": 99,
                            "WHIP": 99,
                        },
                        "yahoo_points_for": None,
                        "extras": {},
                    },
                    {
                        "name": "Beta",
                        "team_key": "T.2",
                        "rank": 2,
                        "stats": {
                            "R": 0,
                            "HR": 0,
                            "RBI": 0,
                            "SB": 0,
                            "AVG": 0,
                            "W": 0,
                            "K": 0,
                            "SV": 0,
                            "ERA": 99,
                            "WHIP": 99,
                        },
                        "yahoo_points_for": None,
                        "extras": {},
                    },
                ],
            }
        ),
    )

    # Minimal blended and ROS projection caches the script reads.
    hitter_row = {
        "name": "Player One",
        "fg_id": "fg1",
        "team": "TBD",
        "positions": "OF",
        "ab": 500,
        "pa": 580,
        "r": 80,
        "hr": 25,
        "rbi": 80,
        "sb": 8,
        "h": 145,
        "avg": 0.290,
        "player_type": "hitter",
    }
    pitcher_row = {
        "name": "Player Two",
        "fg_id": "fg2",
        "team": "TBD",
        "positions": "SP",
        "w": 12,
        "k": 180,
        "sv": 0,
        "ip": 180.0,
        "er": 70,
        "bb": 50,
        "h_allowed": 160,
        "era": 3.50,
        "whip": 1.17,
        "player_type": "pitcher",
    }

    # blended_projections:hitters / pitchers (used as preseason fallback)
    fake_redis.set("blended_projections:hitters", json.dumps([hitter_row]))
    fake_redis.set("blended_projections:pitchers", json.dumps([pitcher_row]))

    # cache:ros_projections (used as ROS source)
    fake_redis.set(
        redis_key(CacheKey.ROS_PROJECTIONS),
        json.dumps({"hitters": [hitter_row], "pitchers": [pitcher_row]}),
    )

    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")

    return fake_redis


def test_backfill_writes_snapshot_per_date(seeded_redis):
    from scripts import backfill_projected_standings_history as backfill

    with patch(
        "fantasy_baseball.data.kv_store.get_kv", return_value=seeded_redis
    ):
        backfill.main(season_year=2026)

    history = redis_store.get_projected_standings_history(seeded_redis)
    assert set(history.keys()) == {"2026-04-01", "2026-04-15"}
    for snap_date, projected in history.items():
        names = {e.team_name for e in projected.entries}
        assert names == {"Alpha", "Beta"}
        assert projected.effective_date.isoformat() == snap_date


def test_backfill_is_idempotent(seeded_redis):
    from scripts import backfill_projected_standings_history as backfill

    with patch(
        "fantasy_baseball.data.kv_store.get_kv", return_value=seeded_redis
    ):
        backfill.main(season_year=2026)
        first = redis_store.get_projected_standings_history(seeded_redis)
        backfill.main(season_year=2026)
        second = redis_store.get_projected_standings_history(seeded_redis)

    assert first.keys() == second.keys()
    for k in first:
        assert first[k] == second[k]
```

- [ ] **Step 2: Run the test, verify it fails**

```
pytest tests/test_scripts/test_backfill_projected_standings_history.py -v
```

Expected: ImportError on `scripts.backfill_projected_standings_history`.

- [ ] **Step 3: Make `scripts/` an importable package**

Verify `scripts/__init__.py` exists. If not, create an empty file:

```
ls scripts/__init__.py
```

If missing:

```
touch scripts/__init__.py
```

- [ ] **Step 4: Implement the backfill script**

Create `scripts/backfill_projected_standings_history.py`:

```python
#!/usr/bin/env python3
"""One-time backfill of `projected_standings_history`.

For every roster snapshot date in `weekly_rosters_history`, build a
`ProjectedStandings` using TODAY'S per-player full-season projection
applied to the historical roster. Writes one snapshot per date to
`projected_standings_history`.

Math note: today's per-player full_season_projection (= ROS + YTD) is
constant in expectation under a constant-rate assumption, so
"extrapolating today's ROS backwards" is equivalent to applying today's
full_season unchanged. Going forward, real per-day snapshots from the
refresh pipeline will carry projection-drift signal that this backfill
cannot.

Idempotent — safe to re-run; same-date entries are overwritten.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config  # noqa: E402
from fantasy_baseball.data.kv_store import get_kv  # noqa: E402
from fantasy_baseball.data.projections import hydrate_roster_entries  # noqa: E402
from fantasy_baseball.data.redis_store import (  # noqa: E402
    write_projected_standings_snapshot,
)
from fantasy_baseball.models.league import League  # noqa: E402
from fantasy_baseball.models.standings import ProjectedStandings  # noqa: E402
from fantasy_baseball.web.season_routes import _load_projections  # noqa: E402

logger = logging.getLogger(__name__)


def main(season_year: int | None = None) -> int:
    """Run the backfill. Returns 0 on success.

    ``season_year`` overrides the config value (test seam).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    year = season_year if season_year is not None else config.season_year

    client = get_kv()

    logger.info("Loading projections...")
    hitters_proj, pitchers_proj, ros_h, ros_p = _load_projections()
    have_ros = not ros_h.empty and not ros_p.empty
    full_h = ros_h if have_ros else None
    full_p = ros_p if have_ros else None

    logger.info("Loading roster + standings history from Redis...")
    league = League.from_redis(year)

    # Group every team's roster history by snapshot date:
    #   rosters_by_date[snap_date_iso][team_name] = Roster
    rosters_by_date: dict[str, dict[str, list]] = {}
    for team in league.teams:
        for roster in team.rosters:
            snap_iso = roster.effective_date.isoformat()
            rosters_by_date.setdefault(snap_iso, {})[team.name] = roster

    if not rosters_by_date:
        logger.warning("No roster history found for season %s — nothing to backfill.", year)
        return 0

    logger.info("Building %d projected snapshots...", len(rosters_by_date))
    for snap_iso in sorted(rosters_by_date):
        team_rosters: dict[str, list] = {}
        for team_name, roster in rosters_by_date[snap_iso].items():
            hydrated = hydrate_roster_entries(
                roster,
                hitters_proj,
                pitchers_proj,
                full_hitters_proj=full_h,
                full_pitchers_proj=full_p,
                context=f"backfill:{snap_iso}:{team_name}",
            )
            if hydrated:
                team_rosters[team_name] = hydrated

        if not team_rosters:
            logger.info("  %s — no rosters resolved, skipping", snap_iso)
            continue

        from datetime import date

        projected = ProjectedStandings.from_rosters(
            team_rosters, effective_date=date.fromisoformat(snap_iso)
        )
        write_projected_standings_snapshot(client, projected)
        logger.info("  %s — wrote %d teams", snap_iso, len(projected.entries))

    logger.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="Override config season_year (default: read from league.yaml).",
    )
    args = parser.parse_args()
    sys.exit(main(season_year=args.season_year))
```

- [ ] **Step 5: Run the tests**

```
pytest tests/test_scripts/test_backfill_projected_standings_history.py -v
```

Expected: both tests green. If `_load_projections` fails because the test fixture's projection rows don't carry every required column, inspect the failure — the seeded `hitter_row`/`pitcher_row` already include the columns `_load_projections` and the matcher need (per `tests/test_web/_refresh_fixture.py` which uses the same shape). The `_name_norm` column gets added by `_load_projections`, not the seed.

- [ ] **Step 6: Run the broader test suite to check nothing regressed**

```
pytest tests/test_data/ tests/test_web/ tests/test_scripts/ -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```
git add scripts/__init__.py scripts/backfill_projected_standings_history.py tests/test_scripts/test_backfill_projected_standings_history.py
git commit -m "feat(scripts): backfill projected_standings_history from roster history"
```

(If `scripts/__init__.py` already existed and you didn't create it, drop it from the `git add`.)

---

## Phase 3 — API: serve the trends payload

### Task 5: Build the trends series builder (TDD)

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`
- Modify: `tests/test_web/test_season_data.py`

The builder is the read-side counterpart of the history hashes. It runs `score_roto` over each snapshot, prefers Yahoo authority for actuals, and transposes to per-team time series.

- [ ] **Step 1: Write failing tests in `tests/test_web/test_season_data.py`**

Append to `tests/test_web/test_season_data.py`:

```python
def test_build_trends_series_empty_history(fake_redis):
    """Empty history hashes → empty payload but valid shape."""
    from fantasy_baseball.web.season_data import build_trends_series

    out = build_trends_series(fake_redis, user_team="Alpha")
    assert out["user_team"] == "Alpha"
    assert "categories" in out and out["categories"]
    assert out["actual"] == {"dates": [], "teams": {}}
    assert out["projected"] == {"dates": [], "teams": {}}


def test_build_trends_series_actual_only(fake_redis):
    """Standings populated, projected empty — actual fills, projected stays empty."""
    import json
    from fantasy_baseball.data import redis_store

    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-15",
        json.dumps(
            {
                "effective_date": "2026-04-15",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {
                            "R": 45,
                            "HR": 12,
                            "RBI": 40,
                            "SB": 8,
                            "AVG": 0.268,
                            "W": 3,
                            "K": 85,
                            "SV": 4,
                            "ERA": 3.21,
                            "WHIP": 1.14,
                        },
                        "yahoo_points_for": 78.5,
                        "extras": {},
                    },
                    {
                        "name": "Beta",
                        "team_key": "T.2",
                        "rank": 2,
                        "stats": {
                            "R": 38,
                            "HR": 9,
                            "RBI": 32,
                            "SB": 6,
                            "AVG": 0.255,
                            "W": 2,
                            "K": 72,
                            "SV": 3,
                            "ERA": 3.85,
                            "WHIP": 1.22,
                        },
                        "yahoo_points_for": 60.0,
                        "extras": {},
                    },
                ],
            }
        ),
    )

    from fantasy_baseball.web.season_data import build_trends_series

    out = build_trends_series(fake_redis, user_team="Alpha")
    assert out["actual"]["dates"] == ["2026-04-15"]
    assert set(out["actual"]["teams"].keys()) == {"Alpha", "Beta"}
    alpha = out["actual"]["teams"]["Alpha"]
    assert alpha["roto_points"] == [78.5]  # Yahoo authority preferred
    assert alpha["stats"]["R"] == [45]
    assert alpha["stats"]["WHIP"] == [1.14]
    assert out["projected"] == {"dates": [], "teams": {}}


def test_build_trends_series_gap_handling(fake_redis):
    """Team appears on day 1 but not day 2 → null on day 2."""
    import json
    from fantasy_baseball.data import redis_store

    base_stats = {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AVG": 0.0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 99.0,
        "WHIP": 99.0,
    }
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-01",
        json.dumps(
            {
                "effective_date": "2026-04-01",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {**base_stats, "R": 10},
                        "yahoo_points_for": 5.0,
                        "extras": {},
                    },
                    {
                        "name": "Beta",
                        "team_key": "T.2",
                        "rank": 2,
                        "stats": {**base_stats, "R": 5},
                        "yahoo_points_for": 4.0,
                        "extras": {},
                    },
                ],
            }
        ),
    )
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-02",
        json.dumps(
            {
                "effective_date": "2026-04-02",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {**base_stats, "R": 20},
                        "yahoo_points_for": 6.0,
                        "extras": {},
                    },
                ],
            }
        ),
    )

    from fantasy_baseball.web.season_data import build_trends_series

    out = build_trends_series(fake_redis, user_team="Alpha")
    assert out["actual"]["dates"] == ["2026-04-01", "2026-04-02"]
    beta = out["actual"]["teams"]["Beta"]
    assert beta["roto_points"][0] == 4.0
    assert beta["roto_points"][1] is None
    assert beta["stats"]["R"][0] == 5
    assert beta["stats"]["R"][1] is None


def test_build_trends_series_projected_uses_score_roto(fake_redis):
    """Projected chart has no Yahoo authority — totals come from score_roto."""
    import json
    from fantasy_baseball.data import redis_store

    base_stats = {
        "R": 800,
        "HR": 200,
        "RBI": 750,
        "SB": 80,
        "AVG": 0.260,
        "W": 70,
        "K": 1400,
        "SV": 50,
        "ERA": 3.80,
        "WHIP": 1.25,
    }
    fake_redis.hset(
        redis_store.PROJECTED_STANDINGS_HISTORY_KEY,
        "2026-04-15",
        json.dumps(
            {
                "effective_date": "2026-04-15",
                "teams": [
                    {"name": "Alpha", "stats": {**base_stats, "R": 880}},
                    {"name": "Beta", "stats": {**base_stats, "R": 820}},
                ],
            }
        ),
    )

    from fantasy_baseball.web.season_data import build_trends_series

    out = build_trends_series(fake_redis, user_team="Alpha")
    assert out["projected"]["dates"] == ["2026-04-15"]
    # With 2 teams, score_roto over 10 categories: each category awards
    # 2 to the leader and 1 to the trailer (avg-tie scoring), so the
    # totals should be in the right order — Alpha (better R) ahead of Beta.
    alpha_total = out["projected"]["teams"]["Alpha"]["roto_points"][0]
    beta_total = out["projected"]["teams"]["Beta"]["roto_points"][0]
    assert alpha_total > beta_total
    assert out["projected"]["teams"]["Alpha"]["stats"]["R"] == [880]
```

- [ ] **Step 2: Run the tests, verify failure**

```
pytest tests/test_web/test_season_data.py -v -k trends
```

Expected: ImportError on `build_trends_series`.

- [ ] **Step 3: Implement `build_trends_series` in `season_data.py`**

Open `src/fantasy_baseball/web/season_data.py`. Add at the bottom of the file:

```python
def build_trends_series(client, *, user_team: str) -> dict:
    """Read both history hashes and return the /api/trends/series payload.

    Shape:
        {
          "user_team": str,
          "categories": list[str],   # ["R", "HR", ..., "WHIP"]
          "actual":    {"dates": [...], "teams": {name: {"roto_points": [...], "stats": {cat: [...]}}}},
          "projected": {"dates": [...], "teams": {name: {"roto_points": [...], "stats": {cat: [...]}}}},
        }

    Per-snapshot per-category totals come from ``score_roto``. For the
    actual series we prefer Yahoo's ``yahoo_points_for`` total when
    every entry on that snapshot has it, matching the /standings page.
    Teams that appear in some snapshots but not others get ``None`` on
    the missing dates so Chart.js renders a gap.
    """
    from typing import cast
    from fantasy_baseball.data.redis_store import (
        get_projected_standings_history,
        get_standings_history,
    )

    categories = [c.value for c in ALL_CATEGORIES]

    actual_history = get_standings_history(client)
    projected_history = get_projected_standings_history(client)

    def _emit_actual() -> dict:
        if not actual_history:
            return {"dates": [], "teams": {}}
        dates = sorted(actual_history.keys())
        all_team_names: set[str] = set()
        for d in dates:
            for entry in actual_history[d].entries:
                all_team_names.add(entry.team_name)

        teams: dict[str, dict] = {
            name: {
                "roto_points": [],
                "stats": {cat: [] for cat in categories},
            }
            for name in all_team_names
        }
        for d in dates:
            standings = actual_history[d]
            roto = score_roto(cast("Any", standings))
            present = {e.team_name: e for e in standings.entries}
            yahoo_authoritative = bool(present) and all(
                e.yahoo_points_for is not None for e in present.values()
            )
            for name in all_team_names:
                entry = present.get(name)
                if entry is None:
                    teams[name]["roto_points"].append(None)
                    for cat in categories:
                        teams[name]["stats"][cat].append(None)
                    continue
                if yahoo_authoritative:
                    teams[name]["roto_points"].append(float(entry.yahoo_points_for))  # type: ignore[arg-type]
                else:
                    teams[name]["roto_points"].append(float(roto[name].total))
                stats_dict = entry.stats.to_dict()
                for cat in categories:
                    teams[name]["stats"][cat].append(stats_dict[cat])
        return {"dates": dates, "teams": teams}

    def _emit_projected() -> dict:
        if not projected_history:
            return {"dates": [], "teams": {}}
        dates = sorted(projected_history.keys())
        all_team_names = set()
        for d in dates:
            for entry in projected_history[d].entries:
                all_team_names.add(entry.team_name)

        teams = {
            name: {
                "roto_points": [],
                "stats": {cat: [] for cat in categories},
            }
            for name in all_team_names
        }
        for d in dates:
            projected = projected_history[d]
            roto = score_roto(cast("Any", projected))
            present = {e.team_name: e for e in projected.entries}
            for name in all_team_names:
                entry = present.get(name)
                if entry is None:
                    teams[name]["roto_points"].append(None)
                    for cat in categories:
                        teams[name]["stats"][cat].append(None)
                    continue
                teams[name]["roto_points"].append(float(roto[name].total))
                stats_dict = entry.stats.to_dict()
                for cat in categories:
                    teams[name]["stats"][cat].append(stats_dict[cat])
        return {"dates": dates, "teams": teams}

    return {
        "user_team": user_team,
        "categories": categories,
        "actual": _emit_actual(),
        "projected": _emit_projected(),
    }
```

Note: the `cast` import is already in `season_data.py` at the top (used by `format_standings_for_display`). If not, add `from typing import Any, cast` at the top.

- [ ] **Step 4: Run the tests, verify they pass**

```
pytest tests/test_web/test_season_data.py -v -k trends
```

Expected: all four green.

- [ ] **Step 5: Commit**

```
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat(season_data): build_trends_series — transpose history into time series"
```

---

### Task 6: Wire `/trends` and `/api/trends/series` routes

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Create: `tests/test_web/test_trends_route.py`

- [ ] **Step 1: Write failing route tests**

Create `tests/test_web/test_trends_route.py`:

```python
"""Coverage for /trends and /api/trends/series."""

import json

import pytest

from fantasy_baseball.data import redis_store


@pytest.fixture
def app(monkeypatch, fake_redis):
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")

    from fantasy_baseball.data import kv_store

    monkeypatch.setattr(kv_store, "get_kv", lambda: fake_redis)

    # Also patch get_kv at the import sites used by season_data and
    # season_routes (functions imported into module namespaces).
    from fantasy_baseball.web import season_data, season_routes  # noqa: F401

    from fantasy_baseball.web.season_app import create_app

    application = create_app()
    application.config["TESTING"] = True
    application.config["SECRET_KEY"] = "test"
    return application


def test_trends_page_renders(app):
    client = app.test_client()
    resp = client.get("/trends")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "chart-actual" in body
    assert "chart-projected" in body


def test_api_trends_series_empty(app):
    client = app.test_client()
    resp = client.get("/api/trends/series")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["user_team"]
    assert payload["categories"] == [
        "R",
        "HR",
        "RBI",
        "SB",
        "AVG",
        "W",
        "K",
        "SV",
        "ERA",
        "WHIP",
    ]
    assert payload["actual"] == {"dates": [], "teams": {}}
    assert payload["projected"] == {"dates": [], "teams": {}}


def test_api_trends_series_with_data(app, fake_redis):
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-15",
        json.dumps(
            {
                "effective_date": "2026-04-15",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {
                            "R": 45,
                            "HR": 12,
                            "RBI": 40,
                            "SB": 8,
                            "AVG": 0.268,
                            "W": 3,
                            "K": 85,
                            "SV": 4,
                            "ERA": 3.21,
                            "WHIP": 1.14,
                        },
                        "yahoo_points_for": 78.5,
                        "extras": {},
                    }
                ],
            }
        ),
    )
    client = app.test_client()
    payload = client.get("/api/trends/series").get_json()
    assert payload["actual"]["dates"] == ["2026-04-15"]
    assert "Alpha" in payload["actual"]["teams"]
    assert payload["actual"]["teams"]["Alpha"]["roto_points"] == [78.5]
```

- [ ] **Step 2: Run the tests, verify they fail**

```
pytest tests/test_web/test_trends_route.py -v
```

Expected: 404 on `/trends` and `/api/trends/series` (routes not registered).

- [ ] **Step 3: Add the route handlers**

In `src/fantasy_baseball/web/season_routes.py`, inside `register_routes(app)`, add two routes anywhere among the existing ones (suggested location: just after the `/standings` block):

```python
    @app.route("/trends")
    def trends():
        meta = read_meta()
        return render_template(
            "season/trends.html",
            meta=meta,
            active_page="trends",
        )

    @app.route("/api/trends/series")
    def api_trends_series():
        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.web.season_data import build_trends_series

        config = _load_config()
        return jsonify(build_trends_series(get_kv(), user_team=config.team_name))
```

- [ ] **Step 4: Run tests, verify they pass**

```
pytest tests/test_web/test_trends_route.py -v
```

Note: the page-render test expects `season/trends.html`. The template is added in Phase 4 — so this test will still 500 with `TemplateNotFound`. To unblock Phase 3, create a stub template now and replace it in Phase 4.

- [ ] **Step 5: Create a minimal stub template so the route renders**

Create `src/fantasy_baseball/web/templates/season/trends.html`:

```html
{% extends "season/base.html" %}
{% block title %}Trends{% endblock %}
{% block content %}
<h1>Trends</h1>
<canvas id="chart-actual"></canvas>
<canvas id="chart-projected"></canvas>
{% endblock %}
```

This stub satisfies `test_trends_page_renders` (it asserts on `chart-actual` / `chart-projected` IDs). Phase 4 replaces it with the real template.

- [ ] **Step 6: Run tests again**

```
pytest tests/test_web/test_trends_route.py -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```
git add src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/templates/season/trends.html tests/test_web/test_trends_route.py
git commit -m "feat(routes): /trends page and /api/trends/series endpoint"
```

---

## Phase 4 — Frontend: charts and tabs

### Task 7: Real `trends.html` template + sidebar nav link

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/trends.html`
- Modify: `src/fantasy_baseball/web/templates/season/base.html`

- [ ] **Step 1: Replace the stub template with the real one**

Overwrite `src/fantasy_baseball/web/templates/season/trends.html`:

```html
{% extends "season/base.html" %}
{% block title %}Trends - Season Dashboard{% endblock %}

{% block head_extra %}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
.trends-section { margin-bottom: 32px; }
.trends-section h2 { margin: 0 0 8px; }
.tab-strip {
  display: flex;
  gap: 4px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.tab-strip button {
  background: var(--surface);
  color: var(--text-secondary);
  border: 1px solid var(--border);
  padding: 4px 10px;
  font-size: 13px;
  cursor: pointer;
  border-radius: 4px;
}
.tab-strip button:hover { background: var(--surface-hover); }
.tab-strip button.active {
  background: var(--accent);
  color: var(--text-on-accent, #fff);
  border-color: var(--accent);
}
.chart-wrapper {
  position: relative;
  height: 360px;
}
</style>
{% endblock %}

{% block content %}
<h1>Trends</h1>
<p style="color: var(--text-secondary); margin-top: 0;">
  Roto and per-stat movement across the season. Click a team in the legend to hide it; hover a line to focus it.
</p>

<section class="trends-section">
  <h2>Actual Standings</h2>
  <nav class="tab-strip" data-target="actual">
    <button class="active" data-tab="roto">Roto Points</button>
    <button data-tab="R">R</button>
    <button data-tab="HR">HR</button>
    <button data-tab="RBI">RBI</button>
    <button data-tab="SB">SB</button>
    <button data-tab="AVG">AVG</button>
    <button data-tab="W">W</button>
    <button data-tab="K">K</button>
    <button data-tab="SV">SV</button>
    <button data-tab="ERA">ERA</button>
    <button data-tab="WHIP">WHIP</button>
  </nav>
  <div class="chart-wrapper"><canvas id="chart-actual"></canvas></div>
</section>

<section class="trends-section">
  <h2>Projected ERoto</h2>
  <nav class="tab-strip" data-target="projected">
    <button class="active" data-tab="roto">Roto Points</button>
    <button data-tab="R">R</button>
    <button data-tab="HR">HR</button>
    <button data-tab="RBI">RBI</button>
    <button data-tab="SB">SB</button>
    <button data-tab="AVG">AVG</button>
    <button data-tab="W">W</button>
    <button data-tab="K">K</button>
    <button data-tab="SV">SV</button>
    <button data-tab="ERA">ERA</button>
    <button data-tab="WHIP">WHIP</button>
  </nav>
  <div class="chart-wrapper"><canvas id="chart-projected"></canvas></div>
</section>

<script src="{{ url_for('static', filename='season_trends.js') }}"></script>
{% endblock %}
```

- [ ] **Step 2: Add the "Trends" link to the sidebar in `base.html`**

Open `src/fantasy_baseball/web/templates/season/base.html`. Find the `Standings` link block:

```html
            <a href="{{ url_for('standings') }}"
               class="nav-link {% if active_page == 'standings' %}active{% endif %}">
                Standings
            </a>
```

Insert immediately after it:

```html
            <a href="{{ url_for('trends') }}"
               class="nav-link {% if active_page == 'trends' %}active{% endif %}">
                Trends
            </a>
```

- [ ] **Step 3: Run the existing route test to confirm the template still renders**

```
pytest tests/test_web/test_trends_route.py -v
```

Expected: green (the test asserts on `chart-actual`/`chart-projected` which the new template still has).

- [ ] **Step 4: Commit**

```
git add src/fantasy_baseball/web/templates/season/trends.html src/fantasy_baseball/web/templates/season/base.html
git commit -m "feat(trends): page template + sidebar link"
```

---

### Task 8: Trends JS — fetch, render, tab switching, hover-highlight

**Files:**
- Create: `src/fantasy_baseball/web/static/season_trends.js`

- [ ] **Step 1: Create the JS module**

Create `src/fantasy_baseball/web/static/season_trends.js`:

```javascript
// /trends — Chart.js line graphs for actual standings + projected ERoto.
//
// Loads /api/trends/series once, builds two charts, and handles tab
// switching (swap dataset.data, then chart.update()), hover-to-highlight
// (dim non-hovered datasets), and click-to-toggle (Chart.js legend
// default).

(function () {
  // 12-color qualitative palette; user team gets its own bold color.
  const PALETTE = [
    "#4e79a7", "#f28e2c", "#76b041", "#bab0ac",
    "#59a14f", "#edc949", "#af7aa1", "#ff9da7",
    "#9c755f", "#3a9da3", "#86bc4f", "#b07aa1",
  ];
  const USER_COLOR = "#e15759";

  let actualChart = null;
  let projectedChart = null;
  let payload = null;

  function colorForTeam(name, userTeam, sortedNames) {
    if (name === userTeam) return USER_COLOR;
    const otherNames = sortedNames.filter((n) => n !== userTeam);
    const i = otherNames.indexOf(name);
    return PALETTE[i % PALETTE.length];
  }

  function buildDatasets(seriesTeams, dates, userTeam, tab) {
    const sortedNames = Object.keys(seriesTeams).sort();
    return sortedNames.map((name) => {
      const series = seriesTeams[name];
      const data = tab === "roto" ? series.roto_points : (series.stats[tab] || []);
      const color = colorForTeam(name, userTeam, sortedNames);
      return {
        label: name,
        data: data.slice(),
        borderColor: color,
        backgroundColor: color,
        borderWidth: name === userTeam ? 4 : 2,
        pointRadius: 2,
        pointHoverRadius: 5,
        spanGaps: false,
        tension: 0.2,
      };
    });
  }

  function buildChart(canvasId, dates, datasets) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    return new Chart(ctx, {
      type: "line",
      data: { labels: dates, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        plugins: {
          legend: { position: "right", onHover: undefined },
          tooltip: { mode: "nearest", intersect: false },
        },
        scales: {
          y: { beginAtZero: false },
          x: { ticks: { autoSkip: true, maxTicksLimit: 10 } },
        },
        onHover: (evt, activeEls, chart) => {
          if (!activeEls || activeEls.length === 0) {
            resetAlpha(chart);
            return;
          }
          const focused = activeEls[0].datasetIndex;
          dimOthers(chart, focused);
        },
      },
    });
  }

  function withAlpha(hex, alpha) {
    // hex like "#abcdef" → "rgba(r,g,b,a)"
    const m = /^#?([a-f\d]{6})$/i.exec(hex);
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    const r = (n >> 16) & 255;
    const g = (n >> 8) & 255;
    const b = n & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function originalColor(ds) {
    return ds._origColor || ds.borderColor;
  }

  function dimOthers(chart, focusedIdx) {
    chart.data.datasets.forEach((ds, i) => {
      if (!ds._origColor) ds._origColor = ds.borderColor;
      const baseColor = ds._origColor;
      if (i === focusedIdx) {
        ds.borderColor = baseColor;
        ds.backgroundColor = baseColor;
      } else {
        ds.borderColor = withAlpha(baseColor, 0.15);
        ds.backgroundColor = withAlpha(baseColor, 0.15);
      }
    });
    chart.update("none");
  }

  function resetAlpha(chart) {
    chart.data.datasets.forEach((ds) => {
      if (ds._origColor) {
        ds.borderColor = ds._origColor;
        ds.backgroundColor = ds._origColor;
      }
    });
    chart.update("none");
  }

  function applyTab(chart, target, tab) {
    const series = payload[target];
    chart.data.datasets.forEach((ds) => {
      delete ds._origColor;
      const team = series.teams[ds.label];
      if (!team) {
        ds.data = [];
        return;
      }
      ds.data = (tab === "roto" ? team.roto_points : team.stats[tab] || []).slice();
    });
    resetAlpha(chart);
    chart.update();
  }

  function wireTabs(navSelector, chart, target) {
    const nav = document.querySelector(navSelector);
    if (!nav) return;
    nav.addEventListener("click", (evt) => {
      const btn = evt.target.closest("button[data-tab]");
      if (!btn) return;
      nav.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyTab(chart, target, btn.dataset.tab);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    fetch("/api/trends/series")
      .then((r) => r.json())
      .then((data) => {
        payload = data;
        const userTeam = data.user_team;

        const actualDatasets = buildDatasets(
          data.actual.teams, data.actual.dates, userTeam, "roto"
        );
        actualChart = buildChart("chart-actual", data.actual.dates, actualDatasets);

        const projectedDatasets = buildDatasets(
          data.projected.teams, data.projected.dates, userTeam, "roto"
        );
        projectedChart = buildChart(
          "chart-projected", data.projected.dates, projectedDatasets
        );

        wireTabs('.tab-strip[data-target="actual"]', actualChart, "actual");
        wireTabs('.tab-strip[data-target="projected"]', projectedChart, "projected");
      });
  });
})();
```

- [ ] **Step 2: Manually verify in the browser**

```
python scripts/run_season_dashboard.py
```

Open http://localhost:5050/trends and check:
- Both charts load with up to 12 lines.
- "Trends" appears in the sidebar; clicking it activates the page.
- Click each tab — y-axis should swap.
- Hover any line — others dim. Mouseout — they restore.
- Click a team name in the legend — that line hides; click again — it returns.
- Your team line is visually thicker than the rest.

Note: if the local KV has no projected_standings_history yet, the projected chart will be empty until either Task 4's backfill is run or the next refresh fires. That's expected.

- [ ] **Step 3: Commit**

```
git add src/fantasy_baseball/web/static/season_trends.js
git commit -m "feat(trends): chart.js renderer with tab switching and hover focus"
```

---

## Phase 5 — Final verification

### Task 9: Run all the gates

**Files:** none

- [ ] **Step 1: Full test suite**

```
pytest -v
```

Expected: all green.

- [ ] **Step 2: Lint**

```
ruff check .
```

Expected: zero violations.

- [ ] **Step 3: Format check**

```
ruff format --check .
```

Expected: clean. If anything drifted: `ruff format .` then re-check.

- [ ] **Step 4: Dead code check**

```
vulture src tests
```

Expected: no NEW findings introduced by this branch. Pre-existing findings unrelated to this work are acceptable.

- [ ] **Step 5: Type check**

Check `pyproject.toml`'s `[tool.mypy].files` list. Each file modified by this branch that's covered there must pass mypy:

```
mypy src/fantasy_baseball/data/redis_store.py src/fantasy_baseball/data/kv_sync.py src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/refresh_pipeline.py
```

Expected: zero errors. If any file isn't in `[tool.mypy].files`, drop it from the command.

- [ ] **Step 6: Run the live refresh pipeline locally to confirm the new write happens for real**

(Per CLAUDE.md memory: "Run refresh before merge — exercise `run_full_refresh` locally before merging refresh-path changes.")

```
python scripts/run_season_dashboard.py
```

In the dashboard, click **Refresh Data**, watch it complete, then in a Python REPL:

```python
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.data.redis_store import get_projected_standings_history
client = get_kv()
hist = get_projected_standings_history(client)
print(sorted(hist.keys()))
```

Expected: at least one snapshot key for today's date.

- [ ] **Step 7: Run the backfill against real data and verify**

```
python scripts/backfill_projected_standings_history.py
```

Re-run the REPL check above; the keys list should now span historical dates back to season start.

- [ ] **Step 8: Browser smoke**

Open http://localhost:5050/trends, confirm both charts populate (the projected chart should now show the full historical range thanks to the backfill).

---

## Self-review notes (covered during plan-writing)

Spec coverage:
- ✓ New hash schema → Task 1
- ✓ Redis helpers (write/get_day/get_history) → Task 1
- ✓ kv_sync inclusion → Task 2
- ✓ Refresh pipeline write → Task 3
- ✓ Backfill script (matches spec's League.from_redis approach) → Task 4
- ✓ /trends page route → Task 6
- ✓ /api/trends/series API + payload shape → Tasks 5–6
- ✓ Yahoo authority preference for actuals → Task 5
- ✓ score_roto for projected → Task 5
- ✓ Gap handling (null on missing dates) → Task 5
- ✓ Template + sidebar nav link → Tasks 6 (stub), 7 (real)
- ✓ Chart.js CDN, tabs, hover-highlight, click-toggle → Task 8
- ✓ User-team distinct color + thicker line → Task 8
- ✓ Tests at every layer → Tasks 1–6
- ✓ No frontend tests (per spec; manual browser check) → Task 8 step 2, Task 9 step 8
- ✓ Phasing constraint (≤5 files per phase) — Phase 1: 5 files, Phase 2: 3 files, Phase 3: 4 files, Phase 4: 3 files. ✓

Type / name consistency:
- `PROJECTED_STANDINGS_HISTORY_KEY` constant referenced consistently across redis_store, kv_sync, tests, and the API layer.
- `build_trends_series(client, *, user_team)` signature consistent between Tasks 5 and 6.
- Payload field names match spec (`user_team`, `categories`, `actual`, `projected`, `dates`, `teams`, `roto_points`, `stats`).
