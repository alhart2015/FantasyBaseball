# ROS-Only Projections for Forward-Looking Decisions

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every roster-decision path (transaction ΔRoto, trade evaluator, waiver/audit, lineup optimizer, projected end-of-season standings) values players by their **rest-of-season-only** projection. End-of-season standings are computed as `current_standings + sum(roster ROS)` — not as `sum(roster full-season)`, which double-counts YTD that may not have been on the team's roster at the time.

**Architecture:** FanGraphs CSVs are ROS-only; `normalize_rest_of_season_to_full_season` adds YTD to produce a full-season *display* projection. We will write **two** Redis blobs (`cache:ros_projections` becomes ROS-only; new `cache:full_season_projections` holds the YTD-added shape for display surfaces). `Player` carries both `rest_of_season` (ROS-only, used by all decision math) and an optional `full_season_projection` (display-only). `project_team_stats` sums ROS-only contribution; `ProjectedStandings.from_rosters` combines `current_standings + ROS contribution` to produce end-of-season totals. Rate stats are recomputed from components — requires `OpportunityStat.AB` to be available on standings.

**Tech Stack:** Python 3.11, pandas, Redis (Upstash on Render / SQLite KV locally), pytest, ruff, vulture, mypy.

---

## Design summary (read first)

**The bug.** `data/projections.py:25` — `normalize_rest_of_season_to_full_season()` adds YTD actuals onto FanGraphs ROS-only counting stats. The result is correctly named "full-season projection" inside that function, but it's then written to `cache:ros_projections` and surfaces as `Player.rest_of_season`. Every consumer that calls itself "ROS" is silently working with `ROS_remaining + YTD`.

**Verified empirically (2026-04-26):** Cruz YTD=19R, cached "ROS"=87R, implied remaining=68R. Cold/IL'd Soto YTD=3R, cached "ROS"=90R, implied remaining=87R. The cached blob is full-season.

**What "correct" looks like.** Forward-looking decisions ignore sunk YTD. When swapping player A out for player B, A's YTD stats stay on Hart's standings (those games already happened with A on Hart's roster). Only A's *future* contribution is at risk. Same for B's incoming contribution: B's YTD stays with B's old team. So the swap delta should diff `A_ros_remaining` and `B_ros_remaining`, not full-season values.

**Counter-example showing the bias.** Hart has hot-start Cruz (87 R full / 68 R remaining) and cold-start Soto (90 R full / 87 R remaining). A trade evaluator using full-season says swapping Cruz for Soto is a +3 R move. ROS-only says it's a +19 R move. The full-season math hides 16 R of already-happened YTD that shouldn't influence the decision.

**A second bug surfaces in `ProjectedStandings.from_rosters`.** Today it sums each player's full-season projection over the *current* roster. That's wrong even as an end-of-season projection: a player picked up mid-season has YTD games that were credited to *their previous fantasy team*, not the new one. Summing full-season inflates the new team's projected end-of-season by every newly-acquired player's pre-acquisition YTD. The correct formulation is `current_standings + sum(current_roster.ROS_remaining)` — the current standings carry the right historical attribution and ROS-remaining is the only future contribution at stake.

**Why this matters for the lineup optimizer.** The optimizer compares hypothetical lineup configurations and picks the one with the best projected team total. If players are valued at full-season (with YTD), a hot-start player is preferred over a cold-start one with similar ROS-only outlook because the hot YTD inflates their full-season number — even though the YTD is locked in regardless of who occupies a slot this week. Switching the player-level math to ROS-only fixes this: the optimizer now picks the player with the best forward-looking projection, which is what "start someone because they project to be good going forward" means.

**Concretely what we change.** Three pieces of math:

1. `apply_swap_delta` ITSELF is mathematically correct: `current - loses + gains` is unit-consistent if all three are in the same units. Switching its `loses_ros`/`gains_ros` parameters to ROS-only and keeping `current_stats` as projected end-of-season (now correctly computed) gives the right answer. **No denominator change.**
2. `project_team_stats(roster)` switches from full-season to ROS-only. It now returns the team's *ROS contribution* — counting stat sums plus rate-stat components (h/ab, er/ip, bh/ip).
3. `ProjectedStandings.from_rosters(rosters, current_standings, date)` adds ROS contribution to `current_standings` per team. Rate stats are recomputed from combined components — requires `current_standings` to expose `AB` (and we already have `IP`).

**Why split the cache (vs. subtract YTD at consumption).** A single source of truth is cleaner than scattering YTD-subtraction logic at every call site. The full-season blob remains useful for *display* surfaces (e.g., player comparison pages) that show "projected season totals" without doing a YTD join on every render.

**Decisions intentionally NOT in scope.**

- We are **not** rewriting historical `delta_roto` values stored on past transactions. New transactions will be scored ROS-only going forward; the migrated `delta_roto` field on old rows stays as-is. (Mark this in the audit page legend.)
- We are **not** changing the draft-mode `eroto_recs.py` use of `apply_swap_delta` — draft has no in-season YTD, so the input semantics are equivalent.
- We are **not** introducing weekly/daily projections to drive lineup decisions. ROS-only is the closest stable input we have; finer-grained projections would be a follow-up.

**Branch hygiene.** Cut a new branch off `main`: `git checkout main && git pull && git checkout -b fix/ros-only-decision-projections`. The current `fix/eroto-recs-rate-stats-and-replacements` branch is unrelated draft work; do NOT layer this on top of it.

---

## File structure

| File | Role | Phase |
|---|---|---|
| `src/fantasy_baseball/data/cache_keys.py` | Add `FULL_SEASON_PROJECTIONS` enum | 1 |
| `src/fantasy_baseball/data/redis_store.py` | Add `get_full_season_projections` / `set_full_season_projections` helpers | 1 |
| `src/fantasy_baseball/data/ros_pipeline.py` | Write ROS-only blob alongside full-season blob | 1 |
| `tests/test_data/test_ros_pipeline.py` | Test dual-write | 1 |
| `tests/test_data/test_redis_store_projections.py` | Test new helpers | 1 |
| `src/fantasy_baseball/models/player.py` | `rest_of_season` becomes ROS-only; add optional `full_season_projection` for display surfaces | 2 |
| `src/fantasy_baseball/data/projections.py` | `match_roster_to_projections` populates `rest_of_season` from ROS frame, `full_season_projection` from full-season frame (display) | 2 |
| `tests/test_data/test_hydrate_roster_entries.py` | Update existing assertions + add ROS-vs-full distinction test | 2 |
| `tests/test_models/test_player.py` (new file if absent) | Test Player parsing/serialization with both fields | 2 |
| `src/fantasy_baseball/scoring.py` | `project_team_stats` sums ROS-only counting stats + rate components | 3 |
| `src/fantasy_baseball/models/standings.py` | `ProjectedStandings.from_rosters` takes `current_standings`; adds it to ROS contribution; rate stats recomputed from combined components | 3 |
| `src/fantasy_baseball/utils/constants.py` | Add `OpportunityStat.AB` if not present (needed for AVG combination) | 3 |
| `src/fantasy_baseball/web/refresh_pipeline.py` | Thread `current_standings` through to `ProjectedStandings.from_rosters` | 3 |
| `src/fantasy_baseball/web/season_data.py` | Same — update call sites of `from_rosters` | 3 |
| `tests/test_scoring.py` / `tests/test_models/test_standings.py` | Test ROS-only summing + standings combination math | 3 |
| `src/fantasy_baseball/trades/evaluate.py` | `player_rest_of_season_stats` now reads `Player.rest_of_season` (ROS-only) | 4 |
| `src/fantasy_baseball/trades/multi_trade.py` | No code change; verify via test | 4 |
| `src/fantasy_baseball/lineup/delta_roto.py` | No code change; verify via test | 4 |
| `tests/test_trades/test_evaluate.py` | Update fixture math to ROS-only expectations | 4 |
| `tests/test_lineup/test_delta_roto.py` (may not exist) | Add ROS-only swap test | 4 |
| `src/fantasy_baseball/analysis/transactions.py` | `_load_projections_for_date_redis` uses ROS-only key; replacement-floor math stays | 5 |
| `src/fantasy_baseball/lineup/waivers.py` | `Player(rest_of_season=ros)` now feeds ROS-only frame | 5 |
| `src/fantasy_baseball/lineup/roster_audit.py` | SGP & FA-gap inputs become ROS-only | 5 |
| `tests/test_analysis/test_transaction_scoring.py` | Update expected ΔRoto values | 5 |
| `tests/test_lineup/test_roster_audit.py` | Update expected SGP-gap values | 5 |
| `scripts/run_season_dashboard.py` | Run end-to-end refresh, verify dashboard | 6 |

---

## Phase 0: Setup + regression test

**Files:**
- Create: `tests/test_data/test_ros_only_regression.py`

- [ ] **Step 1: Cut a fresh branch off main**

```bash
git fetch origin
git checkout main
git pull
git checkout -b fix/ros-only-decision-projections
```

- [ ] **Step 2: Step-0 cleanup pass on touched files**

Per `CLAUDE.md` Pre-Work rule 1 — the touched modules are >300 LOC. Run the linters and clear dead code before starting structural work:

```bash
ruff check --select F,I src/fantasy_baseball/data/projections.py src/fantasy_baseball/data/ros_pipeline.py src/fantasy_baseball/trades/evaluate.py src/fantasy_baseball/analysis/transactions.py src/fantasy_baseball/lineup/optimizer.py src/fantasy_baseball/lineup/roster_audit.py src/fantasy_baseball/lineup/waivers.py src/fantasy_baseball/scoring.py
vulture src/fantasy_baseball/data/projections.py src/fantasy_baseball/data/ros_pipeline.py src/fantasy_baseball/trades/evaluate.py src/fantasy_baseball/analysis/transactions.py src/fantasy_baseball/lineup/optimizer.py src/fantasy_baseball/lineup/roster_audit.py src/fantasy_baseball/lineup/waivers.py src/fantasy_baseball/scoring.py
```

If clean, proceed. If anything surfaces, delete it and commit as `chore: remove dead code in modules touched by ros-only fix` BEFORE starting Phase 1.

- [ ] **Step 3: Write the failing regression test that captures the bug**

Create `tests/test_data/test_ros_only_regression.py`:

```python
"""Regression test for ROS-vs-full-season cache shape.

Captures the bug where ``cache:ros_projections`` held YTD-inflated
full-season stats. After the fix, ``cache:ros_projections`` MUST hold
ROS-remaining-only stats, and ``cache:full_season_projections`` MUST
hold the YTD-added totals.
"""
from __future__ import annotations

import pandas as pd

from fantasy_baseball.data import redis_store as rs


def test_ros_cache_excludes_ytd(tmp_path, monkeypatch):
    """A player with YTD games should have cached ROS == FanGraphs ROS-only.

    Setup: a hitter projected for 100 R remaining-games-only by FanGraphs,
    with 30 R already accumulated YTD. The cached ROS blob should be 100,
    not 130. The cached full-season blob should be 130.
    """
    from fantasy_baseball.data.kv_store import SqliteKVStore

    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "kv.db"))
    kv = SqliteKVStore(tmp_path / "kv.db")
    rs.set_game_log_totals(kv, "hitters", {
        "12345": {"r": 30, "hr": 5, "rbi": 20, "sb": 2, "h": 30, "ab": 100, "pa": 110, "name": "Test Hitter"},
    })

    # Simulate the FanGraphs ROS-only blend (what the CSV blend produces):
    ros_only_df = pd.DataFrame([{
        "name": "Test Hitter", "mlbam_id": 12345, "fg_id": "x",
        "r": 100.0, "hr": 25.0, "rbi": 75.0, "sb": 8.0, "h": 110.0, "ab": 400.0, "pa": 440.0,
        "avg": 0.275, "player_type": "hitter", "team": "X", "adp": 1,
    }])

    # NEW pipeline (Phase 1) MUST write ROS-only and full-season separately.
    from fantasy_baseball.data.ros_pipeline import write_ros_and_full_season
    write_ros_and_full_season(kv, hitters_ros=ros_only_df, pitchers_ros=pd.DataFrame())

    ros_cache = rs.get_ros_projections(kv)
    full_cache = rs.get_full_season_projections(kv)

    ros_row = next(p for p in ros_cache["hitters"] if p["mlbam_id"] == 12345)
    full_row = next(p for p in full_cache["hitters"] if p["mlbam_id"] == 12345)

    assert ros_row["r"] == 100.0, "ros cache must NOT include YTD"
    assert ros_row["ab"] == 400.0
    assert full_row["r"] == 130.0, "full-season cache MUST include YTD"
    assert full_row["ab"] == 500.0
```

- [ ] **Step 4: Run it to verify it fails (function not defined yet)**

```bash
pytest tests/test_data/test_ros_only_regression.py -v
```

Expected: `ImportError` or `AttributeError` because `write_ros_and_full_season`, `get_full_season_projections` don't exist yet.

- [ ] **Step 5: Commit**

```bash
git add tests/test_data/test_ros_only_regression.py
git commit -m "test(ros): regression test for split ROS-only / full-season cache shape"
```

---

## Phase 1: Pipeline writes both blobs

**Files:**
- Modify: `src/fantasy_baseball/data/cache_keys.py`
- Modify: `src/fantasy_baseball/data/redis_store.py`
- Modify: `src/fantasy_baseball/data/ros_pipeline.py`
- Test: `tests/test_data/test_redis_store_projections.py`
- Test: `tests/test_data/test_ros_pipeline.py`

- [ ] **Step 1: Add the new cache key enum entry**

Edit `src/fantasy_baseball/data/cache_keys.py`. Find the `CacheKey` StrEnum and add:

```python
FULL_SEASON_PROJECTIONS = "full_season_projections"
```

right after the existing `ROS_PROJECTIONS = "ros_projections"` line (preserve alphabetical/logical grouping).

- [ ] **Step 2: Write a failing test for the new redis_store helper**

Edit `tests/test_data/test_redis_store_projections.py`. Add a test:

```python
def test_full_season_projections_round_trip(tmp_path):
    from fantasy_baseball.data.kv_store import SqliteKVStore
    from fantasy_baseball.data import redis_store as rs

    kv = SqliteKVStore(tmp_path / "kv.db")
    rs.set_full_season_projections(kv, {
        "hitters": [{"name": "A", "r": 100}],
        "pitchers": [{"name": "B", "k": 200}],
    })
    got = rs.get_full_season_projections(kv)
    assert got == {"hitters": [{"name": "A", "r": 100}],
                   "pitchers": [{"name": "B", "k": 200}]}


def test_full_season_projections_missing_returns_none(tmp_path):
    from fantasy_baseball.data.kv_store import SqliteKVStore
    from fantasy_baseball.data import redis_store as rs

    kv = SqliteKVStore(tmp_path / "kv.db")
    assert rs.get_full_season_projections(kv) is None
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_data/test_redis_store_projections.py::test_full_season_projections_round_trip -v
```

Expected: `AttributeError: module 'fantasy_baseball.data.redis_store' has no attribute 'set_full_season_projections'`.

- [ ] **Step 4: Implement the redis_store helpers**

Edit `src/fantasy_baseball/data/redis_store.py`. After the existing `ROS_PROJECTIONS_KEY = redis_key(CacheKey.ROS_PROJECTIONS)` block (around line 136), add:

```python
FULL_SEASON_PROJECTIONS_KEY = redis_key(CacheKey.FULL_SEASON_PROJECTIONS)


def get_full_season_projections(client) -> dict | None:
    """Read the latest full-season (ROS+YTD) projections from Redis.

    Same shape as :func:`get_ros_projections` —
    ``{"hitters": [...], "pitchers": [...]}`` — but each row's counting
    stats include season-to-date actuals. Used by
    ``ProjectedStandings.from_rosters`` and ``project_team_stats`` for
    end-of-season standings projection. NOT used by forward-looking
    decision paths (transactions, trades, waivers, lineup optimizer) —
    those should call :func:`get_ros_projections`.
    """
    if client is None:
        return None
    raw = client.get(FULL_SEASON_PROJECTIONS_KEY)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Corrupt JSON at Redis key %r; ignoring", FULL_SEASON_PROJECTIONS_KEY)
        return None
    if not isinstance(data, dict):
        return None
    return data


def set_full_season_projections(client, payload: dict) -> None:
    """Overwrite the full-season projections blob.

    No-op when ``client is None`` (unconfigured environments).
    """
    if client is None:
        return
    client.set(FULL_SEASON_PROJECTIONS_KEY, json.dumps(payload))
```

- [ ] **Step 5: Verify the redis_store test passes**

```bash
pytest tests/test_data/test_redis_store_projections.py::test_full_season_projections_round_trip tests/test_data/test_redis_store_projections.py::test_full_season_projections_missing_returns_none -v
```

Expected: PASS.

- [ ] **Step 6: Commit Step 4 progress**

```bash
git add src/fantasy_baseball/data/cache_keys.py src/fantasy_baseball/data/redis_store.py tests/test_data/test_redis_store_projections.py
git commit -m "feat(redis): add cache:full_season_projections helpers"
```

- [ ] **Step 7: Write a failing test for the dual-write pipeline**

Edit `tests/test_data/test_ros_pipeline.py`. Add a test that exercises the new dual-write entry point:

```python
def test_blend_writes_both_ros_and_full_season(tmp_path, monkeypatch):
    """blend_and_cache_ros() must write BOTH cache:ros_projections (ROS-only)
    AND cache:full_season_projections (with YTD added)."""
    import json
    from fantasy_baseball.data.kv_store import SqliteKVStore, _reset_singleton
    from fantasy_baseball.data import redis_store as rs
    from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros

    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "kv.db"))
    _reset_singleton()
    kv = SqliteKVStore(tmp_path / "kv.db")
    # Seed YTD: player 12345 has 30 R already
    rs.set_game_log_totals(kv, "hitters", {
        "12345": {"r": 30, "hr": 5, "rbi": 20, "sb": 2, "h": 30, "ab": 100, "pa": 110, "name": "Test Hitter"},
    })
    rs.set_game_log_totals(kv, "pitchers", {})

    # Build a minimal projections dir with a single date subdir
    proj_dir = tmp_path / "projections"
    date_dir = proj_dir / "2026" / "rest_of_season" / "2026-04-26"
    date_dir.mkdir(parents=True)
    (date_dir / "steamer-hitters.csv").write_text(
        "fg_id,mlbam_id,Name,Team,PA,AB,R,HR,RBI,SB,H,AVG\n"
        "x,12345,Test Hitter,X,440,400,100,25,75,8,110,0.275\n"
    )
    (date_dir / "steamer-pitchers.csv").write_text(
        "fg_id,mlbam_id,Name,Team,IP,W,SO,SV,ER,BB,H,ERA,WHIP\n"
    )

    blend_and_cache_ros(
        projections_dir=proj_dir, systems=["steamer"], weights=None,
        roster_names=None, season_year=2026,
    )

    ros = rs.get_ros_projections(kv)
    full = rs.get_full_season_projections(kv)

    assert ros is not None and full is not None
    ros_row = next(p for p in ros["hitters"] if p.get("mlbam_id") == 12345)
    full_row = next(p for p in full["hitters"] if p.get("mlbam_id") == 12345)
    assert ros_row["r"] == 100.0, "ROS cache must be 100 (CSV value, no YTD added)"
    assert full_row["r"] == 130.0, "Full-season cache must be 100+30=130"
```

- [ ] **Step 8: Run it to verify it fails**

```bash
pytest tests/test_data/test_ros_pipeline.py::test_blend_writes_both_ros_and_full_season -v
```

Expected: FAIL — current pipeline writes only one blob and that blob holds full-season values under `cache:ros_projections`.

- [ ] **Step 9: Modify the pipeline to dual-write**

Edit `src/fantasy_baseball/data/ros_pipeline.py`. The current pipeline normalizes to full-season inside `_normalizer` and writes the result to `ROS_PROJECTIONS`. Change it so the in-memory blend produces ROS-only first, then full-season is computed by adding YTD, and BOTH are written:

```python
"""Rest-of-season projections pipeline: blend CSVs in memory → Redis.

Produces TWO Redis blobs from one CSV blend:
- ``cache:ros_projections`` — ROS-remaining counting stats (FanGraphs
  CSV values, untouched)
- ``cache:full_season_projections`` — same blend plus YTD actuals from
  ``game_log_totals:{hitters,pitchers}``, used by ``project_team_stats``
  for end-of-season standings projection.

Forward-looking decisions (transactions, trades, waivers, lineup
optimizer) read the ROS blob. Standings projection reads the
full-season blob.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.data.projections import (
    blend_projections,
    normalize_rest_of_season_to_full_season,
)
from fantasy_baseball.data.redis_store import (
    get_game_log_totals,
    set_full_season_projections,
)
from fantasy_baseball.models.player import PlayerType


def blend_and_cache_ros(
    projections_dir: Path,
    systems: list[str],
    weights: dict[str, float] | None,
    roster_names: set[str] | None,
    season_year: int,
    progress_cb=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend the latest ROS CSV snapshot in memory and write BOTH Redis blobs.

    Returns the ROS-only ``(hitters_df, pitchers_df)`` — full-season is
    a derived view, callers that want it should read
    ``get_full_season_projections``.
    """
    ros_root = projections_dir / str(season_year) / "rest_of_season"
    if not ros_root.is_dir():
        raise FileNotFoundError(f"ROS snapshot dir missing: {ros_root}")
    date_dirs = sorted(
        (p for p in ros_root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    if not date_dirs:
        raise FileNotFoundError(f"No ROS snapshot dirs under {ros_root}")
    latest = date_dirs[-1]

    client = get_kv()
    hitter_totals = {
        int(k): v for k, v in get_game_log_totals(client, "hitters").items()
    }
    pitcher_totals = {
        int(k): v for k, v in get_game_log_totals(client, "pitchers").items()
    }

    # Blend in pure ROS-only mode — no normalizer.
    hitters_ros, pitchers_ros, _quality = blend_projections(
        latest, systems, weights,
        roster_names=roster_names, progress_cb=progress_cb,
        normalizer=None,
    )

    # Derive full-season by adding YTD actuals.
    hitters_full = normalize_rest_of_season_to_full_season(
        hitters_ros, hitter_totals, PlayerType.HITTER,
    )
    pitchers_full = normalize_rest_of_season_to_full_season(
        pitchers_ros, pitcher_totals, PlayerType.PITCHER,
    )

    from fantasy_baseball.web.season_data import write_cache
    write_cache(CacheKey.ROS_PROJECTIONS, {
        "hitters": hitters_ros.to_dict(orient="records"),
        "pitchers": pitchers_ros.to_dict(orient="records"),
    })
    set_full_season_projections(client, {
        "hitters": hitters_full.to_dict(orient="records"),
        "pitchers": pitchers_full.to_dict(orient="records"),
    })
    return hitters_ros, pitchers_ros
```

NOTE: `blend_projections` currently expects an optional `normalizer` callable. Verify by reading `src/fantasy_baseball/data/projections.py:blend_projections` — if `normalizer=None` is not already supported, add a no-op default. (As of writing, it accepts `normalizer=None` and skips the call when None — confirm before proceeding.)

- [ ] **Step 10: Verify `blend_projections` handles `normalizer=None`**

```bash
grep -n "def blend_projections" src/fantasy_baseball/data/projections.py
grep -n "normalizer" src/fantasy_baseball/data/projections.py
```

Read those lines. If `normalizer` is unconditionally called, change it to:

```python
if normalizer is not None:
    hitters_df, pitchers_df = normalizer(system_name, hitters_df, pitchers_df)
```

If it's already conditional, no change needed.

- [ ] **Step 11: Run the dual-write test and the regression test from Phase 0**

```bash
pytest tests/test_data/test_ros_pipeline.py tests/test_data/test_ros_only_regression.py -v
```

Expected: `test_blend_writes_both_ros_and_full_season` PASSES. `test_ros_cache_excludes_ytd` may still fail because it references `write_ros_and_full_season` which doesn't exist.

If you wrote the regression test referencing a helper that doesn't exist, refactor it to call `blend_and_cache_ros` directly with a tmp projections dir. Update the regression test in place — same intent, real entry point. Commit.

- [ ] **Step 12: Run the full pipeline test suite**

```bash
pytest tests/test_data/ -v
```

Expected: All pass. If `test_ros_pipeline.py` had existing tests that asserted the old (full-season-in-ros-blob) behavior, they will fail. Inspect each: if the test was asserting the bug, update the expected values with a comment noting the math change. Per project rules, justify each test edit explicitly in the commit message.

- [ ] **Step 13: Commit Phase 1**

```bash
git add src/fantasy_baseball/data/ros_pipeline.py tests/test_data/test_ros_pipeline.py tests/test_data/test_ros_only_regression.py
git commit -m "feat(ros): pipeline writes ROS-only and full-season blobs separately

- cache:ros_projections now holds FanGraphs ROS-only counting stats
- cache:full_season_projections (new) holds ROS+YTD, used by ProjectedStandings
- blend_and_cache_ros derives full-season from the same in-memory blend
- adds dual-write regression test"
```

---

## Phase 2: Player model carries both projections

**Files:**
- Modify: `src/fantasy_baseball/models/player.py`
- Modify: `src/fantasy_baseball/data/projections.py` (`match_roster_to_projections`, `hydrate_roster_entries`)
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` (callers of hydrate)
- Test: `tests/test_data/test_hydrate_roster_entries.py`
- Test: `tests/test_models/test_player.py` (new file or existing — check first)

- [ ] **Step 1: Failing test — Player carries both projections**

Edit `tests/test_data/test_hydrate_roster_entries.py`. Add a test:

```python
def test_player_holds_both_ros_and_full_season(tmp_path):
    """match_roster_to_projections must populate Player.rest_of_season
    (ROS-only) and Player.full_season_projection (ROS+YTD) from the
    matching FanGraphs row plus YTD actuals."""
    import pandas as pd
    from fantasy_baseball.data.projections import match_roster_to_projections
    from fantasy_baseball.models.player import Player, PlayerType

    roster = [Player(name="Test Hitter", player_type=PlayerType.HITTER, mlbam_id=12345)]
    ros_hitters = pd.DataFrame([{
        "_name_norm": "test hitter", "name": "Test Hitter", "mlbam_id": 12345,
        "r": 100.0, "hr": 25.0, "rbi": 75.0, "sb": 8.0, "h": 110.0, "ab": 400.0, "pa": 440.0, "avg": 0.275,
    }])
    full_hitters = pd.DataFrame([{
        "_name_norm": "test hitter", "name": "Test Hitter", "mlbam_id": 12345,
        "r": 130.0, "hr": 30.0, "rbi": 95.0, "sb": 10.0, "h": 140.0, "ab": 500.0, "pa": 550.0, "avg": 0.280,
    }])
    matched = match_roster_to_projections(
        roster, hitters_proj=ros_hitters, pitchers_proj=pd.DataFrame(),
        full_hitters_proj=full_hitters, full_pitchers_proj=pd.DataFrame(),
    )
    p = matched[0]
    assert p.rest_of_season.r == 100.0
    assert p.full_season_projection.r == 130.0
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
pytest tests/test_data/test_hydrate_roster_entries.py::test_player_holds_both_ros_and_full_season -v
```

Expected: FAIL — `Player.full_season_projection` doesn't exist; `match_roster_to_projections` doesn't accept the new args.

- [ ] **Step 3: Add `full_season_projection` field to Player**

Edit `src/fantasy_baseball/models/player.py:174`. Add the new field next to `rest_of_season`:

```python
rest_of_season: HitterStats | PitcherStats | None = None
full_season_projection: HitterStats | PitcherStats | None = None
preseason: HitterStats | PitcherStats | None = None
current: HitterStats | PitcherStats | None = None
```

In `Player.from_dict` (line 188), add parsing:

```python
fs_raw = d.get("full_season_projection")
full_season_projection = _make_stats(fs_raw, player_type) if fs_raw is not None else None
```

and pass `full_season_projection=full_season_projection` to the `cls(...)` call.

In `Player.to_dict` (line 247), serialize the field:

```python
if self.full_season_projection is not None:
    d["full_season_projection"] = self.full_season_projection.to_dict()
```

- [ ] **Step 4: Update `match_roster_to_projections` to populate both fields**

Edit `src/fantasy_baseball/data/projections.py`. Find `match_roster_to_projections` and add `full_hitters_proj`/`full_pitchers_proj` parameters. After matching the ROS row, also match the full-season row by `mlbam_id` (preferred) or `_name_norm`, and assign to `player.full_season_projection`.

(Read the existing function first — line ~470 — to understand the matching logic, then mirror it for the full-season frames.)

- [ ] **Step 5: Update `hydrate_roster_entries` and refresh_pipeline call sites**

`hydrate_roster_entries` calls `match_roster_to_projections`. Pass through the new args. In `web/refresh_pipeline.py` find the place that loads ROS via `_load_projections` and add a parallel load of the full-season blob from `get_full_season_projections(client)`. Pass both into the hydration step.

- [ ] **Step 6: Run the failing test from Step 1**

```bash
pytest tests/test_data/test_hydrate_roster_entries.py::test_player_holds_both_ros_and_full_season -v
```

Expected: PASS.

- [ ] **Step 7: Run the full hydrate test file**

```bash
pytest tests/test_data/test_hydrate_roster_entries.py -v
```

Existing tests likely break because `Player.rest_of_season` now holds ROS-only values, but tests asserted full-season. For each failure: confirm the expected value should now be the ROS-only value; update with a comment `# updated: rest_of_season is now ROS-only per ros_only_decision_projections.md`.

- [ ] **Step 8: Commit Phase 2**

```bash
git add src/fantasy_baseball/models/player.py src/fantasy_baseball/data/projections.py src/fantasy_baseball/web/refresh_pipeline.py tests/test_data/test_hydrate_roster_entries.py
git commit -m "feat(player): add full_season_projection field; rest_of_season becomes ROS-only

- Player now carries rest_of_season (ROS-remaining) and full_season_projection (ROS+YTD)
- match_roster_to_projections accepts both projection frames
- refresh pipeline hydrates from both Redis blobs
- updates fixture-based tests; the projection field semantic was the bug"
```

---

## Phase 3: project_team_stats sums ROS-only; ProjectedStandings combines with current_standings

**Files:**
- Modify: `src/fantasy_baseball/scoring.py` (`project_team_stats`, possibly add `_stat` access pattern)
- Modify: `src/fantasy_baseball/models/standings.py` (`ProjectedStandings.from_rosters`, add `_combine_standings_with_ros` helper)
- Modify: `src/fantasy_baseball/utils/constants.py` (add `OpportunityStat.AB` if absent — verify first)
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`, `src/fantasy_baseball/web/season_data.py`, any other call site of `ProjectedStandings.from_rosters`
- Test: `tests/test_scoring.py` (existing or new)
- Test: `tests/test_models/test_standings.py` (existing or new)

This is the load-bearing phase. Two coupled math changes:

1. `project_team_stats(roster)` returns the team's **ROS-only contribution** — counting stat sums plus rate-stat components (h_total, ab_total, er_total, ip_total, bb_total, ha_total). Rate fields (avg, era, whip) are recomputed from the ROS components — they represent the player set's expected forward rate.
2. `ProjectedStandings.from_rosters(rosters, current_standings, date)` combines per team: `end_of_season = current_standings + project_team_stats(roster)`. Counting stats add directly. Rate stats recombine via components — for AVG we need `current_standings.AB`, which means `OpportunityStat.AB` must be on the standings (likely already in Yahoo standings extras; verify in Step 1).

- [ ] **Step 1: Verify `OpportunityStat.AB` and `IP` are present and populated**

```bash
grep -n "class OpportunityStat\|AB\|IP" src/fantasy_baseball/utils/constants.py
```

Expected: both `AB` and `IP` defined. If `AB` is missing, add it:

```python
class OpportunityStat(StrEnum):
    PA = "PA"
    AB = "AB"  # required for end-of-season AVG combination
    IP = "IP"
```

Then verify the Yahoo standings ingest populates `AB` in `entry.extras`. Find the writer (`grep -rn "OpportunityStat.PA" src/`) and confirm `AB` is set alongside `PA`. If not, this is a separate one-line fix to the ingest before Phase 3 can land. **DO NOT proceed past this step until the standings actually carry AB.**

- [ ] **Step 2: Failing test — project_team_stats now sums ROS-only**

Add to `tests/test_scoring.py`:

```python
def test_project_team_stats_sums_ros_only():
    """project_team_stats returns the team's ROS-only contribution.
    Counting stats sum the rest_of_season values; rate stats recompute
    from h/ab (or er/bb/ha/ip) components."""
    from fantasy_baseball.models.player import HitterStats, Player, PlayerType
    from fantasy_baseball.scoring import project_team_stats

    # Hot-YTD player and cold-YTD player with identical ROS-remaining.
    # project_team_stats must produce identical contributions.
    hot = Player(
        name="Hot",
        player_type=PlayerType.HITTER,
        rest_of_season=HitterStats(r=70, hr=20, rbi=60, sb=5, h=100, ab=400, pa=440, avg=0.250),
    )
    cold = Player(
        name="Cold",
        player_type=PlayerType.HITTER,
        rest_of_season=HitterStats(r=70, hr=20, rbi=60, sb=5, h=100, ab=400, pa=440, avg=0.250),
    )
    assert project_team_stats([hot]).r == 70  # NOT full-season
    assert project_team_stats([hot]).r == project_team_stats([cold]).r
```

- [ ] **Step 3: Run; expect FAIL today**

```bash
pytest tests/test_scoring.py::test_project_team_stats_sums_ros_only -v
```

Today: FAIL because `_stat` reads from `rest_of_season` already (so this might actually pass!). Actually verify by reading `_stat` first:

```bash
grep -n "_stat\|rest_of_season" src/fantasy_baseball/scoring.py
```

If `_stat` already reads `rest_of_season`, the test PASSES — but only because `rest_of_season` happens to hold full-season today. After Phase 2 makes it ROS-only, the test still passes for the same reason. The semantic change is centralized in Phase 2; Phase 3 just verifies and documents.

- [ ] **Step 4: Failing test — ProjectedStandings.from_rosters now requires current_standings**

Add to `tests/test_models/test_standings.py`:

```python
def test_projected_standings_combines_standings_with_ros():
    """end_of_season = current_standings + ROS-only roster contribution.
    Rate stats recompute from combined components."""
    from datetime import date
    from fantasy_baseball.models.standings import (
        Standings, StandingsEntry, CategoryStats, ProjectedStandings,
    )
    from fantasy_baseball.models.player import Player, PlayerType, HitterStats
    from fantasy_baseball.utils.constants import OpportunityStat

    # Team with R=100 YTD; AB=400 YTD; AVG=.270 (108 H YTD).
    standings = Standings(
        effective_date=date(2026, 4, 26),
        entries=[StandingsEntry(
            team_name="Test", team_key="t1", rank=1,
            stats=CategoryStats(r=100, hr=20, rbi=80, sb=8, avg=0.270, w=10, k=200, sv=5, era=3.50, whip=1.10),
            extras={OpportunityStat.AB: 400.0, OpportunityStat.IP: 200.0},
        )],
    )
    roster = [Player(
        name="P", player_type=PlayerType.HITTER,
        rest_of_season=HitterStats(r=70, hr=20, rbi=60, sb=5, h=120, ab=480, pa=520, avg=0.250),
    )]
    proj = ProjectedStandings.from_rosters(
        team_rosters={"Test": roster},
        current_standings=standings,
        effective_date=date(2026, 4, 26),
    )
    eos = proj.by_team()["Test"].stats
    # Counting: 100 + 70 = 170 R
    assert eos.r == 170
    # Rate: end_AVG = (108 + 120) / (400 + 480) = 228/880 = 0.2591
    assert abs(eos.avg - 228/880) < 1e-4
```

- [ ] **Step 5: Run; expect FAIL**

```bash
pytest tests/test_models/test_standings.py::test_projected_standings_combines_standings_with_ros -v
```

Expected: FAIL — `from_rosters` doesn't accept `current_standings` yet.

- [ ] **Step 6: Implement `_combine_standings_with_ros` helper in scoring.py**

Add to `src/fantasy_baseball/scoring.py` (next to or above `project_team_stats`):

```python
def _combine_standings_with_ros(
    standings_stats: CategoryStats,
    standings_extras: dict,  # OpportunityStat-keyed
    ros_stats: CategoryStats,
    ros_components: dict[str, float],  # {"h", "ab", "er", "ip", "bb_h_allowed"}
) -> CategoryStats:
    """Build end-of-season CategoryStats by adding ROS contribution to current standings.

    Counting stats sum directly. Rate stats (AVG, ERA, WHIP) recompute
    from combined components: AVG = (standings_H + ros_H) / (standings_AB + ros_AB),
    similarly for ERA (over IP) and WHIP. Standings counting components
    are derived from the rate × volume on the standings side
    (standings_H = standings.AVG * standings.AB, etc.).
    """
    from fantasy_baseball.utils.constants import OpportunityStat
    from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

    standings_AB = float(standings_extras.get(OpportunityStat.AB, 0))
    standings_IP = float(standings_extras.get(OpportunityStat.IP, 0))
    standings_H = standings_stats.avg * standings_AB
    standings_ER = standings_stats.era * standings_IP / 9.0
    standings_BH = standings_stats.whip * standings_IP

    end_AB = standings_AB + ros_components["ab"]
    end_IP = standings_IP + ros_components["ip"]
    end_H = standings_H + ros_components["h"]
    end_ER = standings_ER + ros_components["er"]
    end_BH = standings_BH + ros_components["bb_h_allowed"]

    return CategoryStats(
        r=standings_stats.r + ros_stats.r,
        hr=standings_stats.hr + ros_stats.hr,
        rbi=standings_stats.rbi + ros_stats.rbi,
        sb=standings_stats.sb + ros_stats.sb,
        avg=calculate_avg(end_H, end_AB, default=standings_stats.avg),
        w=standings_stats.w + ros_stats.w,
        k=standings_stats.k + ros_stats.k,
        sv=standings_stats.sv + ros_stats.sv,
        era=calculate_era(end_ER, end_IP, default=standings_stats.era),
        whip=(end_BH / end_IP) if end_IP > 0 else standings_stats.whip,
    )
```

For this helper to receive `ros_components`, change `project_team_stats` to expose them. Either: (a) return a richer object (e.g., a tuple of `CategoryStats + components dict`); or (b) add a sibling function `project_team_components(roster)` that returns the components dict and call both.

Choose (a) for simpler call sites:

```python
@dataclass
class TeamProjection:
    """ROS-only team contribution: roto-shaped stats plus the
    underlying rate-stat components needed for end-of-season combination."""
    stats: CategoryStats
    components: dict[str, float]  # h, ab, er, ip, bb_h_allowed
```

`project_team_stats` returns a `TeamProjection`. Existing callers expecting `CategoryStats` access `.stats`. Update `compute_roto_points` callers similarly — read carefully which interface they expect.

- [ ] **Step 7: Update `ProjectedStandings.from_rosters`**

Edit `src/fantasy_baseball/models/standings.py`. Replace the existing `from_rosters`:

```python
@classmethod
def from_rosters(
    cls,
    team_rosters: Mapping[str, Any],
    current_standings: Standings,
    effective_date: date,
) -> ProjectedStandings:
    """Build end-of-season standings = current standings + ROS contribution per team."""
    from fantasy_baseball.scoring import _combine_standings_with_ros, project_team_stats

    standings_by_team = current_standings.by_team()
    entries: list[ProjectedStandingsEntry] = []
    for tname, roster in team_rosters.items():
        if tname not in standings_by_team:
            raise ValueError(f"team {tname!r} in rosters not in current_standings")
        proj = project_team_stats(roster, displacement=True)
        baseline = standings_by_team[tname]
        end_of_season = _combine_standings_with_ros(
            baseline.stats, baseline.extras, proj.stats, proj.components,
        )
        entries.append(ProjectedStandingsEntry(team_name=tname, stats=end_of_season))
    return cls(effective_date=effective_date, entries=entries)
```

- [ ] **Step 8: Update every caller of `ProjectedStandings.from_rosters`**

```bash
grep -rn "ProjectedStandings.from_rosters\|from_rosters(" src/fantasy_baseball/
```

Pass `current_standings=...` at each call site. Likely callers: `web/refresh_pipeline.py`, `web/season_data.py`. Read each and verify the standings object is in scope at the call site (it should be — the same refresh pipeline that builds rosters also has Standings).

- [ ] **Step 9: Run all tests added in Steps 4 and 2**

```bash
pytest tests/test_models/test_standings.py::test_projected_standings_combines_standings_with_ros tests/test_scoring.py::test_project_team_stats_sums_ros_only -v
```

Expected: both PASS.

- [ ] **Step 10: Run the broader scoring + standings test suite**

```bash
pytest tests/test_scoring.py tests/test_models/ -v
```

Failures here are expected: existing tests called `from_rosters(rosters, date)` without `current_standings`. Update each call site by injecting a minimal `Standings` fixture. **Do not weaken assertions**; their numerical expectations may shift because end-of-season is now `current_standings + ROS` rather than `sum(full_season)`. Hand-verify the new expected value before updating.

- [ ] **Step 11: Run the broader pipeline tests**

```bash
pytest tests/ -v -x --ignore=tests/test_draft
```

Same exercise. Per the `Tests are the guardrail` rule, justify each test edit explicitly with a comment referencing this plan.

- [ ] **Step 12: Commit Phase 3**

```bash
git add src/fantasy_baseball/scoring.py src/fantasy_baseball/models/standings.py src/fantasy_baseball/utils/constants.py src/fantasy_baseball/web/refresh_pipeline.py src/fantasy_baseball/web/season_data.py tests/
git commit -m "fix(standings): ProjectedStandings = current_standings + ROS contribution

- project_team_stats returns ROS-only TeamProjection (stats + components)
- ProjectedStandings.from_rosters takes current_standings; combines via _combine_standings_with_ros
- end-of-season AVG/ERA/WHIP recomputed from combined components
- adds OpportunityStat.AB to surface the AVG denominator from standings
- previously summed full-season per current-roster player, double-counting
  YTD from players that weren't on the team at the time"
```

---

## Phase 4: Swap evaluator + trade evaluator + delta_roto switch to ROS-only

**Files:**
- Modify: `src/fantasy_baseball/trades/evaluate.py` (`player_rest_of_season_stats`, doc comments)
- Test: `tests/test_trades/test_evaluate.py`
- Test: `tests/test_lineup/test_delta_roto.py` (create if absent)

After this phase, `apply_swap_delta` callers automatically receive ROS-only inputs because `player_rest_of_season_stats(p)` reads `p.rest_of_season` — which is now actually ROS-only. The mathematical change happens at the boundary; the swap math is unchanged.

- [ ] **Step 1: Failing test — apply_swap_delta with ROS-only inputs**

Add to `tests/test_trades/test_evaluate.py`:

```python
def test_swap_delta_uses_ros_only_not_full_season():
    """A swap of cold-YTD Soto-archetype for hot-YTD Cruz-archetype should
    score by ROS-remaining only, not by full-season totals that double-count
    YTD already locked into team standings.

    Setup: Hart's projected end-of-season R=900 (a CategoryStats baseline).
    Soto-archetype: 3 R YTD, 87 R remaining (full=90).
    Cruz-archetype: 19 R YTD, 68 R remaining (full=87).
    Swapping Cruz out for Soto should bump Hart's projected R by +19
    (Soto's 87 in vs Cruz's 68 out), NOT +3 (the full-season diff).
    """
    from fantasy_baseball.trades.evaluate import apply_swap_delta

    current = {"R": 900.0, "HR": 200.0, "RBI": 800.0, "SB": 100.0,
               "AVG": 0.260, "W": 80.0, "K": 1300.0, "SV": 40.0,
               "ERA": 3.80, "WHIP": 1.20}
    cruz_ros_only = {"R": 68, "HR": 22, "RBI": 64, "SB": 7, "AVG": 0.255, "ab": 400, "ip": 0,
                     "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}
    soto_ros_only = {"R": 87, "HR": 29, "RBI": 79, "SB": 14, "AVG": 0.290, "ab": 432, "ip": 0,
                     "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}

    after = apply_swap_delta(current, loses_ros=cruz_ros_only, gains_ros=soto_ros_only)
    assert after["R"] == 900.0 - 68 + 87
    assert after["R"] - current["R"] == 19  # NOT 3 (which would be full-season diff)
```

- [ ] **Step 2: Run to confirm passes**

```bash
pytest tests/test_trades/test_evaluate.py::test_swap_delta_uses_ros_only_not_full_season -v
```

Expected: PASS — `apply_swap_delta`'s math is already unit-correct; this test verifies the math is what we want when callers pass ROS-only.

- [ ] **Step 3: Audit `player_rest_of_season_stats` doc + behavior**

Read `src/fantasy_baseball/trades/evaluate.py:250`. Confirm it returns `player.rest_of_season` fields. After Phase 2, `player.rest_of_season` IS ROS-only — no code change needed, only docstring.

Update the docstring to be explicit:

```python
def player_rest_of_season_stats(player: Player) -> dict[str, float]:
    """Return the player's ROS-remaining stats as a flat dict.

    Returns rate stats (AVG, ERA, WHIP) plus counting stats (R, HR, RBI,
    SB, W, K, SV, ab, ip). All values are remaining-games-only — YTD
    actuals already on the team's standings are sunk and not included
    here. For end-of-season display (e.g. player comparison page), call
    sites should read ``player.full_season_projection`` directly. For
    end-of-season *projection math*, ``ProjectedStandings.from_rosters``
    handles the standings + ROS combination — never re-derive that here.
    """
```

Also update `apply_swap_delta`'s docstring (line 90) to clarify the input contract:

```python
"""Project end-of-season team stats after swapping one player for another.

``current_stats`` MUST be the team's projected end-of-season totals
as produced by ``ProjectedStandings.from_rosters`` (= current standings
+ ROS contribution). ``loses_ros`` and ``gains_ros`` MUST be the
players' ROS-remaining-only projections — passing full-season values
here double-counts YTD that's already in current_stats and produces
biased deltas (favors acquiring hot-start players, penalizes acquiring
cold-start players). YTD games already on the team's standings are
sunk; only future contribution is at stake in a swap.
"""
```

- [ ] **Step 4: Update or write test for compute_trade_impact ROS-only behavior**

Open `tests/test_trades/test_evaluate.py`. Find `test_compute_trade_impact_*` (if any). Confirm fixtures pass ROS-only stats. If a fixture has `R=600` for a hitter (clearly full-season), update to a realistic ROS-only number with a comment.

- [ ] **Step 5: Run the trades test suite**

```bash
pytest tests/test_trades/ -v
```

Expected: All pass. Failures here are likely fixture issues or assertions written against the old (full-season-double-counting) numbers — update with comments.

- [ ] **Step 6: Add a delta_roto test if missing**

Check for `tests/test_lineup/test_delta_roto.py`. If missing, create:

```python
"""Tests for lineup.delta_roto.compute_delta_roto."""
from fantasy_baseball.lineup.delta_roto import compute_delta_roto
# Fixture-based test exercising compute_delta_roto with ROS-only stats and
# verifying the resulting DeltaRotoResult has total > 0 when a clearly
# better player is added.
```

(Implement a minimal fixture test — Player A on roster, FA B with strictly higher stats, expect total > 0.)

- [ ] **Step 7: Run lineup test suite**

```bash
pytest tests/test_lineup/ -v
```

Expected: Pass.

- [ ] **Step 8: Commit Phase 4**

```bash
git add src/fantasy_baseball/trades/evaluate.py tests/test_trades/test_evaluate.py tests/test_lineup/
git commit -m "fix(trades): apply_swap_delta + compute_trade_impact use ROS-only player stats

After Phase 2, player.rest_of_season is ROS-only, so player_rest_of_season_stats
returns the right values. Docstrings updated to call out the input contract.
Tests updated to use ROS-only fixture values; the assertions changed because the
underlying math now ignores already-banked YTD."
```

---

## Phase 5: Transaction analyzer + waivers + roster_audit + pace

**Files:**
- Modify: `src/fantasy_baseball/analysis/transactions.py` (verify `_load_projections_for_date_redis` reads ROS-only)
- Modify: `src/fantasy_baseball/lineup/waivers.py`
- Modify: `src/fantasy_baseball/lineup/roster_audit.py`
- Modify: `src/fantasy_baseball/analysis/pace.py` — switch from `player.rest_of_season` to `player.full_season_projection`
- Test: `tests/test_analysis/test_transaction_scoring.py`
- Test: `tests/test_lineup/test_roster_audit.py`
- Test: `tests/test_analysis/test_pace.py` (existing or new)

**Why pace changes too.** `pace.py` computes `rest_of_season_deviation_sgp = (current_ROS - preseason) / sgp_denom` — the intent is "has this player's projection changed since the preseason?" Pre-fix, `current_ROS` was actually full-season (ROS+YTD), so the diff measured projection change. Post-fix, true ROS-remaining naturally declines with elapsed time even when projections haven't moved — the metric becomes meaningless. Switch the read to `player.full_season_projection`, which is the current full-season expectation for that player regardless of fantasy team attribution.

After Phase 1, `cache:ros_projections` is ROS-only — so any consumer reading via `get_ros_projections(client)` is automatically correct, no code change needed beyond updating tests. This phase is mostly a verification phase, with test updates.

- [ ] **Step 1: Verify each consumer reads from `get_ros_projections`**

```bash
grep -rn "get_ros_projections\|cache:ros_projections" src/fantasy_baseball/analysis/transactions.py src/fantasy_baseball/lineup/
```

For each match, confirm:
- it's used for forward-looking decisions (transaction scoring, waiver, audit, swap eval) → ROS-only is correct → no code change
- if any consumer is doing end-of-season projection, it must switch to `get_full_season_projections`

- [ ] **Step 1b: Fix pace.py to read full_season_projection**

Edit `src/fantasy_baseball/analysis/pace.py:274` (and 269 if applicable). Change:

```python
{k: getattr(player.rest_of_season, k, 0) for k in ros_keys}
if player.rest_of_season
```

to:

```python
{k: getattr(player.full_season_projection, k, 0) for k in ros_keys}
if player.full_season_projection
```

The local variable `ros_dict` and the parameter name `rest_of_season_stats` in `compute_player_pace` are fine to keep — internally the value represents "current full-season expectation" but renaming would expand the diff. Add a comment at the call site explaining the semantic.

Verify with the existing pace tests:

```bash
pytest tests/test_analysis/test_pace.py -v 2>&1 | head
```

If tests fail with assertions referencing the post-elapsed-time decline (i.e., they were calibrated against the buggy behavior), update each with a comment noting the metric is now "projection change" rather than "ROS+YTD - preseason".

- [ ] **Step 2: Failing test — transaction scoring uses ROS-only**

Add to `tests/test_analysis/test_transaction_scoring.py`:

```python
def test_score_transaction_ignores_ytd_inflation():
    """ΔRoto for a swap should not include the new player's YTD as 'gain'
    nor the dropped player's YTD as 'loss'. Setup mirrors the Cruz/Soto
    archetype: a hot-start player swapped for a cold-start player should
    show ΔRoto consistent with ROS-remaining diff, not full-season diff."""
    # Build minimal League, drop_txn, add_txn, hitters_proj (ROS-only),
    # call score_transaction, assert delta is in the ROS-only range.
```

(Construct minimal fixtures referencing the Cruz/Soto archetypes from Phase 4. Use `_TEAM_AB`/`_TEAM_IP` baseline.)

- [ ] **Step 3: Run, expect pass given Phase 1 dual-write**

```bash
pytest tests/test_analysis/test_transaction_scoring.py::test_score_transaction_ignores_ytd_inflation -v
```

Expected: PASS — because by this phase, the projections frame fed into `_lookup_player` is ROS-only.

- [ ] **Step 4: Run all transaction-scoring tests; update broken fixtures**

```bash
pytest tests/test_analysis/ -v
```

Existing tests asserting specific ΔRoto numbers will likely fail — they were calibrated against full-season-inflated numbers. For each failure: confirm the new expected value is the ROS-only one (compute by hand from fixtures), update with a comment.

- [ ] **Step 5: Update `tests/test_lineup/test_roster_audit.py` SGP gap fixtures**

Same exercise: the FA-gap and player_sgp values now reflect ROS-only inputs. Fixtures used to inject full-season counting stats; switch to ROS-only counting stats. Update assertions.

- [ ] **Step 6: Update `tests/test_lineup/test_optimizer.py` — optimizer outputs WILL change**

The optimizer calls `project_team_stats(hypothetical, displacement=True)` which now sums ROS-only. **This is the user-facing fix:** start/sit decisions previously favored hot-YTD players because their full-season totals were inflated; now they're scored on ROS-only forward projection. Existing tests asserting specific lineup choices will likely need updates.

For each failing test:
1. Confirm whether the fixture sets `Player.rest_of_season` and (if applicable) `Player.full_season_projection`. After Phase 2, `rest_of_season` is the only field the optimizer reads.
2. If the test asserted "hot-YTD player X gets the slot over cold-start Y", verify whether that assertion is still desirable. If the test was capturing the YTD bias, REWRITE the assertion to use ROS-only projection values that produce the intended winner. If the bias was the test's intent, that intent is now wrong — replace it with a forward-looking version.
3. Comment the change with `# updated for ROS-only optimizer per ros_only_decision_projections.md`.

Add a new positive test:

```python
def test_optimizer_ignores_ytd_when_choosing_starter():
    """Two players competing for one slot. Hot-YTD has more banked stats but
    weaker ROS; cold-YTD has stronger ROS. The optimizer should pick the
    cold-YTD player because we're choosing future contribution, not past."""
    # Construct rosters with a tied slot, hot-YTD vs cold-YTD ROS-only
    # projections, run the optimizer, assert the cold-YTD player is started.
```

- [ ] **Step 7: Run the entire suite**

```bash
pytest tests/ -v
```

Expected: All pass. Fix any straggler tests (with comments).

- [ ] **Step 8: Commit Phase 5**

```bash
git add src/fantasy_baseball/analysis/transactions.py src/fantasy_baseball/lineup/waivers.py src/fantasy_baseball/lineup/roster_audit.py tests/test_analysis/ tests/test_lineup/
git commit -m "fix(decisions): transaction/waiver/audit consumers use ROS-only projections

After Phase 1, cache:ros_projections is ROS-only, so consumers reading via
get_ros_projections automatically work in ROS-only units. Tests updated to
ROS-only fixture values. Numeric expectations changed because YTD is no
longer double-counted in swap deltas."
```

---

## Phase 6: End-to-end verification + dashboard sanity check

**Files:**
- None modified — this phase is verification only.

- [ ] **Step 1: Run the project's full check checklist**

```bash
pytest -v
ruff check .
ruff format --check .
vulture
mypy
```

All four must be clean. If `mypy` is configured for only specific files (per `pyproject.toml [tool.mypy].files`), it should still run on the touched files.

- [ ] **Step 2: Run a local end-to-end refresh**

Per the `Run refresh before merge` user-feedback rule (memory):

```bash
python scripts/run_season_dashboard.py
```

This exercises the full refresh pipeline against live Yahoo + Redis. Verify the dashboard renders. Look for:
- Roster audit page: ΔRoto values should now reflect ROS-only deltas (compare to a known sample by hand)
- Trade evaluator: opening a trade hypothetical should show plausible deltas
- Standings projection: end-of-season totals should still match Yahoo's projection space
- No 500s, no missing data warnings

- [ ] **Step 3: Hand-verify one decision against current Redis numbers**

Write a small ad-hoc script (kept as a one-liner, not committed):

```bash
PYTHONIOENCODING=utf-8 python -c "
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
from fantasy_baseball.data import redis_store as rs
kv = build_explicit_upstash_kv()
ros = rs.get_ros_projections(kv)
full = rs.get_full_season_projections(kv)
# Pick a hot-start player (Cruz) and confirm ros[Cruz].r < full[Cruz].r
cruz_ros = next(p for p in ros['hitters'] if p['name']=='Oneil Cruz')
cruz_full = next(p for p in full['hitters'] if p['name']=='Oneil Cruz')
print(f'Cruz: ROS R={cruz_ros[\"r\"]:.1f}  Full R={cruz_full[\"r\"]:.1f}  YTD R approx={cruz_full[\"r\"] - cruz_ros[\"r\"]:.1f}')
assert cruz_ros['r'] < cruz_full['r'], 'ROS must be strictly less than full when YTD > 0'
"
```

Expected: Cruz ROS R ≈ 68, Full R ≈ 87, YTD ≈ 19. (Numbers will have shifted by run date.)

- [ ] **Step 4: Push branch and open PR**

```bash
git push -u origin fix/ros-only-decision-projections
gh pr create --title "fix: forward-looking decisions use ROS-only projections" --body "$(cat <<'EOF'
## Summary

- `cache:ros_projections` now holds rest-of-season-only projections (FanGraphs CSV values, no YTD added)
- New `cache:full_season_projections` holds ROS+YTD for display surfaces only
- `Player.rest_of_season` is ROS-only (truth-named); optional `full_season_projection` for display
- `project_team_stats` sums ROS-only contribution (counting + rate components)
- `ProjectedStandings.from_rosters(rosters, current_standings, date)` combines current standings + ROS contribution to produce end-of-season totals — fixes a latent YTD-attribution bug for mid-season pickups
- `apply_swap_delta`, transaction scorer, waiver/audit, **and the lineup optimizer** all now value players by ROS-remaining; hot-YTD inflation no longer biases start/sit, drop/add, or trade decisions

## Why

Two coupled bugs:
1. The cached "ROS" was `ROS_remaining + YTD` (verified empirically: Cruz YTD=19R, cached "ROS"=87R). Forward-looking decisions were biased toward hot-start players and against cold-start players by the YTD differential.
2. `ProjectedStandings.from_rosters` summed full-season per current-roster player, attributing pre-acquisition YTD to teams that picked up players mid-season. Correct formulation is `current_standings + sum(roster.ROS_remaining)`.

Fixed structurally: ROS-only is the unit for all decision math; end-of-season is derived once at the standings layer.

## Test plan
- [x] Full pytest suite passes
- [x] Local end-to-end refresh produces a clean dashboard
- [x] Spot-check Cruz / Soto ROS vs full-season numbers in Redis match expected magnitudes
- [ ] Reviewer: open the trade evaluator on a hot/cold pair and sanity-check the delta
EOF
)"
```

- [ ] **Step 5: Self-review against the spec**

Re-read this plan from top. For each design decision (split cache, ROS-only as the decision unit, ProjectedStandings combines standings+ROS, optimizer uses ROS-only via `project_team_stats`), confirm an implementation phase covers it. Note any gaps for follow-up.

---

## Self-review notes

**Spec coverage:** Every consumer in the file-structure table has an implementation phase that touches it. The `eroto_recs.py` and `season_data.py` callers of `apply_swap_delta` are covered by Phase 4 implicitly — they call `player_rest_of_season_stats` which now returns ROS-only. They will need a spot-check during Phase 6 (Step 2). The lineup optimizer is fixed by the Phase 3 change to `project_team_stats`; Phase 5 Step 6 adds explicit verification.

**Placeholder scan:** Step 4 in Phase 4 says "find `test_compute_trade_impact_*`" — that's a directive to read the file before writing the assertion. The test code itself is concrete in the failing test added in Phase 4 Step 1 / Step 6.

**Type consistency:** The new field is consistently named `full_season_projection` (model) ↔ `cache:full_season_projections` (Redis key) ↔ `get_full_season_projections` (helper). The `rest_of_season` field name keeps its semantic — only its data shape changes (intentional behavior change documented in commits).

**Risk:** Tests will fail in droves during Phases 2–5 because numeric expectations were calibrated against the buggy (full-season-everywhere) world. The plan calls for case-by-case judgment ("confirm the new expected value is the ROS-only one") rather than blanket updates. This is per the `Tests are the guardrail` user-feedback rule. If a test failure looks like real broken behavior (e.g., a swap producing a nonsensical negative AVG), STOP and investigate the code, don't update the test.

**Dependency:** Phase 1's pipeline change writes both blobs. After deploying, the first refresh on Render will populate both keys. Until then, `cache:full_season_projections` is missing — `get_full_season_projections` returns None. The display surfaces that read it (player comparison) should fail loudly rather than silently fall back; the math layer doesn't read it at all after the corrected design.

**Critical pre-flight check:** Phase 3 Step 1 verifies `OpportunityStat.AB` is on the standings (needed for end-of-season AVG combination). If Yahoo standings don't surface AB through the existing ingest, that's a one-line fix that has to land **before** Phase 3 — otherwise `_combine_standings_with_ros` produces zero-AB combinations. Don't skip Step 1.

**Optimizer behavior change:** Phase 3 changes the lineup optimizer's outputs because `project_team_stats` now sums ROS-only. This is the user-facing fix the plan exists to deliver. Tests asserting specific lineup choices may flip — that's intentional. The new positive test in Phase 5 Step 6 (hot-YTD vs cold-start ROS comparison) locks in the corrected behavior.
