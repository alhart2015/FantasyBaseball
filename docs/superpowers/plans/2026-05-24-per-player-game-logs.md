# Per-Player Game Logs (Incremental, Box-Score Driven) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist raw per-player per-game logs keyed by player id, derive the existing rollup totals from them, fetch only games that are new or changed since a precise UTC high-water mark, and handle two-way players (Ohtani) correctly.

**Architecture:** The game-log step becomes box-score driven. `game/changes?updatedSince&sportId=1&season&gameType=R` (incremental) or `schedule` (one-time backfill) yields gamePks; each box score is parsed into per-player hitting/pitching rows, upserted by `gamePk` into per-player KV keys, and rolled up into the existing `game_log_totals:*` blobs. A position-player pitching filter keyed on `primaryPosition.code in {"1","Y"}` keeps Ohtani while dropping mop-up innings. A UTC watermark captured at fetch start and persisted only on full success makes the sync idempotent and gap-free. The public `fetch_game_log_totals(season, progress_cb)` signature is preserved, so both callers (`refresh_pipeline._fetch_game_logs`, `season_routes._run_rest_of_season_fetch`) and the refresh test fixture need no changes.

**Tech Stack:** Python 3, `requests` (MLB Stats API), `concurrent.futures` threadpool, the project's `KVStore` abstraction (`get/set` strings via `redis_store` helpers), pytest + `fakeredis`.

**Spec:** `docs/superpowers/specs/2026-05-24-per-player-game-logs-design.md`

**Branch:** `feat/per-player-game-logs`

---

## File Structure

- `src/fantasy_baseball/data/redis_store.py` (MODIFY) — add per-player game-log, watermark, player-position-cache, and dates helpers. This module owns the Redis schema; all new keys go here.
- `src/fantasy_baseball/analysis/game_logs.py` (MODIFY) — extract shared stat-block extractors (`hitter_stats_from_statblock`, `pitcher_stats_from_statblock`) used by both the existing gameLog parsers and the new box-score path. Pure refactor; existing behavior preserved.
- `src/fantasy_baseball/data/mlb_boxscore.py` (CREATE) — pure box-score parsing: iterate players, build per-game rows, decide pitching eligibility. No HTTP.
- `src/fantasy_baseball/data/mlb_game_logs.py` (REWRITE) — the sync engine: HTTP fetchers (changes feed / schedule / boxscore / people), threadpool, watermark, filter application, upsert, rollup derivation. Keeps the public `fetch_game_log_totals`.
- Tests: `tests/test_data/test_redis_store_game_logs.py` (EXTEND), `tests/test_analysis/test_game_logs.py` (EXTEND), `tests/test_data/test_mlb_boxscore.py` (CREATE), `tests/test_data/test_mlb_game_logs_sync.py` (CREATE).

Field names and values used below were verified live against `statsapi.mlb.com` on 2026-05-24 (Ohtani gamePk 776213; primaryPosition codes Ohtani=`Y`, Cole=`1`, Betts=`6`).

---

## PHASE 1 — Schema + Parsers

### Task 1: Redis schema helpers

**Files:**
- Modify: `src/fantasy_baseball/data/redis_store.py` (append after the existing `set_season_progress`, around line 282)
- Test: `tests/test_data/test_redis_store_game_logs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data/test_redis_store_game_logs.py`:

```python
def test_player_game_log_roundtrip(fake_redis):
    payload = {
        "name": "Shohei Ohtani",
        "games": [{"gamePk": 776213, "gameNumber": 1, "date": "2025-09-23",
                   "pa": 4, "ab": 3, "h": 0, "r": 1, "hr": 0, "rbi": 0, "sb": 0}],
    }
    redis_store.set_player_game_log(fake_redis, 2026, "660271", "hitting", payload)
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting") == payload


def test_player_game_log_missing_returns_none(fake_redis):
    assert redis_store.get_player_game_log(fake_redis, 2026, "999", "pitching") is None


def test_player_game_log_rejects_bad_group(fake_redis):
    with pytest.raises(ValueError, match="group must be"):
        redis_store.set_player_game_log(fake_redis, 2026, "1", "fielding", {})


def test_player_game_log_corrupt_returns_none(fake_redis):
    fake_redis.set("game_logs:2026:1:hitting", "not json")
    assert redis_store.get_player_game_log(fake_redis, 2026, "1", "hitting") is None


def test_watermark_roundtrip(fake_redis):
    assert redis_store.get_game_logs_watermark(fake_redis, 2026) is None
    redis_store.set_game_logs_watermark(fake_redis, 2026, "2026-05-24T13:00:00+00:00")
    assert redis_store.get_game_logs_watermark(fake_redis, 2026) == "2026-05-24T13:00:00+00:00"


def test_player_positions_roundtrip(fake_redis):
    assert redis_store.get_player_positions(fake_redis, 2026) == {}
    redis_store.set_player_positions(fake_redis, 2026, {"660271": "Y", "543037": "1"})
    assert redis_store.get_player_positions(fake_redis, 2026) == {"660271": "Y", "543037": "1"}


def test_game_log_dates_dedup_and_sort(fake_redis):
    redis_store.set_game_log_dates(fake_redis, 2026, ["2026-04-02", "2026-04-01", "2026-04-02"])
    assert redis_store.get_game_log_dates(fake_redis, 2026) == ["2026-04-01", "2026-04-02"]


def test_game_log_helpers_noop_on_none_client():
    assert redis_store.get_player_game_log(None, 2026, "1", "hitting") is None
    assert redis_store.set_player_game_log(None, 2026, "1", "hitting", {}) is None
    assert redis_store.get_game_logs_watermark(None, 2026) is None
    assert redis_store.get_player_positions(None, 2026) == {}
    assert redis_store.get_game_log_dates(None, 2026) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_redis_store_game_logs.py -v`
Expected: FAIL with `AttributeError: module 'fantasy_baseball.data.redis_store' has no attribute 'set_player_game_log'`

- [ ] **Step 3: Implement the helpers**

Append to `src/fantasy_baseball/data/redis_store.py`:

```python
# --- Per-player raw game logs (incremental, box-score driven) ---

_GAME_LOG_GROUPS = ("hitting", "pitching")


def _player_game_log_key(season: int, mlbam_id: str, group: str) -> str:
    if group not in _GAME_LOG_GROUPS:
        raise ValueError(f"group must be one of {_GAME_LOG_GROUPS}, got {group!r}")
    return f"game_logs:{season}:{mlbam_id}:{group}"


def get_player_game_log(client, season: int, mlbam_id: str, group: str) -> dict | None:
    """Read one player's per-game log for a group. None on missing/corrupt/None client."""
    if client is None:
        return None
    raw = client.get(_player_game_log_key(season, mlbam_id, group))
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def set_player_game_log(client, season: int, mlbam_id: str, group: str, payload: dict) -> None:
    """Overwrite one player's per-game log for a group. No-op when client is None."""
    if client is None:
        return
    client.set(_player_game_log_key(season, mlbam_id, group), json.dumps(payload))


def _game_logs_watermark_key(season: int) -> str:
    return f"game_logs:{season}:fetched_through_utc"


def get_game_logs_watermark(client, season: int) -> str | None:
    """Read the UTC high-water mark (ISO-8601). None when missing or client is None."""
    if client is None:
        return None
    return client.get(_game_logs_watermark_key(season))


def set_game_logs_watermark(client, season: int, iso_utc: str) -> None:
    """Persist the UTC high-water mark. No-op when client is None."""
    if client is None:
        return
    client.set(_game_logs_watermark_key(season), iso_utc)


def _player_positions_key(season: int) -> str:
    return f"game_logs:{season}:player_pos"


def get_player_positions(client, season: int) -> dict[str, str]:
    """Read the cached {mlbam_id: primaryPosition_code} map. Empty on missing/corrupt/None."""
    if client is None:
        return {}
    raw = client.get(_player_positions_key(season))
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def set_player_positions(client, season: int, positions: dict[str, str]) -> None:
    """Overwrite the cached primaryPosition map. No-op when client is None."""
    if client is None:
        return
    client.set(_player_positions_key(season), json.dumps(positions))


def _game_log_dates_key(season: int) -> str:
    return f"game_logs:{season}:dates"


def get_game_log_dates(client, season: int) -> list[str]:
    """Read the sorted list of distinct ingested game dates. Empty on missing/corrupt/None."""
    if client is None:
        return []
    raw = client.get(_game_log_dates_key(season))
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def set_game_log_dates(client, season: int, dates: list[str]) -> None:
    """Overwrite the distinct game-dates list (deduped + sorted). No-op when client is None."""
    if client is None:
        return
    client.set(_game_log_dates_key(season), json.dumps(sorted(set(dates))))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_redis_store_game_logs.py -v`
Expected: PASS (all, including the pre-existing rollup/season_progress tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/redis_store.py tests/test_data/test_redis_store_game_logs.py
git commit -m "feat(redis): per-player game-log, watermark, position-cache, dates helpers"
```

---

### Task 2: Share stat-block extraction in game_logs.py

**Files:**
- Modify: `src/fantasy_baseball/analysis/game_logs.py:54-90` (the two `parse_*_game_log` functions)
- Test: `tests/test_analysis/test_game_logs.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_analysis/test_game_logs.py`:

```python
from fantasy_baseball.analysis.game_logs import (
    hitter_stats_from_statblock,
    pitcher_stats_from_statblock,
)


def test_hitter_stats_from_statblock():
    stat = {"plateAppearances": 5, "atBats": 4, "hits": 2, "homeRuns": 1,
            "runs": 1, "rbi": 2, "stolenBases": 0}
    assert hitter_stats_from_statblock(stat) == {
        "pa": 5, "ab": 4, "h": 2, "hr": 1, "r": 1, "rbi": 2, "sb": 0}


def test_pitcher_stats_from_statblock_partial_innings():
    stat = {"inningsPitched": "6.1", "strikeOuts": 7, "earnedRuns": 3,
            "baseOnBalls": 2, "hits": 5, "wins": 0, "saves": 0,
            "gamesStarted": 1, "gamesPlayed": 1}
    out = pitcher_stats_from_statblock(stat)
    assert abs(out["ip"] - 6.3333) < 0.01
    assert out["k"] == 7 and out["er"] == 3 and out["h_allowed"] == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_game_logs.py -v`
Expected: FAIL with `ImportError: cannot import name 'hitter_stats_from_statblock'`

- [ ] **Step 3: Refactor to extract the shared functions**

In `src/fantasy_baseball/analysis/game_logs.py`, replace the bodies of `parse_hitter_game_log` and `parse_pitcher_game_log` (lines 54-90) with:

```python
def hitter_stats_from_statblock(stat: dict[str, Any]) -> dict[str, int]:
    """Extract hitter counting stats from an MLB stat block.

    Shared by the gameLog parser (``split["stat"]``) and the box-score
    parser (``player["stats"]["batting"]``) -- both use these field names.
    """
    return {
        "pa": int(stat.get("plateAppearances", 0)),
        "ab": int(stat.get("atBats", 0)),
        "h": int(stat.get("hits", 0)),
        "hr": int(stat.get("homeRuns", 0)),
        "r": int(stat.get("runs", 0)),
        "rbi": int(stat.get("rbi", 0)),
        "sb": int(stat.get("stolenBases", 0)),
    }


def pitcher_stats_from_statblock(stat: dict[str, Any]) -> dict[str, float | int]:
    """Extract pitcher counting stats from an MLB stat block.

    Shared by the gameLog parser and the box-score parser. ``inningsPitched``
    arrives as a string like ``"6.1"`` meaning 6 and 1/3 innings.
    """
    ip_str = str(stat.get("inningsPitched", "0"))
    if "." in ip_str:
        whole, frac = ip_str.split(".")
        ip = int(whole) + int(frac) / 3.0
    else:
        ip = float(ip_str)
    return {
        "ip": round(ip, 4),
        "k": int(stat.get("strikeOuts", 0)),
        "er": int(stat.get("earnedRuns", 0)),
        "bb": int(stat.get("baseOnBalls", 0)),
        "h_allowed": int(stat.get("hits", 0)),
        "w": int(stat.get("wins", 0)),
        "sv": int(stat.get("saves", 0)),
        "gs": int(stat.get("gamesStarted", 0)),
        "g": int(stat.get("gamesPlayed", 0)),
    }


def parse_hitter_game_log(split: dict[str, Any]) -> HitterGameLog:
    """Parse a single hitter game log entry from the MLB API."""
    return {"date": split["date"], **hitter_stats_from_statblock(split["stat"])}  # type: ignore[typeddict-item]


def parse_pitcher_game_log(split: dict[str, Any]) -> PitcherGameLog:
    """Parse a single pitcher game log entry from the MLB API."""
    return {"date": split["date"], **pitcher_stats_from_statblock(split["stat"])}  # type: ignore[typeddict-item]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_game_logs.py -v`
Expected: PASS, including the pre-existing `test_parse_hitter_game_log`, `test_parse_pitcher_game_log`, `test_parse_pitcher_partial_innings` (behavior is unchanged by the refactor).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/game_logs.py tests/test_analysis/test_game_logs.py
git commit -m "refactor(game-logs): share stat-block extraction across parsers"
```

---

### Task 3: Box-score parsing module

**Files:**
- Create: `src/fantasy_baseball/data/mlb_boxscore.py`
- Test: `tests/test_data/test_mlb_boxscore.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data/test_mlb_boxscore.py`:

```python
from fantasy_baseball.data.mlb_boxscore import (
    boxscore_hitter_row,
    boxscore_pitcher_row,
    iter_boxscore_players,
    should_record_pitching,
)

# Verified field names/values from real boxscore gamePk 776213 (Ohtani, two-way).
OHTANI = {
    "person": {"id": 660271, "fullName": "Shohei Ohtani"},
    "stats": {
        "batting": {"plateAppearances": 4, "atBats": 3, "hits": 0, "runs": 1,
                    "homeRuns": 0, "rbi": 0, "stolenBases": 0},
        "pitching": {"inningsPitched": "6.0", "strikeOuts": 8, "earnedRuns": 0,
                     "baseOnBalls": 0, "hits": 5, "wins": 0, "saves": 0,
                     "gamesStarted": 1, "gamesPlayed": 1},
    },
}
BENCH = {"person": {"id": 1, "fullName": "Did Not Play"}, "stats": {"batting": {}, "pitching": {}}}


def _box(*home_players):
    players = {f"ID{p['person']['id']}": p for p in home_players}
    return {"teams": {"home": {"players": players}, "away": {"players": {}}}}


def test_iter_skips_empty_stat_blocks():
    rows = list(iter_boxscore_players(_box(OHTANI, BENCH)))
    by_id = {mlbam: (bat, pit) for mlbam, _name, bat, pit in rows}
    assert by_id["660271"][0] and by_id["660271"][1]   # Ohtani: both populated
    assert not by_id["1"][0] and not by_id["1"][1]      # bench: both empty


def test_boxscore_hitter_row():
    _id, _name, bat, _pit = next(iter(iter_boxscore_players(_box(OHTANI))))
    assert boxscore_hitter_row(bat, 776213, 1, "2025-09-23") == {
        "gamePk": 776213, "gameNumber": 1, "date": "2025-09-23",
        "pa": 4, "ab": 3, "h": 0, "hr": 0, "r": 1, "rbi": 0, "sb": 0}


def test_boxscore_pitcher_row():
    _id, _name, _bat, pit = next(iter(iter_boxscore_players(_box(OHTANI))))
    assert boxscore_pitcher_row(pit, 776213, 1, "2025-09-23") == {
        "gamePk": 776213, "gameNumber": 1, "date": "2025-09-23",
        "ip": 6.0, "k": 8, "er": 0, "bb": 0, "h_allowed": 5, "w": 0, "sv": 0}


def test_should_record_pitching_keeps_pitchers_and_two_way():
    assert should_record_pitching("1") is True    # pitcher
    assert should_record_pitching("Y") is True    # two-way (Ohtani)
    assert should_record_pitching("6") is False   # position player (SS)
    assert should_record_pitching(None) is False  # unknown -> not a pitcher
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_mlb_boxscore.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fantasy_baseball.data.mlb_boxscore'`

- [ ] **Step 3: Implement the module**

Create `src/fantasy_baseball/data/mlb_boxscore.py`:

```python
"""Pure parsing of MLB Stats API box scores into per-player per-game rows.

These functions take an already-fetched box-score JSON dict plus the game
context (gamePk, gameNumber, date) and return per-player rows. All HTTP
lives in ``mlb_game_logs``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fantasy_baseball.analysis.game_logs import (
    hitter_stats_from_statblock,
    pitcher_stats_from_statblock,
)

# primaryPosition.code values that legitimately pitch: Pitcher and Two-Way.
# A position player who mops up in a blowout has a fielding code (2-10, "O", ...)
# and is filtered out. Ohtani is "Y" and is kept. Verified 2026-05-24.
PITCHER_POSITION_CODES = frozenset({"1", "Y"})


def iter_boxscore_players(
    boxscore: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """Yield (mlbam_id, name, batting_block, pitching_block) for each player.

    Empty ``{}`` blocks mean the player did not bat / pitch in this game.
    """
    teams = boxscore.get("teams", {})
    for side in ("home", "away"):
        players = teams.get(side, {}).get("players", {})
        for entry in players.values():
            person = entry.get("person", {})
            mlbam_id = person.get("id")
            if mlbam_id is None:
                continue
            stats = entry.get("stats", {})
            batting = stats.get("batting") or {}
            pitching = stats.get("pitching") or {}
            yield str(mlbam_id), person.get("fullName", ""), batting, pitching


def boxscore_hitter_row(
    batting: dict[str, Any], game_pk: int, game_number: int, date: str
) -> dict[str, Any]:
    """Build a hitting GameRow from a box-score batting block."""
    return {
        "gamePk": game_pk,
        "gameNumber": game_number,
        "date": date,
        **hitter_stats_from_statblock(batting),
    }


def boxscore_pitcher_row(
    pitching: dict[str, Any], game_pk: int, game_number: int, date: str
) -> dict[str, Any]:
    """Build a pitching GameRow from a box-score pitching block."""
    s = pitcher_stats_from_statblock(pitching)
    return {
        "gamePk": game_pk,
        "gameNumber": game_number,
        "date": date,
        "ip": s["ip"],
        "k": s["k"],
        "er": s["er"],
        "bb": s["bb"],
        "h_allowed": s["h_allowed"],
        "w": s["w"],
        "sv": s["sv"],
    }


def should_record_pitching(pos_code: str | None) -> bool:
    """True only for real pitchers ("1") and two-way players ("Y").

    Drops position-player mop-up innings. Returns False for an unknown
    position; the sync engine treats unknown-after-fetch separately (it
    declines to advance the watermark so the game is retried).
    """
    return pos_code in PITCHER_POSITION_CODES
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_mlb_boxscore.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/mlb_boxscore.py tests/test_data/test_mlb_boxscore.py
git commit -m "feat(boxscore): pure per-player box-score parsing + pitching filter"
```

**PHASE 1 GATE:** Run `pytest tests/test_data/test_redis_store_game_logs.py tests/test_analysis/test_game_logs.py tests/test_data/test_mlb_boxscore.py -v` and `ruff check src/fantasy_baseball/data/redis_store.py src/fantasy_baseball/data/mlb_boxscore.py src/fantasy_baseball/analysis/game_logs.py`. All green before Phase 2.

---

## PHASE 2 — Sync Engine

### Task 4: Merge + sum pure helpers

**Files:**
- Create: `src/fantasy_baseball/data/mlb_game_logs.py` (begin the rewrite with the pure helpers; the old contents are fully replaced over Tasks 4-6)
- Test: `tests/test_data/test_mlb_game_logs_sync.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data/test_mlb_game_logs_sync.py`:

```python
from fantasy_baseball.data.mlb_game_logs import (
    _merge_player_games,
    _sum_hitting,
    _sum_pitching,
)


def test_merge_upserts_by_gamepk_and_sorts_by_date():
    existing = {"name": "X", "games": [
        {"gamePk": 2, "gameNumber": 1, "date": "2026-04-02", "pa": 4, "ab": 4, "h": 1,
         "hr": 0, "r": 0, "rbi": 0, "sb": 0}]}
    new_rows = {
        1: {"gamePk": 1, "gameNumber": 1, "date": "2026-04-01", "pa": 3, "ab": 3, "h": 2,
            "hr": 1, "r": 1, "rbi": 2, "sb": 0},
        2: {"gamePk": 2, "gameNumber": 1, "date": "2026-04-02", "pa": 5, "ab": 5, "h": 3,
            "hr": 0, "r": 1, "rbi": 1, "sb": 1},  # correction overwrites gamePk 2
    }
    merged = _merge_player_games(existing, "X", new_rows)
    assert [g["gamePk"] for g in merged["games"]] == [1, 2]   # sorted by date
    assert merged["games"][1]["h"] == 3                        # corrected value won


def test_merge_handles_doubleheader_same_date():
    new_rows = {
        10: {"gamePk": 10, "gameNumber": 1, "date": "2026-07-04", "pa": 4, "ab": 4, "h": 1,
             "hr": 0, "r": 0, "rbi": 0, "sb": 0},
        11: {"gamePk": 11, "gameNumber": 2, "date": "2026-07-04", "pa": 3, "ab": 3, "h": 2,
             "hr": 1, "r": 1, "rbi": 1, "sb": 0},
    }
    merged = _merge_player_games(None, "DH", new_rows)
    assert len(merged["games"]) == 2
    assert {g["gameNumber"] for g in merged["games"]} == {1, 2}


def test_sum_hitting():
    games = [
        {"pa": 4, "ab": 4, "h": 1, "hr": 0, "r": 0, "rbi": 0, "sb": 0},
        {"pa": 5, "ab": 5, "h": 3, "hr": 1, "r": 1, "rbi": 1, "sb": 1},
    ]
    assert _sum_hitting(games) == {"pa": 9, "ab": 9, "h": 4, "hr": 1, "r": 1, "rbi": 1, "sb": 1}


def test_sum_pitching_rounds_ip():
    games = [
        {"ip": 6.3333, "k": 8, "er": 0, "bb": 0, "h_allowed": 5, "w": 0, "sv": 0},
        {"ip": 1.0, "k": 1, "er": 1, "bb": 1, "h_allowed": 2, "w": 0, "sv": 1},
    ]
    out = _sum_pitching(games)
    assert out == {"ip": 7.3333, "k": 9, "er": 1, "bb": 1, "h_allowed": 7, "w": 0, "sv": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_mlb_game_logs_sync.py -v`
Expected: FAIL with `ImportError: cannot import name '_merge_player_games'` (the old module has no such symbol).

- [ ] **Step 3: Replace the module with its new header + pure helpers**

Overwrite `src/fantasy_baseball/data/mlb_game_logs.py` entirely with:

```python
"""Incremental, box-score-driven MLB game-log sync.

Persists raw per-player per-game rows (keyed by player id and group) and
derives the rolled-up ``game_log_totals:*`` blobs that power existing
calcs. Each refresh pulls only games new or changed since a precise UTC
high-water mark; the one-time backfill enumerates the season via schedule.

Two-way players (Ohtani) are handled by parsing box scores -- both their
batting and pitching blocks are recorded -- and a position-player pitching
filter keyed on primaryPosition keeps Ohtani while dropping mop-up innings.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import requests

from fantasy_baseball.data.mlb_boxscore import (
    boxscore_hitter_row,
    boxscore_pitcher_row,
    iter_boxscore_players,
    should_record_pitching,
)
from fantasy_baseball.data.redis_store import (
    get_game_log_dates,
    get_game_log_totals,
    get_game_logs_watermark,
    get_player_game_log,
    get_player_positions,
    set_game_log_dates,
    set_game_log_totals,
    set_game_logs_watermark,
    set_player_game_log,
    set_player_positions,
    set_season_progress,
)

_MLB_API = "https://statsapi.mlb.com/api/v1"
_HITTER_KEYS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")
_PITCHER_KEYS = ("ip", "k", "er", "bb", "h_allowed", "w", "sv")


def _merge_player_games(
    existing: dict[str, Any] | None, name: str, new_rows: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Merge new per-game rows (keyed by gamePk) into a player's stored log.

    New rows overwrite stored rows with the same gamePk (corrections
    self-heal). Games are returned sorted by (date, gamePk).
    """
    by_pk: dict[int, dict[str, Any]] = {}
    if existing:
        for g in existing.get("games", []):
            by_pk[g["gamePk"]] = g
    by_pk.update(new_rows)
    games = sorted(by_pk.values(), key=lambda r: (r["date"], r["gamePk"]))
    resolved_name = name or (existing or {}).get("name", "")
    return {"name": resolved_name, "games": games}


def _sum_hitting(games: list[dict[str, Any]]) -> dict[str, int]:
    out = {k: 0 for k in _HITTER_KEYS}
    for g in games:
        for k in _HITTER_KEYS:
            out[k] += g.get(k, 0) or 0
    return out


def _sum_pitching(games: list[dict[str, Any]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {k: 0 for k in _PITCHER_KEYS}
    for g in games:
        out["ip"] += g.get("ip", 0.0) or 0.0
        for k in ("k", "er", "bb", "h_allowed", "w", "sv"):
            out[k] += g.get(k, 0) or 0
    out["ip"] = round(out["ip"], 4)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_mlb_game_logs_sync.py -v`
Expected: PASS (the four pure-helper tests)

- [ ] **Step 5: Verify the old API surface is intentionally gone, then commit**

Run: `pytest tests/test_web/ -v -k refresh` — Expected: FAIL (the refresh fixture patches `fetch_game_log_totals`, which no longer exists yet). This is expected mid-rewrite; Task 6 restores the symbol. Do NOT fix the fixture.

```bash
git add src/fantasy_baseball/data/mlb_game_logs.py tests/test_data/test_mlb_game_logs_sync.py
git commit -m "feat(game-logs): begin sync engine rewrite with merge/sum helpers"
```

---

### Task 5: HTTP fetchers + regular-season-final filter

**Files:**
- Modify: `src/fantasy_baseball/data/mlb_game_logs.py` (append)
- Test: `tests/test_data/test_mlb_game_logs_sync.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data/test_mlb_game_logs_sync.py`:

```python
from fantasy_baseball.data import mlb_game_logs


def test_is_regular_final():
    final_reg = {"gameType": "R", "status": {"abstractGameState": "Final"}}
    live_reg = {"gameType": "R", "status": {"abstractGameState": "Live"}}
    final_spring = {"gameType": "S", "status": {"abstractGameState": "Final"}}
    assert mlb_game_logs._is_regular_final(final_reg) is True
    assert mlb_game_logs._is_regular_final(live_reg) is False
    assert mlb_game_logs._is_regular_final(final_spring) is False


def test_fetch_changed_games_flattens_dates(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self): ...
        def json(self):
            return {"dates": [{"games": [{"gamePk": 1}, {"gamePk": 2}]},
                              {"games": [{"gamePk": 3}]}]}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _Resp()

    monkeypatch.setattr(mlb_game_logs.requests, "get", fake_get)
    games = mlb_game_logs._fetch_changed_games(2026, "2026-05-24T00:00:00+00:00")
    assert [g["gamePk"] for g in games] == [1, 2, 3]
    assert captured["params"] == {
        "updatedSince": "2026-05-24T00:00:00+00:00", "sportId": 1, "season": 2026}
    assert captured["url"].endswith("/game/changes")


def test_fetch_positions_maps_id_to_code(monkeypatch):
    class _Resp:
        def raise_for_status(self): ...
        def json(self):
            return {"people": [
                {"id": 660271, "primaryPosition": {"code": "Y"}},
                {"id": 543037, "primaryPosition": {"code": "1"}}]}

    monkeypatch.setattr(mlb_game_logs.requests, "get", lambda *a, **k: _Resp())
    assert mlb_game_logs._fetch_positions([660271, 543037]) == {"660271": "Y", "543037": "1"}


def test_fetch_positions_empty_short_circuits(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not call the API for an empty id list")

    monkeypatch.setattr(mlb_game_logs.requests, "get", boom)
    assert mlb_game_logs._fetch_positions([]) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_mlb_game_logs_sync.py -v -k "regular_final or fetch_changed or fetch_positions"`
Expected: FAIL with `AttributeError: ... has no attribute '_is_regular_final'`

- [ ] **Step 3: Implement the fetchers + filter**

Append to `src/fantasy_baseball/data/mlb_game_logs.py`:

```python
def _is_regular_final(game: dict[str, Any]) -> bool:
    """True for a completed regular-season game."""
    return (
        game.get("gameType") == "R"
        and game.get("status", {}).get("abstractGameState") == "Final"
    )


def _game_context(game: dict[str, Any]) -> tuple[int, int, str]:
    """(gamePk, gameNumber, officialDate) from a schedule/changes game dict."""
    game_pk = game["gamePk"]
    game_number = game.get("gameNumber", 1)
    date = game.get("officialDate") or (game.get("gameDate") or "")[:10]
    return game_pk, game_number, date


def _fetch_changed_games(season: int, since_iso: str) -> list[dict[str, Any]]:
    """MLB games (all types) changed since ``since_iso``, scoped to MLB + season."""
    resp = requests.get(
        f"{_MLB_API}/game/changes",
        params={"updatedSince": since_iso, "sportId": 1, "season": season},
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    return [g for d in data.get("dates", []) for g in d.get("games", [])]


def _fetch_season_games(season: int) -> list[dict[str, Any]]:
    """All regular-season MLB games for ``season`` (backfill enumeration)."""
    resp = requests.get(
        f"{_MLB_API}/schedule",
        params={"sportId": 1, "season": season, "gameType": "R"},
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    return [g for d in data.get("dates", []) for g in d.get("games", [])]


def _fetch_boxscore(game_pk: int) -> dict[str, Any]:
    resp = requests.get(f"{_MLB_API}/game/{game_pk}/boxscore", timeout=20)
    resp.raise_for_status()
    return resp.json()


def _fetch_positions(mlbam_ids: list[int]) -> dict[str, str]:
    """Batch primaryPosition.code lookup. {str(id): code}; code may be None."""
    if not mlbam_ids:
        return {}
    resp = requests.get(
        f"{_MLB_API}/people",
        params={"personIds": ",".join(str(i) for i in mlbam_ids)},
        timeout=20,
    )
    resp.raise_for_status()
    out: dict[str, str] = {}
    for person in resp.json().get("people", []):
        out[str(person["id"])] = person.get("primaryPosition", {}).get("code")
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_mlb_game_logs_sync.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/mlb_game_logs.py tests/test_data/test_mlb_game_logs_sync.py
git commit -m "feat(game-logs): MLB changes/schedule/boxscore/people fetchers"
```

---

### Task 6: Sync orchestration + public entry point

**Files:**
- Modify: `src/fantasy_baseball/data/mlb_game_logs.py` (append)
- Test: `tests/test_data/test_mlb_game_logs_sync.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data/test_mlb_game_logs_sync.py`:

```python
from datetime import datetime, timezone

from fantasy_baseball.data import redis_store

# Reusable synthetic box scores (verified field names).
_OHTANI = {
    "person": {"id": 660271, "fullName": "Shohei Ohtani"},
    "stats": {
        "batting": {"plateAppearances": 4, "atBats": 3, "hits": 0, "runs": 1,
                    "homeRuns": 0, "rbi": 0, "stolenBases": 0},
        "pitching": {"inningsPitched": "6.0", "strikeOuts": 8, "earnedRuns": 0,
                     "baseOnBalls": 0, "hits": 5, "wins": 0, "saves": 0,
                     "gamesStarted": 1, "gamesPlayed": 1},
    },
}
# A shortstop (code "6") who mopped up an inning -> pitching must be dropped.
_BETTS_MOPUP = {
    "person": {"id": 605141, "fullName": "Mookie Betts"},
    "stats": {
        "batting": {"plateAppearances": 5, "atBats": 5, "hits": 2, "runs": 1,
                    "homeRuns": 0, "rbi": 1, "stolenBases": 0},
        "pitching": {"inningsPitched": "1.0", "strikeOuts": 0, "earnedRuns": 3,
                     "baseOnBalls": 1, "hits": 2, "wins": 0, "saves": 0,
                     "gamesStarted": 0, "gamesPlayed": 1},
    },
}


def _final(game_pk, date, game_number=1):
    return {"gamePk": game_pk, "gameNumber": game_number, "officialDate": date,
            "gameType": "R", "status": {"abstractGameState": "Final"}}


def _patch_mlb(monkeypatch, *, season_games=None, changed_games=None,
               boxscores=None, positions=None, positions_raises=False):
    monkeypatch.setattr(mlb_game_logs, "_fetch_season_games",
                        lambda s: season_games or [])
    monkeypatch.setattr(mlb_game_logs, "_fetch_changed_games",
                        lambda s, since: changed_games or [])
    monkeypatch.setattr(mlb_game_logs, "_fetch_boxscore",
                        lambda gp: (boxscores or {})[gp])

    def _pos(ids):
        if positions_raises:
            raise RuntimeError("people endpoint down")
        return {str(i): (positions or {}).get(str(i)) for i in ids}

    monkeypatch.setattr(mlb_game_logs, "_fetch_positions", _pos)


NOW = datetime(2026, 5, 24, 13, 0, tzinfo=timezone.utc)


def test_backfill_records_two_way_and_filters_mopup(monkeypatch, fake_redis):
    _patch_mlb(
        monkeypatch,
        season_games=[_final(100, "2026-04-01")],
        boxscores={100: {"teams": {"home": {"players": {
            "ID660271": _OHTANI, "ID605141": _BETTS_MOPUP}}, "away": {"players": {}}}}},
        positions={"660271": "Y", "605141": "6"},
    )
    mlb_game_logs.sync_game_logs(fake_redis, 2026, now_utc=NOW)

    # Ohtani: both halves stored and in both rollups.
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting")["games"]
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "pitching")["games"]
    hitters = redis_store.get_game_log_totals(fake_redis, "hitters")
    pitchers = redis_store.get_game_log_totals(fake_redis, "pitchers")
    assert hitters["660271"]["ab"] == 3 and hitters["660271"]["name"] == "Shohei Ohtani"
    assert pitchers["660271"]["k"] == 8 and pitchers["660271"]["ip"] == 6.0

    # Betts: batting kept, pitching dropped (position player).
    assert hitters["605141"]["h"] == 2
    assert "605141" not in pitchers
    assert redis_store.get_player_game_log(fake_redis, 2026, "605141", "pitching") is None

    # Watermark advanced; games_elapsed reflects the one date.
    assert redis_store.get_game_logs_watermark(fake_redis, 2026) == NOW.isoformat()
    assert redis_store.get_season_progress(fake_redis)["games_elapsed"] == 1


def test_incremental_correction_overwrites_by_gamepk(monkeypatch, fake_redis):
    redis_store.set_game_logs_watermark(fake_redis, 2026, "2026-05-23T13:00:00+00:00")
    redis_store.set_player_game_log(fake_redis, 2026, "660271", "hitting", {
        "name": "Shohei Ohtani",
        "games": [{"gamePk": 100, "gameNumber": 1, "date": "2026-04-01",
                   "pa": 4, "ab": 3, "h": 0, "hr": 0, "r": 1, "rbi": 0, "sb": 0}]})
    # A correction: the box score now shows 2 hits for gamePk 100.
    corrected = {**_OHTANI, "stats": {**_OHTANI["stats"],
                 "batting": {**_OHTANI["stats"]["batting"], "hits": 2}}}
    _patch_mlb(
        monkeypatch,
        changed_games=[_final(100, "2026-04-01")],
        boxscores={100: {"teams": {"home": {"players": {"ID660271": corrected}},
                                   "away": {"players": {}}}}},
        positions={"660271": "Y"},
    )
    mlb_game_logs.sync_game_logs(fake_redis, 2026, now_utc=NOW)
    games = redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting")["games"]
    assert len(games) == 1 and games[0]["h"] == 2          # overwritten, not duplicated
    assert redis_store.get_game_log_totals(fake_redis, "hitters")["660271"]["h"] == 2


def test_watermark_not_advanced_when_position_unresolved(monkeypatch, fake_redis):
    redis_store.set_game_logs_watermark(fake_redis, 2026, "2026-05-23T13:00:00+00:00")
    _patch_mlb(
        monkeypatch,
        changed_games=[_final(100, "2026-04-01")],
        boxscores={100: {"teams": {"home": {"players": {"ID660271": _OHTANI}},
                                   "away": {"players": {}}}}},
        positions_raises=True,   # people endpoint blip
    )
    mlb_game_logs.sync_game_logs(fake_redis, 2026, now_utc=NOW)
    # Batting still stored (idempotent); pitching deferred; watermark unchanged.
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting")["games"]
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "pitching") is None
    assert redis_store.get_game_logs_watermark(fake_redis, 2026) == "2026-05-23T13:00:00+00:00"


def test_fetch_game_log_totals_preserves_public_contract(monkeypatch, fake_redis):
    monkeypatch.setattr(mlb_game_logs, "get_kv", lambda: fake_redis, raising=False)
    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)
    _patch_mlb(
        monkeypatch,
        season_games=[_final(100, "2026-04-01")],
        boxscores={100: {"teams": {"home": {"players": {"ID660271": _OHTANI}},
                                   "away": {"players": {}}}}},
        positions={"660271": "Y"},
    )
    hitters, pitchers, games_elapsed = mlb_game_logs.fetch_game_log_totals(2026)
    assert hitters["660271"]["ab"] == 3
    assert pitchers["660271"]["k"] == 8
    assert games_elapsed == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_mlb_game_logs_sync.py -v -k "backfill or incremental or watermark_not or public_contract"`
Expected: FAIL with `AttributeError: ... has no attribute 'sync_game_logs'`

- [ ] **Step 3: Implement the orchestration + public function**

Append to `src/fantasy_baseball/data/mlb_game_logs.py`:

```python
def _fetch_boxscores(
    games: list[dict[str, Any]], progress_cb
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    """Fetch box scores in parallel. Returns ({gamePk: boxscore}, failed_gamePks)."""
    results: dict[int, dict[str, Any]] = {}
    failed: set[int] = set()

    def _one(game: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        gp = game["gamePk"]
        try:
            return gp, _fetch_boxscore(gp)
        except Exception:
            return gp, None

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = [pool.submit(_one, g) for g in games]
        for i, future in enumerate(as_completed(futures), 1):
            gp, box = future.result()
            if box is None:
                failed.add(gp)
            else:
                results[gp] = box
            if progress_cb and i % 50 == 0:
                progress_cb(f"Box scores: {i}/{len(games)}...")
    return results, failed


def _resolve_positions(client, season: int, pitching_ids: list[str]) -> dict[str, str]:
    """Return {mlbam_id: pos_code} for the given ids, fetching uncached ones.

    On a fetch failure the missing ids stay absent so the caller can retry.
    A fetched-but-null code is cached as null (treated as "not a pitcher").
    """
    cache = get_player_positions(client, season)
    missing = [pid for pid in pitching_ids if pid not in cache]
    if missing:
        try:
            fetched = _fetch_positions([int(p) for p in missing])
        except Exception:
            return cache
        for pid in missing:
            cache[pid] = fetched.get(pid)
        set_player_positions(client, season, cache)
    return cache


def _upsert_and_roll(client, season, group, by_player, sum_fn) -> None:
    """Merge each player's new rows into their stored log and refresh the rollup."""
    if not by_player:
        return
    rollup_type = "hitters" if group == "hitting" else "pitchers"
    rollup = get_game_log_totals(client, rollup_type)
    for mlbam_id, payload in by_player.items():
        existing = get_player_game_log(client, season, mlbam_id, group)
        merged = _merge_player_games(existing, payload["name"], payload["rows"])
        set_player_game_log(client, season, mlbam_id, group, merged)
        rollup[mlbam_id] = {"name": merged["name"], **sum_fn(merged["games"])}
    set_game_log_totals(client, rollup_type, rollup)


def _sync(client, season: int, games: list[dict[str, Any]], now_utc: datetime, progress_cb) -> None:
    boxscores, failed = _fetch_boxscores(games, progress_cb)
    all_ok = not failed
    ctx = {g["gamePk"]: _game_context(g) for g in games}

    hitting: dict[str, dict[str, Any]] = {}
    pitching: dict[str, dict[str, Any]] = {}
    dates: set[str] = set()
    for gp, box in boxscores.items():
        _pk, gnum, date = ctx[gp]
        dates.add(date)
        for mlbam_id, name, batting, pitch in iter_boxscore_players(box):
            if batting:
                h = hitting.setdefault(mlbam_id, {"name": name, "rows": {}})
                h["name"] = name or h["name"]
                h["rows"][gp] = boxscore_hitter_row(batting, gp, gnum, date)
            if pitch:
                p = pitching.setdefault(mlbam_id, {"name": name, "rows": {}})
                p["name"] = name or p["name"]
                p["rows"][gp] = boxscore_pitcher_row(pitch, gp, gnum, date)

    positions = _resolve_positions(client, season, list(pitching.keys()))
    kept_pitching: dict[str, dict[str, Any]] = {}
    for mlbam_id, payload in pitching.items():
        if mlbam_id not in positions:
            all_ok = False  # unresolved (fetch blip) -> retry next run
            continue
        if should_record_pitching(positions[mlbam_id]):
            kept_pitching[mlbam_id] = payload

    _upsert_and_roll(client, season, "hitting", hitting, _sum_hitting)
    _upsert_and_roll(client, season, "pitching", kept_pitching, _sum_pitching)

    if dates:
        merged_dates = set(get_game_log_dates(client, season)) | dates
        set_game_log_dates(client, season, list(merged_dates))
    set_season_progress(
        client,
        games_elapsed=len(get_game_log_dates(client, season)),
        total=162,
        as_of=now_utc.date().isoformat(),
    )

    if all_ok:
        set_game_logs_watermark(client, season, now_utc.isoformat())
    if progress_cb:
        progress_cb(f"Game logs synced: {len(games)} games (clean={all_ok})")


def sync_game_logs(client, season: int, *, progress_cb=None, now_utc: datetime | None = None) -> None:
    """Backfill (no watermark) or incremental (changes feed) sync into ``client``."""
    now_utc = now_utc or datetime.now(timezone.utc)
    watermark = get_game_logs_watermark(client, season)
    if watermark is None:
        if progress_cb:
            progress_cb("No watermark; backfilling full season game logs...")
        games = [g for g in _fetch_season_games(season) if _is_regular_final(g)]
    else:
        if progress_cb:
            progress_cb(f"Incremental game-log sync since {watermark}...")
        games = [g for g in _fetch_changed_games(season, watermark) if _is_regular_final(g)]
    _sync(client, season, games, now_utc, progress_cb)


def fetch_game_log_totals(season: int, progress_cb=None) -> tuple[dict, dict, int]:
    """Sync game logs and return (hitters_totals, pitchers_totals, games_elapsed).

    Public entry point preserved for ``refresh_pipeline`` and
    ``season_routes``; both ignore the return value. The totals are read
    back from the derived rollup so the shape is unchanged.
    """
    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.data.redis_store import get_game_log_totals, get_season_progress

    client = get_kv()
    sync_game_logs(client, season, progress_cb=progress_cb)
    hitters = get_game_log_totals(client, "hitters")
    pitchers = get_game_log_totals(client, "pitchers")
    games_elapsed = get_season_progress(client)["games_elapsed"]
    return hitters, pitchers, games_elapsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_mlb_game_logs_sync.py -v`
Expected: PASS (all sync tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/mlb_game_logs.py tests/test_data/test_mlb_game_logs_sync.py
git commit -m "feat(game-logs): box-score sync engine (backfill + incremental + watermark)"
```

**PHASE 2 GATE:** Run `pytest tests/test_data/ tests/test_analysis/ -v` and `ruff check src/fantasy_baseball/data/mlb_game_logs.py`. All green before Phase 3.

---

## PHASE 3 — Wire-In + Verification

The public `fetch_game_log_totals(season, progress_cb)` is unchanged, so `refresh_pipeline._fetch_game_logs` (`refresh_pipeline.py:760-766`) and `season_routes._run_rest_of_season_fetch` (`season_routes.py:193-213`) need no edits. This phase confirms the integration and the full quality gate.

### Task 7: Confirm the refresh-pipeline integration

**Files:**
- Read/verify: `tests/test_web/_refresh_fixture.py:289,517-520`
- Modify only if the run reveals a break.

- [ ] **Step 1: Run the refresh-pipeline tests**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS. The fixture patches `fantasy_baseball.data.mlb_game_logs.fetch_game_log_totals` with `_fetch_game_logs` (which seeds the canned `game_log_totals:*` blobs) — that symbol still exists with the same signature, and downstream consumers read the unchanged rollup keys.

- [ ] **Step 2: If (and only if) it fails because the patched symbol moved**

The patch target must remain `fantasy_baseball.data.mlb_game_logs.fetch_game_log_totals`. The rewrite keeps it there, so no change should be needed. If a failure points at the import, confirm the function is module-level in `mlb_game_logs.py` (it is, per Task 6). Do not weaken the fixture's assertions.

- [ ] **Step 3: Run the broader web + sgp suites that read the rollup**

Run: `pytest tests/test_web/ tests/test_sgp/test_rankings.py -v`
Expected: PASS (`_load_game_log_totals` and `compute_rankings_from_game_logs` consume the unchanged `game_log_totals:*` shape).

- [ ] **Step 4: Commit (only if a fixture edit was required)**

```bash
git add tests/test_web/_refresh_fixture.py
git commit -m "test(refresh): adapt game-log mock to box-score sync engine"
```

If no edit was needed, skip this commit.

---

### Task 8: Full quality gate + dead-code sweep

**Files:**
- Verify across the repo.

- [ ] **Step 1: Confirm no stale references to removed internals**

The old module's `_HITTER_STATS`, `_PITCHER_STATS`, `_empty_hitter_total`, `_empty_pitcher_total`, and the roster-enumeration loop are gone. Confirm nothing else referenced them:

Run: `grep -rn "_empty_hitter_total\|_empty_pitcher_total\|import statsapi" src/ tests/`
Expected: no matches in `mlb_game_logs.py` (statsapi may legitimately appear elsewhere; confirm none point at the rewritten module).

- [ ] **Step 2: Lint, format, dead-code, types**

```bash
ruff check .
ruff format --check .
vulture
mypy
```
Expected: `ruff check` zero violations; `ruff format --check` no drift (run `ruff format .` if it reports changes); `vulture` introduces no NEW findings beyond pre-existing ones (note any pre-existing ones you see); `mypy` clean for any touched file listed under `[tool.mypy].files` in `pyproject.toml` (check whether `data/mlb_game_logs.py`, `data/mlb_boxscore.py`, `analysis/game_logs.py`, `data/redis_store.py` are covered).

- [ ] **Step 3: Full test suite**

Run: `pytest -n auto`
Expected: PASS. If anything fails, fix the code (not the test) unless the test asserts incidental behavior — in which case stop and surface it.

- [ ] **Step 4: Final commit (formatting/lint fixes, if any)**

```bash
git add -A
git commit -m "chore(game-logs): lint/format/type fixes for per-player sync"
```

---

## Self-Review

**1. Spec coverage:**
- Per-player keyed store -> Task 1 (`set_player_game_log`) + Task 6 (`_upsert_and_roll`). ✓
- Rollup derived, byte-compatible, Ohtani in both -> Task 6 (`_upsert_and_roll`, `test_backfill_records_two_way...`). ✓
- Incremental via changes feed scoped sportId=1 + season + gameType R -> Task 5 (`_fetch_changed_games`, `_is_regular_final`). ✓
- Backfill via schedule -> Task 5 (`_fetch_season_games`) + Task 6 (`sync_game_logs` no-watermark branch). ✓
- Watermark precise UTC, captured at fetch start, advance only on success -> Task 6 (`_sync`, `test_watermark_not_advanced...`). ✓
- Ohtani / position-player filter via primaryPosition -> Task 3 (`should_record_pitching`) + Task 6 (`_resolve_positions`, Betts test). ✓
- Doubleheader via gamePk upsert -> Task 4 (`test_merge_handles_doubleheader_same_date`). ✓
- Correction self-heal -> Task 4 + Task 6 (`test_incremental_correction_overwrites_by_gamepk`). ✓
- `games_elapsed` from dates set (no sorted-set use) -> Task 1 (`*_game_log_dates`) + Task 6 (`_sync`). ✓
- Public signature preserved (no wire-in churn) -> Task 6 + Task 7. ✓
- Refinement vs spec: backfill builds the rollup from the same upsert path (existing key is empty), so `keys()`/`mget()` are not needed — the spec's "full rebuild via keys()+mget()" note is satisfied more cheaply and that risk is moot.

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code and exact commands.

**3. Type consistency:** `_merge_player_games`, `_sum_hitting`, `_sum_pitching`, `_upsert_and_roll`, `_resolve_positions`, `_sync`, `sync_game_logs`, `fetch_game_log_totals`, and the `mlb_boxscore` functions use consistent names and signatures across tasks. Group strings are exactly `"hitting"`/`"pitching"`; rollup types exactly `"hitters"`/`"pitchers"`; stat-key tuples `_HITTER_KEYS`/`_PITCHER_KEYS` match the rollup shape consumed by `_load_game_log_totals`.
