# CacheKey Enum — Design

**Date:** 2026-04-17
**Branch:** `cache-key-enum`
**TODO item:** "Use an enum for cache key names" (TODO.md line 33)

## Problem

`read_cache` and `write_cache` in `src/fantasy_baseball/web/season_data.py` accept bare
strings (`"leverage"`, `"roster"`, `"projections"`, …). A typo produces a silent bug:

- On write, the data lands under the wrong key (or raises `KeyError` from `CACHE_FILES`,
  depending on which string is mistyped).
- On read, the caller gets `None` and falls back to a default, masking the bug.

The `"cache:"` Redis-key prefix is also duplicated across three files
(`season_data.py`, `data/redis_store.py`, `scripts/rescore_transactions.py`), so
the set of valid cache keys has no single source of truth.

## Goals

1. Make typos in cache-key names fail at import time, not silently at runtime.
2. Centralize the list of valid cache keys in one place.
3. Centralize the `"cache:"` Redis-key prefix in one place.
4. Keep the change mechanical — no behavioral change, no new tests required.

## Non-goals

- No mypy expansion. `season_data.py` stays outside `[tool.mypy].files`; that
  expansion is tracked separately under the mypy-migration TODO.
- No change to on-disk JSON filenames, Redis key names, or cache semantics.
- No backward-compat shim accepting `str`. The migration converts every call
  site in the same set of phases.

## Approach

### `CacheKey` enum

Add a `CacheKey(StrEnum)` in `season_data.py` with one member per current
`CACHE_FILES` entry (18 members):

```python
from enum import StrEnum

class CacheKey(StrEnum):
    STANDINGS = "standings"
    ROSTER = "roster"
    PROJECTIONS = "projections"
    LINEUP_OPTIMAL = "lineup_optimal"
    PROBABLE_STARTERS = "probable_starters"
    MONTE_CARLO = "monte_carlo"
    META = "meta"
    RANKINGS = "rankings"
    ROSTER_AUDIT = "roster_audit"
    SPOE = "spoe"
    OPP_ROSTERS = "opp_rosters"
    LEVERAGE = "leverage"
    PENDING_MOVES = "pending_moves"
    TRANSACTION_ANALYZER = "transaction_analyzer"
    TRANSACTIONS = "transactions"
    ROS_PROJECTIONS = "ros_projections"
    POSITIONS = "positions"
```

`StrEnum` (Python 3.11+, which `pyproject.toml` already requires) is a subclass
of `str`. That means a `CacheKey` member substitutes anywhere a string is
expected — f-string interpolation, `json.dumps` keys, dict lookups against
string keys — without `.value` boilerplate.

### API surface

```python
# Before
CACHE_FILES: dict[str, str]
def read_cache(key: str, cache_dir: Path = CACHE_DIR) -> dict | list | None: ...
def write_cache(key: str, data: dict | list, cache_dir: Path = CACHE_DIR) -> None: ...

# After
CACHE_FILES: dict[CacheKey, str]
def read_cache(key: CacheKey, cache_dir: Path = CACHE_DIR) -> dict | list | None: ...
def write_cache(key: CacheKey, data: dict | list, cache_dir: Path = CACHE_DIR) -> None: ...

def redis_key(key: CacheKey) -> str:
    """Return the Redis key for a cache entry (`cache:<name>`)."""
    return f"cache:{key.value}"
```

Callers pass `CacheKey.ROSTER` instead of `"roster"`. Attribute access on the
enum fails at import time if the member name is mistyped, which is the goal.

### Centralizing the Redis prefix

- `data/redis_store.py` currently defines `ROS_PROJECTIONS_KEY = "cache:ros_projections"`.
  Change it to `ROS_PROJECTIONS_KEY = redis_key(CacheKey.ROS_PROJECTIONS)`.
- `scripts/rescore_transactions.py` currently calls
  `redis.delete("cache:transaction_analyzer")` and `redis.delete("cache:transactions")`.
  Change to `redis.delete(redis_key(CacheKey.TRANSACTION_ANALYZER))` and
  `redis.delete(redis_key(CacheKey.TRANSACTIONS))`.

### Internal Redis reads/writes in `season_data.py`

The existing `read_cache`/`write_cache` bodies build the Redis key as
`f"cache:{key}"`. Replace with `redis_key(key)` so the prefix lives in one
helper. Log messages (`[redis] read_cache({key}) failed`, etc.) continue to
interpolate `key` directly — since `CacheKey` is a `StrEnum`, the logged text
is the bare name (`"roster"`), not `"CacheKey.ROSTER"`.

## Call sites

Counts are approximate; the plan captures exact numbers.

| File | Call sites |
|---|---|
| `src/fantasy_baseball/web/refresh_pipeline.py` | 18 |
| `src/fantasy_baseball/web/season_routes.py` | ~30 |
| `src/fantasy_baseball/web/season_data.py` | 3 (internal) |
| `src/fantasy_baseball/data/ros_pipeline.py` | 1 |
| `src/fantasy_baseball/data/redis_store.py` | 1 (constant) |
| `scripts/rescore_transactions.py` | 3 (hardcoded strings) |
| `tests/test_web/test_season_data.py` | many |
| `tests/test_web/test_season_routes.py` | several |
| `tests/test_web/_refresh_fixture.py` | several |
| `tests/test_data/test_ros_pipeline.py` | few |

## Phased execution

Each phase touches ≤5 files per CLAUDE.md's phased-execution rule. After each
phase: run `pytest -v`, `ruff check .`, `ruff format --check .`, and `vulture`.

**Phase 1 — Introduce the enum and update non-web call sites.**
Files: `season_data.py`, `data/redis_store.py`, `data/ros_pipeline.py`,
`scripts/rescore_transactions.py` (4 files).
Work: add `CacheKey`, `redis_key`; change `read_cache`/`write_cache` signatures;
switch `CACHE_FILES` keys; update the three non-web call sites. After this
phase, all `web/` call sites still use bare strings and will fail type-checking
but tests pass (StrEnum is a str subclass).

**Phase 2 — Migrate `refresh_pipeline.py`.**
Files: `src/fantasy_baseball/web/refresh_pipeline.py`,
`tests/test_web/_refresh_fixture.py` (2 files). 18 call sites in the pipeline
plus a handful in the fixture.

**Phase 3 — Migrate `season_routes.py`.**
Files: `src/fantasy_baseball/web/season_routes.py`,
`tests/test_web/test_season_routes.py` (2 files). ~30 call sites in routes
plus a few in tests.

**Phase 4 — Migrate remaining tests; final verification.**
Files: `tests/test_web/test_season_data.py`, `tests/test_data/test_ros_pipeline.py`
(2 files). Run the full end-of-effort checklist.

## Error handling / edge cases

- **String callers after migration:** illegal. Passing a bare `str` to the
  cache functions is a mypy error. At runtime, `CACHE_FILES[key]` raises
  `KeyError` just as it did before — the behavior doesn't regress, and strings
  pointing at valid members still incidentally work (`CACHE_FILES` keys *are*
  `CacheKey` members, which compare equal to their underlying strings). That
  incidental compatibility is acceptable; we enforce type correctness via
  mypy + code review, not runtime rejection.
- **Redis keyspace unchanged:** `f"cache:{CacheKey.ROSTER}"` produces
  `"cache:roster"`, identical to today. No data migration needed.
- **JSON-filename dict unchanged:** `CACHE_FILES[CacheKey.ROSTER]` returns
  `"roster.json"`, same as `CACHE_FILES["roster"]` returned before.
- **Comments/docstrings referencing `cache:roster` etc.:** leave as-is. They
  document on-disk/wire keys, not code symbols.

## Testing

No new tests. Existing tests cover the round-trip path and Redis fallback;
after the migration they exercise the same paths via the enum. The goal of
this change is compile-time / import-time typo detection — there is nothing
new to assert at runtime.

## Rollout

Single PR, four commits (one per phase). No feature flags. No migration
concerns: enum values match existing JSON filenames and Redis keys exactly.
