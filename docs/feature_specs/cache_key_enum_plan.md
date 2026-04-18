# CacheKey Enum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace bare-string cache keys in `read_cache`/`write_cache` with a
`CacheKey(StrEnum)` so typos fail at import time, and centralize the
`"cache:"` Redis-key prefix that's currently duplicated across three files.

**Architecture:** Single `CacheKey` enum co-located with the cache functions
in `src/fantasy_baseball/web/season_data.py`. A `redis_key(CacheKey) -> str`
helper is the one place that builds `"cache:<name>"`. Migration is mechanical
because `StrEnum` values are strings — any code that used
`f"cache:{key}"` continues to work unchanged.

**Tech Stack:** Python 3.11+ `enum.StrEnum`; the existing `upstash_redis`
client and JSON-file cache in `data/cache/`.

**Spec:** `docs/feature_specs/cache_key_enum.md`

---

## File Structure

**Created:** (none — the enum lives alongside the existing cache code)

**Modified:**
- `src/fantasy_baseball/web/season_data.py` — add `CacheKey`, `redis_key`;
  retype `CACHE_FILES`, `read_cache`, `write_cache`; update internal calls.
- `src/fantasy_baseball/data/redis_store.py` — rebuild `ROS_PROJECTIONS_KEY`
  from the enum.
- `src/fantasy_baseball/data/ros_pipeline.py` — pass `CacheKey.ROS_PROJECTIONS`.
- `scripts/rescore_transactions.py` — use `redis_key(CacheKey.TRANSACTIONS)` etc.
- `src/fantasy_baseball/web/refresh_pipeline.py` — 18 call sites.
- `src/fantasy_baseball/web/season_routes.py` — ~30 call sites.
- `tests/test_web/_refresh_fixture.py` — test-side `write_cache` calls.
- `tests/test_web/test_season_routes.py` — test-side `write_cache`/`read_cache`.
- `tests/test_web/test_season_data.py` — round-trip tests.
- `tests/test_data/test_ros_pipeline.py` — incidental `write_cache` comment only.

---

## Phase 1 — Introduce the enum and migrate non-web code

Touches 4 files: `season_data.py`, `redis_store.py`, `ros_pipeline.py`,
`rescore_transactions.py`.

### Task 1.1: Add `CacheKey` enum and `redis_key` helper

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py` (add enum above
  `CACHE_FILES`, around line 59)

- [ ] **Step 1: Add the `StrEnum` import**

At the top of `season_data.py`, change the `from enum import ...` line (add
one if none exists) to include `StrEnum`. The existing imports block is near
lines 1–21. Add the following after the existing stdlib imports:

```python
from enum import StrEnum
```

- [ ] **Step 2: Define `CacheKey` just before `CACHE_FILES`**

Insert this block immediately before the existing `CACHE_FILES = {...}`
definition (currently at line 85):

```python
class CacheKey(StrEnum):
    """Canonical names of every cached payload.

    Typos on member access (e.g. ``CacheKey.LEVARAGE``) raise
    ``AttributeError`` at import time instead of silently reading or writing
    the wrong cache entry.
    """

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


def redis_key(key: "CacheKey") -> str:
    """Return the Redis key for a cache entry (``cache:<name>``)."""
    return f"cache:{key.value}"
```

- [ ] **Step 3: Retype `CACHE_FILES` to use the enum as keys**

Replace the `CACHE_FILES = {...}` block (lines ~85–103) with:

```python
CACHE_FILES: dict[CacheKey, str] = {
    CacheKey.STANDINGS: "standings.json",
    CacheKey.ROSTER: "roster.json",
    CacheKey.PROJECTIONS: "projections.json",
    CacheKey.LINEUP_OPTIMAL: "lineup_optimal.json",
    CacheKey.PROBABLE_STARTERS: "probable_starters.json",
    CacheKey.MONTE_CARLO: "monte_carlo.json",
    CacheKey.META: "meta.json",
    CacheKey.RANKINGS: "rankings.json",
    CacheKey.ROSTER_AUDIT: "roster_audit.json",
    CacheKey.SPOE: "spoe.json",
    CacheKey.OPP_ROSTERS: "opp_rosters.json",
    CacheKey.LEVERAGE: "leverage.json",
    CacheKey.PENDING_MOVES: "pending_moves.json",
    CacheKey.TRANSACTION_ANALYZER: "transaction_analyzer.json",
    CacheKey.TRANSACTIONS: "transactions.json",
    CacheKey.ROS_PROJECTIONS: "ros_projections.json",
    CacheKey.POSITIONS: "positions.json",
}
```

- [ ] **Step 4: Retype `read_cache` and `write_cache` signatures**

In `read_cache` (line 106) and `write_cache` (line 144), change the `key: str`
parameter to `key: CacheKey`. Inside each function body, replace the Redis
key construction `f"cache:{key}"` with `redis_key(key)`. Nothing else in the
bodies needs to change.

`read_cache` becomes:

```python
def read_cache(key: CacheKey, cache_dir: Path = CACHE_DIR) -> dict | list | None:
    """Read a cached JSON file. Falls back to Redis on local miss."""
    path = cache_dir / CACHE_FILES[key]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if cache_dir != CACHE_DIR:
        return None

    redis = _get_redis()
    if not redis:
        return None

    try:
        raw = redis.get(redis_key(key))
        if raw is None:
            return None
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[redis] read_cache({key}) corrupt data, treating as miss")
        return None
    except Exception as e:
        print(f"[redis] read_cache({key}) failed: {e}")
        return None

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[redis] local write-back for {key} failed: {e}")

    return data
```

`write_cache` becomes:

```python
def write_cache(key: CacheKey, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
    """Atomically write a cached JSON file (tmpfile + rename), with Redis write-through."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_FILES[key]
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    if cache_dir == CACHE_DIR:
        redis = _get_redis()
        if redis:
            try:
                redis.set(redis_key(key), json.dumps(data))
            except Exception as e:
                print(f"[redis] write_cache({key}) failed: {e}")
```

- [ ] **Step 5: Update three internal `season_data.py` callers**

Grep within `season_data.py` for `read_cache("` — these are the non-public
internal callers that must move to the enum:

- Line ~169 in `read_meta`: `read_cache("meta", cache_dir)` →
  `read_cache(CacheKey.META, cache_dir)`
- Line ~496: `standings = read_cache("standings") or []` →
  `standings = read_cache(CacheKey.STANDINGS) or []`
- Line ~653: `optimal = read_cache("lineup_optimal")` →
  `optimal = read_cache(CacheKey.LINEUP_OPTIMAL)`

- [ ] **Step 6: Run `season_data` tests**

Run: `pytest tests/test_web/test_season_data.py -v`

Expected: PASS. These tests still pass bare strings to `read_cache` /
`write_cache`; `StrEnum` is a str subclass and `CACHE_FILES` is keyed by
`CacheKey` members which compare equal to the underlying string, so the
tests continue to work during the migration. (They will be migrated in
Phase 4.)

### Task 1.2: Migrate `redis_store.ROS_PROJECTIONS_KEY`

**Files:**
- Modify: `src/fantasy_baseball/data/redis_store.py:188`

- [ ] **Step 1: Replace the hardcoded constant**

`redis_store.py` line 188 currently reads:

```python
ROS_PROJECTIONS_KEY = "cache:ros_projections"
```

Change to:

```python
from fantasy_baseball.web.season_data import CacheKey, redis_key

ROS_PROJECTIONS_KEY = redis_key(CacheKey.ROS_PROJECTIONS)
```

Add the import near the other imports at the top of the file (not inline at
line 188). If this creates a circular import, fall back to a local import
inside any function that needs the value, but check first — `redis_store` is
imported by `season_data`'s `_load_game_log_totals`, but that call happens
inside a function body (line 179), so top-level `redis_store → season_data`
should be safe.

- [ ] **Step 2: Verify `redis_store` still imports cleanly**

Run: `python -c "from fantasy_baseball.data import redis_store; print(redis_store.ROS_PROJECTIONS_KEY)"`

Expected: prints `cache:ros_projections` with no ImportError.

If this step fails with a circular import, move the import into a module-
level lazy pattern:

```python
# at top of redis_store.py, after other imports
from fantasy_baseball.web.season_data import CacheKey, redis_key
```

If top-level fails, instead:

```python
# at the line that previously held ROS_PROJECTIONS_KEY
def _ros_projections_key() -> str:
    from fantasy_baseball.web.season_data import CacheKey, redis_key
    return redis_key(CacheKey.ROS_PROJECTIONS)

ROS_PROJECTIONS_KEY = _ros_projections_key()
```

- [ ] **Step 3: Run the redis-store tests**

Run: `pytest tests/test_data/test_redis_store_projections.py -v`

Expected: PASS (unchanged — the Redis key value is identical).

### Task 1.3: Migrate `ros_pipeline.write_cache` call

**Files:**
- Modify: `src/fantasy_baseball/data/ros_pipeline.py:112-116`

- [ ] **Step 1: Update the lazy import and call**

At lines 112–113 the file currently reads:

```python
    from fantasy_baseball.web.season_data import write_cache
    write_cache("ros_projections", {
```

Change to:

```python
    from fantasy_baseball.web.season_data import CacheKey, write_cache
    write_cache(CacheKey.ROS_PROJECTIONS, {
```

- [ ] **Step 2: Run the ros_pipeline tests**

Run: `pytest tests/test_data/test_ros_pipeline.py -v`

Expected: PASS.

### Task 1.4: Migrate `rescore_transactions.py`

**Files:**
- Modify: `scripts/rescore_transactions.py:72-74`

- [ ] **Step 1: Replace the two hardcoded Redis keys**

Lines 72–73 currently read:

```python
        redis.delete("cache:transaction_analyzer")
        redis.delete("cache:transactions")
        print("Cleared cache:transactions and cache:transaction_analyzer from Redis.")
```

Change to:

```python
        from fantasy_baseball.web.season_data import CacheKey, redis_key

        redis.delete(redis_key(CacheKey.TRANSACTION_ANALYZER))
        redis.delete(redis_key(CacheKey.TRANSACTIONS))
        print("Cleared cache:transactions and cache:transaction_analyzer from Redis.")
```

(Keep the print message literal — it documents the wire keys a user might
search Redis for.)

- [ ] **Step 2: Verify the script imports**

Run: `python -c "import scripts.rescore_transactions"`

Expected: no ImportError. (The script exits normally; we're just verifying
the import path.) If `scripts` isn't importable as a package, skip this
step — the next phase's integration test will exercise the code path.

### Task 1.5: Verify Phase 1 and commit

- [ ] **Step 1: Run the full verification checklist**

Run in sequence:

```bash
pytest -v
ruff check .
ruff format --check .
vulture
```

Expected: all pass. If `ruff format --check` reports drift, run
`ruff format .` and re-verify.

- [ ] **Step 2: Commit Phase 1**

```bash
git add src/fantasy_baseball/web/season_data.py \
        src/fantasy_baseball/data/redis_store.py \
        src/fantasy_baseball/data/ros_pipeline.py \
        scripts/rescore_transactions.py
git commit -m "refactor(cache): introduce CacheKey enum and redis_key helper

Add a StrEnum for the 17 cache keys used by read_cache/write_cache, plus
a redis_key(CacheKey) helper that centralizes the 'cache:' Redis prefix.
Migrate the three non-web call sites (redis_store.ROS_PROJECTIONS_KEY,
ros_pipeline.blend_and_cache_ros, scripts/rescore_transactions.py) to
use the enum. The refresh pipeline and season routes still pass bare
strings; those migrations land in follow-up commits. StrEnum values are
strings, so existing tests continue to exercise the same code paths."
```

- [ ] **Step 3: Wait for user approval before Phase 2**

Per CLAUDE.md phased-execution rule, pause and get explicit approval
before starting Phase 2.

---

## Phase 2 — Migrate `refresh_pipeline.py`

Touches 2 files: `refresh_pipeline.py`, `_refresh_fixture.py`.

### Task 2.1: Update the `season_data` import in `refresh_pipeline.py`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` (top-level imports)

- [ ] **Step 1: Add `CacheKey` to the import**

Find the existing import of `read_cache` / `write_cache` from
`fantasy_baseball.web.season_data` and add `CacheKey`:

```python
from fantasy_baseball.web.season_data import CacheKey, read_cache, write_cache
```

Preserve any other names on the same import line.

### Task 2.2: Migrate all 18 call sites in `refresh_pipeline.py`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` (lines below)

- [ ] **Step 1: Apply the mechanical substitutions**

The full mapping — line number, current call, and replacement:

| Line | Current | Replace first-arg with |
|-----:|---|---|
| 237 | `write_cache("standings", self.standings, self.cache_dir)` | `CacheKey.STANDINGS` |
| 267 | `write_cache("pending_moves", pending_moves, self.cache_dir)` | `CacheKey.PENDING_MOVES` |
| 342 | `ros_cached = read_cache("ros_projections", self.cache_dir)` | `CacheKey.ROS_PROJECTIONS` |
| 486 | `write_cache("opp_rosters", opp_rosters_flat, self.cache_dir)` | `CacheKey.OPP_ROSTERS` |
| 530 | first arg of the multi-line `write_cache("projections", { ... }, self.cache_dir)` at 529–539 | `CacheKey.PROJECTIONS` |
| 616 | `write_cache("rankings", self.rankings_lookup, self.cache_dir)` | `CacheKey.RANKINGS` |
| 626 | `write_cache("roster", roster_flat, self.cache_dir)` | `CacheKey.ROSTER` |
| 673 | `write_cache("lineup_optimal", optimal_data, self.cache_dir)` | `CacheKey.LINEUP_OPTIMAL` |
| 699 | `write_cache("probable_starters", probable_starters, self.cache_dir)` | `CacheKey.PROBABLE_STARTERS` |
| 714 | `write_cache("positions", positions_map, self.cache_dir)` | `CacheKey.POSITIONS` |
| 727 | `write_cache("roster_audit", [e.to_dict() ...], self.cache_dir)` | `CacheKey.ROSTER_AUDIT` |
| 742 | `write_cache("leverage", leverage_by_team, self.cache_dir)` | `CacheKey.LEVERAGE` |
| 818 | first arg of `write_cache("monte_carlo", { ... })` block | `CacheKey.MONTE_CARLO` |
| 848 | `write_cache("spoe", spoe_result, self.cache_dir)` | `CacheKey.SPOE` |
| 867 | `stored_txns = read_cache("transactions", self.cache_dir) or []` | `CacheKey.TRANSACTIONS` |
| 919 | `write_cache("transactions", stored_txns, self.cache_dir)` | `CacheKey.TRANSACTIONS` |
| 921 | `write_cache("transaction_analyzer", cache_data, self.cache_dir)` | `CacheKey.TRANSACTION_ANALYZER` |
| 933 | `write_cache("meta", meta, self.cache_dir)` | `CacheKey.META` |

Use `grep -n 'write_cache\|read_cache' src/fantasy_baseball/web/refresh_pipeline.py`
after the edits to confirm zero remaining bare-string first-args.

- [ ] **Step 2: Verify no bare strings remain**

Run: `grep -nE 'write_cache\("|read_cache\("' src/fantasy_baseball/web/refresh_pipeline.py`

Expected: no output. If any match remains, fix it.

- [ ] **Step 3: Run the refresh-pipeline tests**

Run: `pytest tests/test_web/ -v -k "refresh or pipeline"`

Expected: PASS.

### Task 2.3: Migrate `_refresh_fixture.py`

**Files:**
- Modify: `tests/test_web/_refresh_fixture.py` (around lines 223, 308)

- [ ] **Step 1: Add `CacheKey` to the import**

Find the existing `from fantasy_baseball.web.season_data import ...` line (if
any) and add `CacheKey`. If the fixture imports `write_cache` via
`season_data.write_cache(...)`, import `CacheKey` separately:

```python
from fantasy_baseball.web.season_data import CacheKey
```

- [ ] **Step 2: Update every `write_cache(...)` call**

For each `season_data.write_cache("<key>", ...)` or `write_cache("<key>", ...)`
call in the fixture, replace the first-arg string with `CacheKey.<NAME>`.
The docstring at line 223 references `read_cache("ros_projections")` — that
is documentation text, leave it as-is.

- [ ] **Step 3: Verify no bare-string cache calls remain in the fixture**

Run: `grep -nE 'write_cache\(["'\'']|read_cache\(["'\'']' tests/test_web/_refresh_fixture.py`

Expected: no matches inside function bodies (docstring mentions are fine).

### Task 2.4: Verify Phase 2 and commit

- [ ] **Step 1: Run the full verification checklist**

```bash
pytest -v
ruff check .
ruff format --check .
vulture
```

Expected: all pass.

- [ ] **Step 2: Commit Phase 2**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py \
        tests/test_web/_refresh_fixture.py
git commit -m "refactor(cache): migrate refresh_pipeline to CacheKey enum

Replace bare-string first-args at 18 write_cache/read_cache sites in
refresh_pipeline.py and update the refresh fixture's cache calls to
match. No behavioral change."
```

- [ ] **Step 3: Wait for user approval before Phase 3**

---

## Phase 3 — Migrate `season_routes.py`

Touches 2 files: `season_routes.py`, `test_season_routes.py`.

### Task 3.1: Update the `season_data` import in `season_routes.py`

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (top-level imports)

- [ ] **Step 1: Add `CacheKey` to the import**

Find the existing `from fantasy_baseball.web.season_data import ...` line
and add `CacheKey`:

```python
from fantasy_baseball.web.season_data import CacheKey, read_cache  # plus any other names already imported
```

### Task 3.2: Migrate all ~30 call sites in `season_routes.py`

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (every `read_cache("<key>")`)

- [ ] **Step 1: Apply the mechanical substitutions**

Enumerated mapping (line + key):

| Line | Key string | Replacement |
|-----:|---|---|
| 62 | `"roster"` | `CacheKey.ROSTER` |
| 109 | `"ros_projections"` | `CacheKey.ROS_PROJECTIONS` |
| 208 | `"standings"` | `CacheKey.STANDINGS` |
| 230 | `"projections"` | `CacheKey.PROJECTIONS` |
| 249 | `"monte_carlo"` | `CacheKey.MONTE_CARLO` |
| 287 | `"roster"` | `CacheKey.ROSTER` |
| 288 | `"lineup_optimal"` | `CacheKey.LINEUP_OPTIMAL` |
| 289 | `"probable_starters"` | `CacheKey.PROBABLE_STARTERS` |
| 290 | `"pending_moves"` | `CacheKey.PENDING_MOVES` |
| 299 | `"standings"` | `CacheKey.STANDINGS` |
| 330 | `"roster_audit"` | `CacheKey.ROSTER_AUDIT` |
| 344 | `"roster"` | `CacheKey.ROSTER` |
| 345 | `"opp_rosters"` | `CacheKey.OPP_ROSTERS` |
| 379 | `"standings"` | `CacheKey.STANDINGS` |
| 383 | `"roster"` | `CacheKey.ROSTER` |
| 387 | `"opp_rosters"` | `CacheKey.OPP_ROSTERS` |
| 391 | `"leverage"` | `CacheKey.LEVERAGE` |
| 395 | `"rankings"` | `CacheKey.RANKINGS` |
| 399 | `"projections"` | `CacheKey.PROJECTIONS` |
| 463 | `"ros_projections"` | `CacheKey.ROS_PROJECTIONS` |
| 467 | `"rankings"` | `CacheKey.RANKINGS` |
| 470 | `"positions"` | `CacheKey.POSITIONS` |
| 473 | `"roster"` | `CacheKey.ROSTER` |
| 477 | `"opp_rosters"` | `CacheKey.OPP_ROSTERS` |
| 487 | `"roster_audit"` | `CacheKey.ROSTER_AUDIT` |
| 584 | `"roster"` | `CacheKey.ROSTER` |
| 588 | `"projections"` | `CacheKey.PROJECTIONS` |
| 617 | `"ros_projections"` | `CacheKey.ROS_PROJECTIONS` |
| 667 | `"roster"` | `CacheKey.ROSTER` |
| 672 | `"projections"` | `CacheKey.PROJECTIONS` |
| 679 | `"ros_projections"` | `CacheKey.ROS_PROJECTIONS` |
| 691 | `"positions"` | `CacheKey.POSITIONS` |
| 728 | `"spoe"` | `CacheKey.SPOE` |
| 765 | `"transaction_analyzer"` | `CacheKey.TRANSACTION_ANALYZER` |
| 809 | `"standings"` | `CacheKey.STANDINGS` |
| 831 | `"standings"` | `CacheKey.STANDINGS` |

Line numbers are current as of the grep run at the start of this branch;
if prior phases have shifted them, grep will still find every call.

- [ ] **Step 2: Verify no bare-string cache calls remain**

Run: `grep -nE 'read_cache\("|write_cache\("' src/fantasy_baseball/web/season_routes.py`

Expected: no output. (Comments like `# ROS projections served from the
cache:ros_projections Redis key` at line 106 are fine — the regex only
matches function calls.)

- [ ] **Step 3: Run the season-routes tests**

Run: `pytest tests/test_web/test_season_routes.py -v`

Expected: PASS. Tests still pass bare strings via `write_cache` —
Phase 4 migrates them.

### Task 3.3: Migrate `test_season_routes.py`

**Files:**
- Modify: `tests/test_web/test_season_routes.py` (lines 113–208 approx)

- [ ] **Step 1: Add `CacheKey` to the test's imports**

Find the existing `from fantasy_baseball.web import season_data` (or similar)
and add an explicit `CacheKey` import:

```python
from fantasy_baseball.web.season_data import CacheKey
```

- [ ] **Step 2: Update every `season_data.write_cache(...)` call**

Replace each bare-string first-arg with the matching enum member. The
specific lines from the initial grep:

- L113: `"standings"` → `CacheKey.STANDINGS`
- L114: `"meta"` → `CacheKey.META`
- L148: `"roster"` → `CacheKey.ROSTER`
- L149: `"lineup_optimal"` → `CacheKey.LINEUP_OPTIMAL`
- L150: `"meta"` → `CacheKey.META`
- L192: `"monte_carlo"` → `CacheKey.MONTE_CARLO`
- L203: `"standings"` → `CacheKey.STANDINGS`

- [ ] **Step 3: The `mock_rc.side_effect = lambda k: season_data.read_cache(k, tmp_path)` lines**

These (lines 119, 154, 208 approx) need no change — the `k` they receive
from the route will already be a `CacheKey` after Phase 3 Task 3.2, and the
signature now accepts exactly `CacheKey`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_web/test_season_routes.py -v`

Expected: PASS.

### Task 3.4: Verify Phase 3 and commit

- [ ] **Step 1: Run the full verification checklist**

```bash
pytest -v
ruff check .
ruff format --check .
vulture
```

Expected: all pass.

- [ ] **Step 2: Commit Phase 3**

```bash
git add src/fantasy_baseball/web/season_routes.py \
        tests/test_web/test_season_routes.py
git commit -m "refactor(cache): migrate season_routes to CacheKey enum

Replace bare-string first-args at ~30 read_cache call sites in
season_routes.py and update the matching test fixtures. No
behavioral change."
```

- [ ] **Step 3: Wait for user approval before Phase 4**

---

## Phase 4 — Migrate remaining tests and run final verification

Touches 2 files: `test_season_data.py`, `test_ros_pipeline.py`.

### Task 4.1: Migrate `test_season_data.py`

**Files:**
- Modify: `tests/test_web/test_season_data.py` (lines 11–345 approx)

- [ ] **Step 1: Add `CacheKey` to the test's imports**

```python
from fantasy_baseball.web.season_data import CacheKey, read_cache, write_cache
```

- [ ] **Step 2: Update every `read_cache("standings"...)` and `write_cache("standings"...)` call**

The initial grep lists these lines:

- L13, L14, L19, L26, L36, L37, L38, L262, L274, L276, L287, L289, L301,
  L314, L324, L333, L345

All of them use `"standings"`. Replace with `CacheKey.STANDINGS`.

- [ ] **Step 3: The `assert mock_redis._last_set[0] == "cache:standings"` assertion (line 264)**

This asserts the on-wire Redis key. Leave the string literal as-is — the
Redis key is still `"cache:standings"` after the refactor. The assertion
documents the wire format; tightening it to
`redis_key(CacheKey.STANDINGS)` would just reintroduce the prefix coupling
we're trying to avoid testing against.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_web/test_season_data.py -v`

Expected: PASS.

### Task 4.2: Touch `test_ros_pipeline.py` only if needed

**Files:**
- Inspect: `tests/test_data/test_ros_pipeline.py`

- [ ] **Step 1: Confirm there is nothing to change**

Run: `grep -nE 'write_cache\("|read_cache\("' tests/test_data/test_ros_pipeline.py`

Expected: no output. The earlier grep only showed a docstring comment
(`# ``write_cache(...)`` without an explicit cache_dir uses tmp.`) — no
source-level call needs migration. If this grep unexpectedly returns
lines, migrate them using the same pattern as Task 4.1.

### Task 4.3: Final verification and commit

- [ ] **Step 1: Run the full verification checklist one last time**

```bash
pytest -v
ruff check .
ruff format --check .
vulture
```

Paste the output (or concise summary) into the final message per
CLAUDE.md's Agent Directive 4.

- [ ] **Step 2: Sanity-grep for any remaining bare-string callers**

Run:

```bash
grep -rnE 'read_cache\("|write_cache\("' src/ tests/ scripts/
```

Expected: no matches. If the grep finds anything, finish migrating before
committing.

- [ ] **Step 3: Commit Phase 4**

```bash
git add tests/test_web/test_season_data.py
git commit -m "refactor(cache): migrate test_season_data to CacheKey enum

Final call-site migration — every read_cache/write_cache call in the
repo now passes a CacheKey member. Bare-string cache keys are now a
type error."
```

- [ ] **Step 4: Report completion**

Summarize the four commits to the user and ask whether to open a PR.

---

## Self-review

**Spec coverage:**
- Enum added (Phase 1 Task 1.1). ✓
- `redis_key` helper (Phase 1 Task 1.1). ✓
- `read_cache`/`write_cache` signature change (Phase 1 Task 1.1). ✓
- `CACHE_FILES` retyped (Phase 1 Task 1.1). ✓
- `redis_store.ROS_PROJECTIONS_KEY` migrated (Phase 1 Task 1.2). ✓
- `rescore_transactions.py` migrated (Phase 1 Task 1.4). ✓
- Internal `season_data.py` callers migrated (Phase 1 Task 1.1 Step 5). ✓
- `refresh_pipeline.py` call sites (Phase 2 Task 2.2). ✓
- `season_routes.py` call sites (Phase 3 Task 3.2). ✓
- `ros_pipeline.py` call site (Phase 1 Task 1.3). ✓
- Tests migrated (Phases 2/3/4). ✓
- No mypy expansion (explicitly out of scope per spec). ✓

**Placeholder scan:** no TBDs, no "handle edge cases", all steps contain
either exact code or an exact command. ✓

**Type consistency:** `CacheKey` member names are consistent across every
task (`CacheKey.STANDINGS`, `CacheKey.ROSTER`, etc.); `redis_key()` signature
is defined once and called identically in every usage. ✓
