# Upstash Redis Cache Layer

**Date:** 2026-03-27
**Status:** Approved

## Problem

Render's free tier spins down the web service after 15 minutes of inactivity. The ephemeral filesystem means all local cache files (`data/cache/*.json`) are lost on spin-down, requiring a full re-fetch from Yahoo and MLB APIs on cold start — a process that takes several minutes on Render's free compute.

## Solution

Add an Upstash Redis write-through/fallback layer to the existing `read_cache()` and `write_cache()` functions in `season_data.py`. Upstash is a serverless Redis provider with a free tier (256MB, 10K commands/day) that persists data independently of the Render instance.

## Architecture

### Data flow

```
write_cache("standings", data)
  -> write to local disk (as today)
  -> write to Upstash Redis key "cache:standings"

read_cache("standings")
  -> try local disk (fast path, <1ms)
  -> miss? try Upstash Redis (~50-100ms)
  -> hit from Redis? write back to local disk directly (not via write_cache, to avoid redundant Redis SET)
  -> miss? return None (user triggers manual refresh)
```

### Redis key design

- Keys namespaced as `cache:{key}` (e.g., `cache:standings`, `cache:roster`)
- Values are compact JSON strings (no indentation, unlike pretty-printed local files)
- No TTL — overwritten on each refresh
- 9 keys, ~30KB total

### Client initialization

Lazy singleton via module-level variable with thread-safe double-checked locking. Created on first use, cached for process lifetime. Returns `None` if env vars are missing, enabling graceful degradation for local development.

```python
_redis_client = None
_redis_initialized = False
_redis_lock = threading.Lock()

def _get_redis():
    """Lazy Upstash Redis client. Returns None if not configured."""
    global _redis_client, _redis_initialized
    if _redis_initialized:
        return _redis_client
    with _redis_lock:
        if _redis_initialized:
            return _redis_client
        _redis_initialized = True
        url = os.environ.get("UPSTASH_REDIS_REST_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            from upstash_redis import Redis
            _redis_client = Redis(url=url, token=token)
    return _redis_client
```

### Error handling

All Redis operations wrapped in try/except that catches `Exception`. If Upstash is unreachable or errors, the app continues with local-only behavior. Errors are logged via `print()` (consistent with existing codebase patterns) but never block the user or raise exceptions.

The Redis read path also catches `json.JSONDecodeError` when deserializing the value from Redis, treating corrupt data as a cache miss.

### Modified functions

**`read_cache(key, cache_dir)`:**
1. Try local disk (existing behavior)
2. If None, Redis configured, AND `cache_dir == CACHE_DIR` (skip Redis for tests/non-default dirs): try `redis.get(f"cache:{key}")`
3. If Redis hit: deserialize JSON (catching `JSONDecodeError`), write back to local disk directly (bypassing `write_cache` to avoid redundant Redis SET), return data
4. If Redis miss or error: return None

**`write_cache(key, data, cache_dir)`:**
1. Write to local disk (existing behavior, unchanged)
2. If Redis configured AND `cache_dir == CACHE_DIR`: `redis.set(f"cache:{key}", json.dumps(data))`

### Redis/local isolation

Redis operations are skipped when `cache_dir != CACHE_DIR`. This ensures:
- Tests using `tmp_path` never hit Redis
- Non-default cache directories don't create key conflicts

## Configuration

### Environment variables (Render)

- `UPSTASH_REDIS_REST_URL` — Upstash REST endpoint URL
- `UPSTASH_REDIS_REST_TOKEN` — Upstash REST auth token

Add to `render.yaml` with `sync: false` (set manually in dashboard):
```yaml
- key: UPSTASH_REDIS_REST_URL
  sync: false
- key: UPSTASH_REDIS_REST_TOKEN
  sync: false
```

### Dependencies

Add `upstash-redis` to both:
- `requirements.txt` (used by Render)
- `pyproject.toml` dependencies (used by `pip install -e .`)

The lazy import in `_get_redis()` means the app won't crash if the package is absent and env vars are unset.

## What doesn't change

- Local development works exactly as before (no env vars = no Redis)
- All routes, templates, and refresh logic are untouched
- Cache file format is unchanged
- `read_meta()` benefits automatically (it calls `read_cache("meta")`)

## Known acceptable behaviors

- **Partial Redis writes during refresh:** If Redis fails mid-refresh (e.g., network blip), some keys may be stale while others are current. This is acceptable because: (a) the next successful refresh overwrites everything, (b) stale data is better than no data on cold start, (c) the `meta` key is written last so its timestamp indicates freshness.

## Constraints

- Upstash free tier: 10K commands/day, 256MB storage
- Expected usage per cold start: up to ~20 Redis reads (9 cache keys read across multiple route handlers). Per refresh: 9 Redis writes. Typical daily usage well within 10K limit.
