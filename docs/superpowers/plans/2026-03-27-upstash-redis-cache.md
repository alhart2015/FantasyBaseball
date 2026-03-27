# Upstash Redis Cache Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist dashboard cache data in Upstash Redis so cold starts after Render spin-down serve cached data instead of requiring a multi-minute refresh.

**Architecture:** Write-through/read-fallback layer added to existing `read_cache()` and `write_cache()` in `season_data.py`. Local disk is primary (fast), Redis is secondary (survives spin-down). Lazy singleton client, thread-safe init, graceful degradation when unconfigured.

**Tech Stack:** `upstash-redis` Python SDK, existing Flask/SQLite app

**Spec:** `docs/superpowers/specs/2026-03-27-upstash-redis-cache-design.md`

---

### Task 1: Add `upstash-redis` dependency

**Files:**
- Modify: `requirements.txt:12`
- Modify: `pyproject.toml:9-21`

- [ ] **Step 1: Add to requirements.txt**

Append `upstash-redis` to `requirements.txt`:

```
upstash-redis>=1.0
```

- [ ] **Step 2: Add to pyproject.toml**

Add `"upstash-redis>=1.0"` to the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    ...
    "gunicorn>=21.2",
    "upstash-redis>=1.0",
]
```

- [ ] **Step 3: Install locally**

Run: `pip install -e ".[dev]"`
Expected: installs successfully, `upstash-redis` appears in `pip list`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "deps: add upstash-redis for persistent cache layer"
```

---

### Task 2: Add `_get_redis()` lazy singleton with thread-safe init

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:1-27`
- Test: `tests/test_web/test_season_data.py`

- [ ] **Step 1: Write failing test for `_get_redis` returning None when unconfigured**

Add to `tests/test_web/test_season_data.py`:

```python
from unittest.mock import patch
from fantasy_baseball.web import season_data


def test_get_redis_returns_none_when_unconfigured(monkeypatch):
    """With no env vars, _get_redis() returns None."""
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    # Reset singleton state
    season_data._redis_client = None
    season_data._redis_initialized = False
    result = season_data._get_redis()
    assert result is None
    # Cleanup
    season_data._redis_initialized = False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_data.py::test_get_redis_returns_none_when_unconfigured -v`
Expected: FAIL — `_get_redis` not defined or `_redis_initialized` not found

- [ ] **Step 3: Write failing test for `_get_redis` returning a client when configured**

Add to `tests/test_web/test_season_data.py`:

```python
def test_get_redis_returns_client_when_configured(monkeypatch):
    """With env vars set, _get_redis() returns a Redis client."""
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://fake.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")
    # Reset singleton state
    season_data._redis_client = None
    season_data._redis_initialized = False
    with patch("fantasy_baseball.web.season_data.Redis") as MockRedis:
        MockRedis.return_value = "mock-client"
        result = season_data._get_redis()
        assert result == "mock-client"
        MockRedis.assert_called_once_with(url="https://fake.upstash.io", token="fake-token")
    # Cleanup
    season_data._redis_client = None
    season_data._redis_initialized = False
```

- [ ] **Step 4: Implement `_get_redis()` in season_data.py**

Add after line 15 (`_refresh_status = ...`), before `CACHE_DIR`:

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

Also add the import at the top of the test file so `Redis` can be patched at the right location. The `_get_redis` function does `from upstash_redis import Redis` lazily, so the test patches `fantasy_baseball.web.season_data.Redis` (the name it will be bound to in the module after the first call). Since the lazy import only runs inside `_get_redis`, we need to patch it at the module level where it will land. Update the test to patch correctly:

```python
def test_get_redis_returns_client_when_configured(monkeypatch):
    """With env vars set, _get_redis() returns a Redis client."""
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://fake.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")
    season_data._redis_client = None
    season_data._redis_initialized = False
    with patch("upstash_redis.Redis") as MockRedis:
        MockRedis.return_value = "mock-client"
        result = season_data._get_redis()
        assert result == "mock-client"
        MockRedis.assert_called_once_with(url="https://fake.upstash.io", token="fake-token")
    season_data._redis_client = None
    season_data._redis_initialized = False
```

- [ ] **Step 5: Run both tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py::test_get_redis_returns_none_when_unconfigured tests/test_web/test_season_data.py::test_get_redis_returns_client_when_configured -v`
Expected: both PASS

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat: add thread-safe lazy Redis client singleton"
```

---

### Task 3: Add Redis write-through to `write_cache()`

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:63-77` (write_cache function)
- Test: `tests/test_web/test_season_data.py`

- [ ] **Step 1: Write failing test — write_cache writes to Redis when configured**

Add to `tests/test_web/test_season_data.py`:

```python
def test_write_cache_writes_to_redis(tmp_path, monkeypatch):
    """write_cache with default cache_dir writes to Redis."""
    mock_redis = type("MockRedis", (), {"set": lambda self, k, v: None})()
    mock_redis.set = lambda k, v: setattr(mock_redis, "_last_set", (k, v))
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    data = {"teams": [1, 2, 3]}
    write_cache("standings", data, cache_dir=tmp_path)

    assert mock_redis._last_set[0] == "cache:standings"
    assert json.loads(mock_redis._last_set[1]) == data
```

- [ ] **Step 2: Write failing test — write_cache skips Redis for non-default cache_dir**

```python
from unittest.mock import MagicMock


def test_write_cache_skips_redis_non_default_dir(tmp_path, monkeypatch):
    """write_cache with non-default cache_dir does not touch Redis."""
    mock_redis = MagicMock()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    # tmp_path != CACHE_DIR, so Redis should be skipped
    data = {"v": 1}
    write_cache("standings", data, cache_dir=tmp_path)
    mock_redis.set.assert_not_called()
    assert read_cache("standings", cache_dir=tmp_path) == data
```

- [ ] **Step 2b: Write failing test — write_cache handles Redis network errors**

```python
def test_write_cache_handles_redis_error(tmp_path, monkeypatch):
    """write_cache continues if Redis raises a network error."""
    mock_redis = MagicMock()
    mock_redis.set.side_effect = ConnectionError("Upstash unreachable")
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    data = {"teams": [1, 2, 3]}
    write_cache("standings", data, cache_dir=tmp_path)
    # Local write still succeeded
    assert read_cache("standings", cache_dir=tmp_path) == data
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_web/test_season_data.py::test_write_cache_writes_to_redis tests/test_web/test_season_data.py::test_write_cache_skips_redis_non_default_dir -v`
Expected: FAIL — Redis write not implemented yet

- [ ] **Step 4: Implement Redis write-through in write_cache**

Modify `write_cache()` in `season_data.py` — add after the existing local-disk write logic (after `Path(tmp).rename(path)`, outside the try/except):

```python
def write_cache(key: str, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
    """Atomically write a cached JSON file (tmpfile + rename), with Redis write-through."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_FILES[key]
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # On Windows, must remove target before rename
        if path.exists():
            path.unlink()
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    # Write-through to Redis (only for default cache dir)
    if cache_dir == CACHE_DIR:
        redis = _get_redis()
        if redis:
            try:
                redis.set(f"cache:{key}", json.dumps(data))
            except Exception as e:
                print(f"[redis] write_cache({key}) failed: {e}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: all PASS (including existing tests unchanged)

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat: add Redis write-through to write_cache"
```

---

### Task 4: Add Redis fallback to `read_cache()`

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:54-60` (read_cache function)
- Test: `tests/test_web/test_season_data.py`

- [ ] **Step 1: Write failing test — read_cache falls back to Redis on local miss**

```python
def test_read_cache_falls_back_to_redis(tmp_path, monkeypatch):
    """When local disk has no file, read_cache fetches from Redis and writes back locally."""
    data = {"teams": [1, 2, 3]}
    mock_redis = type("MockRedis", (), {"get": lambda self, k: json.dumps(data)})()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result == data
    # Verify it wrote back to local disk
    local = json.loads((tmp_path / "standings.json").read_text(encoding="utf-8"))
    assert local == data
```

- [ ] **Step 2: Write failing test — read_cache returns None when both local and Redis miss**

```python
def test_read_cache_returns_none_when_both_miss(tmp_path, monkeypatch):
    """When local disk and Redis both miss, returns None."""
    mock_redis = type("MockRedis", (), {"get": lambda self, k: None})()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None
```

- [ ] **Step 3: Write failing test — read_cache handles corrupt Redis data**

```python
def test_read_cache_handles_corrupt_redis_data(tmp_path, monkeypatch):
    """When Redis returns non-JSON, treat as miss."""
    mock_redis = type("MockRedis", (), {"get": lambda self, k: "not-json{{"})()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None
```

- [ ] **Step 4: Write failing test — read_cache skips Redis for non-default cache_dir**

```python
def test_read_cache_skips_redis_non_default_dir(tmp_path, monkeypatch):
    """read_cache with non-default cache_dir does not touch Redis."""
    mock_redis = MagicMock()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    # tmp_path != CACHE_DIR, so Redis should be skipped
    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None
    mock_redis.get.assert_not_called()
```

- [ ] **Step 4b: Write failing test — read_cache handles Redis network errors**

```python
def test_read_cache_handles_redis_error(tmp_path, monkeypatch):
    """read_cache returns None if Redis raises a network error."""
    mock_redis = MagicMock()
    mock_redis.get.side_effect = ConnectionError("Upstash unreachable")
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_web/test_season_data.py::test_read_cache_falls_back_to_redis tests/test_web/test_season_data.py::test_read_cache_returns_none_when_both_miss tests/test_web/test_season_data.py::test_read_cache_handles_corrupt_redis_data tests/test_web/test_season_data.py::test_read_cache_skips_redis_non_default_dir -v`
Expected: FAIL

- [ ] **Step 6: Implement Redis fallback in read_cache**

Replace the `read_cache` function:

```python
def read_cache(key: str, cache_dir: Path = CACHE_DIR) -> dict | list | None:
    """Read a cached JSON file. Falls back to Redis on local miss."""
    path = cache_dir / CACHE_FILES[key]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback to Redis (only for default cache dir)
    if cache_dir != CACHE_DIR:
        return None

    redis = _get_redis()
    if not redis:
        return None

    try:
        raw = redis.get(f"cache:{key}")
        if raw is None:
            return None
        data = json.loads(raw)
    except Exception as e:
        if isinstance(e, json.JSONDecodeError):
            print(f"[redis] read_cache({key}) corrupt data, treating as miss")
        else:
            print(f"[redis] read_cache({key}) failed: {e}")
        return None

    # Write back to local disk for subsequent fast reads
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[redis] local write-back for {key} failed: {e}")

    return data
```

- [ ] **Step 7: Run all tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: all PASS (new tests + all existing tests still pass)

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat: add Redis fallback to read_cache with local write-back"
```

---

### Task 5: Update Render config

**Files:**
- Modify: `render.yaml:7-13`

- [ ] **Step 1: Add env var declarations to render.yaml**

Add after the `ADMIN_PASSWORD` entry:

```yaml
      - key: UPSTASH_REDIS_REST_URL
        sync: false
      - key: UPSTASH_REDIS_REST_TOKEN
        sync: false
```

- [ ] **Step 2: Verify YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('render.yaml'))"`
Expected: no error

- [ ] **Step 3: Commit**

```bash
git add render.yaml
git commit -m "config: add Upstash Redis env vars to render.yaml"
```

---

### Task 6: Run full test suite and verify

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -v`
Expected: all tests pass, no regressions

- [ ] **Step 2: Verify local dev still works without Redis env vars**

Run: `python -c "from fantasy_baseball.web.season_data import _get_redis; assert _get_redis() is None; print('OK: no Redis configured, graceful degradation works')"`
Expected: prints OK message

- [ ] **Step 3: Final commit if any cleanup needed**

If any adjustments were made, commit them.
