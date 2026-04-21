"""Backend-agnostic key-value store for all persistent state.

Two backends, selected by ``RENDER=true``:

- ``UpstashKVStore`` wraps ``upstash_redis.Redis`` (remote, authoritative
  on Render).
- ``SqliteKVStore`` is a file-backed local store used by local
  dashboards and tests.

``get_kv()`` is the single entry point for application code and cannot
reach Upstash off-Render: the ``RENDER`` gate is hard. Scripts that
need to cross the local→remote boundary (``scripts/refresh_remote.py``,
``data/kv_sync``) call ``build_explicit_upstash_kv()`` — the function
is named for exactly the audit trail we want.

The schema mirrors the subset of Redis the app actually uses:
``get/set/keys/mget`` for strings and ``hget/hset/hkeys/hgetall`` for
hashes. If the app ever needs sorted sets, pipelines, or pub/sub the
abstraction has to grow — today it doesn't.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from upstash_redis import Redis

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LOCAL_DB = _PROJECT_ROOT / "data" / "local.db"


def is_remote() -> bool:
    """True when running on Render (``RENDER=true``)."""
    return os.environ.get("RENDER") == "true"


@runtime_checkable
class KVStore(Protocol):
    """Minimal Redis subset the app actually uses. Both backends match."""

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, *, ex: int | None = None) -> None: ...
    def delete(self, key: str) -> int: ...
    def keys(self, pattern: str) -> list[str]: ...
    def mget(self, *keys: str) -> list[str | None]: ...
    def hget(self, hash_name: str, field: str) -> str | None: ...
    def hset(self, hash_name: str, field: str, value: str) -> None: ...
    def hkeys(self, hash_name: str) -> list[str]: ...
    def hgetall(self, hash_name: str) -> dict[str, str]: ...


class UpstashKVStore:
    """Thin pass-through to an ``upstash_redis.Redis`` client."""

    def __init__(self, redis: Redis):
        self._r = redis

    def get(self, key: str) -> str | None:
        return self._r.get(key)

    def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        if ex is None:
            self._r.set(key, value)
        else:
            self._r.set(key, value, ex=ex)

    def delete(self, key: str) -> int:
        return self._r.delete(key)

    def keys(self, pattern: str) -> list[str]:
        return list(self._r.keys(pattern))

    def mget(self, *keys: str) -> list[str | None]:
        if not keys:
            return []
        return list(self._r.mget(*keys))

    def hget(self, hash_name: str, field: str) -> str | None:
        return self._r.hget(hash_name, field)

    def hset(self, hash_name: str, field: str, value: str) -> None:
        self._r.hset(hash_name, field, value)

    def hkeys(self, hash_name: str) -> list[str]:
        return list(self._r.hkeys(hash_name))

    def hgetall(self, hash_name: str) -> dict[str, str]:
        return dict(self._r.hgetall(hash_name))


class SqliteKVStore:
    """File-backed KV store matching the Upstash subset we use.

    Schema:
      - ``kv(key PK, value, expires_at NULL)`` for string keys
      - ``hash_kv(hash_name, field, value; PK hash_name+field)`` for hashes

    TTL on strings is lazy — expired rows are treated as missing on
    read and deleted opportunistically. Matches Upstash's observed
    behavior closely enough for our single TTL use case (job logs).

    Writes serialize through a process-local RLock; WAL mode lets
    readers proceed in parallel. ``check_same_thread=False`` plus the
    lock supports Flask's threaded worker model.
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at REAL
            );
            CREATE TABLE IF NOT EXISTS hash_kv (
                hash_name TEXT NOT NULL,
                field TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (hash_name, field)
            );
            """
        )
        self._lock = threading.RLock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM kv WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            value, expires_at = row
            if expires_at is not None and expires_at < time.time():
                self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
                return None
            return value if value is None else str(value)

    def set(self, key: str, value: str, *, ex: int | None = None) -> None:
        expires_at = (time.time() + ex) if ex is not None else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv(key, value, expires_at) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, expires_at = excluded.expires_at",
                (key, value, expires_at),
            )

    def delete(self, key: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            return cur.rowcount

    def keys(self, pattern: str) -> list[str]:
        like = pattern.replace("*", "%").replace("?", "_")
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT key FROM kv WHERE key LIKE ? AND (expires_at IS NULL OR expires_at >= ?)",
                (like, now),
            ).fetchall()
        return [r[0] for r in rows]

    def mget(self, *keys: str) -> list[str | None]:
        if not keys:
            return []
        now = time.time()
        placeholders = ",".join("?" * len(keys))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key, value, expires_at FROM kv WHERE key IN ({placeholders})",
                keys,
            ).fetchall()
        by_key = {k: v for (k, v, exp) in rows if exp is None or exp >= now}
        return [by_key.get(k) for k in keys]

    def hget(self, hash_name: str, field: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM hash_kv WHERE hash_name = ? AND field = ?",
                (hash_name, field),
            ).fetchone()
        return row[0] if row else None

    def hset(self, hash_name: str, field: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO hash_kv(hash_name, field, value) VALUES(?, ?, ?) "
                "ON CONFLICT(hash_name, field) DO UPDATE SET value = excluded.value",
                (hash_name, field, value),
            )

    def hkeys(self, hash_name: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT field FROM hash_kv WHERE hash_name = ?", (hash_name,)
            ).fetchall()
        return [r[0] for r in rows]

    def hgetall(self, hash_name: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT field, value FROM hash_kv WHERE hash_name = ?", (hash_name,)
            ).fetchall()
        return dict(rows)


_kv_singleton: KVStore | None = None
_kv_singleton_lock = threading.Lock()


def _load_dotenv_if_present() -> None:
    """Load project-root .env so local scripts can reach Upstash
    without callers having to source it. ``setdefault`` — real env
    vars always win."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def get_kv() -> KVStore:
    """Return the KV backend for the current environment.

    On Render: Upstash (reads creds from env).
    Off Render: SQLite at ``data/local.db`` — override path with
    ``FANTASY_LOCAL_KV_PATH`` (tests use this for isolation).

    Cached process-wide. Off-Render there is NO path from this
    function to Upstash regardless of env var values — the
    ``RENDER`` gate is hard.
    """
    global _kv_singleton
    if _kv_singleton is not None:
        return _kv_singleton
    with _kv_singleton_lock:
        if _kv_singleton is not None:
            return _kv_singleton
        _kv_singleton = _build_upstash_kv() if is_remote() else _build_sqlite_kv()
    return _kv_singleton


def _build_upstash_kv() -> UpstashKVStore:
    _load_dotenv_if_present()
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    if not (url and token):
        raise RuntimeError(
            "UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN must be set to build "
            "an Upstash client (on Render they're service env vars; locally they "
            "live in .env)."
        )
    from upstash_redis import Redis

    return UpstashKVStore(Redis(url=url, token=token))


def _build_sqlite_kv() -> SqliteKVStore:
    path_str = os.environ.get("FANTASY_LOCAL_KV_PATH")
    path = Path(path_str) if path_str else _DEFAULT_LOCAL_DB
    return SqliteKVStore(path)


def build_explicit_upstash_kv() -> UpstashKVStore:
    """Build an Upstash KV regardless of ``RENDER``.

    Legitimate callers are tools whose job is explicitly to reach prod
    Upstash from a local process: ``scripts/refresh_remote.py``,
    ``fantasy_baseball.data.kv_sync``, and
    ``scripts/migrate_standings_history.py``. Every other caller must
    use ``get_kv()`` so the environment gate holds.
    """
    return _build_upstash_kv()


def _reset_singleton() -> None:
    """Clear the cached singleton so the next ``get_kv()`` rebuilds
    against the current environment.

    Two legitimate callers: pytest fixtures (per-test isolation) and
    ``scripts/refresh_remote.py`` (flips ``RENDER`` mid-process). App
    code should never need this — ``get_kv()`` is a singleton by
    design."""
    global _kv_singleton
    with _kv_singleton_lock:
        _kv_singleton = None
