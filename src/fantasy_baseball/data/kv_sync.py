"""Copy the remote Upstash KV down to the local SQLite KV.

Use this to pull a fresh snapshot of production state for offline work
(dashboards, scripts, debugging). It is the ONLY path (besides
``scripts/refresh_remote.py``) that crosses the local↔remote boundary,
and it only does so in the safe direction: remote → local.

Design:

- The schema has three hash-typed keys
  (``weekly_rosters_history``, ``standings_history``,
  ``projected_standings_history``); everything else is a string.
  String keys are enumerated via ``keys("*")``; hash names are
  iterated explicitly from the known constants. (The two backends
  don't agree on whether ``keys("*")`` returns hash names — Upstash
  does, our SQLite backend doesn't — so we sidestep the question.)
- The local DB is wiped first (both tables) so the sync leaves no
  stale rows behind. Acceptable because local SQLite is derived state
  — if a script needed uncommitted local writes they'd live in Redis
  anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fantasy_baseball.data.kv_store import (
    KVStore,
    SqliteKVStore,
    build_explicit_upstash_kv,
    get_kv,
    is_remote,
)
from fantasy_baseball.data.redis_store import (
    PROJECTED_STANDINGS_HISTORY_KEY,
    STANDINGS_HISTORY_KEY,
    WEEKLY_ROSTERS_HISTORY_KEY,
)

logger = logging.getLogger(__name__)

_HASH_KEYS: frozenset[str] = frozenset(
    {WEEKLY_ROSTERS_HISTORY_KEY, STANDINGS_HISTORY_KEY, PROJECTED_STANDINGS_HISTORY_KEY}
)

# Read string values in batches via MGET rather than one GET per key.
# Production Upstash holds ~1,300+ string keys (per-player game logs
# dominate); a per-key network GET loop made the startup sync grind for
# minutes. Batching cuts round-trips by ~25x.
#
# Sized by KEY COUNT, but the real ceiling is BYTES: Upstash caps a REST
# request at 10 MB, and game-log values run ~100 KB each, so 50 keys
# stays comfortably under the cap on a typical chunk while keeping the
# request count low. ``_mget_chunked`` halves and retries if a chunk
# still overflows (e.g. an unlucky run of large values).
_MGET_CHUNK = 50


@dataclass(frozen=True)
class SyncStats:
    string_keys: int
    hash_keys: int
    hash_fields: int

    def summary(self) -> str:
        return (
            f"{self.string_keys} string keys, "
            f"{self.hash_keys} hash keys ({self.hash_fields} fields)"
        )


def sync_remote_to_local(
    *,
    remote: KVStore | None = None,
    local: KVStore | None = None,
) -> SyncStats:
    """Overwrite the local KV with a fresh copy of the remote KV.

    Defaults:
      - ``remote``: ``build_explicit_upstash_kv()`` — explicit because
        this crosses the env gate.
      - ``local``: ``get_kv()`` — must resolve to SQLite, which means
        the caller must be off-Render. We refuse to run on Render: the
        remote IS the authoritative store there, so syncing over it
        would be nonsense at best and destructive at worst.
    """
    if is_remote():
        raise RuntimeError(
            "sync_remote_to_local is a local-only operation: on Render the "
            "Upstash KV is authoritative and has nothing to sync to."
        )

    src = remote if remote is not None else build_explicit_upstash_kv()
    dst = local if local is not None else get_kv()

    if isinstance(dst, SqliteKVStore):
        _wipe_sqlite(dst)

    string_keys = [k for k in src.keys("*") if k not in _HASH_KEYS]
    for start in range(0, len(string_keys), _MGET_CHUNK):
        chunk = string_keys[start : start + _MGET_CHUNK]
        for key, value in zip(chunk, _mget_chunked(src, chunk), strict=True):
            if value is not None:
                dst.set(key, value)

    populated_hash_keys = 0
    hash_field_total = 0
    for hash_name in _HASH_KEYS:
        fields = src.hgetall(hash_name)
        if not fields:
            continue
        for field, value in fields.items():
            dst.hset(hash_name, field, value)
        populated_hash_keys += 1
        hash_field_total += len(fields)

    stats = SyncStats(
        string_keys=len(string_keys),
        hash_keys=populated_hash_keys,
        hash_fields=hash_field_total,
    )
    logger.info("sync_remote_to_local complete: %s", stats.summary())
    return stats


def _mget_chunked(src: KVStore, keys: list[str]) -> list[str | None]:
    """MGET ``keys`` in order, halving the batch and retrying on overflow.

    Upstash caps a single REST request at 10 MB. Most batches fit, but a
    run of large values (per-player game logs) can blow the cap, so on
    any failure we split the batch and retry each half; a single key
    always fits. Genuine backend errors (auth, outage) surface fast: the
    left half is evaluated first and re-raises at the leaf (``len == 1``)
    before the right half is attempted, so a persistent error fails after
    ~log2(n) calls rather than hammering every key.
    """
    try:
        return list(src.mget(*keys))
    except Exception:
        if len(keys) <= 1:
            raise
        mid = len(keys) // 2
        return _mget_chunked(src, keys[:mid]) + _mget_chunked(src, keys[mid:])


def _wipe_sqlite(store: SqliteKVStore) -> None:
    """Clear both tables so the sync starts from an empty local DB.

    Reaches into ``_conn``/``_lock`` because the ``KVStore`` protocol
    deliberately has no ``flush`` verb — Upstash callers should never
    be able to flush the remote DB through this abstraction.
    """
    with store._lock:
        store._conn.executescript("DELETE FROM kv; DELETE FROM hash_kv;")
