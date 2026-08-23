"""Copy the remote Upstash KV down to the local SQLite KV.

Use this to pull a fresh snapshot of production state for offline work
(dashboards, scripts, debugging). It is the ONLY path (besides
``scripts/refresh_remote.py``) that crosses the local↔remote boundary,
and it only does so in the safe direction: remote → local.

Design:

- The hash-typed keys are enumerated in ``_HASH_KEYS`` below;
  everything else is a string. String keys are enumerated via
  ``keys("*")``; hash names are iterated explicitly from that set.
  (The two backends don't agree on whether ``keys("*")`` returns hash
  names — Upstash does, our SQLite backend doesn't — so we sidestep
  the question.)
- The local DB is wiped first (both tables) so the sync leaves no
  stale rows behind. Acceptable because local SQLite is derived state
  — if a script needed uncommitted local writes they'd live in Redis
  anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fantasy_baseball.data.kv_store import (
    KVStore,
    SqliteKVStore,
    build_explicit_upstash_kv,
    get_kv,
    is_remote,
)
from fantasy_baseball.data.redis_store import (
    PROJECTED_STANDINGS_HISTORY_KEY,
    ROS_PROJECTION_HISTORY_KEY,
    STANDINGS_HISTORY_KEY,
    WEEKLY_ROSTERS_HISTORY_KEY,
)

logger = logging.getLogger(__name__)

_HASH_KEYS: frozenset[str] = frozenset(
    {
        WEEKLY_ROSTERS_HISTORY_KEY,
        STANDINGS_HISTORY_KEY,
        PROJECTED_STANDINGS_HISTORY_KEY,
        ROS_PROJECTION_HISTORY_KEY,
    }
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


def store_path(client: KVStore) -> Path | None:
    """The absolute file backing ``client``, or None when it has none.

    None is a real answer, not a failure: an Upstash client has no local file,
    and that is exactly the case the guards below must refuse.
    """
    raw = getattr(client, "path", None)
    if raw is None:
        return None
    try:
        return Path(raw).resolve()
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def sync_destination_refusal(
    kv_path: Path | None, *, action: str, recovery: list[str]
) -> str | None:
    """The refusal text when a sync would wipe a non-baseline store, else None.

    THE HAZARD. `sync_remote_to_local` resolves its destination as
    ``local if local is not None else get_kv()`` and then wipes it
    UNCONDITIONALLY -- ``DELETE FROM kv; DELETE FROM hash_kv;`` -- before
    refilling it from Upstash. Every caller that passes ``local=None`` therefore
    aims at whatever ``FANTASY_LOCAL_KV_PATH`` points at, and
    ``scripts/run_manual_refresh.py`` exports that to isolate the manual store.
    Running one of those callers from the same shell destroys the
    hand-transcribed standings and rosters and refills them with the last Yahoo
    snapshot -- no error, no prompt, and a store that still looks populated.

    THE GUARD IS NOT IN THE LIBRARY, deliberately. `FANTASY_LOCAL_KV_PATH` is
    also how the test suite isolates its KV, and
    `test_kv_sync.py::test_default_local_is_get_kv` pins the contract that the
    default destination is simply whatever `get_kv()` returns. Narrowing that
    library-side breaks legitimate callers. The hazard is specific to
    OPERATOR-FACING entry points, so this returns a message and each script
    decides -- but the message and the comparison live here, once, because two
    scripts had each written their own and they had already drifted apart.

    Args:
        kv_path: the destination the caller resolved, or None for a store with
            no local file. PASSED IN rather than resolved here so the caller's
            own startup banner and this guard cannot name different stores, and
            so both are testable without a process-global singleton.
        action: what would be wiped, in the imperative -- "The startup sync",
            "The sync-back". Named because the two callers refuse at different
            points in their run and the operator needs to know which.
        recovery: the caller's own next-step lines. What gets you out of it
            differs per script; the hazard does not.
    """
    # Read off the module at CALL time, not bound at import: relocating the
    # baseline is how the tests keep the real data/local.db out of a test run,
    # and a from-import here would freeze the real path into this function.
    from fantasy_baseball.data import kv_store

    baseline = kv_store._DEFAULT_LOCAL_DB.resolve()
    if kv_path == baseline:
        return None

    target = str(kv_path) if kv_path is not None else "a store with no local file"
    lines = [
        "",
        f"REFUSING TO SYNC: the resolved KV store is {target},",
        f"not the Yahoo baseline {baseline}.",
        f"{action} WIPES its destination (DELETE FROM kv; DELETE FROM hash_kv;) and "
        "refills it from Upstash, so it would destroy this store -- most likely the "
        "hand-transcribed manual store written by scripts/run_manual_refresh.py.",
        "",
        *recovery,
        "See docs/manual-pipeline-runbook.md.",
    ]
    return "\n".join(lines)


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
