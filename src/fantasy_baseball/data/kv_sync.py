"""Copy KV state across the local <-> remote boundary.

``sync_remote_to_local`` pulls a fresh snapshot of production state down for
offline work (dashboards, scripts, debugging). It wipes its destination.

``publish_local_to_remote`` goes the other way, for the manual pipeline: while
Yahoo access is gone the hand-transcribed store (``data/manual.db``) IS the
league state, and publishing it is how Render shows it. It never deletes a
remote key and it writes only what differs -- see its docstring.

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
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY, CacheKey, redis_key
from fantasy_baseball.data.kv_store import (
    _DEFAULT_LOCAL_DB,
    LOCAL_KV_PATH_ENV,
    KVStore,
    SqliteKVStore,
    build_explicit_upstash_kv,
    get_kv,
    is_remote,
)
from fantasy_baseball.data.redis_store import (
    PROJECTED_STANDINGS_HISTORY_KEY,
    REFRESH_LOCK_KEY,
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


def local_destination() -> Path | None:
    """Where a ``local=None`` sync would write, WITHOUT opening the store.

    Read straight off the environment rather than through ``get_kv()``, for two
    reasons that both bit the operator guards:

    - ``get_kv()`` answers according to ``RENDER``. An operator with ``RENDER=true``
      already exported -- which this repo's own CLAUDE.md tells them to do to read
      Upstash -- got the Upstash client, no path, and a guard that refused a
      perfectly legitimate run as "a store with no local file".
    - ``get_kv()`` CONSTRUCTS the store. ``SqliteKVStore.__init__`` mkdirs the parent
      and runs ``CREATE TABLE IF NOT EXISTS``, so merely asking the question created
      an empty database and its WAL sidecars -- underneath a refusal whose own text
      promises nothing was written.

    Returns None only when the resolved path cannot be parsed, which is not a state
    the guards need to distinguish from "not the baseline".
    """
    raw = os.environ.get(LOCAL_KV_PATH_ENV)
    try:
        return Path(raw).resolve() if raw else _DEFAULT_LOCAL_DB.resolve()
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


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
        # The invariant of the guard itself, so both callers state it identically:
        # the one sentence an operator staring at a refusal actually needs.
        "Nothing was deleted.",
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


#: String keys a publish never sends. Each is owned by something other than the
#: store being published:
#:
#: - ``refresh:lock`` is a live mutex. Copying one up would block every refresh
#:   until its TTL ran out -- and the copy would carry no TTL at all.
#: - The two trajectory blobs are written straight to prod by
#:   ``scripts/push_trajectory_board.py``, on their own schedule. The local copy is
#:   whatever that script last wrote with ``--local``, so publishing it could
#:   replace a newer prod board with an older one.
_PUBLISH_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {
        REFRESH_LOCK_KEY,
        redis_key(CacheKey.TRAJECTORY_BOARD),
        redis_key(CacheKey.TRAJECTORY_CHART_DATA),
    }
)

#: Key prefixes a publish never sends. Job logs are written with a TTL, which
#: ``KVStore.set`` cannot read back, so a copy would live forever on the remote.
#: Each environment keeps its own run history.
_PUBLISH_EXCLUDED_PREFIXES: tuple[str, ...] = ("job_log:",)

#: Written LAST, so a publish that dies part-way leaves the page's "Last refresh"
#: on the previous publish rather than claiming a vintage it only half-delivered.
_PUBLISH_LAST: str = redis_key(CacheKey.META)


@dataclass(frozen=True)
class PublishStats:
    """What a publish sent, and what it would have overwritten.

    ``previous`` holds the remote's value for every key and hash field the
    publish CHANGED (None where the remote had nothing), so the caller can save
    it before writing -- the remote carries no history of its own.
    """

    strings_changed: int
    strings_unchanged: int
    strings_excluded: int
    hash_fields_changed: int
    hash_fields_unchanged: int
    remote_only_keys: int
    bytes_sent: int
    previous: dict[str, str | None]
    previous_hash_fields: dict[str, dict[str, str | None]]

    def summary(self) -> str:
        return (
            f"{self.strings_changed} keys changed ({self.strings_unchanged} unchanged, "
            f"{self.strings_excluded} excluded), {self.hash_fields_changed} hash fields "
            f"changed ({self.hash_fields_unchanged} unchanged), "
            f"{self.bytes_sent / 1e6:.1f} MB; {self.remote_only_keys} remote-only keys left alone"
        )


def _publishable(key: str) -> bool:
    return key not in _PUBLISH_EXCLUDED_KEYS and not key.startswith(_PUBLISH_EXCLUDED_PREFIXES)


def publish_local_to_remote(
    *,
    local: KVStore,
    remote: KVStore,
    dry_run: bool = False,
    before_write: Callable[[PublishStats], None] | None = None,
) -> PublishStats:
    """Make ``remote`` match ``local`` for every key ``local`` holds.

    For the manual pipeline, whose store is the only league state there is while
    Yahoo access is gone. Three rules, each for a reason:

    - **Upsert, never delete.** A key only the remote holds stays. Nothing about
      the local store's silence on a key is evidence the remote copy is wrong, and
      a delete is the one write the saved ``previous`` values could not undo.
    - **Only what differs is written.** Values are compared with the remote's
      first, so a publish after a small manual refresh sends a few MB, not the
      whole store -- and the stats say how much actually moved.
    - **The provenance stamp goes first and ``cache:meta`` last.** The stamp is
      what disables the Render refresh button (see ``manual_store_active``), so it
      must be in place before any hand-typed value lands; ``cache:meta`` is what
      the page prints as "Last refresh", so it moves only once everything else has.

    ``dry_run`` does every read and the whole comparison, then writes nothing.
    ``before_write`` is called with the finished comparison after the reads and
    before the first write -- the one moment ``previous`` is both known and still
    true of the remote. An exception from it aborts the publish with nothing sent.
    """
    local_keys = sorted(k for k in local.keys("*") if k not in _HASH_KEYS)
    wanted = [k for k in local_keys if _publishable(k)]
    excluded = len(local_keys) - len(wanted)

    local_values: dict[str, str | None] = {}
    remote_values: dict[str, str | None] = {}
    for start in range(0, len(wanted), _MGET_CHUNK):
        chunk = wanted[start : start + _MGET_CHUNK]
        local_values.update(zip(chunk, _mget_chunked(local, chunk), strict=True))
        remote_values.update(zip(chunk, _mget_chunked(remote, chunk), strict=True))

    changed = [
        k for k in wanted if local_values[k] is not None and local_values[k] != remote_values[k]
    ]
    # Stamp first, meta last; everything else in between, in key order.
    changed.sort(key=lambda k: (k != MANUAL_PROVENANCE_KEY, k == _PUBLISH_LAST, k))

    hash_changes: dict[str, dict[str, str]] = {}
    previous_hash_fields: dict[str, dict[str, str | None]] = {}
    hash_unchanged = 0
    for hash_name in sorted(_HASH_KEYS):
        mine = local.hgetall(hash_name)
        if not mine:
            continue
        theirs = remote.hgetall(hash_name)
        diff = {f: v for f, v in mine.items() if theirs.get(f) != v}
        hash_unchanged += len(mine) - len(diff)
        if diff:
            hash_changes[hash_name] = diff
            previous_hash_fields[hash_name] = {f: theirs.get(f) for f in diff}

    wanted_set = set(wanted)
    remote_only = sum(
        1
        for k in remote.keys("*")
        if k not in _HASH_KEYS and _publishable(k) and k not in wanted_set
    )
    bytes_sent = sum(len(local_values[k] or "") for k in changed) + sum(
        len(v) for diff in hash_changes.values() for v in diff.values()
    )
    stats = PublishStats(
        strings_changed=len(changed),
        strings_unchanged=len(wanted) - len(changed),
        strings_excluded=excluded,
        hash_fields_changed=sum(len(d) for d in hash_changes.values()),
        hash_fields_unchanged=hash_unchanged,
        remote_only_keys=remote_only,
        bytes_sent=bytes_sent,
        previous={k: remote_values[k] for k in changed},
        previous_hash_fields=previous_hash_fields,
    )
    if dry_run:
        return stats
    if before_write is not None:
        before_write(stats)

    # Strings before hashes, except that the meta write is held until the very end.
    for key in changed:
        if key == _PUBLISH_LAST:
            continue
        value = local_values[key]
        assert value is not None  # filtered above; narrows for the type checker
        remote.set(key, value)
    for hash_name, diff in hash_changes.items():
        for field, value in diff.items():
            remote.hset(hash_name, field, value)
    if _PUBLISH_LAST in changed:
        meta = local_values[_PUBLISH_LAST]
        assert meta is not None
        remote.set(_PUBLISH_LAST, meta)

    logger.info("publish_local_to_remote complete: %s", stats.summary())
    return stats
