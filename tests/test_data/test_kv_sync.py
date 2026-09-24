"""Tests for the remote→local KV sync.

Uses two ``SqliteKVStore`` instances as stand-ins for the remote and
local backends. The production code path uses the same ``KVStore``
protocol for both, so the substitution is clean — there is no
Upstash-specific branch inside ``sync_remote_to_local``.
"""

from __future__ import annotations

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY, CacheKey, redis_key
from fantasy_baseball.data.kv_store import SqliteKVStore
from fantasy_baseball.data.kv_sync import (
    SyncStats,
    publish_local_to_remote,
    sync_remote_to_local,
)
from fantasy_baseball.data.redis_store import (
    REFRESH_LOCK_KEY,
    STANDINGS_HISTORY_KEY,
    WEEKLY_ROSTERS_HISTORY_KEY,
)


@pytest.fixture(autouse=True)
def _off_render(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


@pytest.fixture
def remote_kv(tmp_path) -> SqliteKVStore:
    return SqliteKVStore(tmp_path / "remote.db")


@pytest.fixture
def local_kv(tmp_path) -> SqliteKVStore:
    return SqliteKVStore(tmp_path / "local.db")


class _CountingKV:
    """Wraps a ``KVStore`` and counts ``get`` vs ``mget`` calls.

    Used to pin the read strategy of the sync: with ~1,300 prod keys, a
    per-key ``get`` over the network made ``run_season_dashboard`` hang
    for minutes. The sync must batch reads with ``mget`` instead.
    """

    def __init__(self, inner: SqliteKVStore):
        self._inner = inner
        self.get_calls = 0
        self.mget_calls = 0

    def get(self, key):
        self.get_calls += 1
        return self._inner.get(key)

    def mget(self, *keys):
        self.mget_calls += 1
        return self._inner.mget(*keys)

    def set(self, key, value, *, ex=None):
        return self._inner.set(key, value, ex=ex)

    def delete(self, key):
        return self._inner.delete(key)

    def keys(self, pattern):
        return self._inner.keys(pattern)

    def hget(self, hash_name, field):
        return self._inner.hget(hash_name, field)

    def hset(self, hash_name, field, value):
        return self._inner.hset(hash_name, field, value)

    def hkeys(self, hash_name):
        return self._inner.hkeys(hash_name)

    def hgetall(self, hash_name):
        return self._inner.hgetall(hash_name)


def test_copies_string_keys(remote_kv, local_kv):
    remote_kv.set("positions", '{"a": ["OF"]}')
    remote_kv.set("cache:standings", '{"x": 1}')

    stats = sync_remote_to_local(remote=remote_kv, local=local_kv)

    assert local_kv.get("positions") == '{"a": ["OF"]}'
    assert local_kv.get("cache:standings") == '{"x": 1}'
    assert stats.string_keys == 2
    assert stats.hash_keys == 0


def test_string_sync_uses_batched_mget_not_per_key_get(remote_kv, local_kv):
    """Regression: reads must be batched with mget, not one round-trip
    per key. A 250-key remote should produce zero per-key gets and only
    a handful of mget calls -- the original per-key loop made the startup
    sync grind through ~1,360 sequential HTTPS GETs against Upstash."""
    for i in range(250):
        remote_kv.set(f"cache:k{i}", str(i))

    spy = _CountingKV(remote_kv)
    stats = sync_remote_to_local(remote=spy, local=local_kv)

    # Every value still copied, in order, with correct values.
    assert stats.string_keys == 250
    assert local_kv.get("cache:k0") == "0"
    assert local_kv.get("cache:k123") == "123"
    assert local_kv.get("cache:k249") == "249"

    # The load-bearing assertion: batched, not per-key.
    assert spy.get_calls == 0
    assert 1 <= spy.mget_calls <= 5


def test_mget_splits_oversized_batch_and_copies_all(remote_kv, local_kv):
    """Upstash caps a REST request at 10 MB; large game-log values can
    push a batch over. The sync must halve and retry, not fail -- every
    value still lands, with no fallback to per-key gets."""
    for i in range(120):
        remote_kv.set(f"cache:k{i}", str(i))

    class _OverflowKV(_CountingKV):
        def mget(self, *keys):
            if len(keys) > 20:  # stand in for the 10 MB request cap
                raise RuntimeError("ERR max request size exceeded")
            return super().mget(*keys)

    spy = _OverflowKV(remote_kv)
    stats = sync_remote_to_local(remote=spy, local=local_kv)

    assert stats.string_keys == 120
    assert local_kv.get("cache:k0") == "0"
    assert local_kv.get("cache:k119") == "119"
    assert spy.get_calls == 0


def test_mget_reraises_persistent_backend_error(remote_kv, local_kv):
    """A real outage must surface, not loop forever: once a batch can't
    be split further the error propagates."""
    remote_kv.set("cache:a", "1")
    remote_kv.set("cache:b", "2")

    class _BrokenKV(_CountingKV):
        def mget(self, *keys):
            raise RuntimeError("upstash down")

    with pytest.raises(RuntimeError, match="upstash down"):
        sync_remote_to_local(remote=_BrokenKV(remote_kv), local=local_kv)


def test_copies_hash_keys(remote_kv, local_kv):
    remote_kv.hset(WEEKLY_ROSTERS_HISTORY_KEY, "2026-04-07", '[{"a": 1}]')
    remote_kv.hset(WEEKLY_ROSTERS_HISTORY_KEY, "2026-04-14", '[{"b": 2}]')
    remote_kv.hset(STANDINGS_HISTORY_KEY, "2026-04-14", '{"teams": []}')

    stats = sync_remote_to_local(remote=remote_kv, local=local_kv)

    assert local_kv.hgetall(WEEKLY_ROSTERS_HISTORY_KEY) == {
        "2026-04-07": '[{"a": 1}]',
        "2026-04-14": '[{"b": 2}]',
    }
    assert local_kv.hget(STANDINGS_HISTORY_KEY, "2026-04-14") == '{"teams": []}'
    assert stats.hash_keys == 2
    assert stats.hash_fields == 3


def test_wipes_local_before_sync(remote_kv, local_kv):
    """Stale local state must not survive a sync."""
    local_kv.set("stale_string", "old")
    local_kv.hset(WEEKLY_ROSTERS_HISTORY_KEY, "2026-03-30", "stale-hash-field")
    remote_kv.set("fresh_string", "new")

    sync_remote_to_local(remote=remote_kv, local=local_kv)

    assert local_kv.get("stale_string") is None
    assert local_kv.hget(WEEKLY_ROSTERS_HISTORY_KEY, "2026-03-30") is None
    assert local_kv.get("fresh_string") == "new"


def test_empty_remote_produces_empty_local(remote_kv, local_kv):
    local_kv.set("stale", "x")
    stats = sync_remote_to_local(remote=remote_kv, local=local_kv)
    assert local_kv.get("stale") is None
    assert stats == SyncStats(string_keys=0, hash_keys=0, hash_fields=0)


def test_refuses_to_run_on_render(monkeypatch, remote_kv, local_kv):
    monkeypatch.setenv("RENDER", "true")
    with pytest.raises(RuntimeError, match="local-only"):
        sync_remote_to_local(remote=remote_kv, local=local_kv)


def test_default_local_is_get_kv(monkeypatch, remote_kv, tmp_path):
    """When ``local`` is omitted, the sync writes to whatever
    ``get_kv()`` resolves to — SQLite off-Render."""
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "default.db"))
    remote_kv.set("k", "v")

    sync_remote_to_local(remote=remote_kv)

    assert kv_store.get_kv().get("k") == "v"


def test_sync_replicates_projected_standings_history(monkeypatch, tmp_path):
    """Projected standings history is hash-typed and must round-trip
    through sync_remote_to_local."""
    import json

    from fantasy_baseball.data import kv_store, kv_sync, redis_store

    monkeypatch.setenv("RENDER", "")

    remote = kv_store.SqliteKVStore(tmp_path / "remote.db")
    local = kv_store.SqliteKVStore(tmp_path / "local.db")

    remote.hset(
        redis_store.PROJECTED_STANDINGS_HISTORY_KEY,
        "2026-04-15",
        json.dumps({"effective_date": "2026-04-15", "teams": []}),
    )

    kv_sync.sync_remote_to_local(remote=remote, local=local)

    assert (
        local.hget(redis_store.PROJECTED_STANDINGS_HISTORY_KEY, "2026-04-15")
        == '{"effective_date": "2026-04-15", "teams": []}'
    )


# ---------------------------------------------------------------------------
# publish_local_to_remote -- the manual store going UP to prod
# ---------------------------------------------------------------------------

META = redis_key(CacheKey.META)


class _RecordingKV(_CountingKV):
    """Records every write, in order, so the write ORDER can be pinned."""

    def __init__(self, inner):
        super().__init__(inner)
        self.writes: list[str] = []

    def set(self, key, value, *, ex=None):
        self.writes.append(key)
        return super().set(key, value, ex=ex)

    def hset(self, hash_name, field, value):
        self.writes.append(f"{hash_name}#{field}")
        return super().hset(hash_name, field, value)


def test_publish_copies_what_differs_and_skips_what_matches(local_kv, remote_kv):
    local_kv.set("cache:standings", "new")
    local_kv.set("positions", "same")
    remote_kv.set("cache:standings", "old")
    remote_kv.set("positions", "same")

    spy = _RecordingKV(remote_kv)
    stats = publish_local_to_remote(local=local_kv, remote=spy)

    assert remote_kv.get("cache:standings") == "new"
    assert spy.writes == ["cache:standings"], "an unchanged value must not be re-sent"
    assert (stats.strings_changed, stats.strings_unchanged) == (1, 1)
    assert stats.previous == {"cache:standings": "old"}


def test_publish_never_deletes_a_remote_only_key(local_kv, remote_kv):
    local_kv.set("cache:standings", "new")
    remote_kv.set("cache:only_on_prod", "keep me")

    stats = publish_local_to_remote(local=local_kv, remote=remote_kv)

    assert remote_kv.get("cache:only_on_prod") == "keep me"
    assert stats.remote_only_keys == 1


def test_publish_excludes_the_lock_job_logs_and_trajectory_blobs(local_kv, remote_kv):
    """Each excluded key is owned elsewhere: a live mutex, TTL'd logs, an offline push."""
    for key in (
        REFRESH_LOCK_KEY,
        "job_log:manual:2026-09-14:1",
        redis_key(CacheKey.TRAJECTORY_BOARD),
        redis_key(CacheKey.TRAJECTORY_CHART_DATA),
    ):
        local_kv.set(key, "local")
    remote_kv.set(redis_key(CacheKey.TRAJECTORY_BOARD), "newer prod board")

    stats = publish_local_to_remote(local=local_kv, remote=remote_kv)

    assert remote_kv.get(REFRESH_LOCK_KEY) is None
    assert remote_kv.get("job_log:manual:2026-09-14:1") is None
    assert remote_kv.get(redis_key(CacheKey.TRAJECTORY_BOARD)) == "newer prod board"
    assert remote_kv.get(redis_key(CacheKey.TRAJECTORY_CHART_DATA)) is None
    assert stats.strings_excluded == 4
    assert stats.strings_changed == 0


def test_publish_sends_the_stamp_first_and_meta_last(local_kv, remote_kv):
    """The stamp disables Render's Refresh before any hand-typed value lands; meta
    (the page's "Last refresh") moves only once everything else has."""
    local_kv.set("cache:aaa", "1")
    local_kv.set(META, "meta")
    local_kv.set("cache:zzz", "2")
    local_kv.set(MANUAL_PROVENANCE_KEY, '{"seeded": true}')
    local_kv.hset(WEEKLY_ROSTERS_HISTORY_KEY, "2026-09-14", "[]")

    spy = _RecordingKV(remote_kv)
    publish_local_to_remote(local=local_kv, remote=spy)

    assert spy.writes[0] == MANUAL_PROVENANCE_KEY
    assert spy.writes[-1] == META
    assert f"{WEEKLY_ROSTERS_HISTORY_KEY}#2026-09-14" in spy.writes


def test_publish_diffs_hashes_field_by_field(local_kv, remote_kv):
    local_kv.hset(STANDINGS_HISTORY_KEY, "2026-09-07", "same")
    local_kv.hset(STANDINGS_HISTORY_KEY, "2026-09-14", "new")
    remote_kv.hset(STANDINGS_HISTORY_KEY, "2026-09-07", "same")
    remote_kv.hset(STANDINGS_HISTORY_KEY, "2026-08-01", "prod only")

    spy = _RecordingKV(remote_kv)
    stats = publish_local_to_remote(local=local_kv, remote=spy)

    assert spy.writes == [f"{STANDINGS_HISTORY_KEY}#2026-09-14"]
    assert remote_kv.hget(STANDINGS_HISTORY_KEY, "2026-08-01") == "prod only"
    assert (stats.hash_fields_changed, stats.hash_fields_unchanged) == (1, 1)
    assert stats.previous_hash_fields == {STANDINGS_HISTORY_KEY: {"2026-09-14": None}}


def test_publish_dry_run_writes_nothing(local_kv, remote_kv):
    local_kv.set("cache:standings", "new")
    local_kv.hset(STANDINGS_HISTORY_KEY, "2026-09-14", "new")
    spy = _RecordingKV(remote_kv)

    stats = publish_local_to_remote(local=local_kv, remote=spy, dry_run=True)

    assert spy.writes == []
    assert stats.strings_changed == 1 and stats.hash_fields_changed == 1


def test_publish_hook_runs_before_the_first_write_and_can_abort(local_kv, remote_kv):
    """The backup hook is the last moment `previous` is still true of the remote."""
    local_kv.set("cache:standings", "new")
    remote_kv.set("cache:standings", "old")
    spy = _RecordingKV(remote_kv)
    seen: list[dict] = []

    def _hook(stats):
        seen.append(dict(stats.previous))
        assert spy.writes == []
        raise OSError("disk full")

    with pytest.raises(OSError):
        publish_local_to_remote(local=local_kv, remote=spy, before_write=_hook)

    assert seen == [{"cache:standings": "old"}]
    assert remote_kv.get("cache:standings") == "old", "a failed backup must send nothing"


def test_publish_reads_in_batches(local_kv, remote_kv):
    for i in range(120):
        local_kv.set(f"cache:k{i}", str(i))
    spy = _CountingKV(remote_kv)

    publish_local_to_remote(local=local_kv, remote=spy)

    assert spy.get_calls == 0
    assert spy.mget_calls <= 5
