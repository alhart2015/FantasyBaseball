"""Tests for the remote→local KV sync.

Uses two ``SqliteKVStore`` instances as stand-ins for the remote and
local backends. The production code path uses the same ``KVStore``
protocol for both, so the substitution is clean — there is no
Upstash-specific branch inside ``sync_remote_to_local``.
"""

from __future__ import annotations

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.kv_store import SqliteKVStore
from fantasy_baseball.data.kv_sync import (
    SyncStats,
    sync_remote_to_local,
)
from fantasy_baseball.data.redis_store import (
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


def test_copies_string_keys(remote_kv, local_kv):
    remote_kv.set("positions", '{"a": ["OF"]}')
    remote_kv.set("cache:standings", '{"x": 1}')

    stats = sync_remote_to_local(remote=remote_kv, local=local_kv)

    assert local_kv.get("positions") == '{"a": ["OF"]}'
    assert local_kv.get("cache:standings") == '{"x": 1}'
    assert stats.string_keys == 2
    assert stats.hash_keys == 0


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
