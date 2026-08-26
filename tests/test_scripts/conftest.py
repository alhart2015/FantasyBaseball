"""Shared fixtures for the dashboard-launcher tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def seeded_store(tmp_path):
    """Factory for a manual store carrying a live provenance stamp.

    Closes each store it opens: `SqliteKVStore` exposes no `close()`, and on
    Windows a live handle plus its -wal/-shm sidecars can fail `tmp_path`
    cleanup.
    """
    opened = []

    def _make(name: str = "manual.db"):
        from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY
        from fantasy_baseball.data.kv_store import SqliteKVStore

        path = tmp_path / name
        store = SqliteKVStore(path)
        store.set(MANUAL_PROVENANCE_KEY, '{"seeded": "yes"}')
        opened.append(store)
        return path

    yield _make

    for store in opened:
        store._conn.close()
