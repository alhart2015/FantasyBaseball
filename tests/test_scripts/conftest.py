"""Shared fixtures for the tests under ``tests/test_scripts/``."""

from __future__ import annotations

import pytest

from fantasy_baseball.data.kv_store import LOCAL_KV_PATH_ENV
from fantasy_baseball.web.refresh_pipeline import SKIP_YAHOO_ENV


@pytest.fixture(autouse=True)
def _shell_isolation_and_singleton_teardown(monkeypatch):
    """Two gaps the root conftest leaves, both specific to this directory.

    **Inbound.** ``tests/conftest.py::_restore_process_scoped_env`` restores
    these two variables afterwards; it does not clear them beforehand. This
    project's runbook tells the operator to run manual mode in a shell with
    ``FANTASY_LOCAL_KV_PATH`` and ``FB_SKIP_YAHOO`` exported, and these are the
    tests that drive the launchers those variables steer -- so a maintainer's
    own shell would otherwise change what they assert.

    **Teardown.** ``tests/conftest.py::_isolate_kv_from_prod`` is not a
    generator: it resets the KV singleton at SETUP only. Tests here call
    ``get_kv()`` against ``tmp_path``, and that store's connection stays open
    until something drops the reference -- after ``tmp_path`` teardown has
    already tried to unlink the file.
    """
    monkeypatch.delenv(LOCAL_KV_PATH_ENV, raising=False)
    monkeypatch.delenv(SKIP_YAHOO_ENV, raising=False)

    yield

    from fantasy_baseball.data import kv_store

    kv_store._reset_singleton()


@pytest.fixture
def seeded_store(tmp_path):
    """Factory for a manual store carrying a live provenance stamp.

    Closes what it opens: `SqliteKVStore` exposes no `close()`, and on Windows a
    live handle plus its -wal/-shm sidecars can fail `tmp_path` cleanup. This
    covers only the stores the factory itself built; the singleton's is handled
    by the autouse teardown above.
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
