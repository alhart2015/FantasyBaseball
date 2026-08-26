"""Shared fixtures for the dashboard-launcher tests.

Both launcher test files bind KV stores and set the manual-mode environment
variables. `activate_manual_environment` writes them with raw `os.environ[...]`
assignment, which monkeypatch cannot undo unless it already holds an entry for
that name -- so the reset here is explicit rather than incidental.
"""

from __future__ import annotations

import pytest

MANUAL_ENV_VARS = ("FANTASY_LOCAL_KV_PATH", "FB_SKIP_YAHOO")


@pytest.fixture(autouse=True)
def _fresh_kv_singleton():
    """Discard the process-wide KV singleton around every test.

    ``get_kv()`` caches its backend, so a store built against a test's
    ``tmp_path`` would otherwise outlive the directory and be handed to
    whatever runs next in the same worker.
    """
    from fantasy_baseball.data import kv_store

    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


@pytest.fixture(autouse=True)
def _no_manual_env_leak(monkeypatch):
    """Register an undo for every manual-mode variable, set or not.

    `monkeypatch.delenv(..., raising=False)` records the current value so the
    original is restored afterwards. Without this, a test that reaches
    `activate_manual_environment` leaves `FANTASY_LOCAL_KV_PATH` pointing at a
    deleted ``tmp_path`` and `FB_SKIP_YAHOO=1` for the rest of the worker.
    """
    for name in MANUAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def seeded_store(tmp_path):
    """A manual store with a live provenance stamp."""

    def _make(name: str = "manual.db"):
        from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY
        from fantasy_baseball.data.kv_store import SqliteKVStore

        path = tmp_path / name
        SqliteKVStore(path).set(MANUAL_PROVENANCE_KEY, '{"seeded": "yes"}')
        return path

    return _make
