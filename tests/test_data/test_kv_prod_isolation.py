"""The autouse KV guard in conftest must make it impossible for a test to
reach PROD Upstash, even if ``RENDER=true`` leaks into the process (e.g. a
module that sets it at import time landing in an xdist worker -- the exact
leak that clobbered prod META/standings with test fixtures)."""

from fantasy_baseball.data import kv_store


def test_autouse_guard_leaves_tests_on_the_local_store():
    # The autouse guard ran at setup: RENDER is stripped, so the backend is the
    # local SQLite store, never Upstash.
    assert kv_store.is_remote() is False
    assert type(kv_store.get_kv()).__name__ == "SqliteKVStore"


def test_render_leak_remediation_forces_local(monkeypatch):
    # Simulate the import-time leak: RENDER=true would flip get_kv() to Upstash.
    monkeypatch.setenv("RENDER", "true")
    kv_store._reset_singleton()
    assert kv_store.is_remote() is True  # leak active

    # The remediation the autouse guard applies (delenv RENDER + reset the
    # cached singleton) must force the local store -- so a leaked RENDER can
    # never build a real Upstash client and write prod. get_kv() is only called
    # AFTER RENDER is cleared, so this never touches Upstash.
    monkeypatch.delenv("RENDER", raising=False)
    kv_store._reset_singleton()
    assert kv_store.is_remote() is False
    assert type(kv_store.get_kv()).__name__ == "SqliteKVStore"
