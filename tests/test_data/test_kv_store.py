"""Tests for the KV store abstraction.

Two properties matter most:

1. ``get_kv()`` never reaches Upstash off-Render, even with real creds
   in the environment. That is the leak-prevention invariant the whole
   redesign rests on.
2. ``SqliteKVStore`` exposes the same externally-observable behavior as
   ``UpstashKVStore`` for the subset of operations we use, so the
   ``redis_store`` helpers can treat both interchangeably.
"""

from __future__ import annotations

import os
import time

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.kv_store import (
    KVStore,
    SqliteKVStore,
    build_explicit_upstash_kv,
    get_kv,
    is_remote,
)


@pytest.fixture(autouse=True)
def reset_kv_singleton():
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


@pytest.fixture
def tmp_kv(tmp_path) -> SqliteKVStore:
    return SqliteKVStore(tmp_path / "kv.db")


def test_is_remote_off_by_default(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    assert is_remote() is False


def test_is_remote_true_when_render_set(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    assert is_remote() is True


def test_get_kv_returns_sqlite_off_render(monkeypatch, tmp_path):
    """Core leak-prevention invariant: even with Upstash creds in env,
    get_kv() off-Render returns SQLite."""
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://fake.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake")
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "kv.db"))

    kv = get_kv()
    assert isinstance(kv, SqliteKVStore)


def test_get_kv_cached(monkeypatch, tmp_path):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "kv.db"))
    assert get_kv() is get_kv()


def test_get_kv_on_render_requires_creds(monkeypatch):
    """On Render without creds, raise — misconfiguration should be loud.

    We also stub ``_load_dotenv_if_present`` because the repo's .env
    holds real creds for local development; without this stub the
    dotenv loader would repopulate the env after we cleared it."""
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    monkeypatch.setattr(kv_store, "_load_dotenv_if_present", lambda: None)
    with pytest.raises(RuntimeError, match="UPSTASH_REDIS_REST_URL"):
        get_kv()


def test_build_explicit_upstash_kv_works_off_render(monkeypatch):
    """The scripts-only escape hatch should NOT be gated on RENDER — it
    exists precisely to let local scripts cross the boundary.

    Under pytest the builder now fails closed (see
    ``test_build_explicit_upstash_kv_refuses_real_client_under_pytest``), so
    this test opts in explicitly. It uses FAKE creds, so no client can reach
    production even though the guard is lifted."""
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://fake.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake")
    monkeypatch.setenv("FANTASY_ALLOW_UPSTASH_IN_TESTS", "1")
    kv = build_explicit_upstash_kv()
    from fantasy_baseball.data.kv_store import UpstashKVStore

    assert isinstance(kv, UpstashKVStore)


def test_build_explicit_upstash_kv_refuses_real_client_under_pytest(monkeypatch):
    """Fail-closed prod guard. Even with creds present, the builder must
    refuse to construct a real Upstash client under pytest unless a test
    explicitly opts in. This is the regression guard for the incident where a
    test's ros_projections fixture was written to production Upstash (leaked
    ``RENDER=true`` under ``pytest -n auto``), clobbering the live key.

    Fake creds are set so the guard -- not the missing-creds check -- is what
    fires, making the assertion deterministic on machines without a ``.env``."""
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://fake.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake")
    monkeypatch.delenv("FANTASY_ALLOW_UPSTASH_IN_TESTS", raising=False)
    with pytest.raises(RuntimeError, match="under pytest"):
        build_explicit_upstash_kv()
    # And get_kv() routed to remote (leaked RENDER) must not slip past it either.
    monkeypatch.setenv("RENDER", "true")
    kv_store._reset_singleton()
    with pytest.raises(RuntimeError, match="under pytest"):
        kv_store.get_kv()


def test_load_dotenv_strips_wrapping_quotes(monkeypatch, tmp_path):
    """Values wrapped in matching double/single quotes are unquoted —
    otherwise httpx and other downstream consumers reject the value
    because it doesn't start with a recognized scheme/character.

    Regression: ``UPSTASH_REDIS_REST_URL="https://..."`` in .env was
    being stored as the literal string ``"https://..."`` (quotes
    included), so httpx raised UnsupportedProtocol on the first call.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        'UPSTASH_REDIS_REST_URL="https://fake.upstash.io"\n'
        "UPSTASH_REDIS_REST_TOKEN='single-quoted-token'\n"
        "BARE_VALUE=no-quotes\n"
        "MISMATCHED=\"x'\n"
        'EMPTY_QUOTED=""\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(kv_store, "_PROJECT_ROOT", tmp_path)
    for name in (
        "UPSTASH_REDIS_REST_URL",
        "UPSTASH_REDIS_REST_TOKEN",
        "BARE_VALUE",
        "MISMATCHED",
        "EMPTY_QUOTED",
    ):
        monkeypatch.delenv(name, raising=False)

    kv_store._load_dotenv_if_present()

    assert os.environ["UPSTASH_REDIS_REST_URL"] == "https://fake.upstash.io"
    assert os.environ["UPSTASH_REDIS_REST_TOKEN"] == "single-quoted-token"
    assert os.environ["BARE_VALUE"] == "no-quotes"
    # Mismatched quotes: don't strip — the user's value is genuinely weird.
    assert os.environ["MISMATCHED"] == "\"x'"
    # Empty quoted value collapses to empty string.
    assert os.environ["EMPTY_QUOTED"] == ""


def test_load_dotenv_real_env_var_wins_over_file(monkeypatch, tmp_path):
    """``setdefault`` semantics: a value already in the real environment
    is not overwritten by .env (matches the existing contract — real
    env vars always win)."""
    env_file = tmp_path / ".env"
    env_file.write_text('UPSTASH_REDIS_REST_URL="https://from-dotenv"\n', encoding="utf-8")
    monkeypatch.setattr(kv_store, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://from-real-env")

    kv_store._load_dotenv_if_present()

    assert os.environ["UPSTASH_REDIS_REST_URL"] == "https://from-real-env"


# --- SqliteKVStore behavioral tests ---


def test_sqlite_get_set_roundtrip(tmp_kv: SqliteKVStore):
    assert tmp_kv.get("missing") is None
    tmp_kv.set("k", "v")
    assert tmp_kv.get("k") == "v"
    tmp_kv.set("k", "v2")
    assert tmp_kv.get("k") == "v2"


def test_sqlite_ttl_expires(tmp_kv: SqliteKVStore):
    tmp_kv.set("k", "v", ex=1)
    assert tmp_kv.get("k") == "v"

    future = time.time() + 10
    original_time = time.time
    try:
        time.time = lambda: future
        assert tmp_kv.get("k") is None
    finally:
        time.time = original_time


def test_upstash_set_if_absent_reports_skip_as_not_acquired():
    """upstash-redis maps SET's raw response through ``res == "OK"`` (see
    upstash_redis.format.format_set), so a skipped NX write returns ``False``
    -- NOT ``None`` like redis-py/fakeredis. set_if_absent must report that as
    'not acquired'; an ``is not None`` check would treat False as success and
    make the durable refresh lock always 'win' on Render (the one environment
    with the cross-instance race). This stub mirrors the real contract.
    """
    from fantasy_baseball.data.kv_store import UpstashKVStore

    class _StubUpstash:
        def __init__(self):
            self._held = False

        def set(self, key, value, ex=None, nx=False):
            # Mirror format_set: True on a write, False on an NX-skip.
            if nx and self._held:
                return False
            self._held = True
            return True

    kv = UpstashKVStore(_StubUpstash())
    assert kv.set_if_absent("lock", "a") is True
    assert kv.set_if_absent("lock", "b") is False  # skipped -> not acquired


def test_upstash_compare_delete_is_atomic_cas():
    """compare_delete must issue a single Lua CAS-delete (delete only if the
    value still matches), not a get-then-delete, so it can't remove a
    successor's lock. Verify the script is keyed/argued correctly and the
    eval result is coerced to bool.
    """
    from fantasy_baseball.data.kv_store import UpstashKVStore

    class _StubUpstash:
        def __init__(self):
            self.calls = []

        def eval(self, script, keys=None, args=None):
            self.calls.append((script, keys, args))
            # Simulate: deleted 1 row when the token matches "good".
            return 1 if args == ["good"] else 0

    stub = _StubUpstash()
    kv = UpstashKVStore(stub)
    assert kv.compare_delete("refresh:lock", "good") is True
    assert kv.compare_delete("refresh:lock", "stale") is False
    # One eval per call (atomic), keyed on the lock key, arged on the token.
    assert stub.calls[0][1] == ["refresh:lock"]
    assert stub.calls[0][2] == ["good"]
    assert "redis.call('del'" in stub.calls[0][0]


def test_sqlite_set_if_absent_only_writes_when_absent(tmp_kv: SqliteKVStore):
    # set_if_absent is the SETNX primitive the durable refresh lock rests on.
    assert tmp_kv.set_if_absent("lock", "owner-a") is True
    # Second writer is rejected; the first owner's value is untouched.
    assert tmp_kv.set_if_absent("lock", "owner-b") is False
    assert tmp_kv.get("lock") == "owner-a"


def test_sqlite_set_returns_none_and_overwrites(tmp_kv: SqliteKVStore):
    assert tmp_kv.set("k", "v") is None
    assert tmp_kv.set("k", "v2") is None
    assert tmp_kv.get("k") == "v2"


def test_sqlite_compare_delete_only_deletes_on_match(tmp_kv: SqliteKVStore):
    tmp_kv.set("lock", "owner-a")
    # Wrong token: must NOT delete (this is the successor-lock-protection case).
    assert tmp_kv.compare_delete("lock", "owner-b") is False
    assert tmp_kv.get("lock") == "owner-a"
    # Right token: deletes.
    assert tmp_kv.compare_delete("lock", "owner-a") is True
    assert tmp_kv.get("lock") is None


def test_sqlite_set_if_absent_succeeds_again_after_expiry(tmp_kv: SqliteKVStore):
    # An expired lock must be re-acquirable, so a crashed holder self-heals.
    assert tmp_kv.set_if_absent("lock", "owner-a", ex=1) is True
    assert tmp_kv.set_if_absent("lock", "owner-b") is False

    future = time.time() + 10
    original_time = time.time
    try:
        time.time = lambda: future
        # Prior owner's lock has expired -> a new owner may claim it.
        assert tmp_kv.set_if_absent("lock", "owner-b") is True
    finally:
        time.time = original_time
    assert tmp_kv.get("lock") == "owner-b"


def test_sqlite_delete(tmp_kv: SqliteKVStore):
    tmp_kv.set("k", "v")
    assert tmp_kv.delete("k") == 1
    assert tmp_kv.get("k") is None
    assert tmp_kv.delete("missing") == 0


def test_sqlite_keys_glob(tmp_kv: SqliteKVStore):
    tmp_kv.set("cache:foo", "1")
    tmp_kv.set("cache:bar", "2")
    tmp_kv.set("other", "3")
    assert sorted(tmp_kv.keys("cache:*")) == ["cache:bar", "cache:foo"]
    assert sorted(tmp_kv.keys("*")) == ["cache:bar", "cache:foo", "other"]


def test_sqlite_keys_treats_underscore_literally(tmp_kv: SqliteKVStore):
    # `_` is a SQL LIKE single-char wildcard. The key namespace is full of
    # underscores (job_log:*, game_logs:*, full_season_*), so an unescaped
    # `_` in keys() would over-match. job_log:* must not match jobXlog:*.
    tmp_kv.set("job_log:refresh:1", "1")
    tmp_kv.set("jobXlog:refresh:1", "2")
    assert tmp_kv.keys("job_log:*") == ["job_log:refresh:1"]


def test_sqlite_keys_treats_percent_literally(tmp_kv: SqliteKVStore):
    # `%` is a SQL LIKE multi-char wildcard; a literal `%` in the pattern
    # must match only itself, not any run of characters.
    tmp_kv.set("a%b", "1")
    tmp_kv.set("axyzb", "2")
    assert tmp_kv.keys("a%b") == ["a%b"]


def test_sqlite_keys_skips_expired(tmp_kv: SqliteKVStore):
    tmp_kv.set("live", "1")
    tmp_kv.set("dead", "2", ex=1)
    future = time.time() + 10
    original_time = time.time
    try:
        time.time = lambda: future
        assert tmp_kv.keys("*") == ["live"]
    finally:
        time.time = original_time


def test_sqlite_mget(tmp_kv: SqliteKVStore):
    tmp_kv.set("a", "1")
    tmp_kv.set("b", "2")
    assert tmp_kv.mget("a", "missing", "b") == ["1", None, "2"]
    assert tmp_kv.mget() == []


def test_sqlite_hset_hget_roundtrip(tmp_kv: SqliteKVStore):
    assert tmp_kv.hget("h", "field") is None
    tmp_kv.hset("h", "field", "v")
    assert tmp_kv.hget("h", "field") == "v"
    tmp_kv.hset("h", "field", "v2")
    assert tmp_kv.hget("h", "field") == "v2"


def test_sqlite_hkeys_hgetall(tmp_kv: SqliteKVStore):
    tmp_kv.hset("h", "a", "1")
    tmp_kv.hset("h", "b", "2")
    tmp_kv.hset("other", "x", "y")
    assert sorted(tmp_kv.hkeys("h")) == ["a", "b"]
    assert tmp_kv.hgetall("h") == {"a": "1", "b": "2"}
    assert tmp_kv.hkeys("empty") == []
    assert tmp_kv.hgetall("empty") == {}


def test_sqlite_isolation(tmp_path):
    """Two SqliteKVStore instances on different paths do not share data."""
    a = SqliteKVStore(tmp_path / "a.db")
    b = SqliteKVStore(tmp_path / "b.db")
    a.set("k", "va")
    b.set("k", "vb")
    assert a.get("k") == "va"
    assert b.get("k") == "vb"


def test_sqlite_persists_across_instances(tmp_path):
    """Opening the same path twice sees the same data."""
    path = tmp_path / "shared.db"
    SqliteKVStore(path).set("k", "v")
    assert SqliteKVStore(path).get("k") == "v"


def test_kvstore_protocol_accepts_both_backends(tmp_kv: SqliteKVStore):
    """Duck-typed Protocol check — both concrete classes satisfy it."""
    assert isinstance(tmp_kv, KVStore)
