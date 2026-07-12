from pathlib import Path

import fakeredis
import pytest

# Ignore the old test_integration.py file to avoid naming conflict
# with the test_integration/ package directory. Tests were migrated
# to test_integration/test_sgp_pipeline.py.
collect_ignore = ["test_integration.py"]


class _KVFakeRedis(fakeredis.FakeRedis):
    """fakeredis plus the two KVStore methods the app adds on top of the
    plain Redis subset (``set_if_absent`` / ``compare_delete``).

    The real backends (UpstashKVStore via Lua eval, SqliteKVStore via SQL)
    implement these atomically; the test double is single-threaded, so a
    plain get-then-delete is equivalent here. Delegates to fakeredis's native
    SET NX / GET / DELETE so behavior matches the redis-py contract.
    """

    def set_if_absent(self, key, value, *, ex=None):
        return bool(self.set(key, value, nx=True, ex=ex))

    def compare_delete(self, key, expected):
        if self.get(key) == expected:
            return bool(self.delete(key))
        return False


@pytest.fixture(autouse=True)
def _isolate_kv_from_prod(monkeypatch):
    """Fail-closed default: no test can reach PROD Upstash. Two layers,
    because either one alone has a hole:

    1. **Strip ambient Upstash creds.** The repo ``.env`` holds REAL prod
       creds. Alone this is NOT enough: ``_build_upstash_kv`` calls
       ``_load_dotenv_if_present``, which ``setdefault``s those creds right
       back from ``.env``. So a code path that builds an Upstash client
       re-hydrates the creds and can still write PROD -- the documented
       "streak flake" that clobbered remote STREAK_SCORES (team_name="t"),
       and the META/standings clobber (last_refresh="9:00 AM") from a leaked
       ``RENDER=true``.
    2. **Neutralize the RENDER gate.** ``is_remote()``/``get_kv()`` choose the
       backend purely on ``RENDER``. Deleting it forces the local SQLite store
       regardless of creds, and regardless of a module that sets
       ``RENDER=true`` at import time landing in an xdist worker. Resetting the
       cached singleton discards any backend a prior leak already built as
       Upstash so the next ``get_kv()`` rebuilds local.

    Runs before any explicitly-requested fixture of the same scope, so a test
    that legitimately exercises the remote path sets its own FAKE creds /
    ``RENDER`` afterward (its setenv lands after these delenvs) and is
    unaffected.
    """
    from fantasy_baseball.data import kv_store

    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    kv_store._reset_singleton()


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_redis():
    """Per-test isolated in-memory Redis.

    Yields a FakeRedis client. Each test gets a fresh instance so state
    does not leak across tests.
    """
    client = _KVFakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        client.flushall()
