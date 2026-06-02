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
