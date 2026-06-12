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
def _strip_ambient_upstash_creds(monkeypatch):
    """Fail-closed default: no test reaches PROD Upstash via ambient creds.

    The repo ``.env`` holds REAL prod Upstash creds. The first time any
    test builds an Upstash client, ``kv_store._load_dotenv_if_present``
    ``setdefault``s those creds into ``os.environ`` for the rest of the
    process. After that, any later test whose code path reaches
    ``_push_streak_scores_to_remote`` (whose guard reads ``os.environ``
    directly) or otherwise builds an Upstash client would write to PROD
    -- the documented "streak flake" that clobbered remote STREAK_SCORES
    with a fixture payload (team_name="t").

    Stripping the two creds at the start of EVERY test makes those guards
    short-circuit by default. This runs before any explicitly-requested
    fixture of the same scope, so tests that legitimately build an
    Upstash client set their own FAKE creds via ``monkeypatch.setenv``
    and are unaffected (their setenv lands after this delenv).
    """
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)


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
