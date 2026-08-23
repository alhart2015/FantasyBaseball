import os
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
       creds, and an ambient/exported ``UPSTASH_*`` would let a code path that
       builds an Upstash client write PROD -- the documented "streak flake"
       that clobbered remote STREAK_SCORES (team_name="t"), and the
       META/standings clobber (last_refresh="9:00 AM") from a leaked
       ``RENDER=true``. (The other re-hydration route -- ``_build_upstash_kv``
       -> ``_load_dotenv_if_present`` reloading ``.env`` -- is now closed at
       the source: that call is skipped under pytest, and the builder refuses
       to construct a real client anyway.)
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


#: Env vars that select the KV backing file / disable Yahoo. Production code --
#: not just tests -- assigns these (``scripts/run_manual_refresh.py`` sets both
#: in ``_activate_manual_environment``), so any test that drives such a code
#: path can leave them set for whatever runs next in the same process.
_PROCESS_SCOPED_ENV = ("FANTASY_LOCAL_KV_PATH", "FB_SKIP_YAHOO")


@pytest.fixture(autouse=True)
def _restore_process_scoped_env():
    """Put ``_PROCESS_SCOPED_ENV`` back exactly as the test found it.

    A backstop, deliberately independent of ``monkeypatch``. The natural
    monkeypatch spelling for "snapshot a var the code under test might set"
    is ``monkeypatch.delenv(name, raising=False)``, and it silently does
    nothing when the name is not already set: ``delenv`` registers an undo
    only when it actually removes something. So the case this is meant to
    cover -- var absent, code under test assigns it -- is exactly the case
    with no undo registered, and the value escapes.

    The escape is invisible where it happens and fatal somewhere else: a
    leaked ``FB_SKIP_YAHOO=1`` plus a ``FANTASY_LOCAL_KV_PATH`` pointing at a
    torn-down ``tmp_path`` made ``tests/test_web/test_refresh_pipeline.py``
    fail with "FB_SKIP_YAHOO is set but no cached standings exist" -- in a
    file that never touches either variable. With ``pytest -n auto`` and
    ``pytest-randomly``, which worker and which tests got hit changed run to
    run, so it read as flake rather than as pollution.

    Restoring unconditionally here costs two dict lookups per test and makes
    the whole class of leak impossible, wherever the assignment happens.
    """
    saved = {name: os.environ.get(name) for name in _PROCESS_SCOPED_ENV}
    try:
        yield
    finally:
        for name, before in saved.items():
            if before is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = before


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
