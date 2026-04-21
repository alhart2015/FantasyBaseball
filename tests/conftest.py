from pathlib import Path

import fakeredis
import pytest

# Ignore the old test_integration.py file to avoid naming conflict
# with the test_integration/ package directory. Tests were migrated
# to test_integration/test_sgp_pipeline.py.
collect_ignore = ["test_integration.py"]


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def fake_redis():
    """Per-test isolated in-memory Redis.

    Yields a FakeRedis client. Each test gets a fresh instance so state
    does not leak across tests.
    """
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        client.flushall()
