"""Tests for redis_store helpers."""
from fantasy_baseball.data import redis_store


def test_module_imports():
    """Sanity check: module can be imported."""
    assert hasattr(redis_store, "__name__")


def test_get_default_client_returns_none_when_env_unset(monkeypatch):
    """When Upstash env vars are unset, the factory returns None gracefully."""
    import fantasy_baseball.data.redis_store as rs
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    # Reset the module-level cache so the test is deterministic.
    rs._default_client = None
    rs._default_client_initialized = False
    assert rs.get_default_client() is None
