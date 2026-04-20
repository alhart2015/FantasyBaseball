"""Tests for redis_store helpers.

Client-construction behavior moved to ``fantasy_baseball.data.kv_store``
as of the Redis env-gating redesign. See ``tests/test_data/test_kv_store.py``
for leak-prevention and backend-selection coverage.
"""
from fantasy_baseball.data import redis_store


def test_module_imports():
    """Sanity check: module can be imported."""
    assert hasattr(redis_store, "__name__")
