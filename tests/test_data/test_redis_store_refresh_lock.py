"""Tests for the cross-instance refresh lock helpers.

These back the durable lock the daily refresh and ROS-fetch jobs use to
stay mutually exclusive across Render instances / QStash at-least-once
redelivery, where the in-process slot does not reach.
"""

from __future__ import annotations

from fantasy_baseball.data import redis_store


def test_acquire_refresh_lock_is_exclusive(fake_redis):
    assert redis_store.acquire_refresh_lock(fake_redis, "token-a", 60) is True
    # A second instance cannot acquire while the lock is held.
    assert redis_store.acquire_refresh_lock(fake_redis, "token-b", 60) is False


def test_release_refresh_lock_allows_reacquire(fake_redis):
    assert redis_store.acquire_refresh_lock(fake_redis, "token-a", 60) is True
    redis_store.release_refresh_lock(fake_redis, "token-a")
    assert redis_store.acquire_refresh_lock(fake_redis, "token-b", 60) is True


def test_release_refresh_lock_ignores_foreign_token(fake_redis):
    # A holder whose lock already expired (and was re-acquired by someone
    # else) must not delete the new holder's lock when it finally releases.
    assert redis_store.acquire_refresh_lock(fake_redis, "token-a", 60) is True
    redis_store.release_refresh_lock(fake_redis, "stale-token")
    assert redis_store.acquire_refresh_lock(fake_redis, "token-b", 60) is False


def test_acquire_refresh_lock_none_client_returns_false():
    assert redis_store.acquire_refresh_lock(None, "token", 60) is False


def test_release_refresh_lock_none_client_is_noop():
    # Must not raise.
    redis_store.release_refresh_lock(None, "token")
