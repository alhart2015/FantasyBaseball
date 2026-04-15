"""Typed read/write helpers for all Redis keys used by this app.

This module owns the full Redis schema. Every non-cache Redis access in
the codebase should go through here — no inline `redis.get(...)` calls
elsewhere.

Helpers take an explicit client argument so tests can inject a
fakeredis client. Production code uses `get_default_client()` which
lazily reads UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN from the
environment.
"""
from __future__ import annotations

import json
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upstash_redis import Redis

_default_client = None
_default_client_initialized = False
_default_client_lock = threading.Lock()


def get_default_client() -> "Redis | None":
    """Lazy Upstash client for production use. Returns None if unconfigured.

    Thread-safe: uses double-checked locking so concurrent first-access
    from multiple Flask worker threads does not construct the client
    twice. Mirrors the pattern in web/season_data.py::_get_redis().
    """
    global _default_client, _default_client_initialized
    if _default_client_initialized:
        return _default_client
    with _default_client_lock:
        if _default_client_initialized:
            return _default_client
        url = os.environ.get("UPSTASH_REDIS_REST_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            from upstash_redis import Redis
            _default_client = Redis(url=url, token=token)
        _default_client_initialized = True
    return _default_client


POSITIONS_KEY = "positions"


def get_positions(client) -> dict[str, list[str]]:
    """Read the positions map. Returns empty dict when the client is None, the key is missing, or the value is corrupt."""
    if client is None:
        return {}
    raw = client.get(POSITIONS_KEY)
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def set_positions(client, positions: dict[str, list[str]]) -> None:
    """Overwrite the positions map. No-op when the client is None."""
    if client is None:
        return
    client.set(POSITIONS_KEY, json.dumps(positions))
