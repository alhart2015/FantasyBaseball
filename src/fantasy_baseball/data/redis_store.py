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


_BLENDED_PROJ_TYPES = ("hitters", "pitchers")


def _blended_key(player_type: str) -> str:
    if player_type not in _BLENDED_PROJ_TYPES:
        raise ValueError(
            f"player_type must be one of {_BLENDED_PROJ_TYPES}, got {player_type!r}"
        )
    return f"blended_projections:{player_type}"


def get_blended_projections(client, player_type: str) -> list[dict]:
    """Read blended preseason projections for hitters or pitchers."""
    if client is None:
        return []
    raw = client.get(_blended_key(player_type))
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def set_blended_projections(
    client, player_type: str, rows: list[dict]
) -> None:
    """Overwrite blended preseason projections for hitters or pitchers."""
    key = _blended_key(player_type)  # validates player_type
    if client is None:
        return
    client.set(key, json.dumps(rows))


def _game_log_totals_key(player_type: str) -> str:
    if player_type not in _BLENDED_PROJ_TYPES:
        raise ValueError(
            f"player_type must be one of {_BLENDED_PROJ_TYPES}, got {player_type!r}"
        )
    return f"game_log_totals:{player_type}"


def get_game_log_totals(client, player_type: str) -> dict[str, dict]:
    """Read aggregated game log totals. Returns {} on missing key, corrupt JSON, or None client."""
    key = _game_log_totals_key(player_type)
    if client is None:
        return {}
    raw = client.get(key)
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def set_game_log_totals(
    client, player_type: str, totals: dict[str, dict]
) -> None:
    """Overwrite aggregated game log totals for hitters or pitchers."""
    key = _game_log_totals_key(player_type)
    if client is None:
        return
    client.set(key, json.dumps(totals))


SEASON_PROGRESS_KEY = "season_progress"
_DEFAULT_SEASON_PROGRESS = {"games_elapsed": 0, "total": 162, "as_of": None}


def get_season_progress(client) -> dict:
    """Read season progress ({games_elapsed, total, as_of}). Returns defaults on missing or corrupt."""
    if client is None:
        return dict(_DEFAULT_SEASON_PROGRESS)
    raw = client.get(SEASON_PROGRESS_KEY)
    if raw is None:
        return dict(_DEFAULT_SEASON_PROGRESS)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return dict(_DEFAULT_SEASON_PROGRESS)
    if not isinstance(data, dict):
        return dict(_DEFAULT_SEASON_PROGRESS)
    return {
        "games_elapsed": int(data.get("games_elapsed", 0)),
        "total": int(data.get("total", 162)),
        "as_of": data.get("as_of"),
    }


def set_season_progress(
    client, games_elapsed: int, total: int = 162, as_of: str | None = None
) -> None:
    """Overwrite season progress."""
    if client is None:
        return
    client.set(
        SEASON_PROGRESS_KEY,
        json.dumps({"games_elapsed": games_elapsed, "total": total, "as_of": as_of}),
    )


WEEKLY_ROSTERS_HISTORY_KEY = "weekly_rosters_history"


def write_roster_snapshot(
    client,
    snapshot_date: str,
    team: str,
    entries: list[dict],
) -> None:
    """Idempotently replace one team's rows within one snapshot date.

    Reads the existing day's blob, drops any rows whose ``team`` field
    matches the argument, appends the new entries tagged with the team
    name, and writes the merged blob back. No-op when the client is None.
    """
    if client is None:
        return
    raw = client.hget(WEEKLY_ROSTERS_HISTORY_KEY, snapshot_date)
    if raw is None:
        day_rows: list[dict] = []
    else:
        try:
            day_rows = json.loads(raw)
            if not isinstance(day_rows, list):
                day_rows = []
        except json.JSONDecodeError:
            day_rows = []
    day_rows = [row for row in day_rows if row.get("team") != team]
    day_rows.extend({**entry, "team": team} for entry in entries)
    client.hset(
        WEEKLY_ROSTERS_HISTORY_KEY, snapshot_date, json.dumps(day_rows)
    )


def get_weekly_roster_day(client, snapshot_date: str) -> list[dict]:
    """Return the entries for one snapshot date. Empty list on None/missing/corrupt."""
    if client is None:
        return []
    raw = client.hget(WEEKLY_ROSTERS_HISTORY_KEY, snapshot_date)
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def get_latest_weekly_rosters(client) -> list[dict]:
    """Return the entries for the maximum snapshot_date in the hash."""
    if client is None:
        return []
    dates = client.hkeys(WEEKLY_ROSTERS_HISTORY_KEY)
    if not dates:
        return []
    latest = max(dates)
    return get_weekly_roster_day(client, latest)


def get_weekly_roster_history(client) -> dict[str, list[dict]]:
    """Return the entire history as {snapshot_date: [entry, ...]}."""
    if client is None:
        return {}
    raw_map = client.hgetall(WEEKLY_ROSTERS_HISTORY_KEY)
    if not raw_map:
        return {}
    out: dict[str, list[dict]] = {}
    for date, raw in raw_map.items():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            out[date] = data
    return out
