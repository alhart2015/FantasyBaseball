"""Typed read/write helpers for every Redis key used by this app.

This module owns the Redis schema. Every non-``CacheKey`` Redis access
in the codebase goes through here — no inline ``kv.get(...)`` calls
elsewhere.

Helpers take an explicit ``client`` argument (anything satisfying
``KVStore``). Production callers pass ``get_kv()`` from
``fantasy_baseball.data.kv_store``; tests inject a fresh
``SqliteKVStore`` at a ``tmp_path``. On Render ``get_kv()`` resolves
to Upstash; off Render it resolves to ``data/local.db``.
"""

from __future__ import annotations

import json
import logging
import re as _re
from typing import TypedDict

from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.models.standings import Standings

logger = logging.getLogger(__name__)


class SeasonProgress(TypedDict):
    games_elapsed: int
    total: int
    as_of: str | None


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
        raise ValueError(f"player_type must be one of {_BLENDED_PROJ_TYPES}, got {player_type!r}")
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


def set_blended_projections(client, player_type: str, rows: list[dict]) -> None:
    """Overwrite blended preseason projections for hitters or pitchers."""
    key = _blended_key(player_type)  # validates player_type
    if client is None:
        return
    client.set(key, json.dumps(rows))


def _preseason_baseline_key(season_year: int) -> str:
    return f"preseason_baseline:{season_year}"


def get_preseason_baseline(client, season_year: int) -> dict | None:
    """Read the frozen preseason Monte Carlo baseline for ``season_year``.

    Returns ``None`` on missing key, corrupt JSON, non-dict payload, or
    ``client is None``. Shape on success::

        {"base": {...}, "with_management": {...}, "meta": {...}}

    where ``base`` / ``with_management`` are ``run_monte_carlo`` outputs
    captured once per season against Opening-Day rosters + preseason
    projections.
    """
    if client is None:
        return None
    raw = client.get(_preseason_baseline_key(season_year))
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Corrupt JSON at Redis key %r; ignoring",
            _preseason_baseline_key(season_year),
        )
        return None
    if not isinstance(data, dict):
        return None
    return data


def set_preseason_baseline(client, season_year: int, payload: dict) -> None:
    """Overwrite the frozen preseason baseline for ``season_year``.

    The caller is responsible for the payload shape; this helper just
    serializes and stores. No-op when ``client is None`` (e.g. in
    unconfigured environments).
    """
    if client is None:
        return
    client.set(_preseason_baseline_key(season_year), json.dumps(payload))


ROS_PROJECTIONS_KEY = redis_key(CacheKey.ROS_PROJECTIONS)


def get_ros_projections(client) -> dict | None:
    """Read the latest rest-of-season projections snapshot from Redis.

    Returns the parsed ``{"hitters": [...], "pitchers": [...]}`` payload
    or ``None`` on missing key, corrupt JSON, or ``client is None``.

    Reads Redis directly (no disk fallback via ``read_cache``) so tests
    injecting a fake Redis client aren't cross-contaminated by stale
    project-local ``data/cache/*.json`` files.
    """
    if client is None:
        return None
    raw = client.get(ROS_PROJECTIONS_KEY)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Corrupt JSON at Redis key %r; ignoring", ROS_PROJECTIONS_KEY)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _game_log_totals_key(player_type: str) -> str:
    if player_type not in _BLENDED_PROJ_TYPES:
        raise ValueError(f"player_type must be one of {_BLENDED_PROJ_TYPES}, got {player_type!r}")
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


def set_game_log_totals(client, player_type: str, totals: dict[str, dict]) -> None:
    """Overwrite aggregated game log totals for hitters or pitchers."""
    key = _game_log_totals_key(player_type)
    if client is None:
        return
    client.set(key, json.dumps(totals))


SEASON_PROGRESS_KEY = "season_progress"


def _default_season_progress() -> SeasonProgress:
    return {"games_elapsed": 0, "total": 162, "as_of": None}


def get_season_progress(client) -> SeasonProgress:
    """Read season progress ({games_elapsed, total, as_of}). Returns defaults on missing or corrupt."""
    if client is None:
        return _default_season_progress()
    raw = client.get(SEASON_PROGRESS_KEY)
    if raw is None:
        return _default_season_progress()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _default_season_progress()
    if not isinstance(data, dict):
        return _default_season_progress()
    as_of = data.get("as_of")
    return {
        "games_elapsed": int(data.get("games_elapsed", 0)),
        "total": int(data.get("total", 162)),
        "as_of": as_of if isinstance(as_of, str) else None,
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
    client.hset(WEEKLY_ROSTERS_HISTORY_KEY, snapshot_date, json.dumps(day_rows))


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


# Matches Yahoo's " (Batter)" / " (Pitcher)" trailing annotations that
# appear on some dual-eligibility names (e.g. Shohei Ohtani). Mirrors
# the constant in ``data.db`` so get_latest_roster_names normalizes
# identically to the (about-to-be-removed) SQLite get_roster_names.
# TODO(task-11): the duplicated regex in ``data/db.py`` goes away when
# task 11 deletes db.py; this becomes the sole definition.
_PLAYER_SUFFIX_RE = _re.compile(r"\s*\((?:Batter|Pitcher)\)\s*$", _re.IGNORECASE)


def get_latest_roster_names(client) -> set[str] | None:
    """Normalized names of all rostered players from the latest snapshot.

    Strips Yahoo's " (Batter)" / " (Pitcher)" suffixes and normalizes
    (accent-stripped, lowercased). Returns ``None`` when no roster
    snapshots exist, or when *client* is ``None`` (unconfigured Redis).
    """
    from fantasy_baseball.utils.name_utils import normalize_name

    if client is None:
        return None
    entries = get_latest_weekly_rosters(client)
    if not entries:
        return None
    return {normalize_name(_PLAYER_SUFFIX_RE.sub("", e["player_name"])) for e in entries}


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


STANDINGS_HISTORY_KEY = "standings_history"


def write_standings_snapshot(client, standings: Standings) -> None:
    """Write a Standings snapshot keyed by its effective_date. Idempotent overwrite.

    Canonical shape on disk: ``standings.to_json()`` — see spec. No-op
    when ``client`` is None.
    """
    if client is None:
        return
    client.hset(
        STANDINGS_HISTORY_KEY,
        standings.effective_date.isoformat(),
        json.dumps(standings.to_json()),
    )


def get_standings_day(client, snapshot_date: str) -> Standings | None:
    """Return the Standings for one snapshot date, or None if missing/corrupt.

    Raises ValueError if the stored payload is legacy-shape (see
    ``Standings.from_json``); run scripts/migrate_standings_history.py
    to rewrite.
    """
    if client is None:
        return None
    raw = client.hget(STANDINGS_HISTORY_KEY, snapshot_date)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return Standings.from_json(data)


def get_latest_standings(client) -> Standings | None:
    """Return the Standings for the maximum snapshot_date in the hash."""
    if client is None:
        return None
    dates = client.hkeys(STANDINGS_HISTORY_KEY)
    if not dates:
        return None
    return get_standings_day(client, max(dates))


def get_standings_history(client) -> dict[str, Standings]:
    """Return the entire history as {snapshot_date: Standings}.

    Corrupt JSON entries are silently skipped (matches previous behavior).
    Legacy-shape entries raise ValueError — by design; migration script
    rewrites them.
    """
    if client is None:
        return {}
    raw_map = client.hgetall(STANDINGS_HISTORY_KEY)
    if not raw_map:
        return {}
    out: dict[str, Standings] = {}
    for d, raw in raw_map.items():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out[d] = Standings.from_json(data)
    return out
