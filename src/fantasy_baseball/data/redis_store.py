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

from fantasy_baseball.models.standings import ProjectedStandings, Standings

logger = logging.getLogger(__name__)


class SeasonProgress(TypedDict):
    games_elapsed: int
    total: int
    as_of: str | None


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


REFRESH_LOCK_KEY = "refresh:lock"


def acquire_refresh_lock(client, token: str, ttl_seconds: int) -> bool:
    """Try to claim the cross-instance refresh lock (``SET ... NX EX``).

    Returns True if this caller acquired it, False if another holder has it
    (or ``client is None``). The token identifies this holder so
    :func:`release_refresh_lock` only frees a lock it still owns. The TTL is
    a self-heal: a holder that crashes without releasing leaves the lock to
    expire rather than wedging every future job.
    """
    if client is None:
        return False
    return bool(client.set_if_absent(REFRESH_LOCK_KEY, token, ex=ttl_seconds))


def release_refresh_lock(client, token: str) -> None:
    """Release the refresh lock, but only if ``token`` still matches.

    Uses an atomic compare-and-delete so a slow holder whose lock already
    expired (and was re-acquired by another instance) cannot delete the new
    holder's lock -- a plain get-then-delete has a TOCTOU window between the
    check and the delete. No-op when ``client is None``.
    """
    if client is None:
        return
    client.compare_delete(REFRESH_LOCK_KEY, token)


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


PROJECTED_STANDINGS_HISTORY_KEY = "projected_standings_history"


def write_projected_standings_snapshot(client, projected: ProjectedStandings) -> None:
    """Write a ProjectedStandings snapshot keyed by its effective_date.

    Idempotent overwrite — same-day refreshes replace the previous
    snapshot (last-write-wins). No-op when ``client`` is None.
    """
    if client is None:
        return
    client.hset(
        PROJECTED_STANDINGS_HISTORY_KEY,
        projected.effective_date.isoformat(),
        json.dumps(projected.to_json()),
    )


def get_projected_standings_day(client, snapshot_date: str) -> ProjectedStandings | None:
    """Return the ProjectedStandings for one snapshot date, or None if missing/corrupt."""
    if client is None:
        return None
    raw = client.hget(PROJECTED_STANDINGS_HISTORY_KEY, snapshot_date)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return ProjectedStandings.from_json(data)


def get_projected_standings_history(client) -> dict[str, ProjectedStandings]:
    """Return the entire history as {snapshot_date: ProjectedStandings}.

    Corrupt JSON entries are silently skipped.
    """
    if client is None:
        return {}
    raw_map = client.hgetall(PROJECTED_STANDINGS_HISTORY_KEY)
    if not raw_map:
        return {}
    out: dict[str, ProjectedStandings] = {}
    for d, raw in raw_map.items():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out[d] = ProjectedStandings.from_json(data)
    return out


# --- ROS projection history (weekly snapshots for PT-residual calibration) ---

ROS_PROJECTION_HISTORY_KEY = "ros_projection_history"

# Columns retained per player: the playing-time signal (PA / IP) plus the
# counting stats needed to score projected-vs-realized residuals. FanGraphs'
# rates, percentile bands, and ~9k sub-replacement prospect lines are dropped
# so a weekly snapshot stays ~160 KB (free-tier Upstash budget) instead of the
# full ~2.9 MB ROS blob.
_ROS_SNAPSHOT_HITTER_COLS = ("name", "mlbam_id", "pa", "r", "hr", "rbi", "sb", "h", "ab")
_ROS_SNAPSHOT_PITCHER_COLS = ("name", "mlbam_id", "ip", "w", "k", "sv", "er", "bb", "h_allowed")
# Keep anyone rosterable (covers part-time bats and middle relievers); excludes
# the AAA/prospect phantom lines that bloat the blob but never get rostered.
_ROS_SNAPSHOT_MIN_PA = 100.0
_ROS_SNAPSHOT_MIN_IP = 20.0


def _trim_ros_snapshot(ros_blob: dict) -> dict[str, list[dict]]:
    """Reduce a full ROS projection blob to the snapshot's columns and players."""

    def _trim(rows, cols, vol_key, vol_min):
        out = []
        for p in rows or []:
            try:
                vol = float(p.get(vol_key) or 0)
            except (TypeError, ValueError):
                vol = 0.0
            if vol < vol_min:
                continue
            out.append({c: p[c] for c in cols if c in p})
        return out

    return {
        "hitters": _trim(
            ros_blob.get("hitters"), _ROS_SNAPSHOT_HITTER_COLS, "pa", _ROS_SNAPSHOT_MIN_PA
        ),
        "pitchers": _trim(
            ros_blob.get("pitchers"), _ROS_SNAPSHOT_PITCHER_COLS, "ip", _ROS_SNAPSHOT_MIN_IP
        ),
    }


def write_ros_projection_snapshot(client, ros_blob: dict | None, snapshot_date: str) -> None:
    """Archive a trimmed ROS projection snapshot keyed by ``snapshot_date``.

    ``snapshot_date`` is the ROS projection vintage (its ``_ros_snapshot_date``),
    so exactly one entry exists per distinct fetch even if the refresh runs
    several times against it. Idempotent overwrite. No-op when ``client`` is
    None, the blob is empty, or the date is missing/blank.
    """
    if client is None or not ros_blob or not snapshot_date:
        return
    trimmed = _trim_ros_snapshot(ros_blob)
    client.hset(
        ROS_PROJECTION_HISTORY_KEY,
        snapshot_date,
        json.dumps(trimmed, separators=(",", ":")),
    )


def get_ros_projection_history(client) -> dict[str, dict]:
    """Return the entire ROS-projection history as ``{snapshot_date: trimmed_blob}``.

    Corrupt JSON entries are silently skipped. ``{}`` when client is None/empty.
    """
    if client is None:
        return {}
    raw_map = client.hgetall(ROS_PROJECTION_HISTORY_KEY)
    if not raw_map:
        return {}
    out: dict[str, dict] = {}
    for d, raw in raw_map.items():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out[d] = data
    return out


# --- Per-player raw game logs (incremental, box-score driven) ---

_GAME_LOG_GROUPS = ("hitting", "pitching")


def _player_game_log_key(season: int, mlbam_id: str, group: str) -> str:
    if group not in _GAME_LOG_GROUPS:
        raise ValueError(f"group must be one of {_GAME_LOG_GROUPS}, got {group!r}")
    return f"game_logs:{season}:{mlbam_id}:{group}"


def get_player_game_log(client, season: int, mlbam_id: str, group: str) -> dict | None:
    """Read one player's per-game log for a group. None on missing/corrupt/None client."""
    key = _player_game_log_key(season, mlbam_id, group)
    if client is None:
        return None
    raw = client.get(key)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def set_player_game_log(client, season: int, mlbam_id: str, group: str, payload: dict) -> None:
    """Overwrite one player's per-game log for a group. No-op when client is None."""
    key = _player_game_log_key(season, mlbam_id, group)
    if client is None:
        return
    client.set(key, json.dumps(payload))


# Upstash request-size cap is generous, but a 50-key MGET keeps each call well
# under it while collapsing ~550 hitter reads into ~11 round-trips.
_GAME_LOG_MGET_CHUNK = 50


def build_hitter_ytd_game_logs(client, season: int) -> dict[str, dict]:
    """Assemble the hitter per-game payload that ``compute_team_ytd_ab`` consumes.

    Returns ``{mlbam_id: {"name", "type": "hitter", "games": [...]}}`` sourced
    from the per-player ``game_logs:{season}:{id}:hitting`` records, enumerated
    via the ``game_log_totals:hitters`` rollup (the canonical list of every
    hitter we have logs for). Reads are batched with chunked MGET.

    This is the production bridge between Upstash's incrementally-synced game
    logs and the team-YTD AB attribution: the projection layer can pass the
    result as ``compute_team_ytd_ab(..., game_logs=<this>)`` instead of reading
    ``data/roster_game_logs.json`` -- a file nothing in the deployed pipeline
    builds, so AB (and therefore team-YTD AVG) was silently zero in production.

    Rollup ids whose per-player log is missing or corrupt are skipped (no
    fabricated empty entries). Returns ``{}`` when *client* is ``None`` or the
    rollup is empty. ``games`` rows are passed through verbatim;
    ``_load_per_game_hitter_ab`` reads only ``date`` and ``ab``.
    """
    if client is None:
        return {}
    rollup = get_game_log_totals(client, "hitters")
    if not rollup:
        return {}
    ids = list(rollup)
    out: dict[str, dict] = {}
    for start in range(0, len(ids), _GAME_LOG_MGET_CHUNK):
        chunk = ids[start : start + _GAME_LOG_MGET_CHUNK]
        keys = [_player_game_log_key(season, mid, "hitting") for mid in chunk]
        for mid, raw in zip(chunk, client.mget(*keys), strict=True):
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            out[mid] = {
                "name": payload.get("name") or rollup[mid].get("name") or "",
                "type": "hitter",
                "games": payload.get("games") or [],
            }
    return out


def _game_logs_watermark_key(season: int) -> str:
    return f"game_logs:{season}:fetched_through_utc"


def get_game_logs_watermark(client, season: int) -> str | None:
    """Read the UTC high-water mark (ISO-8601). None when missing or client is None."""
    if client is None:
        return None
    raw = client.get(_game_logs_watermark_key(season))
    return raw if isinstance(raw, str) else None


def set_game_logs_watermark(client, season: int, iso_utc: str) -> None:
    """Persist the UTC high-water mark. No-op when client is None."""
    if client is None:
        return
    client.set(_game_logs_watermark_key(season), iso_utc)


def _player_positions_key(season: int) -> str:
    return f"game_logs:{season}:player_pos"


def get_player_positions(client, season: int) -> dict[str, str]:
    """Read the cached {mlbam_id: primaryPosition_code} map. Empty on missing/corrupt/None."""
    if client is None:
        return {}
    raw = client.get(_player_positions_key(season))
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def set_player_positions(client, season: int, positions: dict[str, str]) -> None:
    """Overwrite the cached primaryPosition map. No-op when client is None."""
    if client is None:
        return
    client.set(_player_positions_key(season), json.dumps(positions))


def _game_log_dates_key(season: int) -> str:
    return f"game_logs:{season}:dates"


def get_game_log_dates(client, season: int) -> list[str]:
    """Read the sorted list of distinct ingested game dates. Empty on missing/corrupt/None."""
    if client is None:
        return []
    raw = client.get(_game_log_dates_key(season))
    if raw is None:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def set_game_log_dates(client, season: int, dates: list[str]) -> None:
    """Overwrite the distinct game-dates list (deduped + sorted). No-op when client is None."""
    if client is None:
        return
    client.set(_game_log_dates_key(season), json.dumps(sorted(set(dates))))
