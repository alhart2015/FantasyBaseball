"""Incremental, box-score-driven MLB game-log sync.

Persists raw per-player per-game rows (keyed by player id and group) and
derives the rolled-up ``game_log_totals:*`` blobs that power existing
calcs. Each refresh pulls only games new or changed since a precise UTC
high-water mark; the one-time backfill enumerates the season via schedule.

Two-way players (Ohtani) are handled by parsing box scores -- both their
batting and pitching blocks are recorded -- and a position-player pitching
filter keyed on primaryPosition keeps Ohtani while dropping mop-up innings.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

import requests

from fantasy_baseball.data.mlb_boxscore import (
    boxscore_hitter_row,
    boxscore_pitcher_row,
    iter_boxscore_players,
    should_record_pitching,
)
from fantasy_baseball.data.redis_store import (
    get_game_log_dates,
    get_game_log_totals,
    get_game_logs_watermark,
    get_player_game_log,
    get_player_positions,
    get_season_progress,
    set_game_log_dates,
    set_game_log_totals,
    set_game_logs_watermark,
    set_player_game_log,
    set_player_positions,
    set_season_progress,
)
from fantasy_baseball.utils.time_utils import local_today

_MLB_API = "https://statsapi.mlb.com/api/v1"

_HITTER_KEYS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")
_PITCHER_KEYS = ("ip", "k", "er", "bb", "h_allowed", "w", "sv")
_PITCHER_INT_KEYS = tuple(k for k in _PITCHER_KEYS if k != "ip")


def _merge_player_games(
    existing: dict[str, Any] | None, name: str, new_rows: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Merge new per-game rows (keyed by gamePk) into a player's stored log.

    New rows overwrite stored rows with the same gamePk (corrections
    self-heal). Games are returned sorted by (date, gamePk).
    """
    by_pk: dict[int, dict[str, Any]] = {}
    if existing:
        for g in existing.get("games", []):
            by_pk[g["gamePk"]] = g
    by_pk.update(new_rows)
    games = sorted(by_pk.values(), key=lambda r: (r["date"], r["gamePk"]))
    resolved_name = name or (existing or {}).get("name", "")
    return {"name": resolved_name, "games": games}


def _sum_hitting(games: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {k: 0 for k in _HITTER_KEYS}
    for g in games:
        for k in _HITTER_KEYS:
            out[k] += g.get(k, 0) or 0
    return out


def _sum_pitching(games: list[dict[str, Any]]) -> dict[str, float | int]:
    out: dict[str, float | int] = {k: 0 for k in _PITCHER_KEYS}
    for g in games:
        out["ip"] += g.get("ip", 0.0) or 0.0
        for k in _PITCHER_INT_KEYS:
            out[k] += g.get(k, 0) or 0
    out["ip"] = round(out["ip"], 4)
    return out


def _is_regular_final(game: dict[str, Any]) -> bool:
    """True for a completed regular-season game."""
    return (
        game.get("gameType") == "R" and game.get("status", {}).get("abstractGameState") == "Final"
    )


def _game_context(game: dict[str, Any]) -> tuple[int, int, str]:
    """(gamePk, gameNumber, officialDate) from a schedule/changes game dict."""
    game_pk = game["gamePk"]
    game_number = game.get("gameNumber", 1)
    date = game.get("officialDate") or (game.get("gameDate") or "")[:10]
    return game_pk, game_number, date


def _fetch_changed_games(season: int, since_iso: str) -> list[dict[str, Any]]:
    """MLB games (all types) changed since ``since_iso``, scoped to MLB + season."""
    params: dict[str, str | int] = {
        "updatedSince": since_iso,
        "sportId": 1,
        "season": season,
    }
    resp = requests.get(
        f"{_MLB_API}/game/changes",
        params=params,
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    return [g for d in data.get("dates", []) for g in d.get("games", [])]


def _fetch_season_games(season: int) -> list[dict[str, Any]]:
    """All regular-season MLB games for ``season`` (backfill enumeration)."""
    s_params: dict[str, str | int] = {"sportId": 1, "season": season, "gameType": "R"}
    resp = requests.get(
        f"{_MLB_API}/schedule",
        params=s_params,
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    return [g for d in data.get("dates", []) for g in d.get("games", [])]


def _fetch_boxscore(game_pk: int) -> dict[str, Any]:
    """Fetch one game's box score from the MLB Stats API."""
    resp = requests.get(f"{_MLB_API}/game/{game_pk}/boxscore", timeout=20)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def _fetch_positions(mlbam_ids: list[int]) -> dict[str, str | None]:
    """Batch primaryPosition.code lookup. {str(id): code}; code may be None."""
    if not mlbam_ids:
        return {}
    resp = requests.get(
        f"{_MLB_API}/people",
        params={"personIds": ",".join(str(i) for i in mlbam_ids)},
        timeout=20,
    )
    resp.raise_for_status()
    out: dict[str, str | None] = {}
    for person in resp.json().get("people", []):
        out[str(person["id"])] = person.get("primaryPosition", {}).get("code")
    return out


def _collect_player_rows(
    games: list[dict[str, Any]], progress_cb
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str], set[int]]:
    """Fetch box scores in parallel and parse each into compact per-game rows as it
    completes, never retaining the raw box-score JSON.

    Returns ``(hitting, pitching, dates, failed_gamePks)`` where ``hitting`` and
    ``pitching`` map ``mlbam_id -> {"name", "rows": {gamePk: row}}``.

    Memory: a full box score is ~1-2 MB as a Python dict, and a ``Future`` retains its
    result until the Future itself is released. The submissions are therefore NOT bound
    to a surviving ``futures`` list -- doing so would pin every box score (~1 GB for a
    full-season backfill) and OOM the 512 MB instance, which is exactly the bug this
    function exists to fix. Passing the comprehension straight into ``as_completed`` lets
    it drop each Future as it yields it, and ``del future, box`` releases the parsed box
    score before the next one is awaited, so peak is the compact rows plus only the
    in-flight box scores.
    """
    ctx = {g["gamePk"]: _game_context(g) for g in games}
    hitting: dict[str, dict[str, Any]] = {}
    pitching: dict[str, dict[str, Any]] = {}
    dates: set[str] = set()
    failed: set[int] = set()

    def _one(game: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        gp = game["gamePk"]
        try:
            return gp, _fetch_boxscore(gp)
        except Exception:
            return gp, None

    total = len(games)
    with ThreadPoolExecutor(max_workers=15) as pool:
        for i, future in enumerate(as_completed([pool.submit(_one, g) for g in games]), 1):
            gp, box = future.result()
            if box is None:
                failed.add(gp)
            else:
                _pk, gnum, date = ctx[gp]
                dates.add(date)
                for mlbam_id, name, batting, pitch in iter_boxscore_players(box):
                    if batting:
                        h = hitting.setdefault(mlbam_id, {"name": name, "rows": {}})
                        h["name"] = name or h["name"]
                        h["rows"][gp] = boxscore_hitter_row(batting, gp, gnum, date)
                    if pitch:
                        p = pitching.setdefault(mlbam_id, {"name": name, "rows": {}})
                        p["name"] = name or p["name"]
                        p["rows"][gp] = boxscore_pitcher_row(pitch, gp, gnum, date)
            del future, box  # release the Future and its box score before the next one
            if progress_cb and i % 50 == 0:
                progress_cb(f"Box scores: {i}/{total}...")
    return hitting, pitching, dates, failed


def _resolve_positions(client, season: int, pitching_ids: list[str]) -> dict[str, str]:
    """Return {mlbam_id: pos_code} for the given ids, fetching uncached ones.

    On a fetch failure the missing ids stay absent so the caller can retry
    (declines to advance the watermark). A fetched-but-null code is stored
    as "" -- a resolved "not a pitcher", never retried.
    """
    cache = get_player_positions(client, season)
    missing = [pid for pid in pitching_ids if pid not in cache]
    if missing:
        try:
            fetched = _fetch_positions([int(p) for p in missing])
        except Exception:
            return cache
        for pid in missing:
            cache[pid] = fetched.get(pid) or ""
        set_player_positions(client, season, cache)
    return cache


def _upsert_and_roll(client, season: int, group: str, by_player: dict[str, dict[str, Any]]) -> None:
    """Merge each player's new rows into their stored log and refresh the rollup."""
    if not by_player:
        return
    rollup_type = "hitters" if group == "hitting" else "pitchers"
    rollup = get_game_log_totals(client, rollup_type)
    for mlbam_id, payload in by_player.items():
        existing = get_player_game_log(client, season, mlbam_id, group)
        merged = _merge_player_games(existing, payload["name"], payload["rows"])
        set_player_game_log(client, season, mlbam_id, group, merged)
        totals = (
            _sum_hitting(merged["games"]) if group == "hitting" else _sum_pitching(merged["games"])
        )
        rollup[mlbam_id] = {"name": merged["name"], **totals}
    set_game_log_totals(client, rollup_type, rollup)


def _sync(client, season: int, games: list[dict[str, Any]], now_utc: datetime, progress_cb) -> None:
    hitting, pitching, dates, failed = _collect_player_rows(games, progress_cb)
    all_ok = not failed

    positions = _resolve_positions(client, season, list(pitching.keys()))
    kept_pitching: dict[str, dict[str, Any]] = {}
    for mlbam_id, payload in pitching.items():
        if mlbam_id not in positions:
            all_ok = False  # unresolved (fetch blip) -> retry next run
            continue
        if should_record_pitching(positions[mlbam_id]):
            kept_pitching[mlbam_id] = payload

    _upsert_and_roll(client, season, "hitting", hitting)
    _upsert_and_roll(client, season, "pitching", kept_pitching)

    known_dates = set(get_game_log_dates(client, season))
    if dates:
        known_dates |= dates
        set_game_log_dates(client, season, list(known_dates))
    set_season_progress(
        client,
        games_elapsed=len(known_dates),
        total=162,
        as_of=local_today().isoformat(),
    )

    if all_ok:
        set_game_logs_watermark(client, season, now_utc.isoformat())
    if progress_cb:
        progress_cb(f"Game logs synced: {len(games)} games (clean={all_ok})")


def sync_game_logs(
    client, season: int, *, progress_cb=None, now_utc: datetime | None = None
) -> None:
    """Backfill (no watermark) or incremental (changes feed) sync into ``client``."""
    now_utc = now_utc or datetime.now(UTC)
    watermark = get_game_logs_watermark(client, season)
    if watermark is None:
        if progress_cb:
            progress_cb("No watermark; backfilling full season game logs...")
        games = [g for g in _fetch_season_games(season) if _is_regular_final(g)]
    else:
        if progress_cb:
            progress_cb(f"Incremental game-log sync since {watermark}...")
        games = [g for g in _fetch_changed_games(season, watermark) if _is_regular_final(g)]
    _sync(client, season, games, now_utc, progress_cb)


def fetch_game_log_totals(season: int, progress_cb=None) -> tuple[dict, dict, int]:
    """Sync game logs and return (hitters_totals, pitchers_totals, games_elapsed).

    Public entry point preserved for ``refresh_pipeline`` and ``season_routes``;
    both ignore the return value. Totals are read back from the derived rollup
    so the shape is unchanged. ``get_kv`` is imported lazily so tests that patch
    ``kv_store.get_kv`` take effect at call time.
    """
    from fantasy_baseball.data.kv_store import get_kv

    client = get_kv()
    sync_game_logs(client, season, progress_cb=progress_cb)
    hitters = get_game_log_totals(client, "hitters")
    pitchers = get_game_log_totals(client, "pitchers")
    games_elapsed = get_season_progress(client)["games_elapsed"]
    return hitters, pitchers, games_elapsed
