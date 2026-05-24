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

from typing import Any

import requests

_MLB_API = "https://statsapi.mlb.com/api/v1"

_HITTER_KEYS = ("pa", "ab", "h", "r", "hr", "rbi", "sb")
_PITCHER_KEYS = ("ip", "k", "er", "bb", "h_allowed", "w", "sv")


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
        for k in ("k", "er", "bb", "h_allowed", "w", "sv"):
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
    resp = requests.get(f"{_MLB_API}/game/{game_pk}/boxscore", timeout=20)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return data


def _fetch_positions(mlbam_ids: list[int]) -> dict[str, str]:
    """Batch primaryPosition.code lookup. {str(id): code}; code may be None."""
    if not mlbam_ids:
        return {}
    resp = requests.get(
        f"{_MLB_API}/people",
        params={"personIds": ",".join(str(i) for i in mlbam_ids)},
        timeout=20,
    )
    resp.raise_for_status()
    out: dict[str, str] = {}
    for person in resp.json().get("people", []):
        out[str(person["id"])] = person.get("primaryPosition", {}).get("code")
    return out
