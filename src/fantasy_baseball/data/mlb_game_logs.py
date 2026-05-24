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
