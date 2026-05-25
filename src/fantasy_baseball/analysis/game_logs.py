"""Fetch and cache player game logs from the MLB Stats API."""

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import requests

from fantasy_baseball.models.player import PlayerType

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


class HitterGameLog(TypedDict):
    """One game's worth of hitter stats parsed from the MLB Stats API."""

    date: str
    pa: int
    ab: int
    h: int
    hr: int
    r: int
    rbi: int
    sb: int


class PitcherGameLog(TypedDict):
    """One game's worth of pitcher stats parsed from the MLB Stats API.

    ``ip`` is a float because the API returns values like ``"6.1"`` meaning
    6⅓ innings; we convert the fractional part into thirds.
    """

    date: str
    ip: float
    k: int
    er: int
    bb: int
    h_allowed: int
    w: int
    sv: int
    gs: int
    g: int


# Union of the two game log shapes, for code paths that don't know the
# player type statically (e.g. fetch_player_game_log dispatching on `group`).
GameLog = HitterGameLog | PitcherGameLog


def hitter_stats_from_statblock(stat: dict[str, Any]) -> dict[str, int]:
    """Extract hitter counting stats from an MLB stat block.

    Shared by the gameLog parser (``split["stat"]``) and the box-score
    parser (``player["stats"]["batting"]``) -- both use these field names.
    """
    return {
        "pa": int(stat.get("plateAppearances", 0)),
        "ab": int(stat.get("atBats", 0)),
        "h": int(stat.get("hits", 0)),
        "hr": int(stat.get("homeRuns", 0)),
        "r": int(stat.get("runs", 0)),
        "rbi": int(stat.get("rbi", 0)),
        "sb": int(stat.get("stolenBases", 0)),
    }


def pitcher_stats_from_statblock(stat: dict[str, Any]) -> dict[str, float | int]:
    """Extract pitcher counting stats from an MLB stat block.

    Shared by the gameLog parser and the box-score parser. ``inningsPitched``
    arrives as a string like ``"6.1"`` meaning 6 and 1/3 innings.
    """
    ip_str = str(stat.get("inningsPitched", "0"))
    if "." in ip_str:
        whole, frac = ip_str.split(".")
        ip = int(whole) + int(frac) / 3.0
    else:
        ip = float(ip_str)
    return {
        "ip": round(ip, 4),
        "k": int(stat.get("strikeOuts", 0)),
        "er": int(stat.get("earnedRuns", 0)),
        "bb": int(stat.get("baseOnBalls", 0)),
        "h_allowed": int(stat.get("hits", 0)),
        "w": int(stat.get("wins", 0)),
        "sv": int(stat.get("saves", 0)),
        "gs": int(stat.get("gamesStarted", 0)),
        "g": int(stat.get("gamesPlayed", 0)),
    }


def parse_hitter_game_log(split: dict[str, Any]) -> HitterGameLog:
    """Parse a single hitter game log entry from the MLB API."""
    return {"date": split["date"], **hitter_stats_from_statblock(split["stat"])}  # type: ignore[typeddict-item]


def parse_pitcher_game_log(split: dict[str, Any]) -> PitcherGameLog:
    """Parse a single pitcher game log entry from the MLB API."""
    return {"date": split["date"], **pitcher_stats_from_statblock(split["stat"])}  # type: ignore[typeddict-item]


def fetch_player_game_log(mlbam_id: int, season: int, group: str = "hitting") -> list[GameLog]:
    """Fetch game log from MLB Stats API for one player.

    Returns hitter rows for ``group="hitting"`` and pitcher rows for any
    other value — callers that know the player type can narrow the list
    element type accordingly.
    """
    url = f"{MLB_API_BASE}/people/{mlbam_id}/stats"
    params: dict[str, str | int] = {"stats": "gameLog", "group": group, "season": season}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    if group == "hitting":
        return [parse_hitter_game_log(s) for s in splits]
    return [parse_pitcher_game_log(s) for s in splits]


class PlayerGameLogs(TypedDict):
    """Game logs for a single player, as stored in the cache JSON."""

    name: str
    type: str
    games: list[GameLog]


def fetch_all_game_logs(
    players: list[dict[str, Any]], season: int = 2025, cache_path: Path | None = None
) -> dict[int, PlayerGameLogs]:
    """Fetch game logs for a list of players, with JSON caching.

    Args:
        players: List of dicts with keys 'mlbam_id', 'name', 'type' ('hitter'/'pitcher').
        season: Season year.
        cache_path: Path to JSON cache file.

    Returns:
        Dict keyed by mlbam_id: {'name': str, 'type': str, 'games': [game_log_dicts]}.
    """
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        cached_ids = {int(k) for k in cached}
        requested_ids = {p["mlbam_id"] for p in players}
        if requested_ids.issubset(cached_ids):
            logger.info("Using cached game logs (%d players)", len(cached))
            return {int(k): v for k, v in cached.items()}

    results: dict[int, PlayerGameLogs] = {}
    for i, player in enumerate(players):
        mid = player["mlbam_id"]
        name = player["name"]
        ptype = player["type"]
        group = "hitting" if ptype == PlayerType.HITTER else "pitching"
        try:
            games = fetch_player_game_log(mid, season, group)
            results[mid] = {"name": name, "type": ptype, "games": games}
        except Exception:
            logger.warning("Failed to fetch game log for %s (ID %s)", name, mid)
            results[mid] = {"name": name, "type": ptype, "games": []}
        if (i + 1) % 25 == 0:
            print(f"  Fetched {i + 1}/{len(players)} game logs...")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f)

    return results
