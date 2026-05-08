"""Hitter game log fetch for the streaks project.

This is a streaks-specific parser that captures every column the
`hitter_games` table needs (player_id, name, team, season, plus bb/k that
the existing analysis/game_logs.py omits). The HTTP shape is identical;
only the parsing differs.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import requests

from fantasy_baseball.streaks.models import HitterGame

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def parse_hitter_game_log_full(
    split: dict[str, Any],
    *,
    player_id: int,
    name: str,
    team: str | None,
    season: int,
) -> HitterGame:
    """Parse one /people/{id}/stats?stats=gameLog split into a :class:`HitterGame`.

    Uses the split's ``game.gamePk`` (a unique MLB game identifier) for
    the row PK alongside ``player_id`` so doubleheader games on the same
    date don't collide.
    """
    stat = split.get("stat", {})
    return HitterGame(
        player_id=player_id,
        game_pk=int(split["game"]["gamePk"]),
        name=name,
        team=team,
        season=season,
        date=date.fromisoformat(split["date"]),
        pa=int(stat.get("plateAppearances", 0)),
        ab=int(stat.get("atBats", 0)),
        h=int(stat.get("hits", 0)),
        hr=int(stat.get("homeRuns", 0)),
        r=int(stat.get("runs", 0)),
        rbi=int(stat.get("rbi", 0)),
        sb=int(stat.get("stolenBases", 0)),
        bb=int(stat.get("baseOnBalls", 0)),
        k=int(stat.get("strikeOuts", 0)),
    )


def fetch_hitter_season_game_logs(
    player_id: int, name: str, team: str | None, season: int, timeout: float = 15.0
) -> list[HitterGame]:
    """Fetch one season of game logs for one hitter.

    Returns one :class:`HitterGame` per game played. Empty list if the player
    has no logs.
    """
    url = f"{MLB_API_BASE}/people/{player_id}/stats"
    params: dict[str, str | int] = {
        "stats": "gameLog",
        "group": "hitting",
        "season": season,
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    return [
        parse_hitter_game_log_full(s, player_id=player_id, name=name, team=team, season=season)
        for s in splits
    ]
