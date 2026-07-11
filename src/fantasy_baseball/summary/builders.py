"""Section builders for the daily summary email.

Each builder is a pure function returning a typed section model (or an empty
list). Builders never raise for "no data" -- that is an empty section. They read
KV payloads the morning refresh produced, plus (for last night) per-player game
logs. No builder imports the streaks/dashboard module (it pulls in duckdb).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fantasy_baseball.data.redis_store import get_player_game_log
from fantasy_baseball.summary.crosswalk import player_group
from fantasy_baseball.summary.models import PlayerLine

_HITTER_FIELDS = ("pa", "ab", "h", "hr", "r", "rbi", "sb")
_PITCHER_FIELDS = ("ip", "k", "er", "bb", "w", "sv", "h_allowed")


def build_last_night(
    roster: list[dict[str, Any]],
    xmap: dict[tuple[str, str], int],
    client: Any,
    season: int,
    yesterday: date,
) -> tuple[list[PlayerLine], list[str]]:
    """Box-score lines for rostered players who played on ``yesterday``.

    Returns ``(lines, unmatched_names)``. A player whose name+type is not in the
    crosswalk goes into ``unmatched``; a matched player with no game row for
    ``yesterday`` is omitted (did not play).
    """
    lines: list[PlayerLine] = []
    unmatched: list[str] = []
    target = yesterday.isoformat()

    for entry in roster:
        name = entry.get("name", "")
        positions = entry.get("positions", []) or []
        groups = player_group(positions)

        # A two-way player resolves under whichever type namespace matches; the
        # same person-level MLBAM id serves both game-log groups.
        norm = _normalize(name)
        mlbam: int | None = None
        for group in groups:
            key = (norm, "pitcher" if group == "pitching" else "hitter")
            if key in xmap:
                mlbam = xmap[key]
                break
        if mlbam is None:
            unmatched.append(name)
            continue

        for group in groups:
            log = get_player_game_log(client, season, str(mlbam), group)
            if not log:
                continue
            for row in log.get("games", []):
                if row.get("date") != target:
                    continue
                fields = _HITTER_FIELDS if group == "hitting" else _PITCHER_FIELDS
                stats = {f: _num(row.get(f)) for f in fields}
                lines.append(PlayerLine(name=name, group=group, stats=stats))

    return lines, unmatched


def _normalize(name: str) -> str:
    from fantasy_baseball.utils.name_utils import normalize_name

    return normalize_name(name)


def _num(value: Any) -> float:
    return float(value) if value is not None else 0.0
