"""Pure parsing of MLB Stats API box scores into per-player per-game rows.

These functions take an already-fetched box-score JSON dict plus the game
context (gamePk, gameNumber, date) and return per-player rows. All HTTP
lives in ``mlb_game_logs``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fantasy_baseball.analysis.game_logs import (
    hitter_stats_from_statblock,
    pitcher_stats_from_statblock,
)

# primaryPosition.code values that legitimately pitch: Pitcher and Two-Way.
# A position player who mops up in a blowout has a fielding code (2-10, "O", ...)
# and is filtered out. Ohtani is "Y" and is kept. Verified 2026-05-24.
PITCHER_POSITION_CODES = frozenset({"1", "Y"})


def iter_boxscore_players(
    boxscore: dict[str, Any],
) -> Iterator[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """Yield (mlbam_id, name, batting_block, pitching_block) for each player.

    Empty ``{}`` blocks mean the player did not bat / pitch in this game.
    """
    teams = boxscore.get("teams", {})
    for side in ("home", "away"):
        players = teams.get(side, {}).get("players", {})
        for entry in players.values():
            person = entry.get("person", {})
            mlbam_id = person.get("id")
            if mlbam_id is None:
                continue
            stats = entry.get("stats", {})
            batting = stats.get("batting") or {}
            pitching = stats.get("pitching") or {}
            yield str(mlbam_id), person.get("fullName", ""), batting, pitching


def boxscore_hitter_row(
    batting: dict[str, Any], game_pk: int, game_number: int, date: str
) -> dict[str, Any]:
    """Build a hitting GameRow from a box-score batting block."""
    return {
        "gamePk": game_pk,
        "gameNumber": game_number,
        "date": date,
        **hitter_stats_from_statblock(batting),
    }


def boxscore_pitcher_row(
    pitching: dict[str, Any], game_pk: int, game_number: int, date: str
) -> dict[str, Any]:
    """Build a pitching GameRow from a box-score pitching block."""
    s = pitcher_stats_from_statblock(pitching)
    # gs/g are omitted: box-score rows are keyed by gamePk, so per-game
    # appearance counts (always 0 or 1) are redundant.
    return {
        "gamePk": game_pk,
        "gameNumber": game_number,
        "date": date,
        "ip": s["ip"],
        "k": s["k"],
        "er": s["er"],
        "bb": s["bb"],
        "h_allowed": s["h_allowed"],
        "w": s["w"],
        "sv": s["sv"],
    }


def should_record_pitching(pos_code: str | None) -> bool:
    """True only for real pitchers ("1") and two-way players ("Y").

    Drops position-player mop-up innings. Returns False for an unknown
    position; the sync engine treats unknown-after-fetch separately (it
    declines to advance the watermark so the game is retried).
    """
    return pos_code in PITCHER_POSITION_CODES
