"""Build upcoming projected starts for roster pitchers.

For each in-scope starting pitcher we find the rotation anchor (their
most recent past start within a 14-day lookback) and project the next
starts at every 5th team game until past the scoring window. MLB-
announced probable starters override projections for the same game.

Public API:
    build_team_game_index(probable_pitchers, team_abbrev) -> list[GameSlot]
    find_anchor_index(team_games, pitcher_name, today) -> int | None
    project_start_indices(anchor_index, total_games, step=5) -> list[int]
    compose_pitcher_entries(...)  -> list[StartEntry]

All functions are pure -- no I/O, no global state. The matchup/quality
decoration happens in lineup.matchups via existing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from fantasy_baseball.utils.name_utils import normalize_name


@dataclass(frozen=True)
class GameSlot:
    """One scheduled team-game from the perspective of a single team.

    ``announced_starter`` is the name MLB has listed (or empty string if
    "TBD" / unset). For completed past games, MLB populates this with
    the actual starter, which is the signal used to find rotation anchors.
    """

    date: str  # YYYY-MM-DD
    game_number: int  # 1 for normal games, >1 for second game of doubleheader
    opponent: str  # FanGraphs-normalized opponent abbreviation
    indicator: str  # "@" if away, "vs" if home
    announced_starter: str = ""


@dataclass
class StartEntry:
    """One projected or announced start for a roster pitcher."""

    date: str
    day: str  # "Mon", "Tue", ...
    opponent: str
    indicator: str
    announced: bool = False
    matchup_quality: str = "Fair"  # "Great" | "Fair" | "Tough"
    detail: dict[str, Any] = field(default_factory=dict)


def build_team_game_index(
    probable_pitchers: list[dict[str, Any]],
    team_abbrev: str,
) -> list[GameSlot]:
    """Filter the league-wide probable_pitchers list to one team's games.

    Returns a chronological list (by date, then game_number). Each
    entry exposes the opponent and the announced starter for that team.
    """
    slots: list[GameSlot] = []
    for game in probable_pitchers:
        if game["away_team"] == team_abbrev:
            opponent = game["home_team"]
            indicator = "@"
            starter = game.get("away_pitcher", "") or ""
        elif game["home_team"] == team_abbrev:
            opponent = game["away_team"]
            indicator = "vs"
            starter = game.get("home_pitcher", "") or ""
        else:
            continue

        if starter == "TBD":
            starter = ""

        slots.append(
            GameSlot(
                date=game["date"],
                game_number=int(game.get("game_number", 1) or 1),
                opponent=opponent,
                indicator=indicator,
                announced_starter=starter,
            )
        )

    slots.sort(key=lambda s: (s.date, s.game_number))
    return slots


def find_anchor_index(
    team_games: list[GameSlot],
    pitcher_name: str,
    today: date,
) -> int | None:
    """Most recent index in ``team_games`` where ``pitcher_name`` started.

    Only considers games strictly before ``today``. Name comparison is
    accent/case-insensitive (delegates to normalize_name). Returns
    ``None`` if the pitcher has no eligible past start in the index.
    """
    target = normalize_name(pitcher_name)
    today_iso = today.isoformat()
    anchor: int | None = None
    for i, slot in enumerate(team_games):
        if slot.date >= today_iso:
            continue
        if not slot.announced_starter:
            continue
        if normalize_name(slot.announced_starter) == target:
            anchor = i
    return anchor


def project_start_indices(
    anchor_index: int,
    total_games: int,
    step: int = 5,
) -> list[int]:
    """Return the projected start indices in the team's game stream.

    Starts at ``anchor_index + step`` and steps by ``step`` until
    exceeding ``total_games - 1``. Returns an empty list if anchor_index
    is negative.
    """
    if anchor_index < 0:
        return []
    indices: list[int] = []
    nxt = anchor_index + step
    while nxt < total_games:
        indices.append(nxt)
        nxt += step
    return indices


def compose_pitcher_entries(
    pitcher_name: str,
    team_games: list[GameSlot],
    today: date,
    window_start: date,
    window_end: date,
    matchup_factors: dict[str, dict[str, float]],
    team_stats: dict[str, dict[str, float]],
    ops_rank_map: dict[str, int],
    k_rank_map: dict[str, int],
) -> list[StartEntry]:
    """Build the full list of StartEntry rows for one pitcher.

    Combines:
      - announced starts in ``[window_start, window_end]`` where this
        pitcher is the starter,
      - projected starts (anchor + 5*N) that land inside the window,
        excluding any team-game whose announced starter is someone else.

    Each entry is decorated with the existing matchup_quality + detail
    payload by looking up the opponent in ``matchup_factors`` and
    ``team_stats``. Rows are sorted by date then game_number.
    """
    raise NotImplementedError("Implemented in Task 7")
