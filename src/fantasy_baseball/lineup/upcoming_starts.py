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
from datetime import datetime as _datetime
from typing import Any

import pandas as pd

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


def _day_name(date_iso: str) -> str:
    return _datetime.strptime(date_iso, "%Y-%m-%d").strftime("%a")


def _matchup_quality(
    factors: dict[str, dict[str, float]],
    team_stats: dict[str, dict[str, float]],
    opponent: str,
) -> str:
    if opponent in factors:
        f = factors[opponent]["era_whip_factor"]
        if f <= 0.93:
            return "Great"
        if f >= 1.03:
            return "Tough"
        return "Fair"
    if team_stats:
        avg_ops = sum(s["ops"] for s in team_stats.values()) / max(len(team_stats), 1)
        ops = team_stats.get(opponent, {}).get("ops", avg_ops)
        if ops < avg_ops * 0.95:
            return "Great"
        if ops > avg_ops * 1.05:
            return "Tough"
    return "Fair"


def _build_detail(
    team_stats: dict[str, dict[str, float]],
    ops_rank_map: dict[str, int],
    k_rank_map: dict[str, int],
    opponent: str,
) -> dict[str, Any]:
    opp = team_stats.get(opponent, {})
    raw_k = opp.get("k_pct", 0.0)
    k_display = round(raw_k * 100, 1) if raw_k < 1 else round(raw_k, 1)
    return {
        "ops": round(opp.get("ops", 0.0), 3),
        "ops_rank": ops_rank_map.get(opponent, 0),
        "k_pct": k_display,
        "k_rank": k_rank_map.get(opponent, 0),
    }


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
    target = normalize_name(pitcher_name)
    win_start_iso = window_start.isoformat()
    win_end_iso = window_end.isoformat()

    in_window = lambda d: win_start_iso <= d <= win_end_iso  # noqa: E731

    used_indices: set[int] = set()
    entries: list[StartEntry] = []

    for i, slot in enumerate(team_games):
        if not in_window(slot.date):
            continue
        if not slot.announced_starter:
            continue
        if normalize_name(slot.announced_starter) != target:
            continue
        used_indices.add(i)
        entries.append(
            StartEntry(
                date=slot.date,
                day=_day_name(slot.date),
                opponent=slot.opponent,
                indicator=slot.indicator,
                announced=True,
                matchup_quality=_matchup_quality(matchup_factors, team_stats, slot.opponent),
                detail=_build_detail(team_stats, ops_rank_map, k_rank_map, slot.opponent),
            )
        )

    anchor = find_anchor_index(team_games, pitcher_name, today)
    if anchor is not None:
        for idx in project_start_indices(anchor, len(team_games), step=5):
            if idx in used_indices:
                continue
            slot = team_games[idx]
            if not in_window(slot.date):
                continue
            if slot.announced_starter and normalize_name(slot.announced_starter) != target:
                continue
            entries.append(
                StartEntry(
                    date=slot.date,
                    day=_day_name(slot.date),
                    opponent=slot.opponent,
                    indicator=slot.indicator,
                    announced=False,
                    matchup_quality=_matchup_quality(matchup_factors, team_stats, slot.opponent),
                    detail=_build_detail(team_stats, ops_rank_map, k_rank_map, slot.opponent),
                )
            )

    entries.sort(key=lambda e: e.date)
    return entries


def filter_starting_pitchers(
    roster: list[Any],
    pitchers_proj: pd.DataFrame,
) -> list[Any]:
    """Keep only roster pitchers projected to start at least one game.

    Pitcher-eligible (P / SP / RP) AND projection ``gs > 0``. The
    eligibility gate uses ``is_pitcher`` rather than a strict
    ``Position.SP`` check because some Yahoo leagues use a single
    universal ``P`` slot — every pitcher's eligibility list is just
    ``[P]`` regardless of starter/reliever role, so an SP-only check
    drops the entire pitcher staff. ``gs > 0`` does the real work of
    excluding closers/setup men.

    Players missing from the projection frame, or with a projection row
    that has no ``gs`` column / non-positive gs, are dropped.
    """
    from fantasy_baseball.utils.positions import is_pitcher

    if pitchers_proj is None or pitchers_proj.empty or "gs" not in pitchers_proj.columns:
        return []
    if "_name_norm" not in pitchers_proj.columns:
        # Defensive: refresh pipeline always attaches _name_norm, but tests may not.
        pitchers_proj = pitchers_proj.copy()
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

    gs_by_name = dict(zip(pitchers_proj["_name_norm"], pitchers_proj["gs"], strict=False))

    kept: list[Any] = []
    for player in roster:
        if not is_pitcher(player.positions):
            continue
        gs = gs_by_name.get(normalize_name(player.name), 0.0) or 0.0
        if gs > 0:
            kept.append(player)
    return kept
