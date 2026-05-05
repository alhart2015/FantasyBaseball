import json
import logging
from collections import defaultdict
from datetime import date as _date
from datetime import timedelta as _timedelta
from pathlib import Path
from typing import Any, cast

import statsapi

from fantasy_baseball.utils.time_utils import local_now

logger = logging.getLogger(__name__)

MLB_TO_FANGRAPHS_ABBREV: dict[str, str] = {
    "AZ": "ARI",
    "CWS": "CHW",
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WSH": "WSN",
}


def normalize_team_abbrev(mlb_abbrev: str) -> str:
    """Convert MLB Stats API abbreviation to FanGraphs format.

    Returns the input unchanged if not in the mapping.
    """
    return MLB_TO_FANGRAPHS_ABBREV.get(mlb_abbrev, mlb_abbrev)


def _build_team_name_map() -> dict[str, str]:
    """Build a mapping from team names to FanGraphs-normalized abbreviations.

    Both the short name (e.g. "Yankees") and full name
    (e.g. "New York Yankees") are mapped so that schedule lookups using
    either format will work.
    """
    response = statsapi.get("teams", {"sportId": 1})
    name_map: dict[str, str] = {}
    for team in response.get("teams", []):
        fg_abbrev = normalize_team_abbrev(team["abbreviation"])
        name_map[team["name"]] = fg_abbrev
        name_map[team["teamName"]] = fg_abbrev
    return name_map


def fetch_week_schedule(start_date: str, end_date: str, lookback_days: int = 0) -> dict:
    """Fetch the MLB schedule for a date range and return structured data.

    Filters to regular-season games only (game_type == "R").
    Returns game counts per team (FanGraphs abbreviations), probable
    pitchers, team abbreviation map, and metadata.

    When ``lookback_days > 0``, the actual statsapi call covers
    ``(start_date - lookback_days) .. end_date``. The returned
    ``start_date`` and ``end_date`` fields still reflect the original
    scoring window, not the lookback span. ``games_per_team`` only
    counts games inside ``start_date..end_date`` so existing consumers
    that read it as the per-team count for the scoring week stay correct.

    Each ``probable_pitchers`` entry includes ``game_number`` (defaults
    to 1; >1 marks the second game of a doubleheader). Sorting by
    (date, game_number) gives a stable chronological order.
    """
    fetch_start = start_date
    if lookback_days > 0:
        fetch_start = (_date.fromisoformat(start_date) - _timedelta(days=lookback_days)).isoformat()

    games = statsapi.schedule(fetch_start, end_date)
    team_name_map = _build_team_name_map()

    games_per_team: dict[str, int] = defaultdict(int)
    probable_pitchers: list[dict] = []

    for game in games:
        if game.get("game_type") != "R":
            continue

        away_name = game["away_name"]
        home_name = game["home_name"]
        game_date = game["game_date"]

        away_abbrev = team_name_map.get(away_name, away_name)
        home_abbrev = team_name_map.get(home_name, home_name)

        # Per-team game counts: scoring-week only, so other consumers
        # of games_per_team don't see lookback games.
        if game_date >= start_date:
            games_per_team[away_abbrev] += 1
            games_per_team[home_abbrev] += 1

        away_pitcher = game.get("away_probable_pitcher", "") or "TBD"
        home_pitcher = game.get("home_probable_pitcher", "") or "TBD"

        probable_pitchers.append(
            {
                "date": game_date,
                "game_number": int(game.get("game_num", 1) or 1),
                "away_team": away_abbrev,
                "home_team": home_abbrev,
                "away_pitcher": away_pitcher,
                "home_pitcher": home_pitcher,
            }
        )

    return {
        "games_per_team": dict(games_per_team),
        "probable_pitchers": probable_pitchers,
        "team_abbrev_map": team_name_map,
        "start_date": start_date,
        "end_date": end_date,
        "lookback_days": lookback_days,
        "fetched_at": local_now().isoformat(timespec="seconds"),
    }


def save_schedule_cache(data: dict, path: Path) -> None:
    """Save schedule data to a JSON cache file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_schedule_cache(path: Path) -> dict | None:
    """Load schedule data from a JSON cache file.

    Returns None if the file does not exist.
    """
    if not path.exists():
        return None
    with open(path) as f:
        return cast(dict[str, Any], json.load(f))


def get_week_schedule(
    start_date: str,
    end_date: str,
    cache_path: Path,
    lookback_days: int = 0,
) -> dict | None:
    """Main entry point for fetching the week schedule.

    Tries a live fetch first; on success, caches the result. On API
    failure, falls back to the cache if the cached date range AND
    ``lookback_days`` match the requested ones. Returns None if both
    live and cached data are unavailable or stale.
    """
    try:
        data = fetch_week_schedule(start_date, end_date, lookback_days=lookback_days)
        save_schedule_cache(data, cache_path)
        return data
    except Exception:
        logger.exception("Failed to fetch live week schedule; trying cache")

    cached = load_schedule_cache(cache_path)
    if cached is None:
        return None

    if (
        cached.get("start_date") != start_date
        or cached.get("end_date") != end_date
        or cached.get("lookback_days", 0) != lookback_days
    ):
        logger.warning(
            "Cached schedule (%s-%s, lookback=%s) does not match requested (%s-%s, lookback=%s); ignoring cache",
            cached.get("start_date"),
            cached.get("end_date"),
            cached.get("lookback_days", 0),
            start_date,
            end_date,
            lookback_days,
        )
        return None

    return cached
