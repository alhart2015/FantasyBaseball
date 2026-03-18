"""Fetch roster, standings, and free agents from Yahoo Fantasy API."""

import datetime
import logging

logger = logging.getLogger(__name__)

# Yahoo stat IDs for 5x5 roto categories
YAHOO_STAT_ID_MAP: dict[str, str] = {
    "60": "R",
    "7": "HR",
    "13": "RBI",
    "16": "SB",
    "3": "AVG",
    "28": "W",
    "32": "SV",
    "42": "K",
    "26": "ERA",
    "27": "WHIP",
}


def fetch_roster(league, team_key: str) -> list[dict]:
    """Fetch a team's current roster from Yahoo."""
    team = league.to_team(team_key)
    raw_roster = team.roster()
    return parse_roster(raw_roster)


def parse_roster(raw_roster: list[dict]) -> list[dict]:
    """Normalize raw Yahoo roster data."""
    players = []
    for p in raw_roster:
        players.append({
            "name": p["name"],
            "positions": p.get("eligible_positions", []),
            "selected_position": p.get("selected_position", ""),
            "player_id": p.get("player_id", ""),
        })
    return players


def fetch_standings(league) -> list[dict]:
    """Fetch league standings with cumulative team stats."""
    raw = league.standings()
    return parse_standings(raw, stat_id_map=YAHOO_STAT_ID_MAP)


def parse_standings(raw: dict, stat_id_map: dict[str, str]) -> list[dict]:
    """Normalize raw Yahoo standings data."""
    teams = []
    for team_data in raw.get("teams", []):
        stats = {}
        team_stats = team_data.get("team_stats", {})
        for stat_entry in team_stats.get("stats", []):
            stat = stat_entry.get("stat", {})
            sid = str(stat.get("stat_id", ""))
            if sid in stat_id_map:
                cat = stat_id_map[sid]
                try:
                    stats[cat] = float(stat.get("value", 0))
                except (ValueError, TypeError):
                    stats[cat] = 0.0

        team_standings = team_data.get("team_standings", {})
        teams.append({
            "name": team_data.get("name", ""),
            "team_key": team_data.get("team_key", ""),
            "rank": team_standings.get("rank", 0),
            "stats": stats,
        })
    return teams


def fetch_free_agents(league, position: str, count: int = 50) -> list[dict]:
    """Fetch top free agents at a position."""
    try:
        agents = league.free_agents(position)
        result = []
        for p in agents[:count]:
            result.append({
                "name": p["name"],
                "positions": p.get("eligible_positions", [position]),
                "player_id": p.get("player_id", ""),
            })
        return result
    except (PermissionError, OSError) as exc:
        # Auth failures and critical OS-level errors must surface
        logger.exception(
            "Critical error fetching free agents at position %s", position
        )
        raise
    except Exception:
        # Transient network errors, rate limits, etc. — log and degrade
        logger.exception(
            "Failed to fetch free agents at position %s; returning empty list",
            position,
        )
        return []


def fetch_scoring_period(league) -> tuple[str, str]:
    """Get the current Yahoo scoring period date range.

    Returns (start_date, end_date) as "YYYY-MM-DD" strings.
    Falls back to Monday-Sunday of the current week on error.
    """
    try:
        week = league.current_week()
        start, end = league.week_date_range(week)
        return start.isoformat(), end.isoformat()
    except Exception:
        logger.warning(
            "Failed to get Yahoo scoring period; using Mon-Sun fallback",
            exc_info=True,
        )
        today = datetime.date.today()
        monday = today - datetime.timedelta(days=today.weekday())
        sunday = monday + datetime.timedelta(days=6)
        return monday.isoformat(), sunday.isoformat()
