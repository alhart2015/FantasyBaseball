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
    """Fetch league standings with cumulative roto stats."""
    raw = league.yhandler.get_standings_raw(league.league_id)
    return parse_standings_raw(raw, YAHOO_STAT_ID_MAP)


def parse_standings_raw(
    raw: dict, stat_id_map: dict[str, str],
) -> list[dict]:
    """Parse raw Yahoo standings JSON into a list of team dicts.

    The library's ``standings()`` method omits per-category stat totals,
    so we parse the raw JSON directly.
    """
    # Navigate: fantasy_content.league[1].standings[0].teams
    league_data = raw.get("fantasy_content", {}).get("league", [])
    if len(league_data) < 2:
        return []
    standings_block = league_data[1].get("standings", [{}])
    if not standings_block:
        return []
    raw_teams = standings_block[0].get("teams", {})

    teams = []
    for key in sorted(raw_teams.keys()):
        if key == "count":
            continue
        team_entry = raw_teams[key].get("team", [])
        if not team_entry or len(team_entry) < 2:
            continue

        # First element is a list of metadata dicts
        meta_list = team_entry[0] if isinstance(team_entry[0], list) else []
        team: dict = {"name": "", "team_key": "", "rank": 0, "stats": {}}
        for item in meta_list:
            if isinstance(item, dict):
                if "team_key" in item:
                    team["team_key"] = item["team_key"]
                if "name" in item:
                    team["name"] = item["name"]

        # Second element has team_stats and team_standings
        detail = team_entry[1] if len(team_entry) > 1 else {}
        if isinstance(detail, dict):
            # Parse rank from team_standings
            ts = detail.get("team_standings", {})
            if ts:
                try:
                    team["rank"] = int(ts.get("rank", 0))
                except (ValueError, TypeError):
                    team["rank"] = 0

            # Parse per-category stats from team_stats
            team_stats = detail.get("team_stats", {})
            for stat_entry in team_stats.get("stats", []):
                stat = stat_entry.get("stat", {})
                sid = str(stat.get("stat_id", ""))
                val = stat.get("value", "")
                if sid in YAHOO_STAT_ID_MAP and val != "":
                    try:
                        team["stats"][YAHOO_STAT_ID_MAP[sid]] = float(val)
                    except (ValueError, TypeError):
                        pass

        # Third element (if present) may also have team_standings
        if len(team_entry) > 2 and isinstance(team_entry[2], dict):
            ts = team_entry[2].get("team_standings", {})
            if ts and team["rank"] == 0:
                try:
                    team["rank"] = int(ts.get("rank", 0))
                except (ValueError, TypeError):
                    pass

        teams.append(team)

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
