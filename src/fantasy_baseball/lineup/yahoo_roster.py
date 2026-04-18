"""Fetch roster, standings, and free agents from Yahoo Fantasy API."""

import datetime
import logging
from dataclasses import dataclass, field

from fantasy_baseball.utils.time_utils import local_today

logger = logging.getLogger(__name__)

# Yahoo stat IDs for 5x5 roto categories
YAHOO_STAT_ID_MAP: dict[str, str] = {
    "7": "R",
    "12": "HR",
    "13": "RBI",
    "16": "SB",
    "3": "AVG",
    "28": "W",
    "32": "SV",
    "42": "K",
    "26": "ERA",
    "27": "WHIP",
}


@dataclass
class ParsedStandingsTeam:
    """One team as parsed from Yahoo's raw standings JSON.

    Mirrors the wire-format dict emitted by :func:`parse_standings_raw`
    — ``to_dict()`` is the single source for that mapping. ``stats`` is
    a sparse ``{cat: value}`` mapping (empty pre-season; filled via
    ``_fill_stat_defaults`` at the refresh boundary). ``points_for`` is
    Yahoo's authoritative roto total and is ``None`` when Yahoo hasn't
    scored the week yet (e.g. projected standings).
    """

    name: str = ""
    team_key: str = ""
    rank: int = 0
    stats: dict[str, float] = field(default_factory=dict)
    points_for: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "team_key": self.team_key,
            "rank": self.rank,
            "stats": self.stats,
            "points_for": self.points_for,
        }


def fetch_roster(
    league,
    team_key: str,
    day: datetime.date | None = None,
) -> list[dict]:
    """Fetch a team's roster from Yahoo.

    Args:
        league: Yahoo ``League`` handle.
        team_key: Yahoo team key (e.g. ``"431.l.17492.t.3"``).
        day: If given, fetch the roster as-of that date. Used by the
            refresh pipeline to pull next Tuesday's pre-locked roster
            instead of today's. Yahoo applies transaction effective
            dates server-side, so this is the ground-truth future
            state.
    """
    team = league.to_team(team_key)
    if day is None:
        raw_roster = team.roster()
    else:
        raw_roster = team.roster(day=day)
    return parse_roster(raw_roster)


def parse_roster(raw_roster: list[dict]) -> list[dict]:
    """Normalize raw Yahoo roster data."""
    players = []
    for p in raw_roster:
        entry = {
            "name": p["name"],
            "positions": p.get("eligible_positions", []),
            "selected_position": p.get("selected_position", ""),
            "player_id": p.get("player_id", ""),
            "status": p.get("status", ""),
        }
        players.append(entry)
    return players


def fetch_injuries(league, team_key: str) -> list[dict]:
    """Fetch injured players on a team's roster with injury details.

    Uses the raw Yahoo API to get injury_note and status_full fields
    that the library's roster() method omits.

    Returns list of dicts: {name, status, status_full, injury_note,
    selected_position, player_id, positions}.
    """
    raw = league.yhandler.get(f"team/{team_key}/roster/players")
    return parse_injuries_raw(raw)


def parse_injuries_raw(raw: dict) -> list[dict]:
    """Parse raw Yahoo roster JSON to extract injured players.

    Looks for players with a non-empty ``status`` field (IL15, IL60, DTD, etc.)
    and returns their injury details.
    """
    team_data = raw.get("fantasy_content", {}).get("team", [])
    if len(team_data) < 2:
        return []

    roster_data = team_data[1].get("roster", {})
    players_block = roster_data.get("0", {}).get("players", {})

    injured = []
    for key in sorted(players_block.keys()):
        if key == "count":
            continue
        player = players_block[key].get("player", [])
        if not player:
            continue

        meta = player[0] if isinstance(player[0], list) else []
        info = {
            "name": "", "status": "", "status_full": "",
            "injury_note": "", "player_id": "", "positions": [],
            "selected_position": "",
        }

        for item in meta:
            if not isinstance(item, dict):
                continue
            if "name" in item:
                info["name"] = item["name"].get("full", "")
            if "status" in item:
                info["status"] = item["status"]
                info["status_full"] = item.get("status_full", "")
            if "injury_note" in item:
                info["injury_note"] = item["injury_note"]
            if "player_id" in item:
                info["player_id"] = item["player_id"]
            if "eligible_positions" in item:
                info["positions"] = [
                    ep["position"] for ep in item["eligible_positions"]
                    if isinstance(ep, dict) and "position" in ep
                ]

        # Selected position from second element
        if len(player) > 1 and isinstance(player[1], dict):
            sp = player[1].get("selected_position", [])
            for entry in sp:
                if isinstance(entry, dict) and "position" in entry:
                    info["selected_position"] = entry["position"]

        if info["status"]:
            injured.append(info)

    return injured


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

    Each team dict also carries ``points_for`` — Yahoo's own authoritative
    roto total. Yahoo computes it from full-precision internal stats, so
    it correctly breaks display-level ties (e.g. two teams shown with
    WHIP=1.03 that differ in the fourth decimal) that our local
    ``score_roto`` cannot distinguish.
    """
    # Navigate: fantasy_content.league[1].standings[0].teams
    league_data = raw.get("fantasy_content", {}).get("league", [])
    if len(league_data) < 2:
        return []
    standings_block = league_data[1].get("standings", [{}])
    if not standings_block:
        return []
    raw_teams = standings_block[0].get("teams", {})

    teams: list[ParsedStandingsTeam] = []
    for key in sorted(raw_teams.keys()):
        if key == "count":
            continue
        team_entry = raw_teams[key].get("team", [])
        if not team_entry or len(team_entry) < 2:
            continue

        team = ParsedStandingsTeam()

        # First element is a list of metadata dicts
        meta_list = team_entry[0] if isinstance(team_entry[0], list) else []
        for item in meta_list:
            if isinstance(item, dict):
                if "team_key" in item:
                    team.team_key = item["team_key"]
                if "name" in item:
                    team.name = item["name"]

        # team_standings may live at team_entry[1] or team_entry[2] depending on
        # the Yahoo response shape; check both positions.
        standings_candidates: list[dict] = []
        detail = team_entry[1] if len(team_entry) > 1 else {}
        if isinstance(detail, dict) and detail.get("team_standings"):
            standings_candidates.append(detail["team_standings"])
        if len(team_entry) > 2 and isinstance(team_entry[2], dict):
            extra = team_entry[2].get("team_standings")
            if extra:
                standings_candidates.append(extra)

        for ts in standings_candidates:
            if team.rank == 0:
                try:
                    team.rank = int(ts.get("rank", 0))
                except (ValueError, TypeError):
                    team.rank = 0
            if team.points_for is None:
                raw_pts = ts.get("points_for")
                if raw_pts not in (None, ""):
                    try:
                        team.points_for = float(raw_pts)
                    except (ValueError, TypeError):
                        pass

        # Parse per-category stats from team_stats
        if isinstance(detail, dict):
            team_stats = detail.get("team_stats", {})
            for stat_entry in team_stats.get("stats", []):
                stat = stat_entry.get("stat", {})
                sid = str(stat.get("stat_id", ""))
                val = stat.get("value", "")
                if sid in stat_id_map and val != "":
                    try:
                        team.stats[stat_id_map[sid]] = float(val)
                    except (ValueError, TypeError):
                        pass

        teams.append(team)

    return [t.to_dict() for t in teams]


def fetch_free_agents(league, position: str, count: int = 50) -> list[dict]:
    """Fetch top available players (free agents + waivers) at a position.

    Uses status 'A' (all available) instead of 'FA' (free agents only)
    so that waiver-wire players are included.  Pre-season, all unrostered
    players have waiver status, so 'FA'-only queries return nothing.
    """
    try:
        # _fetch_players('A', ...) returns both FA and W status players.
        # league.free_agents() only returns 'FA', which is empty pre-season.
        agents = league._fetch_players('A', position=position)
        result = []
        for p in agents[:count]:
            result.append({
                "name": p["name"],
                "positions": p.get("eligible_positions", [position]),
                "player_id": p.get("player_id", ""),
                "status": p.get("status", ""),
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
    Falls back to Monday-Sunday of the current week on error
    (common pre-season when no scoring week exists yet).
    """
    try:
        week = league.current_week()
        start, end = league.week_date_range(week)
        return start.isoformat(), end.isoformat()
    except Exception:
        logger.info("No active scoring period (pre-season?) — using Mon-Sun of current week")
        today = local_today()
        monday = today - datetime.timedelta(days=today.weekday())
        sunday = monday + datetime.timedelta(days=6)
        return monday.isoformat(), sunday.isoformat()


def _extract_player_info(player_data: dict) -> tuple[dict, dict]:
    """Extract player metadata and transaction_data from a Yahoo player entry.

    Yahoo nests data as:
        player_data["player"][0] = [list of metadata dicts]  (name, id, positions)
        player_data["player"][1] = {"transaction_data": ...}  (add/drop info)

    transaction_data is a list for adds, a dict for drops. We normalize to a dict.

    Returns:
        (player_info, tdata) where player_info has name, player_id, positions
        and tdata has type, destination/source team info.
    """
    raw_player = player_data.get("player", [])
    meta = raw_player[0] if raw_player and isinstance(raw_player[0], list) else []

    name = ""
    player_id = ""
    positions = []
    for item in meta:
        if not isinstance(item, dict):
            continue
        if "name" in item:
            name = item["name"].get("full", "")
        if "player_id" in item:
            player_id = item["player_id"]
        if "display_position" in item:
            positions = [p.strip() for p in item["display_position"].split(",")]
        if "eligible_positions" in item and not positions:
            positions = [
                ep["position"] for ep in item["eligible_positions"]
                if isinstance(ep, dict) and "position" in ep
            ]

    # transaction_data lives in player[1], not at the player_data level
    tdata = {}
    if len(raw_player) > 1 and isinstance(raw_player[1], dict):
        td_raw = raw_player[1].get("transaction_data", {})
        if isinstance(td_raw, list):
            tdata = td_raw[0] if td_raw else {}
        else:
            tdata = td_raw

    return {"name": name, "player_id": player_id, "positions": positions}, tdata



def fetch_all_transactions(league) -> list[dict]:
    """Fetch all successful add/drop transactions for the season.

    Returns list of flat transaction dicts ready for scoring and DB insertion.
    Only includes successful (completed) transactions, not pending ones.
    """
    try:
        raw = league.transactions("add,drop", "")
        return parse_all_transactions(raw)
    except Exception:
        logger.exception("Failed to fetch transactions; returning empty list")
        return []


def parse_all_transactions(transactions: list[dict]) -> list[dict]:
    """Parse raw Yahoo transactions into flat dicts for DB storage.

    Only includes successful (completed) transactions.
    Each dict has: transaction_id, type, status, timestamp, team, team_key,
    add_name, add_player_id, add_positions, drop_name, drop_player_id,
    drop_positions.
    """
    results = []
    for txn in transactions:
        if txn.get("status") != "successful":
            continue

        add_name = add_pid = add_pos = None
        drop_name = drop_pid = drop_pos = None
        team_name = ""
        team_key = ""

        players = txn.get("players", {})
        for key, player_data in players.items():
            if key == "count" or not isinstance(player_data, dict):
                continue

            player_info, tdata = _extract_player_info(player_data)
            ptype = tdata.get("type", "")
            pos_str = ", ".join(player_info["positions"]) if player_info["positions"] else None

            if ptype == "add":
                add_name = player_info["name"]
                add_pid = player_info["player_id"]
                add_pos = pos_str
                team_name = tdata.get("destination_team_name", team_name)
                team_key = tdata.get("destination_team_key", team_key)
            elif ptype == "drop":
                drop_name = player_info["name"]
                drop_pid = player_info["player_id"]
                drop_pos = pos_str
                if not team_name:
                    team_name = tdata.get("source_team_name", "")
                    team_key = tdata.get("source_team_key", "")

        results.append({
            "transaction_id": txn.get("transaction_id", ""),
            "type": txn.get("type", ""),
            "status": txn.get("status", ""),
            "timestamp": txn.get("timestamp", ""),
            "team": team_name,
            "team_key": team_key,
            "add_name": add_name,
            "add_player_id": add_pid,
            "add_positions": add_pos,
            "drop_name": drop_name,
            "drop_player_id": drop_pid,
            "drop_positions": drop_pos,
        })

    return results
