import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

YAHOO_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]


def fetch_positions_from_yahoo(league) -> dict[str, list[str]]:
    """Fetch player position eligibility from Yahoo Fantasy API.

    Queries both rostered players (via team rosters) and free agents
    to build a comprehensive position map.
    """
    position_maps = []

    # Phase 1: Fetch rostered players from all teams
    try:
        teams = league.teams()
        for team_key in teams:
            try:
                team = league.to_team(team_key)
                roster = team.roster()
                pos_map = {}
                for player in roster:
                    name = player["name"]
                    eligible = player.get("eligible_positions", [])
                    if eligible:
                        pos_map[name] = eligible
                position_maps.append(pos_map)
            except Exception:
                logger.exception(
                    "Failed to fetch roster for team %s; skipping", team_key
                )
                continue
    except Exception:
        logger.exception("Failed to fetch teams; skipping rostered players")

    # Phase 2: Fetch free agents by position
    for pos in YAHOO_POSITIONS:
        try:
            agents = league.free_agents(pos)
            pos_map = {}
            for player in agents:
                name = player["name"]
                eligible = player.get("eligible_positions", [pos])
                pos_map[name] = eligible
            position_maps.append(pos_map)
        except (PermissionError, OSError) as exc:
            # Auth failures and critical OS-level errors should not be hidden
            logger.exception(
                "Critical error fetching free agents for position %s", pos
            )
            raise
        except Exception:
            # Transient network errors, rate limits, etc. — log and skip
            logger.exception(
                "Failed to fetch free agents for position %s; skipping", pos
            )
            continue
    return merge_position_maps(position_maps)


def merge_position_maps(maps: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Merge multiple position maps into one, deduplicating positions."""
    merged: dict[str, list[str]] = {}
    for pos_map in maps:
        for name, positions in pos_map.items():
            if name not in merged:
                merged[name] = []
            for pos in positions:
                if pos not in merged[name]:
                    merged[name].append(pos)
    return merged


def save_positions_cache(positions: dict[str, list[str]], path: Path) -> None:
    """Save position data to a JSON cache file."""
    with open(path, "w") as f:
        json.dump(positions, f, indent=2)


def load_positions_cache(path: Path) -> dict[str, list[str]]:
    """Load position data from a JSON cache file."""
    if not path.exists():
        raise FileNotFoundError(f"Position cache not found: {path}")
    with open(path) as f:
        return json.load(f)
