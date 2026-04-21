import json
import logging
from pathlib import Path
from typing import cast

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


def _extract_positions(player: dict) -> list[str]:
    """Extract position list from a player_details result."""
    raw = player.get("eligible_positions", [])
    positions = []
    for p in raw:
        pos = p["position"] if isinstance(p, dict) else p
        if pos not in positions:
            positions.append(pos)
    return positions


def fetch_missing_keepers(
    league,
    keepers: list[dict],
    existing: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Look up position eligibility for keepers missing from the cache.

    Uses league.player_details() to search by name. Keepers already
    present in *existing* are skipped.

    Results are stored under the keeper's config name (not Yahoo's
    accented name) so that downstream name matching works.

    Players split into batter/pitcher (e.g. Ohtani) produce two
    entries: "Name" for the batter and "Name (Pitcher)" for the
    pitcher half.
    """
    missing_positions: dict[str, list[str]] = {}
    for keeper in keepers:
        name = keeper["name"]
        if name in existing:
            continue
        try:
            results = league.player_details(name)
            if not results:
                logger.warning("No Yahoo results for keeper %r", name)
                continue

            if len(results) >= 2 and _is_batter_pitcher_split(results):
                # Two-way player split (e.g. Ohtani Batter + Pitcher)
                for player in results:
                    positions = _extract_positions(player)
                    if not positions:
                        continue
                    yahoo_name = _get_full_name(player, name)
                    if "(Pitcher)" in yahoo_name or "(pitcher)" in yahoo_name:
                        store_name = f"{name} (Pitcher)"
                    else:
                        store_name = name
                    missing_positions[store_name] = positions
                    logger.info("Found split player %s: %s", store_name, positions)
            else:
                player = results[0]
                positions = _extract_positions(player)
                if positions:
                    # Store under the config name, not Yahoo's accented name
                    missing_positions[name] = positions
                    logger.info("Found keeper %s: %s", name, positions)
                else:
                    logger.warning("Keeper %r found but no positions listed", name)
        except Exception:
            logger.exception("Failed to look up keeper %r", name)
            continue
    return missing_positions


def _get_full_name(player: dict, fallback: str) -> str:
    """Extract the full name string from a player_details result."""
    full_name = player.get("name", {})
    if isinstance(full_name, dict):
        return cast(str, full_name.get("full", fallback))
    return full_name if full_name else fallback


def _is_batter_pitcher_split(results: list[dict]) -> bool:
    """Check if results represent a batter/pitcher split (e.g. Ohtani)."""
    names = []
    for r in results:
        n = _get_full_name(r, "")
        names.append(n.lower())
    has_batter = any("batter" in n or "hitter" in n for n in names)
    has_pitcher = any("pitcher" in n for n in names)
    return has_batter and has_pitcher


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
        return cast(dict[str, list[str]], json.load(f))
