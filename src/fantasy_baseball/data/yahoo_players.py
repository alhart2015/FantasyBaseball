import json
from pathlib import Path

YAHOO_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]


def fetch_positions_from_yahoo(league) -> dict[str, list[str]]:
    """Fetch player position eligibility from Yahoo Fantasy API."""
    position_maps = []
    for pos in YAHOO_POSITIONS:
        try:
            agents = league.free_agents(pos)
            pos_map = {}
            for player in agents:
                name = player["name"]
                eligible = player.get("eligible_positions", [pos])
                pos_map[name] = eligible
            position_maps.append(pos_map)
        except Exception:
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
