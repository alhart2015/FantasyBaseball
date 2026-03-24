"""Fetch player position eligibility from Yahoo and cache to JSON.

Run this once before draft day:
    python scripts/fetch_positions.py
"""
from pathlib import Path
from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.yahoo_players import (
    fetch_missing_keepers,
    fetch_positions_from_yahoo,
    save_positions_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
CACHE_PATH = PROJECT_ROOT / "data" / "player_positions.json"


def main():
    config = load_config(CONFIG_PATH)
    print(f"Connecting to Yahoo Fantasy (league {config.league_id})...")
    session = get_yahoo_session()
    league = get_league(session, league_id=config.league_id, game_key=config.game_code)
    print("Fetching player positions from team rosters + free agents...")
    print("  (this queries each team's roster, then free agents by position)")
    positions = fetch_positions_from_yahoo(league)
    print(f"Fetched {len(positions)} players from rosters + free agents")

    # Phase 3: Look up any keepers missing from the cache
    keepers = config.keepers
    missing = [k["name"] for k in keepers if k["name"] not in positions]
    if missing:
        print(f"  {len(missing)} keepers missing — looking up via player search...")
        for name in missing:
            print(f"    searching: {name}")
        keeper_positions = fetch_missing_keepers(league, keepers, positions)
        positions.update(keeper_positions)
        found = len(keeper_positions)
        still_missing = [k["name"] for k in keepers if k["name"] not in positions]
        print(f"  Found {found}/{len(missing)} missing keepers")
        if still_missing:
            print(f"  WARNING: still missing: {still_missing}")
    else:
        print("  All keepers already in position cache")

    save_positions_cache(positions, CACHE_PATH)
    print(f"Cached {len(positions)} players to {CACHE_PATH}")

    # Show position coverage stats
    pos_counts = {}
    for name, pos_list in positions.items():
        for p in pos_list:
            pos_counts[p] = pos_counts.get(p, 0) + 1
    print("Position coverage:")
    for p in sorted(pos_counts, key=pos_counts.get, reverse=True):
        print(f"  {p:>4}: {pos_counts[p]} players")


if __name__ == "__main__":
    main()
