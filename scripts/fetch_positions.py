"""Fetch player position eligibility from Yahoo and cache to JSON.

Run this once before draft day:
    python scripts/fetch_positions.py
"""
from pathlib import Path
from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.yahoo_players import (
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
    print("Fetching player positions (this may take a minute)...")
    positions = fetch_positions_from_yahoo(league)
    save_positions_cache(positions, CACHE_PATH)
    print(f"Cached {len(positions)} players to {CACHE_PATH}")


if __name__ == "__main__":
    main()
