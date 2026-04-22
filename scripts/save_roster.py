"""Save a weekly roster snapshot from Yahoo.

Usage:
    python scripts/save_roster.py                # auto-detect week number
    python scripts/save_roster.py --week 3       # explicit week number
    python scripts/save_roster.py --week 0       # pre-season snapshot

Connects to Yahoo, fetches the current roster for the user's team,
and saves it to data/rosters/<monday>_hart_roster.json (keyed to Monday
of the current week so each week gets one file).
"""

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
from fantasy_baseball.config import load_config
from fantasy_baseball.lineup.yahoo_roster import fetch_roster
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_pitcher

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
ROSTERS_DIR = PROJECT_ROOT / "data" / "rosters"


def main():
    parser = argparse.ArgumentParser(description="Save weekly roster snapshot from Yahoo")
    parser.add_argument(
        "--week", type=int, default=None, help="Week number (default: auto-detect from Yahoo)"
    )
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    team_name = config.team_name

    print("Connecting to Yahoo...")
    session = get_yahoo_session()
    league = get_league(session, config.league_id, config.game_code)

    # Find user's team key
    teams = league.teams()
    user_team_key = None
    for key, td in teams.items():
        if normalize_name(td["name"]) == normalize_name(team_name):
            user_team_key = key
            break

    if user_team_key is None:
        print(f"Could not find team '{team_name}' in league {config.league_id}")
        sys.exit(1)

    # Determine week number
    week_num = args.week
    if week_num is None:
        # Roto leagues don't have scoring weeks, so derive from calendar.
        # MLB opening day is typically late March; use weeks since then.
        today = date.today()
        opening_day = date(today.year, 3, 20)  # approximate
        week_num = max(0, (today - opening_day).days // 7)

    print(f"Fetching roster for {team_name} (week {week_num})...")
    raw_roster = fetch_roster(league, user_team_key)

    # Build positional mapping: slot -> {name, positions}
    roster = {}
    for p in raw_roster:
        slot = p.get("selected_position", "BN")
        entry = {"name": p["name"], "positions": p.get("positions", [])}

        # Handle multiple players in the same slot (e.g., OF has 4 slots)
        if slot in roster:
            # Append a number to make the key unique: OF -> OF2, OF3, etc.
            i = 2
            while f"{slot}{i}" in roster:
                i += 1
            slot = f"{slot}{i}"
        roster[slot] = entry

    # Use Monday of the current week as the snapshot date
    today = date.today()
    monday = today - timedelta(days=today.weekday())

    snapshot = {
        "snapshot_date": monday.isoformat(),
        "week_num": week_num,
        "team": team_name,
        "league": config.league_id,
        "roster": roster,
    }

    # Save
    ROSTERS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{monday.isoformat()}_hart_roster.json"
    out_path = ROSTERS_DIR / filename

    if out_path.exists():
        answer = input(f"  {out_path} already exists. Overwrite? [y/N] ")
        if answer.lower() != "y":
            print("  Skipped.")
            return

    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    n_players = len(roster)
    n_p = sum(1 for p in roster.values() if is_pitcher(p["positions"]))
    print(f"Saved {n_players} players to {out_path}")
    print(f"  {n_players - n_p} hitters, {n_p} pitchers")


if __name__ == "__main__":
    main()
