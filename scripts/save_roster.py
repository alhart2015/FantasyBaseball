"""Save a weekly roster snapshot from Yahoo.

Usage:
    python scripts/save_roster.py                # auto-detect week number
    python scripts/save_roster.py --week 3       # explicit week number
    python scripts/save_roster.py --week 0       # pre-season snapshot

Connects to Yahoo, fetches the current roster for the user's team,
and saves it to data/rosters/<date>_hart_roster.json.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.lineup.yahoo_roster import fetch_roster
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_pitcher

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
ROSTERS_DIR = PROJECT_ROOT / "data" / "rosters"


def main():
    parser = argparse.ArgumentParser(description="Save weekly roster snapshot from Yahoo")
    parser.add_argument("--week", type=int, default=None,
                        help="Week number (default: auto-detect from Yahoo)")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    team_name = config.team_name

    print(f"Connecting to Yahoo...")
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
        try:
            week_num = league.current_week()
        except Exception:
            print("Could not auto-detect week. Use --week N to specify.")
            sys.exit(1)

    print(f"Fetching roster for {team_name} (week {week_num})...")
    raw_roster = fetch_roster(league, user_team_key)

    # Build snapshot
    roster = []
    for p in raw_roster:
        positions = p.get("positions", [])
        roster.append({"name": p["name"], "positions": positions})

    snapshot = {
        "snapshot_date": date.today().isoformat(),
        "week_num": week_num,
        "team": team_name,
        "league": config.league_id,
        "roster": roster,
    }

    # Save
    ROSTERS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{date.today().isoformat()}_hart_roster.json"
    out_path = ROSTERS_DIR / filename
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Saved {len(roster)} players to {out_path}")
    n_p = sum(1 for p in roster if is_pitcher(p["positions"]))
    print(f"  {len(roster) - n_p} hitters, {n_p} pitchers")


if __name__ == "__main__":
    main()
