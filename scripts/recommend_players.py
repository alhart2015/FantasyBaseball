"""Recommend top available players at a position by leverage-weighted SGP.

Usage:
    python scripts/recommend_players.py -p SP -n 10
    python scripts/recommend_players.py --position RP --count 5
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection, get_blended_projections
from fantasy_baseball.lineup.yahoo_roster import fetch_free_agents, fetch_standings
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_pitcher

import pandas as pd

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"

PITCHER_POSITIONS = {"SP", "RP", "P"}
HITTER_STATS = [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb"), ("AVG", "avg")]
PITCHER_STATS = [("W", "w"), ("K", "k"), ("SV", "sv"), ("ERA", "era"), ("WHIP", "whip")]


def main():
    parser = argparse.ArgumentParser(
        description="Recommend top available players at a position by wSGP",
    )
    parser.add_argument(
        "-p", "--position", required=True,
        help="Position to search (e.g. SP, RP, C, 1B, 2B, 3B, SS, OF)",
    )
    parser.add_argument(
        "-n", "--count", type=int, default=10,
        help="Number of players to show (default: 10)",
    )
    args = parser.parse_args()
    position = args.position.upper()
    count = args.count

    config = load_config(CONFIG_PATH)

    # Connect to Yahoo
    print("Connecting to Yahoo...")
    session = get_yahoo_session()
    league = get_league(session, league_id=config.league_id, game_key=config.game_code)

    # Standings → leverage
    print("Fetching standings...")
    standings = fetch_standings(league)
    leverage = calculate_leverage(standings, config.team_name)

    # Load projections
    print("Loading projections...")
    conn = get_connection()
    hitters_proj, pitchers_proj = get_blended_projections(conn)
    conn.close()

    # Fetch available players at position
    print(f"Fetching available {position} players...")
    available = fetch_free_agents(league, position, count=200)
    if not available:
        print(f"No available players found at {position}.")
        sys.exit(0)

    # Precompute normalized names for projection matching
    if not hitters_proj.empty:
        hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
    if not pitchers_proj.empty:
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

    # Match to projections and score
    is_pitcher_pos = position in PITCHER_POSITIONS
    if is_pitcher_pos:
        search_order = [pitchers_proj, hitters_proj]
    else:
        search_order = [hitters_proj, pitchers_proj]

    scored = []
    seen_names: set[str] = set()
    for player in available:
        name_norm = normalize_name(player["name"])
        if name_norm in seen_names:
            continue
        seen_names.add(name_norm)

        proj_row = None
        for df in search_order:
            if df.empty:
                continue
            matches = df[df["_name_norm"] == name_norm]
            if not matches.empty:
                proj_row = matches.iloc[0].copy()
                break

        if proj_row is None:
            continue

        # RP search = looking for closers; skip pitchers without meaningful saves
        sv = proj_row.get("sv", 0)
        if position == "RP" and (pd.isna(sv) or sv < 1):
            continue

        if "positions" not in proj_row.index:
            proj_row["positions"] = player["positions"]

        wsgp = calculate_weighted_sgp(proj_row, leverage)
        scored.append({"name": player["name"], "proj": proj_row, "wsgp": wsgp})

    scored.sort(key=lambda x: x["wsgp"], reverse=True)
    top = scored[:count]

    if not top:
        print(f"No projected players found at {position}.")
        sys.exit(0)

    # Display
    stat_cols = PITCHER_STATS if is_pitcher_pos else HITTER_STATS
    print()
    print(f"Top {len(top)} available {position} (by wSGP, leverage-weighted)")
    print(f"{'':>4} {'Name':<25} {'Team':<5}", end="")
    for label, _ in stat_cols:
        print(f" {label:>6}", end="")
    print(f" {'wSGP':>7}")
    print("-" * (42 + 7 * len(stat_cols) + 8))

    for i, entry in enumerate(top, 1):
        proj = entry["proj"]
        team = proj.get("team", "")
        print(f"  {i:>2}. {entry['name']:<25} {team:<5}", end="")
        for _, col in stat_cols:
            val = proj.get(col, 0)
            if col in ("avg",):
                print(f" {val:>6.3f}", end="")
            elif col in ("era", "whip"):
                print(f" {val:>6.2f}", end="")
            else:
                print(f" {val:>6.0f}", end="")
        print(f" {entry['wsgp']:>7.2f}")

    # Show leverage context
    print()
    print("Current leverage weights:")
    sorted_lev = sorted(leverage.items(), key=lambda x: x[1], reverse=True)
    for cat, weight in sorted_lev:
        bar = "#" * int(weight * 50)
        print(f"  {cat:>4}: {weight:.3f} {bar}")


if __name__ == "__main__":
    main()
