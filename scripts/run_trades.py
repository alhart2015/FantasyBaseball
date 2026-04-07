"""Trade Recommender — find mutually beneficial 1-for-1 trades.

Usage:
    python scripts/run_trades.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection, get_blended_projections
from fantasy_baseball.data.projections import match_roster_to_projections
from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.sgp.rankings import compute_combined_sgp_rankings
from fantasy_baseball.trades.evaluate import find_trades
from fantasy_baseball.trades.pitch import generate_pitch
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"


def main():
    config = load_config(CONFIG_PATH)
    print(f"Trade Recommender | {config.team_name}")
    print()

    print("Connecting to Yahoo...")
    session = get_yahoo_session()
    league = get_league(session, league_id=config.league_id, game_key=config.game_code)

    teams = league.teams()
    user_team_key = None
    for key, team in teams.items():
        if normalize_name(team["name"]) == normalize_name(config.team_name):
            user_team_key = key
            break
    if not user_team_key:
        print(f"Could not find team '{config.team_name}' in league")
        sys.exit(1)

    print("Fetching standings...")
    standings = fetch_standings(league)
    print(f"Standings: {len(standings)} teams")

    print("Fetching all team rosters...")
    all_rosters_raw = {}
    for key, team in teams.items():
        name = team["name"]
        roster = fetch_roster(league, key)
        all_rosters_raw[name] = roster
        print(f"  {name}: {len(roster)} players")
    print()

    print("Loading projections...")
    conn = get_connection()
    hitters_proj, pitchers_proj = get_blended_projections(conn)
    conn.close()
    hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
    pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

    hart_roster = match_roster_to_projections(
        all_rosters_raw[config.team_name], hitters_proj, pitchers_proj,
    )
    opp_rosters = {}
    for name, roster in all_rosters_raw.items():
        if name == config.team_name:
            continue
        opp_rosters[name] = match_roster_to_projections(roster, hitters_proj, pitchers_proj)

    print(f"Hart roster: {len(hart_roster)} matched")
    for name, roster in opp_rosters.items():
        print(f"  {name}: {len(roster)} matched")
    print()

    print("Computing leverage for all teams...")
    leverage_by_team = {}
    for team in standings:
        leverage_by_team[team["name"]] = calculate_leverage(standings, team["name"])

    rankings = compute_combined_sgp_rankings(hitters_proj, pitchers_proj)

    print("Evaluating trades...")
    trades = find_trades(
        hart_name=config.team_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        standings=standings,
        leverage_by_team=leverage_by_team,
        roster_slots=config.roster_slots,
        rankings=rankings,
        max_results=5,
    )

    print()
    print("=" * 70)
    print(f"TOP {len(trades)} TRADE PROPOSALS")
    print("=" * 70)

    if not trades:
        print("\nNo trades found.")
        return

    for i, trade in enumerate(trades, 1):
        opp = trade["opponent"]
        send_pos = "/".join(trade["send_positions"][:2])
        recv_pos = "/".join(trade["receive_positions"][:2])

        print(f"\n{i}. SEND: {trade['send']:<22} ({send_pos})  ->  {opp}")
        print(f"   GET:  {trade['receive']:<22} ({recv_pos})  <-  {opp}")

        hart_parts = [f"{d:+d} {c}" for c, d in trade["hart_cat_deltas"].items() if d != 0]
        opp_parts = [f"{d:+d} {c}" for c, d in trade["opp_cat_deltas"].items() if d != 0]
        print(f"\n   Hart gains: {trade['hart_delta']:+d} roto pts ({', '.join(hart_parts) if hart_parts else 'no change'})")
        print(f"   They gain:  {trade['opp_delta']:+d} roto pts ({', '.join(opp_parts) if opp_parts else 'no change'})")

        pitch = generate_pitch(
            send_rank=trade.get("send_rank", 0),
            receive_rank=trade.get("receive_rank", 0),
            send_positions=trade.get("send_positions", []),
            receive_positions=trade.get("receive_positions", []),
        )
        print(f"\n   Pitch: \"{pitch}\"")

    print()
    print("Done! Review proposals and propose via Yahoo.")


if __name__ == "__main__":
    main()
