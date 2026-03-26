"""Trade Recommender — find mutually beneficial 1-for-1 trades.

Usage:
    python scripts/run_trades.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.trades.evaluate import find_trades, compute_roto_points_by_cat
from fantasy_baseball.trades.pitch import generate_pitch
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"


def match_roster_to_projections(roster, hitters_proj, pitchers_proj):
    """Match roster player names to projection stats. Returns enriched player dicts."""
    matched = []
    for player in roster:
        name_norm = normalize_name(player["name"])
        positions = player["positions"]

        proj = None
        ptype = None
        if is_hitter(positions) and not hitters_proj.empty:
            matches = hitters_proj[hitters_proj["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = "hitter"
        if proj is None and is_pitcher(positions) and not pitchers_proj.empty:
            matches = pitchers_proj[pitchers_proj["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = "pitcher"
        if proj is None:
            for df, pt in [(hitters_proj, "hitter"), (pitchers_proj, "pitcher")]:
                if df.empty:
                    continue
                matches = df[df["name"].apply(normalize_name) == name_norm]
                if not matches.empty:
                    proj = matches.iloc[0]
                    ptype = pt
                    break

        if proj is not None:
            entry = {"name": player["name"], "positions": positions, "player_type": ptype}
            if ptype == "hitter":
                for col in ["r", "hr", "rbi", "sb", "avg", "h", "ab", "pa"]:
                    entry[col] = float(proj.get(col, 0) or 0)
            else:
                for col in ["w", "k", "sv", "era", "whip", "ip", "er", "bb", "h_allowed"]:
                    entry[col] = float(proj.get(col, 0) or 0)
            matched.append(entry)
    return matched


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
    weights = config.projection_weights if config.projection_weights else None
    hitters_proj, pitchers_proj = blend_projections(
        PROJECTIONS_DIR / str(config.season_year), config.projection_systems, weights,
    )

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

    current_ranks = compute_roto_points_by_cat(standings)

    print("Evaluating trades...")
    trades = find_trades(
        hart_name=config.team_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        standings=standings,
        leverage_by_team=leverage_by_team,
        roster_slots=config.roster_slots,
        max_results=5,
    )

    print()
    print("=" * 70)
    print(f"TOP {len(trades)} TRADE PROPOSALS")
    print("=" * 70)

    if not trades:
        print("\nNo mutually beneficial trades found.")
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

        opp_ranks = current_ranks.get(opp, {})
        pitch = generate_pitch(opp, trade["opp_cat_deltas"], opp_ranks)
        print(f"\n   Pitch: \"{pitch}\"")

    print()
    print("Done! Review proposals and propose via Yahoo.")


if __name__ == "__main__":
    main()
