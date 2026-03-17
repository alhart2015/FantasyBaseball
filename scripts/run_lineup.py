"""In-Season Lineup Optimizer for Fantasy Baseball.

Usage:
    python scripts/run_lineup.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings, fetch_free_agents
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.lineup.waivers import scan_waivers
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher

import pandas as pd

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"

# Default games per week if not specified
DEFAULT_GAMES_PER_WEEK = 6
GAMES_PER_SEASON = 162
WEEKS_PER_SEASON = 27


def scale_by_schedule(player: pd.Series, games_this_week: int) -> pd.Series:
    """Scale counting stat projections by weekly game count.

    Full-season projections assume ~6 games/week average.
    If a player's team plays 7 games, they get proportionally more counting stats.
    Rate stats (AVG, ERA, WHIP) are NOT scaled.
    """
    if games_this_week == DEFAULT_GAMES_PER_WEEK:
        return player

    scale_factor = games_this_week / DEFAULT_GAMES_PER_WEEK
    scaled = player.copy()

    if player.get("player_type") == "hitter":
        for col in ["r", "hr", "rbi", "sb", "ab", "h"]:
            if col in scaled.index:
                scaled[col] = scaled[col] * scale_factor
    elif player.get("player_type") == "pitcher":
        for col in ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]:
            if col in scaled.index:
                scaled[col] = scaled[col] * scale_factor

    return scaled


def main():
    config = load_config(CONFIG_PATH)
    print(f"Lineup Optimizer | {config.team_name}")
    print()

    # Connect to Yahoo
    print("Connecting to Yahoo...")
    session = get_yahoo_session()
    league = get_league(session, league_id=config.league_id, game_key=config.game_code)

    # Find user's team key
    teams = league.teams()
    user_team_key = None
    for key, team in teams.items():
        if normalize_name(team["name"]) == normalize_name(config.team_name):
            user_team_key = key
            break
    if not user_team_key:
        print(f"Could not find team '{config.team_name}' in league")
        sys.exit(1)

    # Fetch roster and standings
    print("Fetching roster and standings...")
    roster = fetch_roster(league, user_team_key)
    standings = fetch_standings(league)

    print(f"Roster: {len(roster)} players")
    print(f"Standings: {len(standings)} teams")
    print()

    # Calculate leverage
    leverage = calculate_leverage(standings, config.team_name)
    print("CATEGORY LEVERAGE (higher = more valuable to target):")
    sorted_lev = sorted(leverage.items(), key=lambda x: x[1], reverse=True)
    for cat, weight in sorted_lev:
        bar = "#" * int(weight * 100)
        print(f"  {cat:>4}: {weight:.3f} {bar}")
    print()

    # Load projections and match to roster
    print("Loading projections...")
    weights = config.projection_weights if config.projection_weights else None
    hitters_proj, pitchers_proj = blend_projections(
        PROJECTIONS_DIR, config.projection_systems, weights,
    )
    positions_cache = load_positions_cache(POSITIONS_PATH)
    norm_positions = {normalize_name(k): v for k, v in positions_cache.items()}

    # Match roster players to projections
    roster_hitters = []
    roster_pitchers = []
    for player in roster:
        name = player["name"]
        name_norm = normalize_name(name)
        positions = player["positions"]

        proj_row = None
        for df in [hitters_proj, pitchers_proj]:
            if df.empty:
                continue
            matches = df[df["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                proj_row = matches.iloc[0].copy()
                break

        if proj_row is None:
            continue

        proj_row["positions"] = positions

        if is_hitter(positions):
            roster_hitters.append(proj_row)
        if is_pitcher(positions):
            roster_pitchers.append(proj_row)

    print(f"Matched: {len(roster_hitters)} hitters, {len(roster_pitchers)} pitchers")
    print()

    # Optimize hitter lineup
    if roster_hitters:
        print("=" * 60)
        print("OPTIMAL HITTER LINEUP")
        print("=" * 60)
        lineup = optimize_hitter_lineup(roster_hitters, leverage)

        # Build lookup for reasoning (Gap 2 fix)
        hitter_wsgp = {}
        for h in roster_hitters:
            hitter_wsgp[h["name"]] = calculate_weighted_sgp(h, leverage)

        starters = set(lineup.values())
        bench_hitters = [h for h in roster_hitters if h["name"] not in starters]

        for slot, name in sorted(lineup.items()):
            wsgp = hitter_wsgp.get(name, 0)
            line = f"  {slot:<8} {name:<25} wSGP: {wsgp:.2f}"

            # For flex slots (IF, UTIL), show reasoning vs best bench player
            if any(slot.startswith(prefix) for prefix in ("IF", "UTIL")):
                if bench_hitters:
                    best_bench = max(bench_hitters, key=lambda h: hitter_wsgp.get(h["name"], 0))
                    bench_wsgp = hitter_wsgp.get(best_bench["name"], 0)
                    delta = wsgp - bench_wsgp
                    if delta > 0:
                        # Find top contributing categories
                        top_cats = _top_category_reasons(
                            roster_hitters, name, best_bench["name"], leverage
                        )
                        line += f"  (over {best_bench['name']}: +{delta:.2f} — {top_cats})"

            print(line)

        if bench_hitters:
            print("  BENCH:")
            for h in sorted(bench_hitters, key=lambda h: hitter_wsgp.get(h["name"], 0), reverse=True):
                print(f"    {h['name']:<25} wSGP: {hitter_wsgp.get(h['name'], 0):.2f}")
        print()

    # Optimize pitcher lineup (Gap 5 fix: use config for slot count)
    pitcher_slots = config.roster_slots.get("P", 9)
    if roster_pitchers:
        print("=" * 60)
        print("OPTIMAL PITCHER LINEUP")
        print("=" * 60)
        starters, bench = optimize_pitcher_lineup(
            roster_pitchers, leverage, slots=pitcher_slots
        )
        print("  START:")
        for p in starters:
            print(f"    {p['name']:<25} wSGP: {p['wsgp']:.2f}")
        if bench:
            print("  BENCH:")
            for p in bench:
                print(f"    {p['name']:<25} wSGP: {p['wsgp']:.2f}")
        print()

    # Waiver wire recommendations (Gap 1 fix)
    print("=" * 60)
    print("WAIVER WIRE RECOMMENDATIONS")
    print("=" * 60)
    all_roster = roster_hitters + roster_pitchers
    if all_roster:
        # Fetch free agents for key positions
        print("Scanning free agents...")
        fa_players = []
        for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
            fas = fetch_free_agents(league, pos, count=25)
            for fa in fas:
                fa_name_norm = normalize_name(fa["name"])
                # Match to projections
                proj_row = None
                for df in [hitters_proj, pitchers_proj]:
                    if df.empty:
                        continue
                    matches = df[df["name"].apply(normalize_name) == fa_name_norm]
                    if not matches.empty:
                        proj_row = matches.iloc[0].copy()
                        break
                if proj_row is not None:
                    proj_row["positions"] = fa["positions"]
                    fa_players.append(proj_row)

        recommendations = scan_waivers(all_roster, fa_players, leverage, max_results=5)
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. ADD {rec['add']:<20} DROP {rec['drop']:<20} "
                      f"gain: +{rec['sgp_gain']:.2f} wSGP")
                # Show category impact
                cats = rec.get("categories", {})
                gains = [f"+{cat}" for cat, v in cats.items() if v > 0.01]
                losses = [f"-{cat}" for cat, v in cats.items() if v < -0.01]
                if gains or losses:
                    print(f"     gains {', '.join(gains)}  |  costs {', '.join(losses)}")
        else:
            print("  No positive-value pickups found.")
    print()

    print("Done! Update your lineup on Yahoo based on these recommendations.")


def _top_category_reasons(
    hitters: list[pd.Series],
    starter_name: str,
    bench_name: str,
    leverage: dict[str, float],
) -> str:
    """Return a short string like 'gains HR, RBI' explaining why starter > bench."""
    from fantasy_baseball.sgp.denominators import get_sgp_denominators
    from fantasy_baseball.sgp.player_value import calculate_counting_sgp

    denoms = get_sgp_denominators()

    starter = next((h for h in hitters if h["name"] == starter_name), None)
    bench = next((h for h in hitters if h["name"] == bench_name), None)
    if starter is None or bench is None:
        return ""

    deltas = {}
    for stat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
        weight = leverage.get(stat, 0)
        if weight > 0:
            s_sgp = calculate_counting_sgp(starter.get(col, 0), denoms[stat]) * weight
            b_sgp = calculate_counting_sgp(bench.get(col, 0), denoms[stat]) * weight
            deltas[stat] = s_sgp - b_sgp

    # Return top 2 positive categories
    top = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
    gains = [cat for cat, d in top if d > 0][:2]
    return f"gains {', '.join(gains)}" if gains else "marginal"


if __name__ == "__main__":
    main()
