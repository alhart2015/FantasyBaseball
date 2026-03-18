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
from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings, fetch_free_agents, fetch_scoring_period
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.lineup.waivers import scan_waivers
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher
from fantasy_baseball.data.mlb_schedule import get_week_schedule

from datetime import datetime as dt
import pandas as pd

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "weekly_schedule.json"

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


def print_probable_starters(
    roster_pitchers: list[pd.Series],
    schedule: dict | None,
) -> None:
    """Print probable starter matchups, flagging two-start pitchers."""
    if not schedule or not roster_pitchers:
        return

    probable = schedule.get("probable_pitchers", [])
    if not probable:
        print("  No probable pitcher data available.")
        return

    # Build pitcher name -> list of starts
    pitcher_starts: dict[str, list[dict]] = {}
    roster_names = {normalize_name(p["name"]) for p in roster_pitchers}

    for game in probable:
        for side, team_key in [("away", "away_team"), ("home", "home_team")]:
            pitcher_name = game.get(f"{side}_pitcher", "TBD")
            if pitcher_name == "TBD":
                continue
            if normalize_name(pitcher_name) not in roster_names:
                continue

            opponent_key = "home_team" if side == "away" else "away_team"
            indicator = "@" if side == "away" else "vs"
            try:
                day = dt.strptime(game["date"], "%Y-%m-%d").strftime("%a")
            except (ValueError, KeyError):
                day = "?"

            if pitcher_name not in pitcher_starts:
                pitcher_starts[pitcher_name] = []
            pitcher_starts[pitcher_name].append({
                "day": day,
                "indicator": indicator,
                "opponent": game[opponent_key],
            })

    if not pitcher_starts:
        print("  No roster pitchers found in probable starters.")
        return

    two_start = {k: v for k, v in pitcher_starts.items() if len(v) >= 2}
    one_start = {k: v for k, v in pitcher_starts.items() if len(v) == 1}

    if two_start:
        print("  ** TWO-START PITCHERS **")
        for name, starts in sorted(two_start.items()):
            matchups = ", ".join(
                f"{s['day']} {s['indicator']} {s['opponent']}" for s in starts
            )
            print(f"    {name:<25} {matchups}")

    if one_start:
        print("  SINGLE START")
        for name, starts in sorted(one_start.items()):
            s = starts[0]
            print(f"    {name:<25} {s['day']} {s['indicator']} {s['opponent']}")

    # Roster pitchers with no announced start
    announced = {normalize_name(k) for k in pitcher_starts.keys()}
    unannounced = [
        p["name"] for p in roster_pitchers
        if normalize_name(p["name"]) not in announced
        and p.get("player_type") == "pitcher"
    ]
    if unannounced:
        print("  NO START ANNOUNCED")
        for name in sorted(unannounced):
            print(f"    {name:<25} TBD")


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

    # Fetch scoring period and MLB schedule
    print("Fetching weekly schedule...")
    period_start, period_end = fetch_scoring_period(league)
    schedule = get_week_schedule(period_start, period_end, SCHEDULE_PATH)
    games_per_team = schedule["games_per_team"] if schedule else {}
    if schedule:
        print(f"Scoring period: {period_start} to {period_end}")
        print(f"Schedule loaded for {len(games_per_team)} teams")
    else:
        print("Schedule unavailable — using default 6 games/week")
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
        games_this_week = DEFAULT_GAMES_PER_WEEK  # updated after projection match

        # Look up hitting and pitching projections separately so two-way
        # players get the correct projection for each role.
        hit_proj = None
        if is_hitter(positions) and not hitters_proj.empty:
            matches = hitters_proj[hitters_proj["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                hit_proj = matches.iloc[0].copy()
                hit_proj["positions"] = positions
                hit_proj["player_type"] = "hitter"
                team = hit_proj.get("team", "")
                games_this_week = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                hit_proj = scale_by_schedule(hit_proj, games_this_week)
                roster_hitters.append(hit_proj)

        pit_proj = None
        if is_pitcher(positions) and not pitchers_proj.empty:
            matches = pitchers_proj[pitchers_proj["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                pit_proj = matches.iloc[0].copy()
                pit_proj["positions"] = positions
                pit_proj["player_type"] = "pitcher"
                team = pit_proj.get("team", "")
                games_this_week = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                pit_proj = scale_by_schedule(pit_proj, games_this_week)
                roster_pitchers.append(pit_proj)

        if hit_proj is None and pit_proj is None:
            # Fallback: try either projection source for players whose
            # position list doesn't clearly indicate hitter vs pitcher.
            for df, ptype, dest in [
                (hitters_proj, "hitter", roster_hitters),
                (pitchers_proj, "pitcher", roster_pitchers),
            ]:
                if df.empty:
                    continue
                matches = df[df["name"].apply(normalize_name) == name_norm]
                if not matches.empty:
                    proj_row = matches.iloc[0].copy()
                    proj_row["positions"] = positions
                    proj_row["player_type"] = ptype
                    team = proj_row.get("team", "")
                    games_this_week = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                    proj_row = scale_by_schedule(proj_row, games_this_week)
                    dest.append(proj_row)
                    break

    print(f"Matched: {len(roster_hitters)} hitters, {len(roster_pitchers)} pitchers")
    print()

    # Optimize hitter lineup
    if roster_hitters:
        print("=" * 60)
        print("OPTIMAL HITTER LINEUP")
        print("=" * 60)
        lineup = optimize_hitter_lineup(roster_hitters, leverage,
                                        roster_slots=config.roster_slots)

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

    # Probable starters display
    if roster_pitchers:
        period_label = f"{period_start} to {period_end}" if schedule else ""
        print("=" * 60)
        print(f"PROBABLE STARTERS THIS WEEK ({period_label})")
        print("=" * 60)
        print_probable_starters(roster_pitchers, schedule)
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
                    team = proj_row.get("team", "")
                    fa_games = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                    proj_row = scale_by_schedule(proj_row, fa_games)
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
