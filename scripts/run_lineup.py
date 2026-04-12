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
from fantasy_baseball.data.db import get_connection, get_blended_projections, get_positions
from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings, fetch_scoring_period
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
from fantasy_baseball.lineup.waivers import scan_waivers, detect_open_slots, fetch_and_match_free_agents
from fantasy_baseball.lineup.matchups import (
    get_team_batting_stats,
    calculate_matchup_factors,
    adjust_pitcher_projection,
    get_probable_starters,
)
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher
from fantasy_baseball.utils.constants import IL_STATUSES
from fantasy_baseball.data.mlb_schedule import get_week_schedule

import pandas as pd

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "weekly_schedule.json"
BATTING_STATS_PATH = PROJECT_ROOT / "data" / "team_batting_stats.json"

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
    matchup_factors: dict[str, dict] | None = None,
) -> None:
    """Print probable starter matchups, flagging two-start pitchers."""
    if not schedule or not roster_pitchers:
        return

    starters = get_probable_starters(roster_pitchers, schedule, matchup_factors)
    if not starters:
        print("  No roster pitchers found in probable starters.")
        return

    pitcher_teams = {p["name"]: p.get("team", "") for p in roster_pitchers}
    two_start = [s for s in starters if s["starts"] >= 2]
    one_start = [s for s in starters if s["starts"] == 1]

    # Map quality badges to CLI labels
    quality_labels = {"Great": " (easy)", "Tough": " (tough)", "Fair": ""}

    if two_start:
        print("  ** TWO-START PITCHERS **")
        for s in sorted(two_start, key=lambda x: x["pitcher"]):
            team = pitcher_teams.get(s["pitcher"], "")
            matchups = ", ".join(
                f"{m['day']} {m['indicator']} {m['opponent']}"
                f"{quality_labels.get(m['matchup_quality'], '')}"
                for m in s["matchups"]
            )
            print(f"    {s['pitcher']:<25} {team:<5} {matchups}")

    if one_start:
        print("  SINGLE START")
        for s in sorted(one_start, key=lambda x: x["pitcher"]):
            team = pitcher_teams.get(s["pitcher"], "")
            m = s["matchups"][0]
            print(f"    {s['pitcher']:<25} {team:<5} {m['day']} {m['indicator']} {m['opponent']}"
                  f"{quality_labels.get(m['matchup_quality'], '')}")

    # Roster pitchers with no announced start
    announced = {normalize_name(s["pitcher"]) for s in starters}
    unannounced = [
        p["name"] for p in roster_pitchers
        if normalize_name(p["name"]) not in announced
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

    il_players = [p for p in roster if p["status"] in IL_STATUSES]
    if il_players:
        print(f"Excluding {len(il_players)} IL player(s):")
        for p in il_players:
            print(f"  {p['name']} ({p['status']})")
        roster = [p for p in roster if p["status"] not in IL_STATUSES]

    print(f"Roster: {len(roster)} players")
    print(f"Standings: {len(standings)} teams")
    print()

    # Read cached projected standings from dashboard (if available)
    projected_standings_snap = None
    from fantasy_baseball.web.season_data import read_cache, _standings_to_snapshot
    cached = read_cache("projections")
    if cached:
        projected_standings = cached.get("projected_standings")
        if projected_standings:
            projected_standings_snap = _standings_to_snapshot(projected_standings)
            print(f"Loaded cached projected standings ({len(projected_standings)} teams)")

    # Fetch scoring period and MLB schedule
    print("Fetching weekly schedule...")
    period_start, period_end = fetch_scoring_period(league)
    schedule = get_week_schedule(period_start, period_end, SCHEDULE_PATH)
    games_per_team = schedule["games_per_team"] if schedule else {}

    # Validate schedule: if fewer than 28 teams or all teams have ≤ 2 games
    # (opening week / partial data), skip scaling to avoid asymmetric comparisons
    # where some players get scaled down while others default to 6.
    if games_per_team:
        max_games = max(games_per_team.values()) if games_per_team else 0
        if len(games_per_team) < 28 or max_games <= 2:
            print(f"Scoring period: {period_start} to {period_end}")
            print(f"Schedule incomplete ({len(games_per_team)} teams, max {max_games} games) "
                  f"— skipping schedule scaling")
            games_per_team = {}
        else:
            print(f"Scoring period: {period_start} to {period_end}")
            print(f"Schedule loaded for {len(games_per_team)} teams")
    else:
        print("Schedule unavailable — using default 6 games/week")
    print()

    # Fetch team batting stats for matchup adjustments
    print("Fetching team batting stats...")
    team_batting = get_team_batting_stats(BATTING_STATS_PATH)
    matchup_factors = calculate_matchup_factors(team_batting) if team_batting else {}
    if matchup_factors:
        print(f"Matchup factors loaded for {len(matchup_factors)} teams")
    else:
        print("Team batting stats unavailable — no matchup adjustments")
    print()

    # Calculate leverage
    standings_snap = _standings_to_snapshot(standings)
    leverage = calculate_leverage(
        standings_snap, config.team_name,
        projected_standings=projected_standings_snap,
    )
    print("CATEGORY LEVERAGE (higher = more valuable to target):")
    sorted_lev = sorted(leverage.items(), key=lambda x: x[1], reverse=True)
    for cat, weight in sorted_lev:
        bar = "#" * int(weight * 100)
        print(f"  {cat:>4}: {weight:.3f} {bar}")
    print()

    # Load projections and match to roster
    print("Loading projections...")
    conn = get_connection()
    hitters_proj, pitchers_proj = get_blended_projections(conn)
    positions_cache = get_positions(conn)
    conn.close()
    # Precompute normalized names for projection matching (avoids repeated
    # apply(normalize_name) on every lookup — ~800x fewer calls).
    if not hitters_proj.empty:
        hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
    if not pitchers_proj.empty:
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)
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
            matches = hitters_proj[hitters_proj["_name_norm"] == name_norm]
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
            matches = pitchers_proj[pitchers_proj["_name_norm"] == name_norm]
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
                matches = df[df["_name_norm"] == name_norm]
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

    # Map roster pitchers to their weekly opponents using probable starters
    pitcher_matchups: dict[str, list[dict]] = {}
    if schedule and matchup_factors:
        probable = schedule.get("probable_pitchers", [])
        for game in probable:
            for side, opp_key in [("away", "home_team"), ("home", "away_team")]:
                pitcher_name = game.get(f"{side}_pitcher", "TBD")
                if pitcher_name == "TBD":
                    continue
                opp_abbrev = game[opp_key]
                if opp_abbrev in matchup_factors:
                    pitcher_matchups.setdefault(pitcher_name, []).append(
                        matchup_factors[opp_abbrev]
                    )

    # Apply matchup adjustments to roster pitchers
    adjusted_count = 0
    for i, p in enumerate(roster_pitchers):
        name_norm = normalize_name(p["name"])
        for prob_name, factors in pitcher_matchups.items():
            if normalize_name(prob_name) == name_norm:
                roster_pitchers[i] = adjust_pitcher_projection(p, factors)
                adjusted_count += 1
                break
    if adjusted_count:
        print(f"Applied matchup adjustments to {adjusted_count} pitchers")
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
        print_probable_starters(roster_pitchers, schedule, matchup_factors)
        print()

    # Waiver wire recommendations
    print("=" * 60)
    print("WAIVER WIRE RECOMMENDATIONS")
    print("=" * 60)

    open_hitter_slots, open_pitcher_slots, open_bench_slots = detect_open_slots(
        roster, config.roster_slots,
    )
    open_slots = open_hitter_slots + open_pitcher_slots + open_bench_slots
    if open_slots:
        parts = []
        if open_hitter_slots:
            parts.append(f"{open_hitter_slots} hitter")
        if open_pitcher_slots:
            parts.append(f"{open_pitcher_slots} pitcher")
        if open_bench_slots:
            parts.append(f"{open_bench_slots} bench")
        print(f"  Empty slots: {', '.join(parts)} — will recommend pure adds")

    all_roster = roster_hitters + roster_pitchers
    if all_roster or open_slots > 0:
        print("Scanning free agents...")
        fa_players, fa_fetched = fetch_and_match_free_agents(
            league, hitters_proj, pitchers_proj,
            on_position_loaded=lambda pos, n: print(f"    {pos}: loaded {n} players"),
        )

        if fa_fetched == 0:
            print("  No available players returned by Yahoo")
        else:
            print(f"  Found {fa_fetched} free agents, {len(fa_players)} matched projections")

        recommendations = scan_waivers(
            all_roster, fa_players, leverage, max_results=5,
            open_hitter_slots=open_hitter_slots,
            open_pitcher_slots=open_pitcher_slots,
            open_bench_slots=open_bench_slots,
            roster_slots=config.roster_slots,
        )
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                if rec["drop"].startswith("(empty"):
                    print(f"  {i}. ADD {rec['add']:<20} {rec['drop']}  "
                          f"value: +{rec['sgp_gain']:.2f} wSGP")
                else:
                    print(f"  {i}. ADD {rec['add']:<20} DROP {rec['drop']:<20} "
                          f"gain: +{rec['sgp_gain']:.2f} wSGP")
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
