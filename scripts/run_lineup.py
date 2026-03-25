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
from fantasy_baseball.lineup.matchups import (
    get_team_batting_stats,
    calculate_matchup_factors,
    adjust_pitcher_projection,
)
from fantasy_baseball.analysis.game_logs import fetch_all_game_logs
from fantasy_baseball.analysis.recency import predict_reliability_blend
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher
from fantasy_baseball.data.mlb_schedule import get_week_schedule

from datetime import datetime as dt
import pandas as pd

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
POSITIONS_PATH = PROJECT_ROOT / "data" / "player_positions.json"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "weekly_schedule.json"
BATTING_STATS_PATH = PROJECT_ROOT / "data" / "team_batting_stats.json"
ROSTER_GAME_LOGS_PATH = PROJECT_ROOT / "data" / "roster_game_logs.json"

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


def apply_recency_blend(player: pd.Series, game_log: list[dict], cutoff: str) -> pd.Series:
    """Blend a player's projection with actual stats using reliability weighting.

    Converts the projection to per-PA/IP rates, runs predict_reliability_blend,
    then converts blended rates back to counting stats using the projected volume.
    """
    blended = player.copy()

    if player.get("player_type") == "hitter":
        pa = player.get("pa", 0)
        ab = player.get("ab", 0)
        if pa <= 0 or ab <= 0:
            return blended
        proj_rates = {
            "hr_per_pa": player.get("hr", 0) / pa,
            "r_per_pa": player.get("r", 0) / pa,
            "rbi_per_pa": player.get("rbi", 0) / pa,
            "sb_per_pa": player.get("sb", 0) / pa,
            "avg": player.get("avg", 0) if "avg" in player.index else player.get("h", 0) / ab,
        }
        rates = predict_reliability_blend(proj_rates, game_log, cutoff)
        blended["hr"] = rates["hr_per_pa"] * pa
        blended["r"] = rates["r_per_pa"] * pa
        blended["rbi"] = rates["rbi_per_pa"] * pa
        blended["sb"] = rates["sb_per_pa"] * pa
        blended["avg"] = rates["avg"]
        blended["h"] = rates["avg"] * ab

    elif player.get("player_type") == "pitcher":
        ip = player.get("ip", 0)
        if ip <= 0:
            return blended
        gs = player.get("gs", 0) if "gs" in player.index else 0
        g = player.get("g", 0) if "g" in player.index else 0
        proj_rates = {
            "k_per_ip": player.get("k", 0) / ip,
            "era": player.get("era", 0) if "era" in player.index else player.get("er", 0) * 9 / ip,
            "whip": player.get("whip", 0) if "whip" in player.index else (player.get("bb", 0) + player.get("h_allowed", 0)) / ip,
            "w_per_gs": player.get("w", 0) / gs if gs > 0 else 0,
            "sv_per_g": player.get("sv", 0) / g if g > 0 else 0,
        }
        rates = predict_reliability_blend(proj_rates, game_log, cutoff)
        blended["k"] = rates["k_per_ip"] * ip
        blended["era"] = rates["era"]
        blended["whip"] = rates["whip"]
        blended["er"] = rates["era"] * ip / 9
        blended["bb"] = rates["whip"] * ip * 0.4  # approximate split
        blended["h_allowed"] = rates["whip"] * ip * 0.6
        if gs > 0:
            blended["w"] = rates["w_per_gs"] * gs
        if g > 0:
            blended["sv"] = rates["sv_per_g"] * g

    return blended


def print_probable_starters(
    roster_pitchers: list[pd.Series],
    schedule: dict | None,
    matchup_factors: dict[str, dict] | None = None,
) -> None:
    """Print probable starter matchups, flagging two-start pitchers."""
    if not schedule or not roster_pitchers:
        return

    probable = schedule.get("probable_pitchers", [])
    if not probable:
        print("  No probable pitcher data available.")
        return

    # Build pitcher name -> list of starts, and name -> team lookup
    pitcher_starts: dict[str, list[dict]] = {}
    roster_names = {normalize_name(p["name"]) for p in roster_pitchers}
    pitcher_teams = {p["name"]: p.get("team", "") for p in roster_pitchers}

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

            quality = ""
            if matchup_factors and game[opponent_key] in matchup_factors:
                f = matchup_factors[game[opponent_key]]["era_whip_factor"]
                if f <= 0.93:
                    quality = " (easy)"
                elif f <= 0.97:
                    quality = " (lean)"
                elif f >= 1.07:
                    quality = " (tough)"
                elif f >= 1.03:
                    quality = " (hard)"

            if pitcher_name not in pitcher_starts:
                pitcher_starts[pitcher_name] = []
            pitcher_starts[pitcher_name].append({
                "day": day,
                "indicator": indicator,
                "opponent": game[opponent_key],
                "quality": quality,
            })

    if not pitcher_starts:
        print("  No roster pitchers found in probable starters.")
        return

    two_start = {k: v for k, v in pitcher_starts.items() if len(v) >= 2}
    one_start = {k: v for k, v in pitcher_starts.items() if len(v) == 1}

    if two_start:
        print("  ** TWO-START PITCHERS **")
        for name, starts in sorted(two_start.items()):
            team = pitcher_teams.get(name, "")
            matchups = ", ".join(
                f"{s['day']} {s['indicator']} {s['opponent']}{s.get('quality', '')}" for s in starts
            )
            print(f"    {name:<25} {team:<5} {matchups}")

    if one_start:
        print("  SINGLE START")
        for name, starts in sorted(one_start.items()):
            team = pitcher_teams.get(name, "")
            s = starts[0]
            print(f"    {name:<25} {team:<5} {s['day']} {s['indicator']} {s['opponent']}{s.get('quality', '')}")

    # Roster pitchers with no announced start
    announced = {normalize_name(k) for k in pitcher_starts.keys()}
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

    print(f"Roster: {len(roster)} players")
    print(f"Standings: {len(standings)} teams")
    print()

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

    # Fetch game logs and apply recency-weighted blending
    print("Fetching roster game logs for recency blending...")

    # Build (name, type) -> MLBAMID lookup from raw projection CSVs (blended
    # projections don't carry MLBAMID through, so we read the raw files).
    # Keyed by (normalized_name, player_type) to avoid collisions when a
    # hitter and pitcher share the same name (e.g. Julio Rodriguez).
    mlbamid_lookup: dict[tuple[str, str], int] = {}
    for csv_path in PROJECTIONS_DIR.glob("*.csv"):
        try:
            csv_lower = csv_path.name.lower()
            if "hitter" in csv_lower:
                ptype = "hitter"
            elif "pitcher" in csv_lower:
                ptype = "pitcher"
            else:
                continue
            raw_df = pd.read_csv(csv_path, usecols=lambda c: c in ("Name", "MLBAMID"))
            if "MLBAMID" in raw_df.columns and "Name" in raw_df.columns:
                for _, row in raw_df.dropna(subset=["MLBAMID"]).iterrows():
                    mlbamid_lookup[(normalize_name(row["Name"]), ptype)] = int(row["MLBAMID"])
        except Exception:
            pass

    roster_players_for_logs = []
    for p in roster_hitters:
        mid = mlbamid_lookup.get((normalize_name(p["name"]), "hitter"))
        if mid:
            roster_players_for_logs.append({"mlbam_id": mid, "name": p["name"], "type": "hitter"})
    for p in roster_pitchers:
        mid = mlbamid_lookup.get((normalize_name(p["name"]), "pitcher"))
        if mid:
            roster_players_for_logs.append({"mlbam_id": mid, "name": p["name"], "type": "pitcher"})

    blended_count = 0
    if roster_players_for_logs:
        game_logs = fetch_all_game_logs(
            roster_players_for_logs, cache_path=ROSTER_GAME_LOGS_PATH,
        )
        today = dt.now().strftime("%Y-%m-%d")

        # Build mlbam_id lookup by normalized name
        name_to_log = {}
        for mid, entry in game_logs.items():
            name_to_log[normalize_name(entry["name"])] = entry.get("games", [])

        for i, p in enumerate(roster_hitters):
            log = name_to_log.get(normalize_name(p["name"]), [])
            if log:
                roster_hitters[i] = apply_recency_blend(p, log, today)
                blended_count += 1
        for i, p in enumerate(roster_pitchers):
            log = name_to_log.get(normalize_name(p["name"]), [])
            if log:
                roster_pitchers[i] = apply_recency_blend(p, log, today)
                blended_count += 1

    if blended_count:
        print(f"Applied reliability-weighted recency blend to {blended_count} players")
    else:
        print("No game logs available — using projections only")
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

    # Waiver wire recommendations (Gap 1 fix)
    print("=" * 60)
    print("WAIVER WIRE RECOMMENDATIONS")
    print("=" * 60)

    # Detect empty active roster slots by type using selected_position.
    # Yahoo returns position names like "Util" (not "UTIL"), "SP"/"RP" (not "P"),
    # so we normalize to lowercase for matching.
    il_positions = {"il", "il+", "dl", "dl+"}
    bench_positions = {"bn"}

    # Classify each roster player's slot
    filled_hitter = 0
    filled_pitcher = 0
    filled_bench = 0
    filled_il = 0
    for p in roster:
        slot = (p.get("selected_position") or "").lower()
        if slot in il_positions:
            filled_il += 1
        elif slot in bench_positions:
            filled_bench += 1
        elif is_pitcher([slot.upper()]) or slot in ("sp", "rp", "p"):
            filled_pitcher += 1
        elif slot:  # any other non-empty slot is a hitter position
            filled_hitter += 1

    total_hitter_slots = sum(
        v for k, v in config.roster_slots.items()
        if k.lower() not in {"p", "bn", "il", "il+", "dl", "dl+"}
    )
    total_pitcher_slots = config.roster_slots.get("P", 0)
    total_bench_slots = config.roster_slots.get("BN", 0)

    open_hitter_slots = max(0, total_hitter_slots - filled_hitter)
    open_pitcher_slots = max(0, total_pitcher_slots - filled_pitcher)
    open_bench_slots = max(0, total_bench_slots - filled_bench)
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
        # Fetch free agents for key positions
        print("Scanning free agents...")
        fa_players = []
        fa_fetched = 0
        fa_seen_names: set[str] = set()
        FA_PER_POSITION = 100
        for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
            fas = fetch_free_agents(league, pos, count=FA_PER_POSITION)
            print(f"    {pos}: loaded {len(fas)} players")
            fa_fetched += len(fas)
            for fa in fas:
                fa_name_norm = normalize_name(fa["name"])
                if fa_name_norm in fa_seen_names:
                    continue
                fa_seen_names.add(fa_name_norm)
                # Match to projections — check the appropriate projection
                # source first based on position to avoid name collisions
                proj_row = None
                if pos in ("SP", "RP"):
                    search_order = [pitchers_proj, hitters_proj]
                else:
                    search_order = [hitters_proj, pitchers_proj]
                for df in search_order:
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

        if fa_fetched == 0:
            print("  No available players returned by Yahoo")
            # Fallback: use projections to find best unrostered players
            print("  Falling back to projection-based recommendations...")
            roster_names = {normalize_name(p["name"]) for p in roster}
            for df in [hitters_proj, pitchers_proj]:
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    if normalize_name(row["name"]) not in roster_names:
                        candidate = row.copy()
                        if "positions" not in candidate.index:
                            candidate["positions"] = []
                        team = candidate.get("team", "")
                        fa_games = games_per_team.get(team, DEFAULT_GAMES_PER_WEEK)
                        candidate = scale_by_schedule(candidate, fa_games)
                        fa_players.append(candidate)
            print(f"  {len(fa_players)} unrostered players found in projections")
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
            if fa_fetched == 0:
                print("  (based on projections — verify availability on Yahoo)")
            for i, rec in enumerate(recommendations, 1):
                if rec["drop"].startswith("(empty"):
                    print(f"  {i}. ADD {rec['add']:<20} {rec['drop']}  "
                          f"value: +{rec['sgp_gain']:.2f} wSGP")
                else:
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
