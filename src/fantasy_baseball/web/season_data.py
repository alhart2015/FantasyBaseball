"""Cache management and data assembly for the season dashboard."""

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS as INVERSE_CATS
from fantasy_baseball.utils.positions import PITCHER_POSITIONS

_refresh_lock = threading.Lock()
_refresh_status = {"running": False, "progress": "", "error": None}

_redis_client = None
_redis_initialized = False
_redis_lock = threading.Lock()


def _get_redis():
    """Lazy Upstash Redis client. Returns None if not configured."""
    global _redis_client, _redis_initialized
    if _redis_initialized:
        return _redis_client
    with _redis_lock:
        if _redis_initialized:
            return _redis_client
        url = os.environ.get("UPSTASH_REDIS_REST_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            from upstash_redis import Redis
            _redis_client = Redis(url=url, token=token)
        _redis_initialized = True
    return _redis_client


def get_refresh_status() -> dict:
    with _refresh_lock:
        return dict(_refresh_status)


def _set_refresh_progress(msg: str) -> None:
    with _refresh_lock:
        _refresh_status["progress"] = msg

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

# Defaults for early-season teams missing stats
_STAT_DEFAULTS = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
                  "W": 0, "K": 0, "SV": 0, "ERA": 99.0, "WHIP": 99.0}


def _fill_stat_defaults(standings: list[dict]) -> None:
    """Ensure every team has all 10 stat keys (early season some are missing)."""
    for t in standings:
        filled = dict(_STAT_DEFAULTS)
        filled.update(t["stats"])
        t["stats"] = filled

CACHE_FILES = {
    "standings": "standings.json",
    "roster": "roster.json",
    "projections": "projections.json",
    "lineup_optimal": "lineup_optimal.json",
    "probable_starters": "probable_starters.json",
    "waivers": "waivers.json",
    "trades": "trades.json",
    "monte_carlo": "monte_carlo.json",
    "meta": "meta.json",
}


def read_cache(key: str, cache_dir: Path = CACHE_DIR) -> dict | list | None:
    """Read a cached JSON file. Returns None if missing or corrupt."""
    path = cache_dir / CACHE_FILES[key]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_cache(key: str, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
    """Atomically write a cached JSON file (tmpfile + rename)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_FILES[key]
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # On Windows, must remove target before rename
        if path.exists():
            path.unlink()
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_meta(cache_dir: Path = CACHE_DIR) -> dict:
    """Read cache metadata (last refresh time, week, etc.). Returns empty dict if missing."""
    return read_cache("meta", cache_dir) or {}


def format_standings_for_display(
    standings: list[dict], user_team_name: str
) -> dict:
    """Transform raw standings cache into display-ready structure with roto points and color codes.

    Args:
        standings: List of team dicts from fetch_standings(), each with "name" and "stats" keys.
        user_team_name: The authenticated user's team name for highlighting.

    Returns:
        {"teams": [...]} where each team has roto_points, is_user flag, color_classes, and rank.
    """
    if not standings:
        return {"teams": []}

    _fill_stat_defaults(standings)

    all_stats = {t["name"]: t["stats"] for t in standings}
    roto = score_roto(all_stats)

    cat_ranks = _compute_category_ranks(standings)
    num_teams = len(standings)

    teams = []
    for t in standings:
        name = t["name"]
        is_user = name == user_team_name
        roto_pts = roto[name]

        color_classes = {}
        if is_user:
            for cat in ALL_CATEGORIES:
                rank = cat_ranks[cat][name]
                if rank <= 3:
                    color_classes[cat] = "cat-top"
                elif rank > num_teams - 3:
                    color_classes[cat] = "cat-bottom"
                else:
                    color_classes[cat] = ""
        else:
            color_classes = {cat: "" for cat in ALL_CATEGORIES}

        teams.append({
            "name": name,
            "stats": t["stats"],
            "roto_points": roto_pts,
            "is_user": is_user,
            "color_classes": color_classes,
        })

    teams.sort(key=lambda t: t["roto_points"]["total"], reverse=True)

    for i, t in enumerate(teams):
        t["rank"] = i + 1

    return {"teams": teams}


def format_monte_carlo_for_display(
    mc_data: dict, user_team_name: str
) -> dict:
    """Format Monte Carlo results for template display.

    Returns dict with:
      - teams: list sorted by median_pts desc, each with median_pts, p10, p90,
               first_pct, top3_pct, is_user
      - category_risk: list of dicts with cat, median_pts, p10, p90,
                       top3_pct, bot3_pct, risk_class
    """
    if not mc_data or "team_results" not in mc_data:
        return {"teams": [], "category_risk": []}

    teams = []
    for name, res in mc_data["team_results"].items():
        teams.append({
            "name": name,
            "median_pts": res["median_pts"],
            "p10": res["p10"],
            "p90": res["p90"],
            "first_pct": res["first_pct"],
            "top3_pct": res["top3_pct"],
            "is_user": name == user_team_name,
        })
    teams.sort(key=lambda t: t["median_pts"], reverse=True)

    risk = []
    for cat, data in mc_data.get("category_risk", {}).items():
        if data["top3_pct"] >= 50:
            risk_class = "cat-top"
        elif data["bot3_pct"] >= 30:
            risk_class = "cat-bottom"
        else:
            risk_class = ""
        risk.append({
            "cat": cat,
            "median_pts": data["median_pts"],
            "p10": data["p10"],
            "p90": data["p90"],
            "top3_pct": data["top3_pct"],
            "bot3_pct": data["bot3_pct"],
            "risk_class": risk_class,
        })

    return {"teams": teams, "category_risk": risk}


HITTER_SLOTS_ORDER = ["C", "1B", "2B", "3B", "SS", "IF", "OF", "OF", "OF", "OF",
                       "UTIL", "UTIL", "BN", "IL"]


def format_lineup_for_display(
    roster: list[dict], optimal: dict | None
) -> dict:
    """Format roster + optimizer output for the lineup template."""
    hitters = []
    pitchers = []

    for p in roster:
        pos = p.get("selected_position", "BN")
        is_pitcher = pos in PITCHER_POSITIONS or (
            pos == "BN" and set(p.get("positions", [])).issubset(PITCHER_POSITIONS | {"BN"})
        )
        entry = {
            "name": p["name"],
            "positions": p.get("positions", []),
            "selected_position": pos,
            "player_id": p.get("player_id", ""),
            "status": p.get("status", ""),
            "wsgp": p.get("wsgp", 0),
            "games": p.get("games_this_week", 0),
            "is_bench": pos in ("BN", "IL", "DL"),
            "is_il": "IL" in p.get("status", "") or pos == "IL",
        }
        if is_pitcher:
            pitchers.append(entry)
        else:
            hitters.append(entry)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h["selected_position"], 99), -h["wsgp"]))
    pitchers.sort(key=lambda p: (p["is_bench"], -p["wsgp"]))

    moves = optimal.get("moves", []) if optimal else []

    return {
        "hitters": hitters,
        "pitchers": pitchers,
        "is_optimal": len(moves) == 0,
        "moves": moves,
    }


def run_optimize() -> dict:
    """Re-run lineup optimizer from cached data. Returns moves list."""
    optimal = read_cache("lineup_optimal")
    if optimal:
        return {"moves": optimal.get("moves", []), "is_optimal": len(optimal.get("moves", [])) == 0}
    return {"moves": [], "is_optimal": True}


def compute_trade_standings_impact(
    trade: dict, standings: list[dict], user_team_name: str
) -> dict:
    """Compute before/after roto standings for a trade.

    Returns dict with:
      - before: {user_team: {cat: points}, opp_team: {cat: points}}
      - after: {user_team: {cat: points}, opp_team: {cat: points}}
      - before_stats: {user_team: {cat: stat}, opp_team: {cat: stat}}
      - after_stats: {user_team: {cat: stat}, opp_team: {cat: stat}}
      - categories: list of category names
    """
    opp_name = trade["opponent"]

    all_stats_before = {t["name"]: dict(t["stats"]) for t in standings}
    roto_before = score_roto(all_stats_before)

    all_stats_after = {t["name"]: dict(t["stats"]) for t in standings}

    if "hart_stats_after" in trade and "opp_stats_after" in trade:
        all_stats_after[user_team_name] = trade["hart_stats_after"]
        all_stats_after[opp_name] = trade["opp_stats_after"]
    else:
        for cat in ALL_CATEGORIES:
            hart_delta = trade.get("hart_cat_deltas", {}).get(cat, 0)
            opp_delta = trade.get("opp_cat_deltas", {}).get(cat, 0)
            all_stats_after[user_team_name][cat] += hart_delta
            all_stats_after[opp_name][cat] += opp_delta

    roto_after = score_roto(all_stats_after)

    return {
        "before": {
            user_team_name: roto_before[user_team_name],
            opp_name: roto_before[opp_name],
        },
        "after": {
            user_team_name: roto_after[user_team_name],
            opp_name: roto_after[opp_name],
        },
        "before_stats": {
            user_team_name: all_stats_before[user_team_name],
            opp_name: all_stats_before[opp_name],
        },
        "after_stats": {
            user_team_name: all_stats_after[user_team_name],
            opp_name: all_stats_after[opp_name],
        },
        "categories": ALL_CATEGORIES,
    }


def _compute_category_ranks(standings: list[dict]) -> dict[str, dict[str, int]]:
    """Compute per-category rank for each team (1 = best).

    For inverse categories (ERA, WHIP), lower value = rank 1.
    """
    ranks = {}
    for cat in ALL_CATEGORIES:
        reverse = cat not in INVERSE_CATS
        sorted_teams = sorted(standings, key=lambda t: t["stats"][cat], reverse=reverse)
        ranks[cat] = {t["name"]: i + 1 for i, t in enumerate(sorted_teams)}
    return ranks




def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    """Connect to Yahoo, fetch all data, run computations, and write cache files.

    Sets refresh status throughout so the UI can poll progress.
    """
    with _refresh_lock:
        _refresh_status["running"] = True
        _refresh_status["progress"] = "Starting..."
        _refresh_status["error"] = None

    try:
        # Lazy imports — only loaded when refresh actually runs
        from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
        from fantasy_baseball.config import load_config
        from fantasy_baseball.data.mlb_schedule import get_week_schedule
        from fantasy_baseball.data.db import get_connection as get_db_connection, get_blended_projections
        from fantasy_baseball.lineup.leverage import calculate_leverage
        from fantasy_baseball.lineup.matchups import calculate_matchup_factors, get_team_batting_stats
        from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
        from fantasy_baseball.lineup.waivers import fetch_and_match_free_agents, scan_waivers, detect_open_slots
        from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings, fetch_scoring_period
        from fantasy_baseball.trades.evaluate import find_trades
        from fantasy_baseball.trades.pitch import generate_pitch
        from fantasy_baseball.utils.name_utils import normalize_name

        import pandas as pd

        project_root = Path(__file__).resolve().parents[3]

        # --- Step 1: Auth + league ---
        _set_refresh_progress("Authenticating with Yahoo...")
        sc = get_yahoo_session()
        config = load_config(project_root / "config" / "league.yaml")
        league = get_league(sc, config.league_id, config.game_code)

        # --- Step 2: Find user's team key ---
        _set_refresh_progress("Finding team...")
        teams = league.teams()
        user_team_key = None
        for key, team_info in teams.items():
            if team_info.get("name") == config.team_name:
                user_team_key = key
                break
        if user_team_key is None:
            # Fall back to first team if not found by name
            user_team_key = next(iter(teams))

        # --- Step 3: Fetch standings + roster ---
        _set_refresh_progress("Fetching standings...")
        standings = fetch_standings(league)
        _fill_stat_defaults(standings)
        write_cache("standings", standings, cache_dir)

        _set_refresh_progress("Fetching roster...")
        roster_raw = fetch_roster(league, user_team_key)

        # --- Step 4: Read projections from SQLite ---
        _set_refresh_progress("Loading projections...")
        db_conn = get_db_connection()
        hitters_proj, pitchers_proj = get_blended_projections(db_conn)
        db_conn.close()
        hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

        # --- Step 5: Leverage weights ---
        _set_refresh_progress("Calculating leverage weights...")
        leverage = calculate_leverage(standings, config.team_name)

        # --- Step 6: Match roster players to projections, compute wSGP ---
        _set_refresh_progress("Matching roster to projections...")
        from fantasy_baseball.data.projections import match_roster_to_projections

        matched = match_roster_to_projections(roster_raw, hitters_proj, pitchers_proj)
        # Build lookup of matched players, add wSGP
        matched_names = set()
        roster_with_proj = []
        for entry in matched:
            entry["wsgp"] = calculate_weighted_sgp(pd.Series(entry), leverage)
            matched_names.add(normalize_name(entry["name"]))
            roster_with_proj.append(entry)
        # Include unmatched players with wsgp=0
        for player in roster_raw:
            if normalize_name(player["name"]) not in matched_names:
                entry = dict(player)
                entry["wsgp"] = 0.0
                roster_with_proj.append(entry)

        write_cache("roster", roster_with_proj, cache_dir)

        # --- Step 7: Run lineup optimizer ---
        _set_refresh_progress("Optimizing lineup...")
        hitter_players = []
        pitcher_players = []
        for p in roster_with_proj:
            positions = p.get("positions", [])
            if set(positions) & PITCHER_POSITIONS:
                pitcher_players.append(pd.Series(p))
            else:
                hitter_players.append(pd.Series(p))

        optimal_hitters = optimize_hitter_lineup(
            hitter_players, leverage, config.roster_slots
        )
        optimal_pitchers_starters, optimal_pitchers_bench = optimize_pitcher_lineup(
            pitcher_players, leverage
        )

        # --- Step 8: Compare optimal to current, find moves ---
        _set_refresh_progress("Computing lineup moves...")
        moves = []
        for slot, player_name in optimal_hitters.items():
            for p in roster_with_proj:
                if p["name"] == player_name:
                    current_slot = p.get("selected_position", "BN")
                    base_slot = slot.split("_")[0]
                    # Case-insensitive compare (Yahoo returns "Util", optimizer uses "UTIL")
                    if current_slot.upper() != base_slot.upper():
                        moves.append({
                            "action": "START",
                            "player": player_name,
                            "slot": base_slot,
                            "reason": f"wSGP: {p.get('wsgp', 0):.1f}",
                        })
                    break

        optimal_data = {
            "hitter_lineup": optimal_hitters,
            "pitcher_starters": [p["name"] for p in optimal_pitchers_starters],
            "pitcher_bench": [p["name"] for p in optimal_pitchers_bench],
            "moves": moves,
        }
        write_cache("lineup_optimal", optimal_data, cache_dir)

        # --- Step 9: Probable starters ---
        _set_refresh_progress("Fetching schedule and matchup data...")
        start_date, end_date = fetch_scoring_period(league)
        schedule_cache_path = project_root / "data" / "weekly_schedule.json"
        schedule = get_week_schedule(start_date, end_date, schedule_cache_path)

        batting_stats_cache_path = project_root / "data" / "team_batting_stats.json"
        team_stats = get_team_batting_stats(batting_stats_cache_path)
        matchup_factors = calculate_matchup_factors(team_stats)

        pitcher_roster_for_schedule = [
            p for p in roster_with_proj
            if set(p.get("positions", [])) & PITCHER_POSITIONS
        ]
        from fantasy_baseball.lineup.matchups import get_probable_starters
        probable_starters = get_probable_starters(
            pitcher_roster_for_schedule, schedule or {},
            matchup_factors=matchup_factors, team_stats=team_stats,
        )
        write_cache("probable_starters", probable_starters, cache_dir)

        # --- Step 10: Scan waivers ---
        _set_refresh_progress("Scanning waivers...")
        open_h, open_p, open_b = detect_open_slots(roster_raw, config.roster_slots)
        fa_players, _ = fetch_and_match_free_agents(
            league, hitters_proj, pitchers_proj
        )
        roster_series = [pd.Series(p) for p in roster_with_proj]
        waiver_recs = scan_waivers(
            roster_series,
            fa_players,
            leverage,
            max_results=10,
            open_hitter_slots=open_h,
            open_pitcher_slots=open_p,
            open_bench_slots=open_b,
            roster_slots=config.roster_slots,
        )
        write_cache("waivers", waiver_recs, cache_dir)

        # --- Step 11: Find trades + generate pitches ---
        _set_refresh_progress("Evaluating trades...")
        # Build opponent rosters (fetch top players from each team)
        opp_rosters: dict[str, list[dict]] = {}
        cat_ranks = _compute_category_ranks(standings)
        leverage_by_team: dict[str, dict] = {}
        for team in standings:
            tname = team["name"]
            leverage_by_team[tname] = calculate_leverage(standings, tname)

        all_raw_rosters = {config.team_name: roster_raw}
        for key, team_info in teams.items():
            tname = team_info.get("name", "")
            if tname == config.team_name or key == user_team_key:
                continue
            try:
                opp_raw = fetch_roster(league, key)
                all_raw_rosters[tname] = opp_raw
                opp_proj_list = match_roster_to_projections(
                    opp_raw, hitters_proj, pitchers_proj
                )
                if opp_proj_list:
                    opp_rosters[tname] = opp_proj_list
            except Exception:
                pass  # Skip teams we can't fetch

        hart_roster_for_trades = [
            p for p in roster_with_proj
            if p.get("player_type") in ("hitter", "pitcher")
        ]
        trade_proposals = find_trades(
            hart_name=config.team_name,
            hart_roster=hart_roster_for_trades,
            opp_rosters=opp_rosters,
            standings=standings,
            leverage_by_team=leverage_by_team,
            roster_slots=config.roster_slots,
            max_results=10,
        )

        # Attach trade pitches
        for trade in trade_proposals:
            opp_name = trade["opponent"]
            opp_cat_ranks = cat_ranks  # use league-wide ranks as proxy
            opp_team_ranks = {cat: opp_cat_ranks[cat].get(opp_name, 5) for cat in ALL_CATEGORIES}
            trade["pitch"] = generate_pitch(
                opp_name,
                trade.get("opp_cat_deltas", {}),
                opp_team_ranks,
            )

        write_cache("trades", trade_proposals, cache_dir)

        # --- Step 12: Project full-season standings from rosters ---
        _set_refresh_progress("Projecting standings...")
        from fantasy_baseball.scoring import project_team_stats

        all_team_rosters = {config.team_name: roster_with_proj}
        all_team_rosters.update(opp_rosters)

        projected_standings = []
        for tname, roster in all_team_rosters.items():
            proj_stats = project_team_stats(roster)
            projected_standings.append({
                "name": tname,
                "team_key": "",
                "rank": 0,
                "stats": proj_stats,
            })

        write_cache("projections", {"projected_standings": projected_standings}, cache_dir)

        # --- Step 13: Monte Carlo simulation ---
        from fantasy_baseball.simulation import run_monte_carlo

        h_slots = sum(v for k, v in config.roster_slots.items()
                      if k not in ("P", "BN", "IL", "DL"))
        p_slots = config.roster_slots.get("P", 9)

        mc_rosters = {}
        for tname, roster in all_team_rosters.items():
            mc_rosters[tname] = [pd.Series(p) for p in roster]

        base_mc = run_monte_carlo(
            mc_rosters, h_slots, p_slots, config.team_name,
            n_iterations=1000, use_management=False,
            progress_cb=lambda i: _set_refresh_progress(f"Monte Carlo: iteration {i}/1000..."),
        )
        mgmt_mc = run_monte_carlo(
            mc_rosters, h_slots, p_slots, config.team_name,
            n_iterations=1000, use_management=True,
            progress_cb=lambda i: _set_refresh_progress(f"MC + Roster Mgmt: iteration {i}/1000..."),
        )

        write_cache("monte_carlo", {"base": base_mc, "with_management": mgmt_mc}, cache_dir)

        # --- Step 14: Update SQLite database ---
        _set_refresh_progress("Updating database...")
        from fantasy_baseball.data.db import (
            append_roster_snapshot,
            append_standings_snapshot,
            create_tables,
            get_connection,
        )

        snapshot_date = start_date  # Monday of scoring week
        db_conn = get_connection()
        create_tables(db_conn)  # idempotent — ensures tables exist
        try:
            # Append all team rosters for this week
            week_num = None
            for tname, raw_roster in all_raw_rosters.items():
                append_roster_snapshot(db_conn, raw_roster, snapshot_date, week_num, tname)

            # Append current standings snapshot
            append_standings_snapshot(db_conn, standings, config.season_year, snapshot_date)
        finally:
            db_conn.close()

        # --- Step 15: Write meta ---
        _set_refresh_progress("Finalizing...")
        meta = {
            "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "start_date": start_date,
            "end_date": end_date,
            "team_name": config.team_name,
        }
        write_cache("meta", meta, cache_dir)

        _set_refresh_progress("Done")

    except Exception as exc:
        with _refresh_lock:
            _refresh_status["error"] = str(exc)
        raise
    finally:
        with _refresh_lock:
            _refresh_status["running"] = False


def run_mlb_fetch() -> None:
    """Fetch all MLB player game logs and store in SQLite.

    Manages refresh status so the UI can poll progress. Mirrors the
    pattern used by run_full_refresh().
    """
    with _refresh_lock:
        _refresh_status["running"] = True
        _refresh_status["progress"] = "Starting MLB data fetch..."
        _refresh_status["error"] = None

    try:
        from fantasy_baseball.config import load_config
        from fantasy_baseball.data.db import (
            create_tables,
            fetch_and_load_game_logs,
            get_connection,
        )

        project_root = Path(__file__).resolve().parents[3]
        config = load_config(project_root / "config" / "league.yaml")

        conn = get_connection()
        create_tables(conn)
        try:
            new_rows = fetch_and_load_game_logs(
                conn, config.season_year,
                progress_cb=_set_refresh_progress,
            )
        finally:
            conn.close()

        _set_refresh_progress(f"Done — {new_rows} new game log rows added")

    except Exception as exc:
        with _refresh_lock:
            _refresh_status["error"] = str(exc)
        raise
    finally:
        with _refresh_lock:
            _refresh_status["running"] = False
