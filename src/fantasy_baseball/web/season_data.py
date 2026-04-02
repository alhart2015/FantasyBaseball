"""Cache management and data assembly for the season dashboard."""

import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES, HITTER_PROJ_KEYS, INVERSE_STATS as INVERSE_CATS, PITCHER_PROJ_KEYS,
)
from fantasy_baseball.utils.positions import PITCHER_POSITIONS

_refresh_lock = threading.Lock()
_refresh_status = {"running": False, "progress": "", "error": None}

_opponent_cache: dict = {}
OPPONENT_CACHE_TTL_SECONDS = 900  # 15 minutes


def clear_opponent_cache() -> None:
    """Clear the opponent lineup in-memory cache (called on full refresh)."""
    _opponent_cache.clear()


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
    "buy_low": "buy_low.json",
    "rankings": "rankings.json",
}


def read_cache(key: str, cache_dir: Path = CACHE_DIR) -> dict | list | None:
    """Read a cached JSON file. Falls back to Redis on local miss."""
    path = cache_dir / CACHE_FILES[key]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback to Redis (only for default cache dir)
    if cache_dir != CACHE_DIR:
        return None

    redis = _get_redis()
    if not redis:
        return None

    try:
        raw = redis.get(f"cache:{key}")
        if raw is None:
            return None
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[redis] read_cache({key}) corrupt data, treating as miss")
        return None
    except Exception as e:
        print(f"[redis] read_cache({key}) failed: {e}")
        return None

    # Write back to local disk for subsequent fast reads
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[redis] local write-back for {key} failed: {e}")

    return data


def write_cache(key: str, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
    """Atomically write a cached JSON file (tmpfile + rename), with Redis write-through."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_FILES[key]
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    # Write-through to Redis (only for default cache dir)
    if cache_dir == CACHE_DIR:
        redis = _get_redis()
        if redis:
            try:
                redis.set(f"cache:{key}", json.dumps(data))
            except Exception as e:
                print(f"[redis] write_cache({key}) failed: {e}")


def read_meta(cache_dir: Path = CACHE_DIR) -> dict:
    """Read cache metadata (last refresh time, week, etc.). Returns empty dict if missing."""
    return read_cache("meta", cache_dir) or {}


def _load_game_log_totals(season_year: int) -> tuple[dict, dict]:
    """Load aggregated game log totals from SQLite, with Redis fallback/write-through.

    Returns (hitter_logs, pitcher_logs) where each is {normalized_name: {stat: value}}.
    On Render, SQLite game_logs may be empty after a cold start — falls back to Redis.
    After loading from SQLite, writes to Redis so the data survives spin-downs.
    """
    from fantasy_baseball.data.db import get_connection as get_db_connection
    from fantasy_baseball.utils.name_utils import normalize_name

    hitter_logs = {}
    pitcher_logs = {}

    # Try SQLite first
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT name, SUM(pa) as pa, SUM(ab) as ab, SUM(h) as h, "
            "SUM(r) as r, SUM(hr) as hr, SUM(rbi) as rbi, SUM(sb) as sb "
            "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
            "GROUP BY name", (season_year,)
        ).fetchall()
        for row in rows:
            norm = normalize_name(row["name"])
            hitter_logs[norm] = {
                "pa": row["pa"] or 0, "ab": row["ab"] or 0, "h": row["h"] or 0,
                "r": row["r"] or 0, "hr": row["hr"] or 0, "rbi": row["rbi"] or 0, "sb": row["sb"] or 0,
            }

        rows = conn.execute(
            "SELECT name, SUM(ip) as ip, SUM(k) as k, SUM(w) as w, SUM(sv) as sv, "
            "SUM(er) as er, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
            "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
            "GROUP BY name", (season_year,)
        ).fetchall()
        for row in rows:
            norm = normalize_name(row["name"])
            pitcher_logs[norm] = {
                "ip": row["ip"] or 0, "k": row["k"] or 0, "w": row["w"] or 0, "sv": row["sv"] or 0,
                "er": row["er"] or 0, "bb": row["bb"] or 0, "h_allowed": row["h_allowed"] or 0,
            }
    finally:
        conn.close()

    # If SQLite had data, write through to Redis for persistence
    if hitter_logs or pitcher_logs:
        redis = _get_redis()
        if redis:
            try:
                redis.set("game_log_totals:hitters", json.dumps(hitter_logs))
                redis.set("game_log_totals:pitchers", json.dumps(pitcher_logs))
            except Exception as e:
                print(f"[redis] game_log_totals write failed: {e}")
        return hitter_logs, pitcher_logs

    # SQLite empty — fall back to Redis (cold start on Render)
    redis = _get_redis()
    if redis:
        try:
            raw_h = redis.get("game_log_totals:hitters")
            raw_p = redis.get("game_log_totals:pitchers")
            if raw_h:
                hitter_logs = json.loads(raw_h)
            if raw_p:
                pitcher_logs = json.loads(raw_p)
            if hitter_logs or pitcher_logs:
                print("[redis] loaded game log totals from Redis (SQLite was empty)")
        except Exception as e:
            print(f"[redis] game_log_totals read failed: {e}")

    return hitter_logs, pitcher_logs


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
        for cat in ALL_CATEGORIES:
            rank = cat_ranks[cat][name]
            if rank <= 2:
                color_classes[cat] = "rank-top"
            elif rank <= 4:
                color_classes[cat] = "rank-high"
            elif rank <= 6:
                color_classes[cat] = "rank-mid"
            elif rank <= 8:
                color_classes[cat] = "rank-low"
            else:
                color_classes[cat] = "rank-bottom"

        teams.append({
            "name": name,
            "team_key": t.get("team_key", ""),
            "stats": t["stats"],
            "roto_points": roto_pts,
            "is_user": is_user,
            "color_classes": color_classes,
        })

    teams.sort(key=lambda t: t["roto_points"]["total"], reverse=True)

    for i, t in enumerate(teams):
        t["rank"] = i + 1

    return {"teams": teams}


def get_teams_list(
    standings: list[dict], user_team_name: str
) -> dict:
    """Build a team list for the opponent selector dropdown.

    Args:
        standings: Raw standings cache (list of team dicts with name, team_key, rank).
        user_team_name: The user's team name for flagging.

    Returns:
        {"teams": [...], "user_team_key": str | None}
    """
    if not standings:
        return {"teams": [], "user_team_key": None}

    user_team_key = None
    teams = []
    for t in standings:
        is_user = t["name"] == user_team_name
        if is_user:
            user_team_key = t.get("team_key", "")
        teams.append({
            "name": t["name"],
            "team_key": t.get("team_key", ""),
            "rank": t.get("rank", 0),
            "is_user": is_user,
        })

    teams.sort(key=lambda t: t["rank"])
    return {"teams": teams, "user_team_key": user_team_key}


def build_opponent_lineup(
    roster: list[dict],
    opponent_name: str,
    standings: list[dict],
    hitters_proj: "pd.DataFrame",
    pitchers_proj: "pd.DataFrame",
    ros_hitters: "pd.DataFrame",
    ros_pitchers: "pd.DataFrame",
    user_leverage: dict[str, float],
    season_year: int,
) -> dict:
    """Build a fully enriched opponent lineup (projections, pace, dual wSGP).

    Args:
        roster: Raw roster from fetch_roster().
        opponent_name: Opponent team name (for leverage calculation).
        standings: Raw standings cache.
        hitters_proj: Blended hitter projections (with _name_norm column).
        pitchers_proj: Blended pitcher projections (with _name_norm column).
        ros_hitters: ROS hitter projections (may be empty DataFrame).
        ros_pitchers: ROS pitcher projections (may be empty DataFrame).
        user_leverage: User's leverage weights.
        season_year: Season year for game log lookup.

    Returns:
        Dict with "hitters" and "pitchers" lists, each entry containing
        projection stats, pace data, and dual wSGP (wsgp_them, wsgp_you).
    """
    import pandas as pd

    from fantasy_baseball.analysis.pace import compute_player_pace
    from fantasy_baseball.data.projections import match_roster_to_projections
    from fantasy_baseball.lineup.leverage import calculate_leverage
    from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
    from fantasy_baseball.utils.name_utils import normalize_name

    # Match roster to projections
    matched = match_roster_to_projections(roster, hitters_proj, pitchers_proj)

    # ROS projection lookup
    has_ros = not ros_hitters.empty or not ros_pitchers.empty
    if has_ros:
        ros_matched = match_roster_to_projections(roster, ros_hitters, ros_pitchers)
        ros_lookup = {normalize_name(p["name"]): p for p in ros_matched}
    else:
        ros_lookup = {}

    # Opponent leverage
    opp_leverage = calculate_leverage(standings, opponent_name)

    # Load game log totals for pace
    hitter_logs, pitcher_logs = _load_game_log_totals(season_year)

    # Build enriched entries
    matched_names = set()
    enriched = []
    for entry in matched:
        player_series = pd.Series(entry)
        entry["wsgp_them"] = calculate_weighted_sgp(player_series, opp_leverage)
        entry["wsgp_you"] = calculate_weighted_sgp(player_series, user_leverage)
        norm = normalize_name(entry["name"])
        matched_names.add(norm)

        # ROS projection tooltip data
        ros_entry = ros_lookup.get(norm)
        if ros_entry:
            entry["ros"] = {
                k: ros_entry.get(k, 0)
                for k in (["r", "hr", "rbi", "sb", "avg"]
                          if entry.get("player_type") == "hitter"
                          else ["w", "k", "sv", "era", "whip"])
            }

        # Pace data
        ptype = entry.get("player_type", "hitter")
        if ptype == "hitter":
            actuals = hitter_logs.get(norm, {})
        else:
            actuals = pitcher_logs.get(norm, {})
        proj_keys = HITTER_PROJ_KEYS if ptype == "hitter" else PITCHER_PROJ_KEYS
        projected = {k: entry.get(k, 0) for k in proj_keys}
        entry["stats"] = compute_player_pace(actuals, projected, ptype)

        enriched.append(entry)

    # Include unmatched players
    for player in roster:
        if normalize_name(player["name"]) not in matched_names:
            entry = dict(player)
            entry["wsgp_them"] = 0.0
            entry["wsgp_you"] = 0.0
            entry["stats"] = {}
            enriched.append(entry)

    # Split into hitters and pitchers
    hitters = []
    pitchers = []
    for p in enriched:
        pos = p.get("selected_position", "BN")
        is_pitcher = pos in PITCHER_POSITIONS or (
            pos == "BN" and set(p.get("positions", [])).issubset(
                PITCHER_POSITIONS | {"BN"}
            )
        )
        if is_pitcher:
            pitchers.append(p)
        else:
            hitters.append(p)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h.get("selected_position", "").upper(), 99),
                                -h.get("wsgp_them", 0)))
    pitchers.sort(key=lambda p: (p.get("selected_position", "") in ("BN", "IL", "DL"),
                                 -p.get("wsgp_them", 0)))

    return {"hitters": hitters, "pitchers": pitchers}


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
            "stats": p.get("stats", {}),
            "ros": p.get("ros"),
            "rank": p.get("rank", {}),
            "preseason": p.get("preseason"),
        }
        if is_pitcher:
            pitchers.append(entry)
        else:
            hitters.append(entry)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h["selected_position"].upper(), 99), -h["wsgp"]))
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
    trade: dict, standings: list[dict], user_team_name: str,
    projected_standings: list[dict] | None = None,
) -> dict:
    """Compute before/after roto standings for a trade.

    Uses projected end-of-season standings as the baseline when available,
    so the before/after comparison reflects end-of-season impact.

    Returns dict with:
      - before: {user_team: {cat: points}, opp_team: {cat: points}}
      - after: {user_team: {cat: points}, opp_team: {cat: points}}
      - before_stats: {user_team: {cat: stat}, opp_team: {cat: stat}}
      - after_stats: {user_team: {cat: stat}, opp_team: {cat: stat}}
      - categories: list of category names
    """
    baseline = projected_standings if projected_standings is not None else standings
    opp_name = trade["opponent"]

    all_stats_before = {t["name"]: dict(t["stats"]) for t in baseline}
    roto_before = score_roto(all_stats_before)

    all_stats_after = {t["name"]: dict(t["stats"]) for t in baseline}

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
        cat_ranks = {}
        prev_val = None
        prev_rank = 0
        for i, t in enumerate(sorted_teams):
            val = t["stats"][cat]
            if val != prev_val:
                prev_rank = i + 1
                prev_val = val
            cat_ranks[t["name"]] = prev_rank
        ranks[cat] = cat_ranks
    return ranks




def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    """Connect to Yahoo, fetch all data, run computations, and write cache files.

    Sets refresh status throughout so the UI can poll progress.
    """
    with _refresh_lock:
        _refresh_status["running"] = True
        _refresh_status["progress"] = "Starting..."
        _refresh_status["error"] = None

    from fantasy_baseball.web.job_logger import JobLogger
    logger = JobLogger("refresh")

    def _progress(msg):
        _set_refresh_progress(msg)
        logger.log(msg)

    try:
        # Lazy imports — only loaded when refresh actually runs
        from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
        from fantasy_baseball.config import load_config
        from fantasy_baseball.data.mlb_schedule import get_week_schedule
        from fantasy_baseball.data.db import (
            create_tables, fetch_and_load_game_logs,
            get_connection as get_db_connection, get_blended_projections,
            get_ros_projections, load_ros_projections,
        )
        from fantasy_baseball.lineup.leverage import calculate_leverage
        from fantasy_baseball.lineup.matchups import calculate_matchup_factors, get_team_batting_stats
        from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
        from fantasy_baseball.lineup.waivers import fetch_and_match_free_agents, scan_waivers, detect_open_slots
        from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings, fetch_scoring_period
        from fantasy_baseball.trades.evaluate import find_trades
        from fantasy_baseball.trades.pitch import generate_pitch
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.analysis.pace import compute_player_pace
        from fantasy_baseball.analysis.buy_low import find_buy_low_candidates

        import pandas as pd

        project_root = Path(__file__).resolve().parents[3]

        # --- Step 1: Auth + league ---
        _progress("Authenticating with Yahoo...")
        sc = get_yahoo_session()
        config = load_config(project_root / "config" / "league.yaml")
        league = get_league(sc, config.league_id, config.game_code)

        # --- Step 2: Find user's team key ---
        _progress("Finding team...")
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
        _progress("Fetching standings...")
        standings = fetch_standings(league)
        _fill_stat_defaults(standings)
        write_cache("standings", standings, cache_dir)
        _progress(f"Fetched standings for {len(standings)} teams")

        _progress("Fetching roster...")
        roster_raw = fetch_roster(league, user_team_key)
        _progress(f"Fetched roster: {len(roster_raw)} players")

        # --- Step 4: Read projections from SQLite ---
        _progress("Loading projections...")
        db_conn = get_db_connection()
        create_tables(db_conn)
        try:
            hitters_proj, pitchers_proj = get_blended_projections(db_conn)

            # Load ROS projections (skip if latest snapshot already in DB)
            _progress("Loading ROS projections...")
            projections_dir = project_root / "data" / "projections"
            ros_dir = projections_dir / str(config.season_year) / "ros"
            if ros_dir.is_dir():
                latest_on_disk = max(
                    (d.name for d in ros_dir.iterdir() if d.is_dir()), default=None,
                )
                if latest_on_disk:
                    existing = db_conn.execute(
                        "SELECT COUNT(*) FROM ros_blended_projections "
                        "WHERE year = ? AND snapshot_date = ?",
                        (config.season_year, latest_on_disk),
                    ).fetchone()[0]
                    if existing == 0:
                        from fantasy_baseball.data.db import get_roster_names
                        ros_roster_names = None
                        try:
                            ros_roster_names = get_roster_names(db_conn)
                        except Exception:
                            pass
                        load_ros_projections(
                            db_conn, projections_dir,
                            config.projection_systems, config.projection_weights,
                            roster_names=ros_roster_names, progress_cb=_progress,
                        )

            ros_hitters, ros_pitchers = get_ros_projections(db_conn)
        finally:
            db_conn.close()
        hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)
        _progress(f"Loaded {len(hitters_proj)} hitter + {len(pitchers_proj)} pitcher projections")
        has_ros = not ros_hitters.empty or not ros_pitchers.empty
        preseason_hitters = hitters_proj
        preseason_pitchers = pitchers_proj
        if has_ros:
            ros_hitters["_name_norm"] = ros_hitters["name"].apply(normalize_name)
            ros_pitchers["_name_norm"] = ros_pitchers["name"].apply(normalize_name)
            _progress(f"Loaded {len(ros_hitters)} ROS hitters + {len(ros_pitchers)} ROS pitchers")
            # Use ROS projections as primary — they're the most current estimates
            hitters_proj = ros_hitters
            pitchers_proj = ros_pitchers
        else:
            _progress("WARNING: No ROS projections available — falling back to preseason")

        # --- Step 4b: Fetch opponent rosters ---
        _progress("Fetching opponent rosters...")
        from fantasy_baseball.data.projections import match_roster_to_projections

        opp_rosters: dict[str, list[dict]] = {}
        all_raw_rosters = {config.team_name: roster_raw}

        def _fetch_opp(key_and_info):
            key, team_info = key_and_info
            tname = team_info.get("name", "")
            try:
                opp_raw = fetch_roster(league, key)
                opp_proj_list = match_roster_to_projections(
                    opp_raw, hitters_proj, pitchers_proj
                )
                return (tname, opp_raw, opp_proj_list)
            except Exception:
                return None

        opp_items = [
            (key, info) for key, info in teams.items()
            if info.get("name", "") != config.team_name and key != user_team_key
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(_fetch_opp, opp_items):
                if result is None:
                    continue
                tname, opp_raw, opp_proj_list = result
                all_raw_rosters[tname] = opp_raw
                if opp_proj_list:
                    opp_rosters[tname] = opp_proj_list
        _progress(f"Fetched {len(opp_rosters)} opponent rosters")

        # --- Step 4c: Build projected standings ---
        _progress("Projecting end-of-season standings...")
        from fantasy_baseball.scoring import project_team_stats

        # Match user roster to projections (wSGP added later after leverage)
        matched = match_roster_to_projections(roster_raw, hitters_proj, pitchers_proj)

        all_team_rosters = {config.team_name: matched}
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
        _progress(f"Projected standings for {len(projected_standings)} teams")

        # --- Step 5: Leverage weights ---
        _progress("Calculating leverage weights...")
        leverage = calculate_leverage(
            standings, config.team_name,
            projected_standings=projected_standings,
        )

        # --- Step 6: Match roster players to projections, compute wSGP ---
        _progress("Matching roster to projections...")

        # Match preseason projections for tooltip comparison (main stats are ROS)
        preseason_matched = match_roster_to_projections(
            roster_raw, preseason_hitters, preseason_pitchers,
        )
        preseason_lookup = {normalize_name(p["name"]): p for p in preseason_matched}

        # Build lookup of matched players, add wSGP
        matched_names = set()
        roster_with_proj = []
        for entry in matched:
            entry["wsgp"] = calculate_weighted_sgp(pd.Series(entry), leverage)
            norm = normalize_name(entry["name"])
            matched_names.add(norm)
            # Attach preseason projection stats for tooltip comparison
            pre_entry = preseason_lookup.get(norm)
            if pre_entry:
                entry["preseason"] = {
                    k: pre_entry.get(k, 0)
                    for k in (["r", "hr", "rbi", "sb", "avg"] if entry.get("player_type") == "hitter"
                              else ["w", "k", "sv", "era", "whip"])
                }
            roster_with_proj.append(entry)
        # Include unmatched players with wsgp=0
        for player in roster_raw:
            if normalize_name(player["name"]) not in matched_names:
                entry = dict(player)
                entry["wsgp"] = 0.0
                roster_with_proj.append(entry)
        _progress(f"Matched {len(roster_with_proj)} players to projections")

        # --- Step 6b: Fetch MLB game logs ---
        _progress("Fetching MLB game logs...")
        gl_conn = get_db_connection()
        create_tables(gl_conn)
        try:
            fetch_and_load_game_logs(
                gl_conn, config.season_year,
                progress_cb=_progress,
            )
        finally:
            gl_conn.close()

        # --- Step 6c: Compute season-to-date pace vs projections ---
        _progress("Computing player pace...")
        hitter_logs, pitcher_logs = _load_game_log_totals(config.season_year)

        # Attach pace data to each roster player (pace compares actuals vs preseason)
        for entry in roster_with_proj:
            norm = normalize_name(entry["name"])
            if "player_type" in entry:
                ptype = entry["player_type"]
            else:
                ptype = "pitcher" if set(entry.get("positions", [])) & PITCHER_POSITIONS else "hitter"
            if ptype == "hitter":
                actuals = hitter_logs.get(norm, {})
            else:
                actuals = pitcher_logs.get(norm, {})
            proj_keys = HITTER_PROJ_KEYS if ptype == "hitter" else PITCHER_PROJ_KEYS
            pre = preseason_lookup.get(norm, {})
            projected = {k: pre.get(k, 0) for k in proj_keys}
            entry["stats"] = compute_player_pace(actuals, projected, ptype)

        # --- Step 6d: Compute SGP rankings ---
        _progress("Computing SGP rankings...")
        from fantasy_baseball.sgp.rankings import (
            compute_sgp_rankings, compute_rankings_from_game_logs,
            rank_key, rank_key_from_positions,
        )

        ros_ranks = compute_sgp_rankings(hitters_proj, pitchers_proj)
        preseason_ranks = compute_sgp_rankings(preseason_hitters, preseason_pitchers)
        current_ranks = compute_rankings_from_game_logs(hitter_logs, pitcher_logs)

        # Build combined lookup: {name::player_type: {ros, preseason, current}}
        all_keys = set(ros_ranks) | set(preseason_ranks) | set(current_ranks)
        rankings_lookup = {}
        for key in all_keys:
            rankings_lookup[key] = {
                "ros": ros_ranks.get(key),
                "preseason": preseason_ranks.get(key),
                "current": current_ranks.get(key),
            }

        write_cache("rankings", rankings_lookup, cache_dir)
        _progress(f"Ranked {len(ros_ranks)} ROS, {len(preseason_ranks)} preseason, {len(current_ranks)} current")

        # Attach ranks to roster players
        for entry in roster_with_proj:
            key = rank_key(entry["name"], entry.get("player_type", "hitter"))
            entry["rank"] = rankings_lookup.get(key, {})

        write_cache("roster", roster_with_proj, cache_dir)

        # --- Step 7: Run lineup optimizer ---
        _progress("Optimizing lineup...")
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
        _progress("Computing lineup moves...")
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
        _progress("Fetching schedule and matchup data...")
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
        _progress("Scanning waivers...")
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
        # Attach ranks to waiver recommendations
        for rec in waiver_recs:
            rec["add_rank"] = rankings_lookup.get(
                rank_key_from_positions(rec["add"], rec.get("add_positions", [])), {})
            rec["drop_rank"] = rankings_lookup.get(
                rank_key_from_positions(rec["drop"], rec.get("drop_positions", [])), {})

        write_cache("waivers", waiver_recs, cache_dir)

        # --- Step 11: Find trades + generate pitches ---
        _progress("Evaluating trades...")
        cat_ranks = _compute_category_ranks(standings)
        leverage_by_team: dict[str, dict] = {}
        for team in standings:
            tname = team["name"]
            leverage_by_team[tname] = calculate_leverage(
                standings, tname, projected_standings=projected_standings,
            )

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
            projected_standings=projected_standings,
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

        # Attach ranks to trade proposals
        for trade in trade_proposals:
            trade["send_rank"] = rankings_lookup.get(
                rank_key_from_positions(trade["send"], trade.get("send_positions", [])), {})
            trade["receive_rank"] = rankings_lookup.get(
                rank_key_from_positions(trade["receive"], trade.get("receive_positions", [])), {})

        write_cache("trades", trade_proposals, cache_dir)

        # --- Step 11b: Compute buy-low candidates ---
        _progress("Finding buy-low candidates...")
        all_game_logs = {**hitter_logs, **pitcher_logs}

        buy_low_trade_targets = []
        for tname, opp_roster in opp_rosters.items():
            candidates = find_buy_low_candidates(
                opp_roster, all_game_logs, leverage, owner=tname,
            )
            buy_low_trade_targets.extend(candidates)
        buy_low_trade_targets.sort(key=lambda c: c["avg_z"])

        buy_low_free_agents = find_buy_low_candidates(
            [s.to_dict() for s in fa_players],
            all_game_logs, leverage, owner="Free Agent",
        )

        # Attach ranks to buy-low candidates
        for candidate in buy_low_trade_targets + buy_low_free_agents:
            candidate["rank"] = rankings_lookup.get(
                rank_key(candidate["name"], candidate.get("player_type", "hitter")), {})

        write_cache("buy_low", {
            "trade_targets": buy_low_trade_targets,
            "free_agents": buy_low_free_agents,
        }, cache_dir)

        # --- Step 13: Monte Carlo simulation ---
        from fantasy_baseball.simulation import run_monte_carlo

        h_slots = sum(v for k, v in config.roster_slots.items()
                      if k not in ("P", "BN", "IL", "DL"))
        p_slots = config.roster_slots.get("P", 9)

        mc_rosters = {}
        for tname, roster in all_team_rosters.items():
            mc_rosters[tname] = roster

        base_mc = run_monte_carlo(
            mc_rosters, h_slots, p_slots, config.team_name,
            n_iterations=1000, use_management=False,
            progress_cb=lambda i: _progress(f"Monte Carlo: iteration {i}/1000..."),
        )
        _progress("Pre-season Monte Carlo complete")
        mgmt_mc = run_monte_carlo(
            mc_rosters, h_slots, p_slots, config.team_name,
            n_iterations=1000, use_management=True,
            progress_cb=lambda i: _progress(f"MC + Roster Mgmt: iteration {i}/1000..."),
        )
        _progress("Pre-season + Mgmt Monte Carlo complete")

        # --- Step 13b: ROS Monte Carlo simulation ---
        ros_mc = None
        ros_mgmt_mc = None
        if has_ros:
            from fantasy_baseball.simulation import run_ros_monte_carlo
            from datetime import date

            season_start = date.fromisoformat(config.season_start)
            season_end = date.fromisoformat(config.season_end)
            total_days = (season_end - season_start).days
            remaining_days = max(0, (season_end - date.today()).days)
            fraction_remaining = remaining_days / total_days if total_days > 0 else 0

            # Build ROS rosters for all teams
            ros_mc_rosters = {}
            # User's team (matched uses ROS projections since step 4c)
            if matched:
                ros_mc_rosters[config.team_name] = matched

            # Opponent teams
            for tname, opp_raw in all_raw_rosters.items():
                if tname == config.team_name:
                    continue
                opp_ros = match_roster_to_projections(
                    opp_raw, ros_hitters, ros_pitchers,
                )
                if opp_ros:
                    ros_mc_rosters[tname] = opp_ros

            # Build actual standings dict
            actual_standings_dict = {
                s["name"]: s["stats"] for s in standings
            }

            if ros_mc_rosters:
                ros_mc = run_ros_monte_carlo(
                    team_rosters=ros_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=fraction_remaining,
                    h_slots=h_slots, p_slots=p_slots,
                    user_team_name=config.team_name,
                    n_iterations=1000, use_management=False,
                    progress_cb=lambda i: _progress(
                        f"Current MC: iteration {i}/1000..."
                    ),
                )
                _progress("Current Monte Carlo complete")
                ros_mgmt_mc = run_ros_monte_carlo(
                    team_rosters=ros_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=fraction_remaining,
                    h_slots=h_slots, p_slots=p_slots,
                    user_team_name=config.team_name,
                    n_iterations=1000, use_management=True,
                    progress_cb=lambda i: _progress(
                        f"Current MC + Mgmt: iteration {i}/1000..."
                    ),
                )
                _progress("Current + Mgmt Monte Carlo complete")

        write_cache("monte_carlo", {
            "base": base_mc,
            "with_management": mgmt_mc,
            "ros": ros_mc,
            "ros_with_management": ros_mgmt_mc,
        }, cache_dir)

        # --- Step 14: Update SQLite database ---
        _progress("Updating database...")
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
        _progress("Finalizing...")
        meta = {
            "last_refresh": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "start_date": start_date,
            "end_date": end_date,
            "team_name": config.team_name,
        }
        write_cache("meta", meta, cache_dir)

        logger.finish("ok")
        _progress("Done")
        clear_opponent_cache()

    except Exception as exc:
        with _refresh_lock:
            _refresh_status["error"] = str(exc)
        logger.finish("error", str(exc))
        raise
    finally:
        with _refresh_lock:
            _refresh_status["running"] = False


