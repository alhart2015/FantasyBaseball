"""Route handlers for the season dashboard."""

import functools
import hmac
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import ALL_CATEGORIES, RATE_STATS
from fantasy_baseball.web.season_data import read_cache, read_meta

_config = None


def _get_search_db():
    """Get a SQLite connection for player search queries."""
    from fantasy_baseball.data.db import get_connection
    return get_connection()


def _get_admin_password():
    return os.environ.get("ADMIN_PASSWORD", "dev")


def _require_auth(f):
    """Decorator that requires admin password for protected routes.

    Supports two auth methods:
    - Session cookie (browser login via /login)
    - Bearer token header (for automated jobs like QStash cron)
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("authenticated"):
            return f(*args, **kwargs)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            if hmac.compare_digest(token, _get_admin_password()):
                return f(*args, **kwargs)
        if request.is_json or request.content_type == "application/json":
            return jsonify({"error": "Authentication required"}), 401
        return redirect(url_for("login", next=request.path))
    return wrapper


def _load_config():
    global _config
    if _config is None:
        from fantasy_baseball.config import load_config
        config_path = Path(__file__).resolve().parents[3] / "config" / "league.yaml"
        _config = load_config(config_path)
    return _config


def _get_latest_ros_snapshot(conn, season: int) -> str | None:
    """Return the most recent ROS snapshot date, or None."""
    row = conn.execute(
        "SELECT MAX(snapshot_date) as d FROM ros_blended_projections WHERE year = ?",
        (season,),
    ).fetchone()
    return row["d"] if row and row["d"] else None


def _build_roster_maps(conn, team_name: str):
    """Build position and ownership maps.

    Positions come from the Redis ``positions`` key (broad coverage
    including free agents) overlaid with ``weekly_rosters`` (more current
    for rostered players).  Returns (pos_map, owner_map) where pos_map
    maps normalized names to position lists and owner_map maps normalized
    names to ``"roster"`` (user's team) or an opponent team name.
    """
    from fantasy_baseball.utils.name_utils import normalize_name
    from fantasy_baseball.data.redis_store import get_positions, get_default_client

    # Base positions from Redis (covers FAs)
    pos_map: dict[str, list[str]] = {
        normalize_name(k): v for k, v in get_positions(get_default_client()).items()
    }

    owner_map: dict[str, str] = {}

    for p in (read_cache("roster") or []):
        owner_map[normalize_name(p["name"])] = "roster"

    # Weekly rosters override positions (fresher) and provide ownership
    for r in conn.execute(
        "SELECT player_name, positions, team FROM weekly_rosters "
        "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM weekly_rosters)"
    ).fetchall():
        norm = normalize_name(r["player_name"])
        if r["positions"]:
            pos_map[norm] = [p.strip() for p in r["positions"].split(",")]
        if r["team"] != team_name and norm not in owner_map:
            owner_map[norm] = r["team"]

    return pos_map, owner_map


def _get_leverage() -> dict[str, float]:
    """Return per-category leverage weights (uniform fallback if no standings)."""
    from fantasy_baseball.lineup.leverage import calculate_leverage
    from fantasy_baseball.web.season_data import _standings_to_snapshot

    standings_raw = read_cache("standings") or []
    config = _load_config()
    proj_cache = read_cache("projections") or {}
    if standings_raw:
        standings_snap = _standings_to_snapshot(standings_raw)
        proj_raw = proj_cache.get("projected_standings")
        proj_snap = _standings_to_snapshot(proj_raw) if proj_raw else None
        return calculate_leverage(
            standings_snap, config.team_name,
            projected_standings=proj_snap,
        )
    return {c: 1.0 / 10 for c in ALL_CATEGORIES}


def _load_yahoo_league():
    """Get Yahoo league object and user team key."""
    from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
    config = _load_config()
    sc = get_yahoo_session()
    league = get_league(sc, config.league_id, config.game_code)
    teams = league.teams()
    user_team_key = None
    for key, team_info in teams.items():
        if team_info.get("name") == config.team_name:
            user_team_key = key
            break
    if user_team_key is None:
        user_team_key = next(iter(teams))
    return league, user_team_key


def _load_projections():
    """Load projections from SQLite. Returns (hitters, pitchers, rest_of_season_hitters, rest_of_season_pitchers)."""
    from fantasy_baseball.data.db import (
        get_connection as get_db_connection, get_blended_projections, get_rest_of_season_projections,
    )
    from fantasy_baseball.utils.name_utils import normalize_name
    db_conn = get_db_connection()
    try:
        hitters, pitchers = get_blended_projections(db_conn)
        rest_of_season_hitters, rest_of_season_pitchers = get_rest_of_season_projections(db_conn)
    finally:
        db_conn.close()
    hitters["_name_norm"] = hitters["name"].apply(normalize_name)
    pitchers["_name_norm"] = pitchers["name"].apply(normalize_name)
    if not rest_of_season_hitters.empty:
        rest_of_season_hitters["_name_norm"] = rest_of_season_hitters["name"].apply(normalize_name)
    if not rest_of_season_pitchers.empty:
        rest_of_season_pitchers["_name_norm"] = rest_of_season_pitchers["name"].apply(normalize_name)
    return hitters, pitchers, rest_of_season_hitters, rest_of_season_pitchers


def _run_rest_of_season_fetch() -> None:
    """Background worker for ROS projection fetch + quality checks."""
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.fangraphs_fetch import fetch_rest_of_season_projections
    from fantasy_baseball.data.db import (
        create_tables, fetch_and_load_game_logs,
        get_connection as get_db_connection,
        get_roster_names, load_rest_of_season_projections,
    )
    from fantasy_baseball.web.job_logger import JobLogger

    logger = JobLogger("rest_of_season_fetch")
    project_root = Path(__file__).resolve().parents[3]
    config = load_config(project_root / "config" / "league.yaml")
    projections_dir = project_root / "data" / "projections"

    try:
        # Refresh game_logs FIRST so that normalize_rest_of_season_to_full_season has
        # current accumulated actuals to add. On Render's ephemeral filesystem
        # game_logs is wiped every deploy and only gets populated during the
        # dashboard refresh — without this step, rest_of_season_fetch runs against an
        # empty game_logs table, normalize_rest_of_season_to_full_season early-returns
        # via its `if not game_log_totals` guard, and the resulting snapshot
        # is silently un-normalized (matches preseason values).
        gl_conn = get_db_connection()
        create_tables(gl_conn)
        try:
            logger.log("Refreshing MLB game logs (so normalization has actuals to add)")
            fetch_and_load_game_logs(
                gl_conn, config.season_year, progress_cb=logger.log,
            )
        finally:
            gl_conn.close()

        logger.log(f"Fetching ROS projections for {len(config.projection_systems)} systems")
        results = fetch_rest_of_season_projections(
            projections_dir, config.projection_systems, config.season_year,
            progress_cb=logger.log,
        )

        for system, status in results.items():
            logger.log(f"  {system}: {status}")

        # Load roster names for quality checks
        quality_warnings = []

        def _quality_cb(msg):
            logger.log(msg)
            if msg.startswith("QUALITY:"):
                quality_warnings.append(msg)

        db_conn = get_db_connection()
        create_tables(db_conn)
        try:
            roster_names = get_roster_names(db_conn)
            if roster_names:
                logger.log(f"Loaded {len(roster_names)} rostered players for quality checks")

            logger.log("Loading into SQLite...")
            load_rest_of_season_projections(
                db_conn, projections_dir,
                config.projection_systems, config.projection_weights,
                roster_names=roster_names, progress_cb=_quality_cb,
            )

            # Persist blended ROS projections to Redis so the refresh
            # pipeline can read them even if Render spins up a new instance
            # (ephemeral filesystem is wiped between instances).
            from fantasy_baseball.data.db import get_rest_of_season_projections
            from fantasy_baseball.web.season_data import write_cache
            ros_h, ros_p = get_rest_of_season_projections(db_conn)
            if not ros_h.empty or not ros_p.empty:
                ros_data = {
                    "hitters": ros_h.to_dict(orient="records"),
                    "pitchers": ros_p.to_dict(orient="records"),
                }
                write_cache("ros_projections", ros_data)
                logger.log(f"Persisted {len(ros_h)} ROS hitters + {len(ros_p)} ROS pitchers to Redis")
        finally:
            db_conn.close()

        # Write standalone quality report
        if quality_warnings:
            q_logger = JobLogger("projection_quality")
            for w in quality_warnings:
                q_logger.log(w)
            exclusions = [w for w in quality_warnings if "EXCLUDE" in w]
            q_logger.finish(
                "warning" if exclusions else "ok",
                f"{len(quality_warnings)} warnings, {len(exclusions)} exclusions",
            )

        failed = [s for s, v in results.items() if v != "ok"]
        if failed:
            logger.finish("error", f"Failed systems: {', '.join(failed)}")
        else:
            logger.finish("ok")

    except Exception as exc:
        logger.finish("error", str(exc))


def register_routes(app: Flask) -> None:

    @app.route("/")
    def index():
        return redirect(url_for("standings"))

    @app.route("/standings")
    def standings():
        meta = read_meta()
        raw_standings = read_cache("standings")
        config = _load_config()
        standings_data = None
        projected_data = None
        mc_data = None
        mc_mgmt_data = None
        rest_of_season_mc_data = None
        rest_of_season_mgmt_mc_data = None

        if raw_standings:
            from fantasy_baseball.web.season_data import (
                format_standings_for_display,
                format_monte_carlo_for_display,
                _standings_to_snapshot,
            )

            standings_data = format_standings_for_display(
                _standings_to_snapshot(raw_standings), config.team_name
            )

            raw_projected = read_cache("projections")
            if raw_projected and "projected_standings" in raw_projected:
                projected_data = format_standings_for_display(
                    _standings_to_snapshot(raw_projected["projected_standings"]),
                    config.team_name,
                )

            raw_mc = read_cache("monte_carlo")
            if raw_mc:
                mc_data = format_monte_carlo_for_display(
                    raw_mc.get("base", raw_mc), config.team_name
                )
                if "with_management" in raw_mc:
                    mc_mgmt_data = format_monte_carlo_for_display(
                        raw_mc["with_management"], config.team_name
                    )
                if "rest_of_season" in raw_mc and raw_mc["rest_of_season"]:
                    rest_of_season_mc_data = format_monte_carlo_for_display(
                        raw_mc["rest_of_season"], config.team_name
                    )
                if "rest_of_season_with_management" in raw_mc and raw_mc["rest_of_season_with_management"]:
                    rest_of_season_mgmt_mc_data = format_monte_carlo_for_display(
                        raw_mc["rest_of_season_with_management"], config.team_name
                    )

        return render_template(
            "season/standings.html",
            meta=meta,
            active_page="standings",
            standings=standings_data,
            projected=projected_data,
            mc=mc_data,
            mc_mgmt=mc_mgmt_data,
            rest_of_season_mc=rest_of_season_mc_data,
            rest_of_season_mgmt_mc=rest_of_season_mgmt_mc_data,
            categories=ALL_CATEGORIES,
        )

    @app.route("/lineup")
    def lineup():
        meta = read_meta()
        roster_raw = read_cache("roster")
        optimal_raw = read_cache("lineup_optimal")
        starters_raw = read_cache("probable_starters")
        pending_moves_raw = read_cache("pending_moves") or []

        lineup_data = None
        if roster_raw:
            from fantasy_baseball.web.season_data import format_lineup_for_display
            lineup_data = format_lineup_for_display(roster_raw, optimal_raw)

        # Build teams list for opponent selector dropdown
        from fantasy_baseball.web.season_data import get_teams_list
        standings_raw = read_cache("standings")
        config = _load_config()
        teams_data = get_teams_list(standings_raw or [], config.team_name)

        # Check if a specific team was requested via query param
        selected_team_key = request.args.get("team", teams_data.get("user_team_key", ""))

        return render_template(
            "season/lineup.html",
            meta=meta,
            active_page="lineup",
            lineup=lineup_data,
            starters=starters_raw,
            pending_moves=pending_moves_raw,
            teams=teams_data["teams"],
            user_team_key=teams_data.get("user_team_key", ""),
            selected_team_key=selected_team_key,
        )

    @app.route("/api/optimize", methods=["POST"])
    def api_optimize():
        from fantasy_baseball.web.season_data import run_optimize
        try:
            result = run_optimize()
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/roster-audit")
    def roster_audit():
        meta = read_meta()
        audit_raw = read_cache("roster_audit")
        return render_template(
            "season/roster_audit.html",
            meta=meta,
            active_page="roster_audit",
            audit=audit_raw or [],
            categories=ALL_CATEGORIES,
        )

    @app.route("/waivers-trades")
    def waivers_trades():
        meta = read_meta()

        # Build player name list for trade search autocomplete
        roster_raw = read_cache("roster") or []
        opp_rosters_raw = read_cache("opp_rosters") or {}
        my_players = sorted(set(p.get("name", "") for p in roster_raw if p.get("name")))
        opp_players = sorted(set(
            p.get("name", "")
            for players in opp_rosters_raw.values()
            for p in players
            if p.get("name")
        ))

        return render_template(
            "season/waivers_trades.html",
            meta=meta,
            active_page="waivers_trades",
            my_players=my_players,
            opp_players=opp_players,
        )

    @app.route("/api/trade-search", methods=["POST"])
    def api_trade_search():
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.trades.evaluate import (
            search_trades_away, search_trades_for,
        )

        data = request.get_json(silent=True) or {}
        player_name = data.get("player_name", "").strip()
        mode = data.get("mode", "")

        if not player_name:
            return jsonify({"error": "player_name is required"}), 400
        if mode not in ("away", "for"):
            return jsonify({"error": "mode must be 'away' or 'for'"}), 400

        config = _load_config()
        standings_raw = read_cache("standings")
        if not standings_raw:
            return jsonify({"error": "No standings data. Run a refresh first."}), 404

        roster_raw = read_cache("roster")
        if not roster_raw:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404

        opp_rosters_raw = read_cache("opp_rosters")
        if not opp_rosters_raw:
            return jsonify({"error": "No opponent roster data. Run a refresh first."}), 404

        leverage_raw = read_cache("leverage")
        if not leverage_raw:
            return jsonify({"error": "No leverage data. Run a refresh first."}), 404

        rankings_raw = read_cache("rankings")
        if not rankings_raw:
            return jsonify({"error": "No rankings data. Run a refresh first."}), 404

        proj_cache = read_cache("projections") or {}
        projected_standings = proj_cache.get("projected_standings")

        hart_roster = [Player.from_dict(p) for p in roster_raw]
        opp_rosters = {
            tname: [Player.from_dict(p) for p in players]
            for tname, players in opp_rosters_raw.items()
        }

        # Rankings cache stores {key: {ros: int, preseason: int, current: int}}
        # The search functions expect {key: int} (ROS rank only)
        flat_rankings = {}
        for key, val in rankings_raw.items():
            if isinstance(val, dict):
                ros = val.get("rest_of_season")
                if ros is not None:
                    flat_rankings[key] = ros
            elif isinstance(val, int):
                flat_rankings[key] = val

        kwargs = dict(
            player_name=player_name,
            hart_name=config.team_name,
            hart_roster=hart_roster,
            opp_rosters=opp_rosters,
            standings=standings_raw,
            leverage_by_team=leverage_raw,
            roster_slots=config.roster_slots,
            rankings=flat_rankings,
            projected_standings=projected_standings,
        )

        if mode == "away":
            results = search_trades_away(**kwargs)
        else:
            results = search_trades_for(**kwargs)

        return jsonify(results)

    @app.route("/players")
    def player_search():
        meta = read_meta()
        return render_template(
            "season/players.html",
            meta=meta,
            active_page="players",
        )

    @app.route("/api/players/search")
    def api_player_search():
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.utils.time_utils import local_today
        from fantasy_baseball.analysis.pace import compute_player_pace
        from fantasy_baseball.utils.constants import HITTER_PROJ_KEYS, PITCHER_PROJ_KEYS
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats, RankInfo
        from fantasy_baseball.sgp.rankings import lookup_rank

        query = request.args.get("q", "").strip()
        if len(query) < 2:
            return jsonify([])

        conn = _get_search_db()
        try:
            season = local_today().year
            snapshot = _get_latest_ros_snapshot(conn, season)
            if not snapshot:
                return jsonify([])

            # Search ROS projections by name (case-insensitive LIKE)
            like_pattern = f"%{query}%"
            rest_of_season_rows = conn.execute(
                "SELECT * FROM ros_blended_projections "
                "WHERE year = ? AND snapshot_date = ? AND name LIKE ? "
                "ORDER BY CASE WHEN adp IS NOT NULL THEN adp ELSE 9999 END ASC "
                "LIMIT 25",
                (season, snapshot, like_pattern),
            ).fetchall()

            if not rest_of_season_rows:
                return jsonify([])

            # Load preseason projections for comparison (match by fg_id, not name)
            fg_ids = [r["fg_id"] for r in rest_of_season_rows if r["fg_id"]]
            preseason_map = {}
            if fg_ids:
                placeholders = ",".join("?" * len(fg_ids))
                for r in conn.execute(
                    f"SELECT * FROM blended_projections WHERE year = ? AND fg_id IN ({placeholders})",
                    (season, *fg_ids),
                ).fetchall():
                    preseason_map[r["fg_id"]] = dict(r)

            # Load game log totals for pace
            hitter_logs = {}
            for r in conn.execute(
                "SELECT name, SUM(pa) as pa, SUM(ab) as ab, SUM(h) as h, "
                "SUM(r) as r, SUM(hr) as hr, SUM(rbi) as rbi, SUM(sb) as sb "
                "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
                "GROUP BY name", (season,),
            ).fetchall():
                hitter_logs[normalize_name(r["name"])] = dict(r)

            pitcher_logs = {}
            for r in conn.execute(
                "SELECT name, SUM(ip) as ip, SUM(k) as k, SUM(w) as w, SUM(sv) as sv, "
                "SUM(er) as er, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
                "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
                "GROUP BY name", (season,),
            ).fetchall():
                pitcher_logs[normalize_name(r["name"])] = dict(r)

            config = _load_config()
            leverage = _get_leverage()
            pos_map, owner_map = _build_roster_maps(conn, config.team_name)
            rankings_cache = read_cache("rankings") or {}

            # Use cached roster wSGP for rostered players (includes recency blending)
            roster_wsgp = {}
            roster_cache = read_cache("roster") or []
            for rp in roster_cache:
                rn = normalize_name(rp.get("name", ""))
                if rp.get("wsgp"):
                    roster_wsgp[rn] = rp["wsgp"]

            # Build results
            results = []
            for row in rest_of_season_rows:
                rest_of_season_dict = dict(row)
                name = rest_of_season_dict["name"]
                norm = normalize_name(name)
                ptype = rest_of_season_dict["player_type"]
                fg_id = rest_of_season_dict.get("fg_id")

                # Preseason stats
                pre = preseason_map.get(fg_id, {})

                # Pace
                pace = None
                logs = hitter_logs if ptype == PlayerType.HITTER else pitcher_logs
                actuals = logs.get(norm)
                if actuals:
                    proj_keys = HITTER_PROJ_KEYS if ptype == PlayerType.HITTER else PITCHER_PROJ_KEYS
                    projected = {k: pre.get(k, 0) or 0 for k in proj_keys}
                    if any(v > 0 for v in projected.values()):
                        pace = compute_player_pace(actuals, projected, ptype)

                # Ownership
                owner = owner_map.get(norm)
                ownership = "Your roster" if owner == "roster" else (owner or "Free Agent")

                rank = lookup_rank(rankings_cache, fg_id, name, ptype)

                stats_cls = HitterStats if ptype == PlayerType.HITTER else PitcherStats
                player = Player(
                    name=name,
                    player_type=ptype,
                    team=rest_of_season_dict.get("team", ""),
                    positions=pos_map.get(norm, []),
                    rest_of_season=stats_cls.from_dict(rest_of_season_dict),
                    preseason=stats_cls.from_dict(pre) if pre else None,
                    rank=RankInfo.from_dict(rank),
                    pace=pace,
                )
                cached = roster_wsgp.get(norm)
                if cached is not None:
                    player.wsgp = cached
                else:
                    player.compute_wsgp(leverage)

                result = player.to_dict()
                result["ownership"] = ownership
                results.append(result)

            return jsonify(results)
        finally:
            conn.close()

    @app.route("/api/players/browse")
    def api_player_browse():
        """Return all ROS-projected players with stats, rank, SGP, wSGP, ownership.

        Reads from Redis caches (ros_projections, roster, opp_rosters,
        rankings, leverage) so the page works without a local SQLite DB.
        """
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.sgp.rankings import lookup_rank
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats, RankInfo

        ros_cache = read_cache("ros_projections")
        if not ros_cache:
            return jsonify([])

        config = _load_config()
        rankings_cache = read_cache("rankings") or {}
        leverage = _get_leverage()

        # Build position and ownership maps from Redis caches
        pos_map: dict[str, list[str]] = read_cache("positions") or {}
        owner_map: dict[str, str] = {}
        roster_wsgp: dict[str, float] = {}

        for rp in (read_cache("roster") or []):
            norm = normalize_name(rp.get("name", ""))
            owner_map[norm] = "roster"
            if rp.get("wsgp"):
                roster_wsgp[norm] = rp["wsgp"]

        for team_name_opp, team_roster in (read_cache("opp_rosters") or {}).items():
            for rp in team_roster:
                norm = normalize_name(rp.get("name", ""))
                if norm not in owner_map:
                    owner_map[norm] = team_name_opp

        players = []
        for pool in [ros_cache.get("hitters", []), ros_cache.get("pitchers", [])]:
            for d in pool:
                name = d.get("name", "")
                norm = normalize_name(name)
                ptype = d.get("player_type", "")
                if not ptype:
                    continue
                fg_id = d.get("fg_id")
                team = d.get("team")
                if team != team and isinstance(team, float):
                    team = ""  # NaN team

                stats_cls = HitterStats if ptype == PlayerType.HITTER else PitcherStats
                ros = stats_cls.from_dict(d)
                ros.compute_sgp()

                rank_info = lookup_rank(rankings_cache, fg_id, name, ptype)

                p = Player(
                    name=name,
                    player_type=ptype,
                    team=team or "",
                    fg_id=fg_id,
                    positions=pos_map.get(norm, []),
                    rest_of_season=ros,
                    rank=RankInfo.from_dict(rank_info),
                )
                cached = roster_wsgp.get(norm)
                if cached is not None:
                    p.wsgp = cached
                else:
                    p.compute_wsgp(leverage)

                result = {
                    "name": name,
                    "team": p.team,
                    "player_type": ptype,
                    "fg_id": fg_id,
                    "positions": p.positions,
                    "owner": owner_map.get(norm),
                    "rank": p.rank.rest_of_season,
                    "sgp": round(ros.sgp, 2),
                    "wsgp": round(p.wsgp, 2),
                }

                if ptype == PlayerType.HITTER:
                    result.update({"R": ros.r, "HR": ros.hr, "RBI": ros.rbi,
                                   "SB": ros.sb, "AVG": ros.avg,
                                   "h": ros.h, "ab": ros.ab})
                else:
                    result.update({"W": ros.w, "K": ros.k, "SV": ros.sv,
                                   "ERA": ros.era, "WHIP": ros.whip,
                                   "ip": ros.ip, "er": ros.er,
                                   "bb": ros.bb, "h_allowed": ros.h_allowed})

                players.append(result)

        return jsonify(players)

    @app.route("/api/players/compare")
    def api_player_compare():
        """Return projected standings before/after swapping a roster player."""
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats
        from fantasy_baseball.utils.name_utils import normalize_name

        roster_player = request.args.get("roster_player")
        other_name = request.args.get("other_player")
        other_type = request.args.get("other_type")

        if not roster_player or not other_name or not other_type:
            return jsonify({"error": "roster_player, other_player, and other_type are required"}), 400

        roster_cache = read_cache("roster")
        if not roster_cache:
            return jsonify({"error": "No roster data available"}), 404

        proj_cache = read_cache("projections") or {}
        projected_standings = proj_cache.get("projected_standings")
        if not projected_standings:
            return jsonify({"error": "No projected standings available"}), 404

        user_roster = [Player.from_dict(p) for p in roster_cache]

        def _float(key, default=0.0):
            try:
                return float(request.args.get(key, default))
            except (TypeError, ValueError):
                return default

        other_player = Player.from_dict({
            "name": other_name,
            "player_type": other_type,
            "r": _float("other_r"), "hr": _float("other_hr"),
            "rbi": _float("other_rbi"), "sb": _float("other_sb"),
            "h": _float("other_h"), "ab": _float("other_ab"),
            "w": _float("other_w"), "k": _float("other_k"),
            "sv": _float("other_sv"), "ip": _float("other_ip"),
            "er": _float("other_er"), "bb": _float("other_bb"),
            "h_allowed": _float("other_ha"),
        })

        # Look up roster player's ROS from ros_projections — the same
        # source the browse page uses.  This prevents the delta from
        # diverging when ros_projections is updated after a refresh.
        roster_player_projection = None
        ros_cache = read_cache("ros_projections") or {}
        target_norm = normalize_name(roster_player)
        for pool_key in ("hitters", "pitchers"):
            for d in ros_cache.get(pool_key, []):
                if normalize_name(d.get("name", "")) == target_norm:
                    ptype = "hitter" if pool_key == "hitters" else "pitcher"
                    roster_player_projection = Player.from_dict({
                        **d, "player_type": ptype,
                    })
                    break
            if roster_player_projection:
                break

        config = _load_config()

        from fantasy_baseball.web.season_data import compute_comparison_standings
        result = compute_comparison_standings(
            roster_player_name=roster_player,
            other_player=other_player,
            user_roster=user_roster,
            projected_standings=projected_standings,
            user_team_name=config.team_name,
            roster_player_projection=roster_player_projection,
        )

        if "error" in result:
            return jsonify(result), 404

        return jsonify(result)

    @app.route("/luck")
    def luck():
        meta = read_meta()
        spoe_cache = read_cache("spoe") or {}
        latest = spoe_cache.get("snapshot_date")
        results = spoe_cache.get("results", [])

        spoe_data = []
        if results:
            teams = {}
            for r in results:
                team = r["team"]
                if team not in teams:
                    teams[team] = {"team": team, "categories": {}}
                if r["category"] == "total":
                    teams[team]["total_spoe"] = r["spoe"]
                    teams[team]["projected_pts"] = r["projected_pts"]
                    teams[team]["actual_pts"] = r["actual_pts"]
                else:
                    teams[team]["categories"][r["category"]] = r

            spoe_data = sorted(
                teams.values(),
                key=lambda t: t.get("actual_pts", 0),
                reverse=True,
            )

        return render_template(
            "season/luck.html",
            meta=meta,
            active_page="luck",
            spoe_data=spoe_data,
            snapshot_date=meta.get("last_refresh", latest),
            categories=ALL_CATEGORIES,
            rate_stats=RATE_STATS,
        )

    @app.route("/transactions")
    def transactions():
        meta = read_meta()
        txn_cache = read_cache("transaction_analyzer") or {}
        config = _load_config()
        return render_template(
            "season/transactions.html",
            meta=meta,
            active_page="transactions",
            txn_data=txn_cache.get("teams", []),
            user_team=config.team_name,
        )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            password = request.form.get("password", "")
            admin_pw = _get_admin_password()
            if admin_pw and hmac.compare_digest(password, admin_pw):
                session["authenticated"] = True
                next_url = request.args.get("next", url_for("standings"))
                return redirect(next_url)
            error = "Wrong password"
        return render_template("season/login.html", error=error, active_page=None, meta=read_meta())

    @app.route("/logout")
    def logout():
        session.pop("authenticated", None)
        return redirect(url_for("standings"))

    @app.route("/sql", methods=["GET", "POST"])
    @_require_auth
    def sql_runner():
        meta = read_meta()
        query = ""
        columns = None
        rows = None
        row_count = None
        error = None

        if request.method == "POST":
            query = request.form.get("query", "").strip()
            query_params: tuple = ()
            table_name = request.form.get("schema_table", "").strip()

            if request.form.get("action") == "tables":
                query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            elif request.form.get("action") == "schema" and table_name:
                query = "SELECT sql FROM sqlite_master WHERE type='table' AND name=?"
                query_params = (table_name,)

            if query:
                from fantasy_baseball.data.db import get_connection
                conn = get_connection()
                try:
                    cursor = conn.execute(query, query_params)
                    if cursor.description:
                        columns = [d[0] for d in cursor.description]
                        rows = cursor.fetchall()
                        row_count = len(rows)
                    else:
                        conn.commit()
                        row_count = cursor.rowcount
                except Exception as e:
                    error = str(e)
                finally:
                    conn.close()

        return render_template(
            "season/sql.html",
            meta=meta,
            active_page="sql",
            query=query,
            columns=columns,
            rows=rows,
            row_count=row_count,
            error=error,
        )

    @app.route("/logs")
    @_require_auth
    def logs():
        meta = read_meta()
        from fantasy_baseball.web.job_logger import get_all_logs
        job_logs = get_all_logs()
        return render_template(
            "season/logs.html",
            meta=meta,
            active_page="logs",
            job_logs=job_logs,
        )

    @app.route("/api/teams")
    def api_teams():
        from fantasy_baseball.web.season_data import get_teams_list
        standings = read_cache("standings")
        config = _load_config()
        if not standings:
            return jsonify({"teams": [], "user_team_key": None})
        return jsonify(get_teams_list(standings, config.team_name))

    @app.route("/api/opponent/<team_key>/lineup")
    @_require_auth
    def api_opponent_lineup(team_key):
        import time
        from fantasy_baseball.lineup.leverage import calculate_leverage
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster
        from fantasy_baseball.web.season_data import (
            _opponent_cache, OPPONENT_CACHE_TTL_SECONDS,
            _standings_to_snapshot,
            build_opponent_lineup,
        )

        # Check cache
        cached = _opponent_cache.get(team_key)
        if cached and (time.time() - cached["fetched_at"]) < OPPONENT_CACHE_TTL_SECONDS:
            return jsonify(cached["data"])

        # Need standings for leverage + team name lookup
        standings = read_cache("standings")
        if not standings:
            return jsonify({"error": "No standings data. Run a refresh first."}), 404

        # Find opponent name from team_key
        opponent = next((t for t in standings if t.get("team_key") == team_key), None)
        if not opponent:
            return jsonify({"error": f"Team key {team_key} not found"}), 404

        config = _load_config()

        try:
            league, _ = _load_yahoo_league()
            roster = fetch_roster(league, team_key)
        except Exception as e:
            return jsonify({"error": f"Failed to fetch roster: {e}"}), 500

        try:
            hitters_proj, pitchers_proj, rest_of_season_hitters, rest_of_season_pitchers = _load_projections()
        except Exception as e:
            return jsonify({"error": f"Failed to load projections: {e}"}), 500

        standings_snap = _standings_to_snapshot(standings)
        user_leverage = calculate_leverage(standings_snap, config.team_name)

        lineup = build_opponent_lineup(
            roster=roster,
            opponent_name=opponent["name"],
            standings=standings,
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            rest_of_season_hitters=rest_of_season_hitters,
            rest_of_season_pitchers=rest_of_season_pitchers,
            user_leverage=user_leverage,
            season_year=config.season_year,
        )

        response_data = {
            "team_name": opponent["name"],
            "team_key": team_key,
            "rank": opponent.get("rank", 0),
            "hitters": lineup["hitters"],
            "pitchers": lineup["pitchers"],
            "hitter_totals": lineup["hitter_totals"],
            "pitcher_totals": lineup["pitcher_totals"],
        }

        _opponent_cache[team_key] = {
            "data": response_data,
            "fetched_at": time.time(),
        }

        return jsonify(response_data)

    @app.route("/api/refresh", methods=["POST"])
    @_require_auth
    def api_refresh():
        from fantasy_baseball.web.season_data import get_refresh_status, run_full_refresh
        status = get_refresh_status()
        if status["running"]:
            return jsonify({"status": "already_running"})
        thread = threading.Thread(target=run_full_refresh, daemon=True)
        thread.start()
        return jsonify({"status": "started"})

    @app.route("/api/refresh-status")
    def api_refresh_status():
        from fantasy_baseball.web.season_data import get_refresh_status
        return jsonify(get_refresh_status())

    @app.route("/api/fetch-ros-projections", methods=["POST"])
    @_require_auth
    def api_fetch_rest_of_season_projections():
        """Kick off a ROS projection fetch in a background thread.

        Used to be synchronous but now also refreshes game_logs first
        (so normalize_rest_of_season_to_full_season has actuals to add), which can
        take 90-300 seconds on a fresh deploy. Pushes past gunicorn's
        120s timeout. Threaded to match the api_refresh pattern.
        Results written to job log (visible on /logs).
        """
        thread = threading.Thread(target=_run_rest_of_season_fetch, daemon=True)
        thread.start()
        return jsonify({"status": "started"})

