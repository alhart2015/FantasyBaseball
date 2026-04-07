"""Route handlers for the season dashboard."""

import functools
import hmac
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import ALL_CATEGORIES
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

    Positions come from the ``positions`` table (broad coverage including
    free agents) overlaid with ``weekly_rosters`` (more current for
    rostered players).  Returns (pos_map, owner_map) where pos_map maps
    normalized names to position lists and owner_map maps normalized
    names to ``"roster"`` (user's team) or an opponent team name.
    """
    from fantasy_baseball.utils.name_utils import normalize_name
    from fantasy_baseball.data.db import get_positions

    # Base positions from the positions table (covers FAs)
    pos_map: dict[str, list[str]] = {
        normalize_name(k): v for k, v in get_positions(conn).items()
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

    standings = read_cache("standings") or []
    config = _load_config()
    proj_cache = read_cache("projections") or {}
    if standings:
        return calculate_leverage(
            standings, config.team_name,
            projected_standings=proj_cache.get("projected_standings"),
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
    """Load projections from SQLite. Returns (hitters, pitchers, ros_hitters, ros_pitchers)."""
    from fantasy_baseball.data.db import (
        get_connection as get_db_connection, get_blended_projections, get_ros_projections,
    )
    from fantasy_baseball.utils.name_utils import normalize_name
    db_conn = get_db_connection()
    try:
        hitters, pitchers = get_blended_projections(db_conn)
        ros_hitters, ros_pitchers = get_ros_projections(db_conn)
    finally:
        db_conn.close()
    hitters["_name_norm"] = hitters["name"].apply(normalize_name)
    pitchers["_name_norm"] = pitchers["name"].apply(normalize_name)
    if not ros_hitters.empty:
        ros_hitters["_name_norm"] = ros_hitters["name"].apply(normalize_name)
    if not ros_pitchers.empty:
        ros_pitchers["_name_norm"] = ros_pitchers["name"].apply(normalize_name)
    return hitters, pitchers, ros_hitters, ros_pitchers


def _run_ros_fetch() -> None:
    """Background worker for ROS projection fetch + quality checks."""
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.fangraphs_fetch import fetch_ros_projections
    from fantasy_baseball.data.db import (
        create_tables, get_connection as get_db_connection,
        get_roster_names, load_ros_projections,
    )
    from fantasy_baseball.web.job_logger import JobLogger

    logger = JobLogger("ros_fetch")
    project_root = Path(__file__).resolve().parents[3]
    config = load_config(project_root / "config" / "league.yaml")
    projections_dir = project_root / "data" / "projections"

    try:
        logger.log(f"Fetching ROS projections for {len(config.projection_systems)} systems")
        results = fetch_ros_projections(
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
            load_ros_projections(
                db_conn, projections_dir,
                config.projection_systems, config.projection_weights,
                roster_names=roster_names, progress_cb=_quality_cb,
            )
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
        ros_mc_data = None
        ros_mgmt_mc_data = None

        if raw_standings:
            from fantasy_baseball.web.season_data import (
                format_standings_for_display,
                format_monte_carlo_for_display,
            )

            standings_data = format_standings_for_display(
                raw_standings, config.team_name
            )

            raw_projected = read_cache("projections")
            if raw_projected and "projected_standings" in raw_projected:
                projected_data = format_standings_for_display(
                    raw_projected["projected_standings"], config.team_name
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
                if "ros" in raw_mc and raw_mc["ros"]:
                    ros_mc_data = format_monte_carlo_for_display(
                        raw_mc["ros"], config.team_name
                    )
                if "ros_with_management" in raw_mc and raw_mc["ros_with_management"]:
                    ros_mgmt_mc_data = format_monte_carlo_for_display(
                        raw_mc["ros_with_management"], config.team_name
                    )

        return render_template(
            "season/standings.html",
            meta=meta,
            active_page="standings",
            standings=standings_data,
            projected=projected_data,
            mc=mc_data,
            mc_mgmt=mc_mgmt_data,
            ros_mc=ros_mc_data,
            ros_mgmt_mc=ros_mgmt_mc_data,
            categories=ALL_CATEGORIES,
        )

    @app.route("/lineup")
    def lineup():
        meta = read_meta()
        roster_raw = read_cache("roster")
        optimal_raw = read_cache("lineup_optimal")
        starters_raw = read_cache("probable_starters")

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
        waivers_raw = read_cache("waivers")
        trades_raw = read_cache("trades")
        buy_low_raw = read_cache("buy_low") or {}
        return render_template(
            "season/waivers_trades.html",
            meta=meta,
            active_page="waivers_trades",
            waivers=waivers_raw or [],
            trades=trades_raw or [],
            buy_low_targets=buy_low_raw.get("trade_targets", []),
            buy_low_free_agents=buy_low_raw.get("free_agents", []),
            categories=ALL_CATEGORIES,
        )

    @app.route("/api/trade/<int:idx>/standings")
    def api_trade_standings(idx):
        trades_raw = read_cache("trades")
        if not trades_raw or idx >= len(trades_raw):
            return jsonify({"error": "Trade not found"}), 404

        standings_raw = read_cache("standings")
        if not standings_raw:
            return jsonify({"error": "No standings data"}), 404

        proj_cache = read_cache("projections") or {}
        projected_standings = proj_cache.get("projected_standings")

        from fantasy_baseball.web.season_data import compute_trade_standings_impact
        config = _load_config()
        result = compute_trade_standings_impact(
            trade=trades_raw[idx], standings=standings_raw,
            user_team_name=config.team_name,
            projected_standings=projected_standings,
        )
        return jsonify(result)

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
        from datetime import date
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.analysis.pace import compute_player_pace
        from fantasy_baseball.utils.constants import HITTER_PROJ_KEYS, PITCHER_PROJ_KEYS
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats, RankInfo
        from fantasy_baseball.sgp.rankings import lookup_rank

        query = request.args.get("q", "").strip()
        if len(query) < 2:
            return jsonify([])

        conn = _get_search_db()
        try:
            season = date.today().year
            snapshot = _get_latest_ros_snapshot(conn, season)
            if not snapshot:
                return jsonify([])

            # Search ROS projections by name (case-insensitive LIKE)
            like_pattern = f"%{query}%"
            ros_rows = conn.execute(
                "SELECT * FROM ros_blended_projections "
                "WHERE year = ? AND snapshot_date = ? AND name LIKE ? "
                "ORDER BY CASE WHEN adp IS NOT NULL THEN adp ELSE 9999 END ASC "
                "LIMIT 25",
                (season, snapshot, like_pattern),
            ).fetchall()

            if not ros_rows:
                return jsonify([])

            # Load preseason projections for comparison (match by fg_id, not name)
            fg_ids = [r["fg_id"] for r in ros_rows if r["fg_id"]]
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

            # Build results
            results = []
            for ros in ros_rows:
                ros_dict = dict(ros)
                name = ros_dict["name"]
                norm = normalize_name(name)
                ptype = ros_dict["player_type"]
                fg_id = ros_dict.get("fg_id")

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
                    team=ros_dict.get("team", ""),
                    positions=pos_map.get(norm, []),
                    ros=stats_cls.from_dict(ros_dict),
                    preseason=stats_cls.from_dict(pre) if pre else None,
                    rank=RankInfo.from_dict(rank),
                    pace=pace,
                )
                player.compute_wsgp(leverage)

                result = player.to_dict()
                result["ownership"] = ownership
                results.append(result)

            return jsonify(results)
        finally:
            conn.close()

    @app.route("/api/players/browse")
    def api_player_browse():
        """Return all ROS-projected players with stats, rank, SGP, wSGP, ownership."""
        from datetime import date
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.sgp.rankings import lookup_rank
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats, RankInfo

        conn = _get_search_db()
        try:
            season = date.today().year
            snapshot = _get_latest_ros_snapshot(conn, season)
            if not snapshot:
                return jsonify([])

            all_rows = conn.execute(
                "SELECT * FROM ros_blended_projections WHERE year = ? AND snapshot_date = ?",
                (season, snapshot),
            ).fetchall()

            config = _load_config()
            pos_map, owner_map = _build_roster_maps(conn, config.team_name)
            rankings_cache = read_cache("rankings") or {}
            leverage = _get_leverage()

            # Actual PA/BF from game logs for significance indicators
            actual_pa: dict[str, float] = {}
            for r in conn.execute(
                "SELECT name, SUM(pa) as pa FROM game_logs "
                "WHERE season = ? AND player_type = 'hitter' GROUP BY name",
                (season,),
            ).fetchall():
                actual_pa[normalize_name(r["name"])] = r["pa"] or 0

            actual_pitcher_logs: dict[str, dict] = {}
            for r in conn.execute(
                "SELECT name, SUM(ip) as ip, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
                "FROM game_logs WHERE season = ? AND player_type = 'pitcher' GROUP BY name",
                (season,),
            ).fetchall():
                actual_pitcher_logs[normalize_name(r["name"])] = {
                    "ip": r["ip"] or 0, "bb": r["bb"] or 0, "h_allowed": r["h_allowed"] or 0,
                }

            players = []
            for row in all_rows:
                d = dict(row)
                name = d["name"]
                norm = normalize_name(name)
                ptype = d["player_type"]
                fg_id = d.get("fg_id")

                stats_cls = HitterStats if ptype == PlayerType.HITTER else PitcherStats
                ros = stats_cls.from_dict(d)
                ros.compute_sgp()

                rank_info = lookup_rank(rankings_cache, fg_id, name, ptype)

                p = Player(
                    name=name,
                    player_type=ptype,
                    team=d.get("team", ""),
                    fg_id=fg_id,
                    positions=pos_map.get(norm, []),
                    ros=ros,
                    rank=RankInfo.from_dict(rank_info),
                )
                p.compute_wsgp(leverage)

                result = {
                    "name": name,
                    "team": p.team,
                    "player_type": ptype,
                    "positions": p.positions,
                    "owner": owner_map.get(norm),
                    "rank": p.rank.ros,
                    "sgp": round(ros.sgp, 2),
                    "wsgp": round(p.wsgp, 2),
                }

                if ptype == PlayerType.HITTER:
                    result.update({"R": ros.r, "HR": ros.hr, "RBI": ros.rbi,
                                   "SB": ros.sb, "AVG": ros.avg,
                                   "h": ros.h, "ab": ros.ab})
                    actual_obj = HitterStats(pa=actual_pa.get(norm, 0))
                    result["significant"] = actual_obj.significant_dict()
                else:
                    result.update({"W": ros.w, "K": ros.k, "SV": ros.sv,
                                   "ERA": ros.era, "WHIP": ros.whip,
                                   "ip": ros.ip, "er": ros.er,
                                   "bb": ros.bb, "h_allowed": ros.h_allowed})
                    logs = actual_pitcher_logs.get(norm, {})
                    actual_obj = PitcherStats(
                        ip=logs.get("ip", 0),
                        bb=logs.get("bb", 0),
                        h_allowed=logs.get("h_allowed", 0),
                    )
                    result["significant"] = actual_obj.significant_dict()

                players.append(result)

            return jsonify(players)
        finally:
            conn.close()

    @app.route("/api/players/compare")
    def api_player_compare():
        """Return projected standings before/after swapping a roster player."""
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

        def _float(key, default=0.0):
            try:
                return float(request.args.get(key, default))
            except (TypeError, ValueError):
                return default

        other_player = {
            "name": other_name,
            "player_type": other_type,
            "r": _float("other_r"), "hr": _float("other_hr"),
            "rbi": _float("other_rbi"), "sb": _float("other_sb"),
            "h": _float("other_h"), "ab": _float("other_ab"),
            "w": _float("other_w"), "k": _float("other_k"),
            "sv": _float("other_sv"), "ip": _float("other_ip"),
            "er": _float("other_er"), "bb": _float("other_bb"),
            "h_allowed": _float("other_ha"),
        }

        config = _load_config()

        from fantasy_baseball.web.season_data import compute_comparison_standings
        result = compute_comparison_standings(
            roster_player_name=roster_player,
            other_player=other_player,
            user_roster=roster_cache,
            projected_standings=projected_standings,
            user_team_name=config.team_name,
        )

        if "error" in result:
            return jsonify(result), 404

        return jsonify(result)

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
            hitters_proj, pitchers_proj, ros_hitters, ros_pitchers = _load_projections()
        except Exception as e:
            return jsonify({"error": f"Failed to load projections: {e}"}), 500

        user_leverage = calculate_leverage(standings, config.team_name)

        lineup = build_opponent_lineup(
            roster=roster,
            opponent_name=opponent["name"],
            standings=standings,
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            ros_hitters=ros_hitters,
            ros_pitchers=ros_pitchers,
            user_leverage=user_leverage,
            season_year=config.season_year,
        )

        response_data = {
            "team_name": opponent["name"],
            "team_key": team_key,
            "rank": opponent.get("rank", 0),
            "hitters": lineup["hitters"],
            "pitchers": lineup["pitchers"],
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
    def api_fetch_ros_projections():
        """Run ROS projection fetch synchronously.

        Requires gunicorn --timeout 120 to avoid worker kill during the
        ~30s FanGraphs fetch. Results written to job log (visible on /logs).
        """
        _run_ros_fetch()
        return jsonify({"status": "done"})

