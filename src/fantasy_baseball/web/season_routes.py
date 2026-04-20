"""Route handlers for the season dashboard."""

import functools
import hmac
import logging
import os
import threading
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import ALL_CATEGORIES, RATE_STATS
from fantasy_baseball.web.season_data import CacheKey, read_cache, read_meta

log = logging.getLogger(__name__)

_config = None


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


def _compute_worst_roster_by_position() -> dict[str, str]:
    """Cache-backed ``{pool_pos: worst_roster_player_name}``. Empty if roster
    cache is missing."""
    from fantasy_baseball.lineup.roster_audit import worst_roster_by_position
    from fantasy_baseball.models.player import Player

    roster_raw = read_cache(CacheKey.ROSTER) or []
    if not roster_raw:
        return {}
    roster = [Player.from_dict(p) for p in roster_raw]
    return worst_roster_by_position(roster)


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
    """Load projections from Redis. Returns (hitters, pitchers, rest_of_season_hitters, rest_of_season_pitchers)."""
    import pandas as pd
    from fantasy_baseball.data.redis_store import (
        get_blended_projections, get_default_client,
    )
    from fantasy_baseball.utils.name_utils import normalize_name
    from fantasy_baseball.web.season_data import read_cache

    client = get_default_client()
    hitters_raw = get_blended_projections(client, "hitters")
    pitchers_raw = get_blended_projections(client, "pitchers")
    if client is not None and not hitters_raw and not pitchers_raw:
        log.warning(
            "blended_projections:hitters and blended_projections:pitchers are both empty - "
            "run scripts/build_db.py to populate preseason projections before dashboard use."
        )
    hitters = pd.DataFrame(hitters_raw)
    pitchers = pd.DataFrame(pitchers_raw)

    # ROS projections served from the cache:ros_projections Redis key (existing write-through).
    # Task 7 migrates the ROS pipeline off SQLite staging; this request-path reader already
    # goes through Redis cache.
    ros_cache = read_cache(CacheKey.ROS_PROJECTIONS) or {}
    rest_of_season_hitters = pd.DataFrame(ros_cache.get("hitters", []))
    rest_of_season_pitchers = pd.DataFrame(ros_cache.get("pitchers", []))

    if "name" in hitters.columns:
        hitters["_name_norm"] = hitters["name"].apply(normalize_name)
    if "name" in pitchers.columns:
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
    from fantasy_baseball.data.mlb_game_logs import fetch_game_log_totals
    from fantasy_baseball.web.job_logger import JobLogger

    logger = JobLogger("rest_of_season_fetch")
    project_root = Path(__file__).resolve().parents[3]
    config = load_config(project_root / "config" / "league.yaml")
    projections_dir = project_root / "data" / "projections"

    try:
        # Refresh game_logs FIRST so that normalize_rest_of_season_to_full_season has
        # current accumulated actuals to add. On Render's ephemeral filesystem
        # Redis is the only persistent store — without this step, rest_of_season_fetch
        # could run against stale game_log_totals and normalize_rest_of_season_to_full_season
        # would early-return via its `if not game_log_totals` guard, silently leaving
        # the resulting snapshot un-normalized (matches preseason values).
        logger.log("Refreshing MLB game logs (so normalization has actuals to add)")
        fetch_game_log_totals(config.season_year, progress_cb=logger.log)

        logger.log(f"Fetching ROS projections for {len(config.projection_systems)} systems")
        results = fetch_rest_of_season_projections(
            projections_dir, config.projection_systems, config.season_year,
            progress_cb=logger.log,
        )

        for system, status in results.items():
            logger.log(f"  {system}: {status}")

        # Load roster names from Redis for quality checks
        quality_warnings = []

        def _quality_cb(msg):
            logger.log(msg)
            if msg.startswith("QUALITY:"):
                quality_warnings.append(msg)

        from fantasy_baseball.data.redis_store import (
            get_default_client, get_latest_roster_names,
        )
        roster_names = get_latest_roster_names(get_default_client())
        if roster_names:
            logger.log(f"Loaded {len(roster_names)} rostered players for quality checks")

        logger.log("Blending ROS projections → Redis...")
        from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros
        ros_h, ros_p = blend_and_cache_ros(
            projections_dir, config.projection_systems, config.projection_weights,
            roster_names, config.season_year, progress_cb=_quality_cb,
        )
        logger.log(f"Persisted {len(ros_h)} ROS hitters + {len(ros_p)} ROS pitchers to Redis")

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
        raw_standings = read_cache(CacheKey.STANDINGS)
        config = _load_config()
        standings_data = None
        preseason_data = None
        current_projected_data = None
        mc_data = None
        mc_mgmt_data = None
        rest_of_season_mc_data = None
        rest_of_season_mgmt_mc_data = None
        baseline_meta = None

        if raw_standings:
            from fantasy_baseball.web.season_data import (
                format_standings_for_display,
                format_monte_carlo_for_display,
                _standings_to_snapshot,
            )

            standings_data = format_standings_for_display(
                _standings_to_snapshot(raw_standings), config.team_name
            )

            raw_projected = read_cache(CacheKey.PROJECTIONS)
            if raw_projected:
                preseason_standings = raw_projected.get(
                    "preseason_standings",
                    raw_projected.get("projected_standings"),
                )
                if preseason_standings:
                    preseason_data = format_standings_for_display(
                        _standings_to_snapshot(preseason_standings),
                        config.team_name,
                        team_sds=raw_projected.get("preseason_team_sds"),
                    )
                if "projected_standings" in raw_projected:
                    current_projected_data = format_standings_for_display(
                        _standings_to_snapshot(raw_projected["projected_standings"]),
                        config.team_name,
                        team_sds=raw_projected.get("team_sds"),
                    )

            raw_mc = read_cache(CacheKey.MONTE_CARLO)
            if raw_mc:
                baseline_meta = raw_mc.get("baseline_meta")
                if raw_mc.get("base"):
                    mc_data = format_monte_carlo_for_display(
                        raw_mc["base"], config.team_name
                    )
                if raw_mc.get("with_management"):
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
            preseason=preseason_data,
            current_projected=current_projected_data,
            mc=mc_data,
            mc_mgmt=mc_mgmt_data,
            baseline_meta=baseline_meta,
            rest_of_season_mc=rest_of_season_mc_data,
            rest_of_season_mgmt_mc=rest_of_season_mgmt_mc_data,
            categories=ALL_CATEGORIES,
        )

    @app.route("/lineup")
    def lineup():
        meta = read_meta()
        roster_raw = read_cache(CacheKey.ROSTER)
        optimal_raw = read_cache(CacheKey.LINEUP_OPTIMAL)
        starters_raw = read_cache(CacheKey.PROBABLE_STARTERS)
        pending_moves_raw = read_cache(CacheKey.PENDING_MOVES) or []

        lineup_data = None
        if roster_raw:
            from fantasy_baseball.web.season_data import format_lineup_for_display
            lineup_data = format_lineup_for_display(roster_raw, optimal_raw)

        # Build teams list for opponent selector dropdown
        from fantasy_baseball.web.season_data import get_teams_list
        standings_raw = read_cache(CacheKey.STANDINGS)
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
        audit_raw = read_cache(CacheKey.ROSTER_AUDIT)
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
        roster_raw = read_cache(CacheKey.ROSTER) or []
        opp_rosters_raw = read_cache(CacheKey.OPP_ROSTERS) or {}
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
            my_roster_data=roster_raw or [],
            opp_rosters_data=opp_rosters_raw or {},
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
        standings_raw = read_cache(CacheKey.STANDINGS)
        if not standings_raw:
            return jsonify({"error": "No standings data. Run a refresh first."}), 404

        roster_raw = read_cache(CacheKey.ROSTER)
        if not roster_raw:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404

        opp_rosters_raw = read_cache(CacheKey.OPP_ROSTERS)
        if not opp_rosters_raw:
            return jsonify({"error": "No opponent roster data. Run a refresh first."}), 404

        leverage_raw = read_cache(CacheKey.LEVERAGE)
        if not leverage_raw:
            return jsonify({"error": "No leverage data. Run a refresh first."}), 404

        rankings_raw = read_cache(CacheKey.RANKINGS)
        if not rankings_raw:
            return jsonify({"error": "No rankings data. Run a refresh first."}), 404

        proj_cache = read_cache(CacheKey.PROJECTIONS) or {}
        projected_standings = proj_cache.get("projected_standings")
        team_sds = proj_cache.get("team_sds")

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
            team_sds=team_sds,
        )

        if mode == "away":
            results = search_trades_away(**kwargs)
        else:
            results = search_trades_for(**kwargs)

        return jsonify(results)

    @app.route("/api/waiver-search")
    @_require_auth
    def api_waiver_search():
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.trades.multi_trade import build_waiver_pool
        from fantasy_baseball.utils.name_utils import normalize_name

        query = (request.args.get("q") or "").strip()
        if len(query) < 2:
            return jsonify([])

        roster_raw = read_cache(CacheKey.ROSTER) or []
        opp_rosters_raw = read_cache(CacheKey.OPP_ROSTERS) or {}
        ros_cache = read_cache(CacheKey.ROS_PROJECTIONS) or {}

        hart_roster = [Player.from_dict(p) for p in roster_raw]
        opp_rosters = {n: [Player.from_dict(p) for p in ps] for n, ps in opp_rosters_raw.items()}
        pool = build_waiver_pool(hart_roster, opp_rosters, ros_cache)

        q_norm = normalize_name(query)
        matches = [
            {
                "key": key,
                "name": p.name,
                "player_type": p.player_type,
                "positions": p.positions,
            }
            for key, p in pool.items()
            if q_norm in normalize_name(p.name)
        ]
        matches.sort(key=lambda m: m["name"])
        return jsonify(matches[:20])

    @app.route("/api/evaluate-trade", methods=["POST"])
    @_require_auth
    def api_evaluate_trade():
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.trades.multi_trade import (
            TradeProposal,
            build_waiver_pool,
            evaluate_multi_trade,
        )

        data = request.get_json(silent=True) or {}
        opponent = (data.get("opponent") or "").strip()
        if not opponent:
            return jsonify({"error": "opponent is required"}), 400

        config = _load_config()
        roster_raw = read_cache(CacheKey.ROSTER)
        opp_rosters_raw = read_cache(CacheKey.OPP_ROSTERS)
        if roster_raw is None or opp_rosters_raw is None:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404
        if opponent not in opp_rosters_raw:
            return jsonify({"error": f"Unknown opponent: {opponent}"}), 400

        proj_cache = read_cache(CacheKey.PROJECTIONS) or {}
        projected_standings = proj_cache.get("projected_standings")
        team_sds = proj_cache.get("team_sds")
        if not projected_standings:
            return jsonify({"error": "No projected standings. Run a refresh first."}), 404

        ros_cache = read_cache(CacheKey.ROS_PROJECTIONS) or {}

        hart_roster = [Player.from_dict(p) for p in roster_raw]
        opp_rosters = {
            n: [Player.from_dict(p) for p in ps] for n, ps in opp_rosters_raw.items()
        }
        waiver_pool = build_waiver_pool(hart_roster, opp_rosters, ros_cache)

        proposal = TradeProposal(
            opponent=opponent,
            send=list(data.get("send") or []),
            receive=list(data.get("receive") or []),
            my_drops=list(data.get("my_drops") or []),
            opp_drops=list(data.get("opp_drops") or []),
            my_adds=list(data.get("my_adds") or []),
            my_active_ids=set(data.get("my_active_ids") or []),
        )

        result = evaluate_multi_trade(
            proposal=proposal,
            hart_name=config.team_name,
            hart_roster=hart_roster,
            opp_rosters=opp_rosters,
            waiver_pool=waiver_pool,
            projected_standings=projected_standings,
            team_sds=team_sds,
            roster_slots=config.roster_slots,
        )
        return jsonify(
            {
                "legal": result.legal,
                "reason": result.reason,
                "delta_total": round(result.delta_total, 2),
                "categories": {
                    cat: {
                        "before": round(cd.before, 2),
                        "after": round(cd.after, 2),
                        "delta": round(cd.delta, 2),
                    }
                    for cat, cd in result.categories.items()
                },
            }
        )

    @app.route("/players")
    def player_search():
        meta = read_meta()
        return render_template(
            "season/players.html",
            meta=meta,
            active_page="players",
        )

    @app.route("/api/players/browse")
    def api_player_browse():
        """Return all ROS-projected players with stats, rank, SGP, ownership,
        and — for FAs — precomputed ΔRoto vs the worst roster player at
        their position (pulled from the roster_audit cache).

        Reads from Redis caches (ros_projections, roster, opp_rosters,
        rankings, roster_audit) so the page works without a local SQLite DB.
        """
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.sgp.rankings import lookup_rank
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats, RankInfo
        from fantasy_baseball.lineup.roster_audit import fa_target_positions

        ros_cache = read_cache(CacheKey.ROS_PROJECTIONS)
        if not ros_cache:
            return jsonify([])

        rankings_cache = read_cache(CacheKey.RANKINGS) or {}

        # Build position and ownership maps from Redis caches
        pos_map: dict[str, list[str]] = read_cache(CacheKey.POSITIONS) or {}
        owner_map: dict[str, str] = {}

        for rp in (read_cache(CacheKey.ROSTER) or []):
            norm = normalize_name(rp.get("name", ""))
            owner_map[norm] = "roster"

        for team_name_opp, team_roster in (read_cache(CacheKey.OPP_ROSTERS) or {}).items():
            for rp in team_roster:
                norm = normalize_name(rp.get("name", ""))
                if norm not in owner_map:
                    owner_map[norm] = team_name_opp

        # Precomputed ΔRoto: roster_audit already evaluated the top-N FAs at
        # each roster slot.  Index every (drop_player, fa) pair so the browse
        # page can surface ΔRoto directly without recomputation, then pair
        # each FA with the worst-SGP roster player at their position.
        audit_raw = read_cache(CacheKey.ROSTER_AUDIT) or []
        audit_index: dict[tuple[str, str], dict] = {}
        for entry in audit_raw:
            drop_name = entry.get("player")
            if not drop_name:
                continue
            for c in entry.get("candidates") or []:
                fa_name = c.get("name")
                dr = c.get("delta_roto")
                if fa_name and dr:
                    audit_index[(drop_name, fa_name)] = dr
        worst_by_pos = _compute_worst_roster_by_position()

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

                # ΔRoto: FAs only, max over eligible positions against the
                # worst roster player at each pool position.
                owner = owner_map.get(norm)
                delta_roto = None
                if owner is None:
                    targets = fa_target_positions(ptype, p.positions, ros.sv if ptype == PlayerType.PITCHER else 0.0)
                    for target_pos in targets:
                        drop_name = worst_by_pos.get(target_pos)
                        if not drop_name:
                            continue
                        dr = audit_index.get((drop_name, name))
                        if dr is None:
                            continue
                        if delta_roto is None or dr["total"] > delta_roto["total"]:
                            delta_roto = dr

                result = {
                    "name": name,
                    "team": p.team,
                    "player_type": ptype,
                    "fg_id": fg_id,
                    "positions": p.positions,
                    "owner": owner,
                    "rank": p.rank.rest_of_season,
                    "sgp": round(ros.sgp, 2),
                    "delta_roto": delta_roto,
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

        roster_cache = read_cache(CacheKey.ROSTER)
        if not roster_cache:
            return jsonify({"error": "No roster data available"}), 404

        proj_cache = read_cache(CacheKey.PROJECTIONS) or {}
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
        ros_cache = read_cache(CacheKey.ROS_PROJECTIONS) or {}
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
            team_sds=proj_cache.get("team_sds"),
        )

        if "error" in result:
            return jsonify(result), 404

        return jsonify(result)

    @app.route("/api/players/delta_roto")
    def api_player_delta_roto():
        """Live-compute ΔRoto for an FA vs the worst-SGP roster player at
        that FA's position.

        Used by the players page for FAs outside the precomputed roster_audit
        pools (top 5 C/1B/2B/3B/SS, top 15 OF, top 20 SP, top 10 RP). Returns
        the best delta_roto.to_dict() across the FA's eligible positions.
        """
        from fantasy_baseball.lineup.delta_roto import compute_delta_roto
        from fantasy_baseball.lineup.roster_audit import fa_target_positions
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.utils.name_utils import normalize_name

        player_name = request.args.get("player")
        player_type = request.args.get("player_type")
        if not player_name or not player_type:
            return jsonify({"error": "player and player_type are required"}), 400

        roster_raw = read_cache(CacheKey.ROSTER)
        if not roster_raw:
            return jsonify({"error": "No roster data available"}), 404
        user_roster = [Player.from_dict(p) for p in roster_raw]

        proj_cache = read_cache(CacheKey.PROJECTIONS) or {}
        projected_standings = proj_cache.get("projected_standings")
        if not projected_standings:
            return jsonify({"error": "No projected standings available"}), 404

        # Resolve the FA's ROS projection from ros_projections (same source
        # the browse page uses, so totals line up with the table row).
        ros_cache = read_cache(CacheKey.ROS_PROJECTIONS) or {}
        pool_key = "pitchers" if player_type == "pitcher" else "hitters"
        target_norm = normalize_name(player_name)
        fa_player = None
        for d in ros_cache.get(pool_key, []):
            if normalize_name(d.get("name", "")) == target_norm:
                fa_player = Player.from_dict({**d, "player_type": player_type})
                break
        if fa_player is None:
            return jsonify({"error": f"{player_name} not found in projections"}), 404

        # Worst roster player at the FA's target positions
        pos_map = read_cache(CacheKey.POSITIONS) or {}
        fa_positions = pos_map.get(target_norm, [])
        sv = fa_player.rest_of_season.sv if player_type == "pitcher" else 0.0
        targets = fa_target_positions(player_type, fa_positions, sv)

        worst_by_pos = _compute_worst_roster_by_position()
        config = _load_config()
        team_sds = proj_cache.get("team_sds")

        best = None
        for target_pos in targets:
            drop_name = worst_by_pos.get(target_pos)
            if not drop_name:
                continue
            try:
                dr = compute_delta_roto(
                    drop_name=drop_name,
                    add_player=fa_player,
                    user_roster=user_roster,
                    projected_standings=projected_standings,
                    team_name=config.team_name,
                    team_sds=team_sds,
                )
            except (ValueError, KeyError) as exc:
                log.warning("live ΔRoto failed for %s vs %s: %s", player_name, drop_name, exc)
                continue
            d = dr.to_dict()
            if best is None or d["total"] > best["total"]:
                best = d

        if best is None:
            return jsonify({"error": "No comparable roster player at this position"}), 404
        return jsonify({"delta_roto": best})

    @app.route("/luck")
    def luck():
        meta = read_meta()
        spoe_cache = read_cache(CacheKey.SPOE) or {}
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
        txn_cache = read_cache(CacheKey.TRANSACTION_ANALYZER) or {}
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
        standings = read_cache(CacheKey.STANDINGS)
        config = _load_config()
        if not standings:
            return jsonify({"teams": [], "user_team_key": None})
        return jsonify(get_teams_list(standings, config.team_name))

    @app.route("/api/opponent/<team_key>/lineup")
    @_require_auth
    def api_opponent_lineup(team_key):
        import time
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster
        from fantasy_baseball.web.season_data import (
            _opponent_cache, OPPONENT_CACHE_TTL_SECONDS,
            build_opponent_lineup,
        )

        # Check cache
        cached = _opponent_cache.get(team_key)
        if cached and (time.time() - cached["fetched_at"]) < OPPONENT_CACHE_TTL_SECONDS:
            return jsonify(cached["data"])

        # Need standings for team name lookup
        standings = read_cache(CacheKey.STANDINGS)
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

        lineup = build_opponent_lineup(
            roster=roster,
            opponent_name=opponent["name"],
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            rest_of_season_hitters=rest_of_season_hitters,
            rest_of_season_pitchers=rest_of_season_pitchers,
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
        from fantasy_baseball.web.refresh_pipeline import get_refresh_status, run_full_refresh
        status = get_refresh_status()
        if status["running"]:
            return jsonify({"status": "already_running"})
        thread = threading.Thread(target=run_full_refresh, daemon=True)
        thread.start()
        return jsonify({"status": "started"})

    @app.route("/api/refresh-status")
    def api_refresh_status():
        from fantasy_baseball.web.refresh_pipeline import get_refresh_status
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

