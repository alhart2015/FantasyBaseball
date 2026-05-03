"""Route handlers for the season dashboard."""

import functools
import hmac
import logging
import math
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from fantasy_baseball.lineup.roster_audit import (
    HITTER_SOURCE_POSITIONS,
    POSITION_POOL_SIZES,
    RP_SV_THRESHOLD,
)
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.standings import (
    ProjectedStandings,
    Standings,
    StandingsEntry,
)
from fantasy_baseball.scoring import team_sds_from_json
from fantasy_baseball.utils.constants import ALL_CATEGORIES, RATE_STATS, Category
from fantasy_baseball.web.season_data import (
    CacheKey,
    read_cache_dict,
    read_cache_list,
    read_meta,
)


def _standings_from_cache(raw: dict) -> Standings:
    """Build a typed ``Standings`` from the canonical cache payload.

    The refresh pipeline writes ``Standings.to_json()`` (shape
    ``{"effective_date", "teams": [...]}``). Route handlers call this
    to rehydrate on read.
    """
    return Standings.from_json(raw)


def _projected_from_cache(raw: dict) -> ProjectedStandings:
    """Build a typed ``ProjectedStandings`` from the canonical cache payload."""
    return ProjectedStandings.from_json(raw)


def _team_sds_from_cache(raw: dict | None) -> dict[str, dict[Category, float]] | None:
    """Deserialize a cached ``team_sds`` payload (or ``None``) into typed form."""
    return team_sds_from_json(raw) if raw else None


def _projected_as_standings(raw: dict) -> Standings:
    """Adapt a cached ``ProjectedStandings`` into a ``Standings`` for display.

    ``format_standings_for_display`` takes a ``Standings``, but the
    projected/preseason standings column renders the same shape from
    a ``ProjectedStandings`` payload. We fill ``team_key``/``rank`` with
    placeholders and let the display layer compute rank from roto totals.
    """
    projected = ProjectedStandings.from_json(raw)
    return Standings(
        effective_date=projected.effective_date,
        entries=[
            StandingsEntry(
                team_name=e.team_name,
                team_key="",
                rank=0,
                stats=e.stats,
            )
            for e in projected.entries
        ],
    )


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
        if (
            request.path.startswith("/api/")
            or request.is_json
            or request.content_type == "application/json"
        ):
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

    roster_raw = read_cache_list(CacheKey.ROSTER) or []
    if not roster_raw:
        return {}
    roster = [Player.from_dict(p) for p in roster_raw]
    return worst_roster_by_position(roster)


def _load_yahoo_league():
    """Get Yahoo league object and user team key."""
    from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
    from fantasy_baseball.lineup.yahoo_roster import fetch_teams, find_user_team_key

    config = _load_config()
    sc = get_yahoo_session()
    league = get_league(sc, config.league_id, config.game_code)
    return league, find_user_team_key(fetch_teams(league), config.team_name)


def load_projections():
    """Load projections from Redis. Returns (hitters, pitchers, rest_of_season_hitters, rest_of_season_pitchers)."""
    import pandas as pd

    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.data.redis_store import get_blended_projections
    from fantasy_baseball.utils.name_utils import normalize_name

    client = get_kv()
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
    ros_cache = read_cache_dict(CacheKey.ROS_PROJECTIONS) or {}
    rest_of_season_hitters = pd.DataFrame(ros_cache.get("hitters", []))
    rest_of_season_pitchers = pd.DataFrame(ros_cache.get("pitchers", []))

    if "name" in hitters.columns:
        hitters["_name_norm"] = hitters["name"].apply(normalize_name)
    if "name" in pitchers.columns:
        pitchers["_name_norm"] = pitchers["name"].apply(normalize_name)
    if not rest_of_season_hitters.empty:
        rest_of_season_hitters["_name_norm"] = rest_of_season_hitters["name"].apply(normalize_name)
    if not rest_of_season_pitchers.empty:
        rest_of_season_pitchers["_name_norm"] = rest_of_season_pitchers["name"].apply(
            normalize_name
        )
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
            projections_dir,
            config.projection_systems,
            config.season_year,
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

        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import get_latest_roster_names

        roster_names = get_latest_roster_names(get_kv())
        if roster_names:
            logger.log(f"Loaded {len(roster_names)} rostered players for quality checks")

        logger.log("Blending ROS projections → Redis...")
        from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros

        ros_h, ros_p = blend_and_cache_ros(
            projections_dir,
            config.projection_systems,
            config.projection_weights,
            roster_names,
            config.season_year,
            progress_cb=_quality_cb,
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


def _optimize_one_side(
    roster,
    projected_standings,
    team_name,
    roster_slots,
    team_sds,
    optimize_hitter_lineup,
    optimize_pitcher_lineup,
    il_slots,
):
    """Run hitter + pitcher optimizers; return ``{player_key: zone}`` dict.

    IL players are passed through unchanged. All non-active hitters/pitchers
    land on BN. Slot labels match the trade-builder UI's slot IDs: when
    ``roster_slots[label]`` is 1 the label is bare (``"3B"``); when > 1 it
    is suffixed with a 1-based index (``"OF1".."OF4"``, ``"UTIL1"``,
    ``"P1".."P9"``). The frontend's ``slotList()`` follows the same
    convention, so the panel's ``pl.zone === slotId`` lookup matches.
    """
    from fantasy_baseball.trades.multi_trade import player_key

    out: dict[str, str] = {}

    # IL passthrough.
    for p in roster:
        if p.selected_position in il_slots:
            out[player_key(p)] = "IL"

    hitters = [
        p
        for p in roster
        if p.player_type == PlayerType.HITTER and p.selected_position not in il_slots
    ]
    pitchers = [
        p
        for p in roster
        if p.player_type == PlayerType.PITCHER and p.selected_position not in il_slots
    ]

    hitter_assignments = optimize_hitter_lineup(
        hitters,
        roster,
        projected_standings,
        team_name,
        roster_slots=roster_slots,
        team_sds=team_sds,
    )
    pitcher_starters, pitcher_bench = optimize_pitcher_lineup(
        pitchers,
        roster,
        projected_standings,
        team_name,
        slots=roster_slots.get("P", 9),
        team_sds=team_sds,
    )

    slot_counters: dict[str, int] = {}
    assigned_hitter_keys = set()
    for a in hitter_assignments:
        # `a.slot` is a Position enum; .value is the canonical slot label.
        base = a.slot.value if hasattr(a.slot, "value") else str(a.slot)
        if roster_slots.get(base, 1) > 1:
            slot_counters[base] = slot_counters.get(base, 0) + 1
            slot_label = f"{base}{slot_counters[base]}"
        else:
            slot_label = base
        out[player_key(a.player)] = slot_label
        assigned_hitter_keys.add(player_key(a.player))
    for p in hitters:
        if player_key(p) not in assigned_hitter_keys:
            out[player_key(p)] = "BN"

    n_pslots = roster_slots.get("P", 9)
    for i, starter in enumerate(pitcher_starters[:n_pslots]):
        # `starter` is a PitcherStarter dataclass — has a .player attribute.
        target = starter.player if hasattr(starter, "player") else starter
        slot = f"P{i + 1}" if n_pslots > 1 else "P"
        out[player_key(target)] = slot
    for p in pitcher_bench:
        out[player_key(p)] = "BN"

    return out


def register_routes(app: Flask) -> None:

    @app.route("/")
    def index():
        return redirect(url_for("standings"))

    @app.route("/standings")
    def standings():
        meta = read_meta()
        raw_standings = read_cache_dict(CacheKey.STANDINGS)
        config = _load_config()
        standings_data = None
        preseason_data = None
        current_projected_data = None
        mc_data = None
        mc_mgmt_data = None
        rest_of_season_mc_data = None
        rest_of_season_mgmt_mc_data = None
        baseline_meta = None
        raw_breakdown = None

        if raw_standings:
            from fantasy_baseball.web.season_data import (
                format_monte_carlo_for_display,
                format_standings_for_display,
            )

            standings_data = format_standings_for_display(
                _standings_from_cache(raw_standings), config.team_name
            )

            raw_projected = read_cache_dict(CacheKey.PROJECTIONS)
            if raw_projected:
                preseason_standings = raw_projected.get(
                    "preseason_standings",
                    raw_projected.get("projected_standings"),
                )
                if preseason_standings:
                    preseason_data = format_standings_for_display(
                        _projected_as_standings(preseason_standings),
                        config.team_name,
                        team_sds=_team_sds_from_cache(raw_projected.get("preseason_team_sds")),
                    )
                if "projected_standings" in raw_projected:
                    current_projected_data = format_standings_for_display(
                        _projected_as_standings(raw_projected["projected_standings"]),
                        config.team_name,
                        team_sds=_team_sds_from_cache(raw_projected.get("team_sds")),
                    )

            raw_breakdown = read_cache_dict(CacheKey.STANDINGS_BREAKDOWN)

            raw_mc = read_cache_dict(CacheKey.MONTE_CARLO)
            if raw_mc:
                baseline_meta = raw_mc.get("baseline_meta")
                if raw_mc.get("base"):
                    mc_data = format_monte_carlo_for_display(raw_mc["base"], config.team_name)
                if raw_mc.get("with_management"):
                    mc_mgmt_data = format_monte_carlo_for_display(
                        raw_mc["with_management"], config.team_name
                    )
                if raw_mc.get("rest_of_season"):
                    rest_of_season_mc_data = format_monte_carlo_for_display(
                        raw_mc["rest_of_season"], config.team_name
                    )
                if raw_mc.get("rest_of_season_with_management"):
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
            standings_breakdown=raw_breakdown,
            mc=mc_data,
            mc_mgmt=mc_mgmt_data,
            baseline_meta=baseline_meta,
            rest_of_season_mc=rest_of_season_mc_data,
            rest_of_season_mgmt_mc=rest_of_season_mgmt_mc_data,
            categories=[c.value for c in ALL_CATEGORIES],
            all_categories=ALL_CATEGORIES,
        )

    @app.route("/trends")
    def trends():
        meta = read_meta()
        return render_template(
            "season/trends.html",
            meta=meta,
            active_page="trends",
            categories=[c.value for c in ALL_CATEGORIES],
        )

    @app.route("/api/trends/series")
    def api_trends_series():
        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.web.season_data import build_trends_series

        config = _load_config()
        return jsonify(build_trends_series(get_kv(), user_team=config.team_name))

    @app.route("/lineup")
    def lineup():
        meta = read_meta()
        roster_raw = read_cache_list(CacheKey.ROSTER)
        optimal_raw = read_cache_dict(CacheKey.LINEUP_OPTIMAL)
        starters_raw = read_cache_list(CacheKey.PROBABLE_STARTERS)
        pending_moves_raw = read_cache_list(CacheKey.PENDING_MOVES) or []

        lineup_data = None
        if roster_raw:
            from fantasy_baseball.web.season_data import format_lineup_for_display

            lineup_data = format_lineup_for_display(roster_raw, optimal_raw)

        # Build teams list for opponent selector dropdown
        from fantasy_baseball.web.season_data import get_teams_list

        standings_raw = read_cache_dict(CacheKey.STANDINGS)
        config = _load_config()
        standings_typed = (
            _standings_from_cache(standings_raw)
            if standings_raw
            else Standings(effective_date=date.min, entries=[])
        )
        teams_data = get_teams_list(standings_typed, config.team_name)

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
        audit_raw = read_cache_list(CacheKey.ROSTER_AUDIT)
        return render_template(
            "season/roster_audit.html",
            meta=meta,
            active_page="roster_audit",
            audit=audit_raw or [],
            categories=[c.value for c in ALL_CATEGORIES],
            all_categories=ALL_CATEGORIES,
        )

    @app.route("/waivers-trades")
    def waivers_trades():
        meta = read_meta()

        # Build player name list for trade search autocomplete
        roster_raw = read_cache_list(CacheKey.ROSTER) or []
        opp_rosters_raw = read_cache_dict(CacheKey.OPP_ROSTERS) or {}
        my_players = sorted(set(p.get("name", "") for p in roster_raw if p.get("name")))
        opp_players = sorted(
            set(
                p.get("name", "")
                for players in opp_rosters_raw.values()
                for p in players
                if p.get("name")
            )
        )

        config = _load_config()
        return render_template(
            "season/waivers_trades.html",
            meta=meta,
            active_page="waivers_trades",
            my_players=my_players,
            opp_players=opp_players,
            my_roster_data=roster_raw or [],
            opp_rosters_data=opp_rosters_raw or {},
            roster_slots=dict(config.roster_slots),
            config=config,
        )

    @app.route("/api/trade-search", methods=["POST"])
    def api_trade_search():
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.trades.evaluate import (
            search_trades_away,
            search_trades_for,
        )

        data = request.get_json(silent=True) or {}
        player_name = data.get("player_name", "").strip()
        mode = data.get("mode", "")

        if not player_name:
            return jsonify({"error": "player_name is required"}), 400
        if mode not in ("away", "for"):
            return jsonify({"error": "mode must be 'away' or 'for'"}), 400

        config = _load_config()
        standings_raw = read_cache_dict(CacheKey.STANDINGS)
        if not standings_raw:
            return jsonify({"error": "No standings data. Run a refresh first."}), 404

        roster_raw = read_cache_list(CacheKey.ROSTER)
        if not roster_raw:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404

        opp_rosters_raw = read_cache_dict(CacheKey.OPP_ROSTERS)
        if not opp_rosters_raw:
            return jsonify({"error": "No opponent roster data. Run a refresh first."}), 404

        leverage_raw = read_cache_dict(CacheKey.LEVERAGE)
        if not leverage_raw:
            return jsonify({"error": "No leverage data. Run a refresh first."}), 404

        rankings_raw = read_cache_dict(CacheKey.RANKINGS)
        if not rankings_raw:
            return jsonify({"error": "No rankings data. Run a refresh first."}), 404

        proj_cache = read_cache_dict(CacheKey.PROJECTIONS) or {}
        projected_standings_raw = proj_cache.get("projected_standings")

        # Convert cached list[dict] to typed Standings / ProjectedStandings
        # at the boundary. The trades/evaluate API takes typed objects so
        # post-Phase-3.2 callers no longer pass raw cache dicts through.
        standings = _standings_from_cache(standings_raw)
        projected_standings = (
            _projected_from_cache(projected_standings_raw) if projected_standings_raw else None
        )
        team_sds = _team_sds_from_cache(proj_cache.get("team_sds"))

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
            standings=standings,
            leverage_by_team=leverage_raw,
            roster_slots=config.roster_slots,
            rankings=flat_rankings,
            projected_standings=projected_standings,
            team_sds=team_sds,
        )

        results: list[Any]
        if mode == "away":
            results = search_trades_away(**kwargs)
        else:
            results = search_trades_for(**kwargs)

        # Decorate each candidate with canonical name::player_type keys
        # so client code (Compare button) can build /players?compare=...
        # URLs without re-deriving the player_type lookup. Fail loudly on
        # missing player_type — defaulting would silently miscategorize
        # pitcher trades as hitter trades (CLAUDE.md: never key on bare names).
        for group in results:
            for cand in group.get("candidates", []):
                cand["send_key"] = f"{cand['send']}::{cand['send_player_type']}"
                cand["receive_key"] = f"{cand['receive']}::{cand['receive_player_type']}"

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

        roster_raw = read_cache_list(CacheKey.ROSTER) or []
        opp_rosters_raw = read_cache_dict(CacheKey.OPP_ROSTERS) or {}
        ros_cache = read_cache_dict(CacheKey.ROS_PROJECTIONS) or {}

        hart_roster = [Player.from_dict(p) for p in roster_raw]
        opp_rosters = {n: [Player.from_dict(p) for p in ps] for n, ps in opp_rosters_raw.items()}
        pool = build_waiver_pool(hart_roster, opp_rosters, ros_cache)

        q_norm = normalize_name(query)
        matches: list[dict[str, Any]] = [
            {
                "key": key,
                "name": p.name,
                "player_type": p.player_type,
                "positions": p.positions,
            }
            for key, p in pool.items()
            if q_norm in normalize_name(p.name)
        ]
        matches.sort(key=lambda m: str(m["name"]))
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
        roster_raw = read_cache_list(CacheKey.ROSTER)
        opp_rosters_raw = read_cache_dict(CacheKey.OPP_ROSTERS)
        if roster_raw is None or opp_rosters_raw is None:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404
        if opponent not in opp_rosters_raw:
            return jsonify({"error": f"Unknown opponent: {opponent}"}), 400

        proj_cache = read_cache_dict(CacheKey.PROJECTIONS) or {}
        projected_standings_raw = proj_cache.get("projected_standings")
        team_sds = _team_sds_from_cache(proj_cache.get("team_sds"))
        if not projected_standings_raw:
            return jsonify({"error": "No projected standings. Run a refresh first."}), 404

        ros_cache = read_cache_dict(CacheKey.ROS_PROJECTIONS) or {}

        hart_roster = [Player.from_dict(p) for p in roster_raw]
        opp_rosters = {n: [Player.from_dict(p) for p in ps] for n, ps in opp_rosters_raw.items()}
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
            projected_standings=_projected_from_cache(projected_standings_raw),
            team_sds=team_sds,
            roster_slots=config.roster_slots,
        )

        def _serialize_view(view) -> dict:
            return {
                "delta_total": round(view.delta_total, 2),
                "categories": {
                    cat: {
                        "before": round(cv.before, 4),
                        "after": round(cv.after, 4),
                        "delta": round(cv.delta, 4),
                    }
                    for cat, cv in view.categories.items()
                },
            }

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
                "roto": _serialize_view(result.roto),
                "ev_roto": _serialize_view(result.ev_roto),
                "stat_totals": _serialize_view(result.stat_totals),
            }
        )

    @app.route("/api/optimize-trade-lineup", methods=["POST"])
    @_require_auth
    def api_optimize_trade_lineup():
        from fantasy_baseball.lineup.optimizer import (
            optimize_hitter_lineup,
            optimize_pitcher_lineup,
        )
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.models.positions import IL_SLOTS
        from fantasy_baseball.trades.multi_trade import (
            TradeProposal,
            build_waiver_pool,
            evaluate_multi_trade,
            player_key,
        )

        data = request.get_json(silent=True) or {}
        opponent = (data.get("opponent") or "").strip()
        if not opponent:
            return jsonify({"error": "opponent is required"}), 400

        config = _load_config()
        roster_raw = read_cache_list(CacheKey.ROSTER)
        opp_rosters_raw = read_cache_dict(CacheKey.OPP_ROSTERS)
        if roster_raw is None or opp_rosters_raw is None:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404
        if opponent not in opp_rosters_raw:
            return jsonify({"error": f"Unknown opponent: {opponent}"}), 400

        proj_cache = read_cache_dict(CacheKey.PROJECTIONS) or {}
        projected_standings_raw = proj_cache.get("projected_standings")
        team_sds = _team_sds_from_cache(proj_cache.get("team_sds"))
        if not projected_standings_raw:
            return jsonify({"error": "No projected standings. Run a refresh first."}), 404

        ros_cache = read_cache_dict(CacheKey.ROS_PROJECTIONS) or {}
        hart_roster = [Player.from_dict(p) for p in roster_raw]
        opp_rosters = {n: [Player.from_dict(p) for p in ps] for n, ps in opp_rosters_raw.items()}
        waiver_pool = build_waiver_pool(hart_roster, opp_rosters, ros_cache)

        proposal = TradeProposal(
            opponent=opponent,
            send=list(data.get("send") or []),
            receive=list(data.get("receive") or []),
            my_drops=list(data.get("my_drops") or []),
            opp_drops=list(data.get("opp_drops") or []),
            my_adds=list(data.get("my_adds") or []),
        )

        # Resolve key lists into Player objects for the post-trade rosters.
        my_idx = {player_key(p): p for p in hart_roster}
        opp_idx = {player_key(p): p for p in opp_rosters[opponent]}
        try:
            sent = [my_idx[k] for k in proposal.send]
            received = [opp_idx[k] for k in proposal.receive]
            my_drops_p = [my_idx[k] for k in proposal.my_drops]
            opp_drops_p = [opp_idx[k] for k in proposal.opp_drops]
            my_adds_p = [waiver_pool[k] for k in proposal.my_adds]
        except KeyError as exc:
            return jsonify({"ok": False, "reason": f"Unknown player key: {exc}"})

        sent_keys = {player_key(p) for p in sent}
        received_keys = {player_key(p) for p in received}
        my_drop_keys = {player_key(p) for p in my_drops_p}
        opp_drop_keys = {player_key(p) for p in opp_drops_p}

        hart_post = (
            [p for p in hart_roster if player_key(p) not in (sent_keys | my_drop_keys)]
            + received
            + my_adds_p
        )
        opp_post = [
            p for p in opp_rosters[opponent] if player_key(p) not in (received_keys | opp_drop_keys)
        ] + sent

        # Legality check: reuse the full evaluator so size mismatches surface
        # the same way the trade evaluator UI reports them.
        legality_proposal = TradeProposal(
            opponent=opponent,
            send=proposal.send,
            receive=proposal.receive,
            my_drops=proposal.my_drops,
            opp_drops=proposal.opp_drops,
            my_adds=proposal.my_adds,
            my_active_ids=set(),
        )
        projected = _projected_from_cache(projected_standings_raw)
        result = evaluate_multi_trade(
            proposal=legality_proposal,
            hart_name=config.team_name,
            hart_roster=hart_roster,
            opp_rosters=opp_rosters,
            waiver_pool=waiver_pool,
            projected_standings=projected,
            team_sds=team_sds,
            roster_slots=config.roster_slots,
        )
        if not result.legal:
            return jsonify({"ok": False, "reason": result.reason or "Trade is not legal"})

        try:
            my_slots = _optimize_one_side(
                hart_post,
                projected,
                config.team_name,
                config.roster_slots,
                team_sds,
                optimize_hitter_lineup,
                optimize_pitcher_lineup,
                IL_SLOTS,
            )
            opp_slots = _optimize_one_side(
                opp_post,
                projected,
                opponent,
                config.roster_slots,
                team_sds,
                optimize_hitter_lineup,
                optimize_pitcher_lineup,
                IL_SLOTS,
            )
        except ValueError as exc:
            return jsonify({"ok": False, "reason": str(exc)})

        return jsonify({"ok": True, "my_slots": my_slots, "opp_slots": opp_slots})

    @app.route("/players")
    def player_search():
        meta = read_meta()
        return render_template(
            "season/players.html",
            meta=meta,
            active_page="players",
        )

    _ALL_VARIANTS = {"ALL", "ALL_HIT", "ALL_PIT"}
    _VALID_POS = set(HITTER_SOURCE_POSITIONS) | {"SP", "RP"} | _ALL_VARIANTS

    def _ros_eligible_at(d: dict, ptype: PlayerType, pos_list: list[str], pos: str) -> bool:
        if pos == "ALL":
            return True
        if pos == "ALL_HIT":
            return ptype == PlayerType.HITTER
        if pos == "ALL_PIT":
            return ptype == PlayerType.PITCHER
        if pos in HITTER_SOURCE_POSITIONS:
            return ptype == PlayerType.HITTER and pos in pos_list
        if pos == "SP":
            return ptype == PlayerType.PITCHER and (d.get("sv") or 0) < RP_SV_THRESHOLD
        if pos == "RP":
            return ptype == PlayerType.PITCHER and (d.get("sv") or 0) >= RP_SV_THRESHOLD
        return False

    def _default_fa_limit(pos: str) -> int:
        if pos in _ALL_VARIANTS:
            return 20
        return POSITION_POOL_SIZES[pos]

    def _split_rostered_and_fa(
        ros_cache: dict,
        pos_map: dict[str, list[str]],
        owner_map: dict[str, str],
        pos: str,
    ) -> tuple[list[tuple[dict, PlayerType, float]], list[tuple[dict, PlayerType, float]]]:
        """Walk ros_projections once; return (rostered, fa) lists of
        (projection_dict, ptype, sgp) triples eligible for ``pos``.

        SGP is computed once during the walk so the handler can sort and
        build records without re-running ``compute_sgp`` on the survivors.
        """
        from fantasy_baseball.models.player import HitterStats, PitcherStats
        from fantasy_baseball.utils.name_utils import normalize_name

        rostered: list[tuple[dict, PlayerType, float]] = []
        fas: list[tuple[dict, PlayerType, float]] = []
        for pool_key, ptype in (
            ("hitters", PlayerType.HITTER),
            ("pitchers", PlayerType.PITCHER),
        ):
            for d in ros_cache.get(pool_key, []):
                norm = normalize_name(d.get("name", ""))
                pos_list = pos_map.get(norm, [])
                if not _ros_eligible_at(d, ptype, pos_list, pos):
                    continue
                ros: HitterStats | PitcherStats
                if ptype == PlayerType.HITTER:
                    ros = HitterStats.from_dict(d)
                else:
                    ros = PitcherStats.from_dict(d)
                ros.compute_sgp()
                sgp = ros.sgp if ros.sgp is not None else 0.0
                bucket = rostered if owner_map.get(norm) else fas
                bucket.append((d, ptype, sgp))
        return rostered, fas

    def _build_player_record(
        d: dict,
        ptype: PlayerType,
        owner_map: dict[str, str],
        pos_map: dict[str, list[str]],
        rankings_cache: dict,
        audit_index: dict[tuple[str, str], dict],
        worst_by_pos: dict[str, str],
        sgp_hint: float | None = None,
    ) -> dict[str, Any]:
        """Build the per-player browse-page record from a ros_projections row.

        When ``sgp_hint`` is provided the cached SGP is used directly,
        avoiding a redundant ``compute_sgp`` call. Safe because
        ``HitterStats.compute_sgp`` / ``PitcherStats.compute_sgp`` only
        assign ``self.sgp`` and return it.
        """
        from fantasy_baseball.lineup.roster_audit import fa_target_positions
        from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, RankInfo
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.sgp.rankings import lookup_rank
        from fantasy_baseball.utils.name_utils import normalize_name

        name = d.get("name", "")
        norm = normalize_name(name)
        fg_id = d.get("fg_id")
        team = d.get("team")
        if isinstance(team, float) and math.isnan(team):
            team = ""

        ros: HitterStats | PitcherStats
        if ptype == PlayerType.HITTER:
            ros = HitterStats.from_dict(d)
        else:
            ros = PitcherStats.from_dict(d)
        if sgp_hint is not None:
            ros.sgp = sgp_hint
        else:
            ros.compute_sgp()

        rank_info = lookup_rank(rankings_cache, fg_id, name, ptype)
        positions = [Position.parse(pos) for pos in pos_map.get(norm, [])]
        p = Player(
            name=name,
            player_type=ptype,
            team=team or "",
            fg_id=fg_id,
            positions=positions,
            rest_of_season=ros,
            rank=RankInfo.from_dict(rank_info),
        )

        owner = owner_map.get(norm)
        delta_roto = None
        if owner is None:
            sv_for_targets = ros.sv if isinstance(ros, PitcherStats) else 0.0
            targets = fa_target_positions(ptype, p.positions, sv_for_targets)
            for target_pos in targets:
                drop_name = worst_by_pos.get(target_pos)
                if not drop_name:
                    continue
                dr = audit_index.get((drop_name, name))
                if dr is None:
                    continue
                if delta_roto is None or dr["total"] > delta_roto["total"]:
                    delta_roto = dr

        result: dict[str, Any] = {
            "name": name,
            "team": p.team,
            "player_type": ptype,
            "fg_id": fg_id,
            "positions": p.positions,
            "owner": owner,
            "rank": p.rank.rest_of_season,
            "sgp": round(ros.sgp, 2) if ros.sgp is not None else None,
            "delta_roto": delta_roto,
        }
        if isinstance(ros, HitterStats):
            result.update(
                {
                    "R": ros.r,
                    "HR": ros.hr,
                    "RBI": ros.rbi,
                    "SB": ros.sb,
                    "AVG": ros.avg,
                    "h": ros.h,
                    "ab": ros.ab,
                }
            )
        else:
            result.update(
                {
                    "W": ros.w,
                    "K": ros.k,
                    "SV": ros.sv,
                    "ERA": ros.era,
                    "WHIP": ros.whip,
                    "ip": ros.ip,
                    "er": ros.er,
                    "bb": ros.bb,
                    "h_allowed": ros.h_allowed,
                }
            )
        return result

    def _browse_context() -> dict[str, Any]:
        """Read the caches the per-player builder needs in one place."""
        from fantasy_baseball.utils.name_utils import normalize_name

        ros_cache = read_cache_dict(CacheKey.ROS_PROJECTIONS) or {}
        rankings_cache = read_cache_dict(CacheKey.RANKINGS) or {}
        pos_map: dict[str, list[str]] = read_cache_dict(CacheKey.POSITIONS) or {}

        owner_map: dict[str, str] = {}
        for rp in read_cache_list(CacheKey.ROSTER) or []:
            owner_map[normalize_name(rp.get("name", ""))] = "roster"
        for team_name_opp, team_roster in (read_cache_dict(CacheKey.OPP_ROSTERS) or {}).items():
            for rp in team_roster:
                norm = normalize_name(rp.get("name", ""))
                if norm not in owner_map:
                    owner_map[norm] = team_name_opp

        audit_raw = read_cache_list(CacheKey.ROSTER_AUDIT) or []
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

        return {
            "ros_cache": ros_cache,
            "rankings_cache": rankings_cache,
            "pos_map": pos_map,
            "owner_map": owner_map,
            "audit_index": audit_index,
            "worst_by_pos": _compute_worst_roster_by_position(),
        }

    @app.route("/api/players/browse")
    def api_player_browse():
        """Position-scoped player browse.

        Query params:
          pos        — required. C/1B/2B/3B/SS/OF/SP/RP/ALL_HIT/ALL_PIT/ALL.
          fa_limit   — optional. Defaults to POSITION_POOL_SIZES[pos] for
                       specific positions, 20 for ALL variants.
          fa_offset  — optional. Defaults to 0. Paginates the SGP-sorted FA
                       pool. Rostered players are returned only when offset=0.
        """
        pos = request.args.get("pos")
        if not pos or pos not in _VALID_POS:
            return jsonify({"error": "pos must be one of " + ", ".join(sorted(_VALID_POS))}), 400
        try:
            fa_offset = int(request.args.get("fa_offset", 0))
        except ValueError:
            return jsonify({"error": "fa_offset must be an integer"}), 400
        if fa_offset < 0:
            return jsonify({"error": "fa_offset must be >= 0"}), 400
        try:
            fa_limit = int(request.args.get("fa_limit", _default_fa_limit(pos)))
        except ValueError:
            return jsonify({"error": "fa_limit must be an integer"}), 400
        if fa_limit <= 0:
            return jsonify({"error": "fa_limit must be > 0"}), 400

        ctx = _browse_context()
        if not ctx["ros_cache"]:
            return jsonify({"players": [], "has_more_fa": False, "next_fa_offset": fa_offset})

        rostered, fas = _split_rostered_and_fa(
            ctx["ros_cache"],
            ctx["pos_map"],
            ctx["owner_map"],
            pos,
        )
        fas.sort(key=lambda t: t[2], reverse=True)

        fa_slice = fas[fa_offset : fa_offset + fa_limit]
        has_more_fa = (fa_offset + fa_limit) < len(fas)
        next_fa_offset = fa_offset + len(fa_slice)

        rows: list[dict[str, Any]] = []
        if fa_offset == 0:
            for d, ptype, sgp in rostered:
                rows.append(
                    _build_player_record(
                        d,
                        ptype,
                        ctx["owner_map"],
                        ctx["pos_map"],
                        ctx["rankings_cache"],
                        ctx["audit_index"],
                        ctx["worst_by_pos"],
                        sgp_hint=sgp,
                    )
                )
        for d, ptype, sgp in fa_slice:
            rows.append(
                _build_player_record(
                    d,
                    ptype,
                    ctx["owner_map"],
                    ctx["pos_map"],
                    ctx["rankings_cache"],
                    ctx["audit_index"],
                    ctx["worst_by_pos"],
                    sgp_hint=sgp,
                )
            )

        return jsonify(
            {
                "players": rows,
                "has_more_fa": has_more_fa,
                "next_fa_offset": next_fa_offset,
            }
        )

    @app.route("/api/players/compare")
    def api_player_compare():
        """Return projected standings before/after swapping a roster player."""
        from fantasy_baseball.models.player import Player
        from fantasy_baseball.utils.name_utils import normalize_name

        roster_player = request.args.get("roster_player")
        other_name = request.args.get("other_player")
        other_type = request.args.get("other_type")

        if not roster_player or not other_name or not other_type:
            return jsonify(
                {"error": "roster_player, other_player, and other_type are required"}
            ), 400

        roster_cache = read_cache_list(CacheKey.ROSTER)
        if not roster_cache:
            return jsonify({"error": "No roster data available"}), 404

        proj_cache = read_cache_dict(CacheKey.PROJECTIONS) or {}
        projected_standings = proj_cache.get("projected_standings")
        if not projected_standings:
            return jsonify({"error": "No projected standings available"}), 404

        user_roster = [Player.from_dict(p) for p in roster_cache]

        def _float(key, default=0.0):
            try:
                return float(request.args.get(key, default))
            except (TypeError, ValueError):
                return default

        other_player = Player.from_dict(
            {
                "name": other_name,
                "player_type": other_type,
                "r": _float("other_r"),
                "hr": _float("other_hr"),
                "rbi": _float("other_rbi"),
                "sb": _float("other_sb"),
                "h": _float("other_h"),
                "ab": _float("other_ab"),
                "w": _float("other_w"),
                "k": _float("other_k"),
                "sv": _float("other_sv"),
                "ip": _float("other_ip"),
                "er": _float("other_er"),
                "bb": _float("other_bb"),
                "h_allowed": _float("other_ha"),
            }
        )

        # Look up roster player's ROS from ros_projections — the same
        # source the browse page uses.  This prevents the delta from
        # diverging when ros_projections is updated after a refresh.
        roster_player_projection = None
        ros_cache = read_cache_dict(CacheKey.ROS_PROJECTIONS) or {}
        target_norm = normalize_name(roster_player)
        for pool_key in ("hitters", "pitchers"):
            for d in ros_cache.get(pool_key, []):
                if normalize_name(d.get("name", "")) == target_norm:
                    ptype = PlayerType.HITTER if pool_key == "hitters" else PlayerType.PITCHER
                    roster_player_projection = Player.from_dict(
                        {
                            **d,
                            "player_type": ptype,
                        }
                    )
                    break
            if roster_player_projection:
                break

        config = _load_config()

        from fantasy_baseball.web.season_data import compute_comparison_standings

        result = compute_comparison_standings(
            roster_player_name=roster_player,
            other_player=other_player,
            user_roster=user_roster,
            projected_standings=_projected_from_cache(projected_standings),
            user_team_name=config.team_name,
            roster_player_projection=roster_player_projection,
            team_sds=_team_sds_from_cache(proj_cache.get("team_sds")),
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
        from fantasy_baseball.models.player import PitcherStats, Player
        from fantasy_baseball.utils.name_utils import normalize_name

        player_name = request.args.get("player")
        player_type_arg = request.args.get("player_type")
        if not player_name or not player_type_arg:
            return jsonify({"error": "player and player_type are required"}), 400
        try:
            player_type = PlayerType(player_type_arg)
        except ValueError:
            return jsonify({"error": f"invalid player_type: {player_type_arg}"}), 400

        roster_raw = read_cache_list(CacheKey.ROSTER)
        if not roster_raw:
            return jsonify({"error": "No roster data available"}), 404
        user_roster = [Player.from_dict(p) for p in roster_raw]

        proj_cache = read_cache_dict(CacheKey.PROJECTIONS) or {}
        projected_standings_raw = proj_cache.get("projected_standings")
        if not projected_standings_raw:
            return jsonify({"error": "No projected standings available"}), 404

        # Resolve the FA's ROS projection from ros_projections (same source
        # the browse page uses, so totals line up with the table row).
        ros_cache = read_cache_dict(CacheKey.ROS_PROJECTIONS) or {}
        pool_key = "pitchers" if player_type == PlayerType.PITCHER else "hitters"
        target_norm = normalize_name(player_name)
        fa_player = None
        for d in ros_cache.get(pool_key, []):
            if normalize_name(d.get("name", "")) == target_norm:
                fa_player = Player.from_dict({**d, "player_type": player_type})
                break
        if fa_player is None:
            return jsonify({"error": f"{player_name} not found in projections"}), 404

        # Worst roster player at the FA's target positions
        pos_map = read_cache_dict(CacheKey.POSITIONS) or {}
        fa_positions = pos_map.get(target_norm, [])
        fa_ros = fa_player.rest_of_season
        sv = fa_ros.sv if isinstance(fa_ros, PitcherStats) else 0.0
        targets = fa_target_positions(player_type, fa_positions, sv)

        worst_by_pos = _compute_worst_roster_by_position()
        config = _load_config()
        team_sds = _team_sds_from_cache(proj_cache.get("team_sds"))
        projected_standings = _projected_from_cache(projected_standings_raw)

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
        spoe_cache = read_cache_dict(CacheKey.SPOE) or {}
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
            categories=[c.value for c in ALL_CATEGORIES],
            all_categories=ALL_CATEGORIES,
            rate_stats={c.value for c in RATE_STATS},
        )

    @app.route("/transactions")
    def transactions():
        meta = read_meta()
        txn_cache = read_cache_dict(CacheKey.TRANSACTION_ANALYZER) or {}
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

        standings = read_cache_dict(CacheKey.STANDINGS)
        config = _load_config()
        if not standings:
            return jsonify({"teams": [], "user_team_key": None})
        return jsonify(get_teams_list(_standings_from_cache(standings), config.team_name))

    @app.route("/api/opponent/<team_key>/lineup")
    @_require_auth
    def api_opponent_lineup(team_key):
        import time

        from fantasy_baseball.lineup.yahoo_roster import fetch_roster
        from fantasy_baseball.web.season_data import (
            OPPONENT_CACHE_TTL_SECONDS,
            _opponent_cache,
            build_opponent_lineup,
        )

        # Check cache
        cached = _opponent_cache.get(team_key)
        if cached and (time.time() - cached["fetched_at"]) < OPPONENT_CACHE_TTL_SECONDS:
            return jsonify(cached["data"])

        # Need standings for team name lookup
        standings_raw = read_cache_dict(CacheKey.STANDINGS)
        if not standings_raw:
            return jsonify({"error": "No standings data. Run a refresh first."}), 404

        standings = _standings_from_cache(standings_raw)

        # Find opponent entry from team_key
        opponent = next((e for e in standings.entries if e.team_key == team_key), None)
        if opponent is None:
            return jsonify({"error": f"Team key {team_key} not found"}), 404

        try:
            league, _ = _load_yahoo_league()
            roster = fetch_roster(league, team_key)
        except Exception as e:
            return jsonify({"error": f"Failed to fetch roster: {e}"}), 500

        try:
            hitters_proj, pitchers_proj, rest_of_season_hitters, rest_of_season_pitchers = (
                load_projections()
            )
        except Exception as e:
            return jsonify({"error": f"Failed to load projections: {e}"}), 500

        lineup = build_opponent_lineup(
            roster=roster,
            opponent_name=opponent.team_name,
            hitters_proj=hitters_proj,
            pitchers_proj=pitchers_proj,
            rest_of_season_hitters=rest_of_season_hitters,
            rest_of_season_pitchers=rest_of_season_pitchers,
        )

        # Render the same Jinja partials the user roster uses, so opponents
        # inherit rank badges + full tooltips without duplicate JS rendering.
        hitters_html = render_template(
            "season/_lineup_hitters_tbody.html",
            players=lineup["hitters"],
            totals=lineup["hitter_totals"],
        )
        pitchers_html = render_template(
            "season/_lineup_pitchers_tbody.html",
            players=lineup["pitchers"],
            totals=lineup["pitcher_totals"],
        )

        response_data = {
            "team_name": opponent.team_name,
            "team_key": team_key,
            "rank": opponent.rank,
            "hitters_html": hitters_html,
            "pitchers_html": pitchers_html,
        }

        _opponent_cache[team_key] = {
            "data": response_data,
            "fetched_at": time.time(),
        }

        return jsonify(response_data)

    @app.route("/api/refresh", methods=["POST"])
    @_require_auth
    def api_refresh():
        from fantasy_baseball.web.refresh_pipeline import (
            run_full_refresh,
            try_acquire_refresh_slot,
        )

        if not try_acquire_refresh_slot():
            return jsonify({"status": "already_running"})
        thread = threading.Thread(target=run_full_refresh, daemon=True)
        thread.start()
        return jsonify({"status": "started"})

    @app.route("/api/refresh-status")
    def api_refresh_status():
        from fantasy_baseball.web.refresh_pipeline import get_refresh_status

        return jsonify(get_refresh_status())

    @app.route("/api/sync-from-remote", methods=["POST"])
    @_require_auth
    def api_sync_from_remote():
        """Pull the Upstash KV down to the local SQLite KV.

        Lightweight alternative to ``/api/refresh`` for local dev: no
        Yahoo fetch, no recomputation — just a fresh copy of whatever
        the most recent QStash-driven refresh wrote to Upstash. Only
        meaningful off-Render; on Render the Upstash KV is authoritative
        and has nothing to sync to.
        """
        from fantasy_baseball.data.kv_store import is_remote
        from fantasy_baseball.data.kv_sync import sync_remote_to_local

        if is_remote():
            return jsonify(
                {"error": "sync-from-remote is local-only; on Render Upstash is authoritative"}
            ), 400

        try:
            stats = sync_remote_to_local()
        except Exception as exc:
            log.warning(f"sync_remote_to_local failed: {exc}")
            return jsonify({"error": str(exc)}), 500

        return jsonify({"ok": True, "summary": stats.summary()})

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
