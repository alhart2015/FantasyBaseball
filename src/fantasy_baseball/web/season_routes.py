"""Route handlers for the season dashboard."""

import threading
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

from fantasy_baseball.utils.constants import ALL_CATEGORIES
from fantasy_baseball.web.season_data import read_cache, read_meta

_config = None


def _load_config():
    global _config
    if _config is None:
        from fantasy_baseball.config import load_config
        config_path = Path(__file__).resolve().parents[3] / "config" / "league.yaml"
        _config = load_config(config_path)
    return _config


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

        return render_template(
            "season/standings.html",
            meta=meta,
            active_page="standings",
            standings=standings_data,
            projected=projected_data,
            mc=mc_data,
            mc_mgmt=mc_mgmt_data,
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

        return render_template(
            "season/lineup.html",
            meta=meta,
            active_page="lineup",
            lineup=lineup_data,
            starters=starters_raw,
        )

    @app.route("/api/optimize", methods=["POST"])
    def api_optimize():
        from fantasy_baseball.web.season_data import run_optimize
        try:
            result = run_optimize()
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/waivers-trades")
    def waivers_trades():
        meta = read_meta()
        waivers_raw = read_cache("waivers")
        trades_raw = read_cache("trades")
        return render_template(
            "season/waivers_trades.html",
            meta=meta,
            active_page="waivers_trades",
            waivers=waivers_raw or [],
            trades=trades_raw or [],
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

        from fantasy_baseball.web.season_data import compute_trade_standings_impact
        config = _load_config()
        result = compute_trade_standings_impact(trade=trades_raw[idx], standings=standings_raw, user_team_name=config.team_name)
        return jsonify(result)

    @app.route("/sql", methods=["GET", "POST"])
    def sql_runner():
        meta = read_meta()
        query = ""
        columns = None
        rows = None
        row_count = None
        error = None

        if request.method == "POST":
            query = request.form.get("query", "").strip()
            table_name = request.form.get("schema_table", "").strip()

            if request.form.get("action") == "tables":
                query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            elif request.form.get("action") == "schema" and table_name:
                query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"

            if query:
                from fantasy_baseball.data.db import get_connection
                conn = get_connection()
                try:
                    cursor = conn.execute(query)
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

    @app.route("/api/refresh", methods=["POST"])
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

    @app.route("/api/fetch-mlb", methods=["POST"])
    def api_fetch_mlb():
        from fantasy_baseball.web.season_data import get_refresh_status, run_mlb_fetch

        status = get_refresh_status()
        if status["running"]:
            return jsonify({"status": "already_running"})
        thread = threading.Thread(target=run_mlb_fetch, daemon=True)
        thread.start()
        return jsonify({"status": "started"})
