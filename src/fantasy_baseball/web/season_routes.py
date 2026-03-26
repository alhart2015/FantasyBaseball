"""Route handlers for the season dashboard."""

from pathlib import Path

from flask import Flask, redirect, render_template, url_for

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
            categories=["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"],
        )

    @app.route("/lineup")
    def lineup():
        meta = read_meta()
        return render_template("season/lineup.html", meta=meta, active_page="lineup")

    @app.route("/waivers-trades")
    def waivers_trades():
        meta = read_meta()
        return render_template(
            "season/waivers_trades.html", meta=meta, active_page="waivers_trades"
        )
