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
        standings_data = None
        if raw_standings:
            from fantasy_baseball.web.season_data import format_standings_for_display
            config = _load_config()
            standings_data = format_standings_for_display(
                raw_standings, config.team_name
            )
        return render_template(
            "season/standings.html",
            meta=meta,
            active_page="standings",
            standings=standings_data,
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
