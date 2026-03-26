"""Route handlers for the season dashboard."""

from flask import Flask, redirect, render_template, url_for

from fantasy_baseball.web.season_data import read_cache, read_meta


def register_routes(app: Flask) -> None:

    @app.route("/")
    def index():
        return redirect(url_for("standings"))

    @app.route("/standings")
    def standings():
        meta = read_meta()
        return render_template("season/standings.html", meta=meta, active_page="standings")

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
