"""Season dashboard Flask application."""

from pathlib import Path

from flask import Flask

from fantasy_baseball.web.season_routes import register_routes


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    register_routes(app)
    return app
