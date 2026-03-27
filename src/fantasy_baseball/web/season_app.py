"""Season dashboard Flask application."""

import os
from pathlib import Path

from flask import Flask

from fantasy_baseball.web.season_routes import register_routes


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
    register_routes(app)
    return app
