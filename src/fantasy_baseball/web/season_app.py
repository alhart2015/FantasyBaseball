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

    # Ensure database tables exist (handles first deploy or missing DB)
    from fantasy_baseball.data.db import get_connection, create_tables
    conn = get_connection()
    create_tables(conn)
    conn.close()

    register_routes(app)
    return app
