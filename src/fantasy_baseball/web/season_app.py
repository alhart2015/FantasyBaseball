"""Season dashboard Flask application."""

import logging
import os
from datetime import timedelta
from pathlib import Path

from flask import Flask

from fantasy_baseball.utils.rate_stats import format_ip
from fantasy_baseball.web.season_routes import register_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

# Keep users signed in for a month so the site doesn't re-prompt every
# browser restart. The session cookie itself carries the auth flag; Flask
# extends its expiry on each request once ``session.permanent`` is set.
SESSION_LIFETIME = timedelta(days=30)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")
    app.permanent_session_lifetime = SESSION_LIFETIME
    # Render terminates TLS in front of gunicorn, so production cookies
    # must be Secure. Local dev runs on plain http://localhost so the
    # flag would prevent the cookie from being set at all.
    if os.environ.get("RENDER"):
        app.config["SESSION_COOKIE_SECURE"] = True
    app.jinja_env.filters["format_ip"] = format_ip
    register_routes(app)
    return app
