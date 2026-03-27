"""WSGI entry point for gunicorn."""
from fantasy_baseball.web.season_app import create_app

app = create_app()
