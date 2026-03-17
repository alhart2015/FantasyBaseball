"""Flask application for the draft dashboard.

Serves a read-only browser dashboard that visualises the current draft
state.  State is read from a JSON file written atomically by the CLI
loop in ``run_draft.py``.
"""
from pathlib import Path

from flask import Flask, jsonify, render_template

from fantasy_baseball.draft.state import read_state

DEFAULT_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "draft_state.json"


def create_app(state_path: Path | None = None) -> Flask:
    """Application factory.

    Parameters
    ----------
    state_path:
        Path to the ``draft_state.json`` file.  Defaults to
        ``<project_root>/data/draft_state.json``.
    """
    if state_path is None:
        state_path = DEFAULT_STATE_PATH

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["STATE_PATH"] = state_path

    @app.route("/")
    def index():
        state = read_state(app.config["STATE_PATH"])
        return render_template("dashboard.html", state=state)

    @app.route("/api/state")
    def api_state():
        state = read_state(app.config["STATE_PATH"])
        return jsonify(state)

    return app
