"""Flask application for the draft dashboard.

Serves a read-only browser dashboard that visualises the current draft
state.  State is read from a JSON file written atomically by the CLI
loop in ``run_draft.py``.

Endpoints
---------
``GET /``
    Renders the dashboard HTML page.

``GET /api/board``
    Returns the full player board (300+ rows).  Intended to be fetched
    **once** on page load and cached by the client.

``GET /api/state``
    Returns draft state.  Accepts an optional ``?since=<version>`` query
    parameter:

    * **No ``since``**: returns the full state dict including
      ``available_players`` (backward compatible with older clients).
    * **``since=<int>``**: returns a delta containing only the fields
      that changed since that version.  If the requested version is stale
      (i.e. more than one version behind the current delta file), the
      full state is returned with ``full_state: true`` so the client can
      reset.
"""

import logging
import os
from pathlib import Path
from typing import Any, cast

import yaml
from flask import Flask, jsonify, render_template, request

from fantasy_baseball.draft import draft_controller
from fantasy_baseball.draft.state import read_board, read_delta, read_state, write_state

DEFAULT_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "draft_state.json"


def _load_league_yaml() -> dict[str, Any]:
    path = os.environ.get("DRAFT_LEAGUE_YAML_PATH") or str(
        Path(__file__).resolve().parents[3] / "config" / "league.yaml"
    )
    with open(path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def _teams_by_position(league_yaml: dict[str, Any]) -> dict[int, str]:
    teams = league_yaml["draft"]["teams"]
    if isinstance(teams, dict):
        return {int(k): v for k, v in teams.items()}
    return {i + 1: v for i, v in enumerate(teams)}


def _resolve_keeper_factory(league_yaml: dict[str, Any]):  # noqa: ARG001
    """Returns a keeper-resolver callable.

    For now, uses a placeholder that echoes the name. Phase 4+ wires the
    real board-backed resolver (draft.search.find_player_by_name) that
    consumes ``league_yaml`` and the ``team`` argument to scope the lookup.
    """

    def _resolver(name: str, _team: str) -> tuple[str, str, str]:
        return (f"{name}::hitter", name, "OF")

    return _resolver


def _register_writer_routes(app):
    @app.post("/api/new-draft")
    def new_draft():
        league_yaml = _load_league_yaml()
        try:
            state = draft_controller.start_new_draft(
                league_yaml,
                resolve_keeper=_resolve_keeper_factory(league_yaml),
            )
        except draft_controller.UnresolvedKeeperError as e:
            return jsonify({"error": str(e)}), 400
        write_state(state, app.config["STATE_PATH"])
        return jsonify(state)

    @app.post("/api/pick")
    def record_pick():
        body = request.get_json(silent=True) or {}
        required = ("player_id", "player_name", "position", "team")
        missing = [k for k in required if k not in body]
        if missing:
            return jsonify({"error": f"missing fields: {missing}"}), 400
        league_yaml = _load_league_yaml()
        state = draft_controller.resume_or_init(app.config["STATE_PATH"])
        try:
            new_state = draft_controller.apply_pick(
                state,
                player_id=body["player_id"],
                player_name=body["player_name"],
                position=body["position"],
                team=body["team"],
                teams_by_position=_teams_by_position(league_yaml),
            )
        except draft_controller.WrongTeamError as e:
            return jsonify({"error": str(e)}), 409
        except draft_controller.AlreadyDraftedError as e:
            return jsonify({"error": str(e)}), 409
        write_state(new_state, app.config["STATE_PATH"])
        return jsonify(new_state)

    @app.post("/api/undo")
    def undo():
        league_yaml = _load_league_yaml()
        state = draft_controller.resume_or_init(app.config["STATE_PATH"])
        new_state = draft_controller.undo_pick(
            state,
            teams_by_position=_teams_by_position(league_yaml),
        )
        write_state(new_state, app.config["STATE_PATH"])
        return jsonify(new_state)

    @app.post("/api/on-the-clock")
    def override_on_the_clock():
        body = request.get_json(silent=True) or {}
        if "team" not in body:
            return jsonify({"error": "missing fields: ['team']"}), 400
        state = draft_controller.resume_or_init(app.config["STATE_PATH"])
        new_state = {**state, "on_the_clock": body["team"]}
        write_state(new_state, app.config["STATE_PATH"])
        return jsonify(new_state)

    @app.post("/api/reset")
    def reset():
        body = request.get_json(silent=True) or {}
        if body.get("confirm") != "RESET":
            return jsonify({"error": "missing confirm"}), 400
        for p in (
            app.config["STATE_PATH"],
            app.config["BOARD_PATH"],
            app.config["DELTA_PATH"],
        ):
            Path(p).unlink(missing_ok=True)
        return jsonify({"reset": True})

    @app.get("/api/recs")
    def recs():
        # Real implementation lands in Phase 5.
        return jsonify({"error": "not yet implemented"}), 501


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

    state_path = Path(state_path)
    board_path = state_path.with_name(state_path.stem + "_board" + state_path.suffix)
    delta_path = state_path.with_name(state_path.stem + "_delta" + state_path.suffix)

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["STATE_PATH"] = state_path
    app.config["BOARD_PATH"] = board_path
    app.config["DELTA_PATH"] = delta_path

    # Suppress Flask/werkzeug request logging (htmx polls every 2s)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.route("/")
    def index():
        state = read_state(app.config["STATE_PATH"])
        roster_slots = state.get("roster_slots", {})
        return render_template("dashboard.html", state=state, roster_slots=roster_slots)

    @app.route("/api/board")
    def api_board():
        """Return the full player board (fetched once by the client)."""
        board = read_board(app.config["BOARD_PATH"])
        return jsonify(board)

    @app.route("/api/state")
    def api_state():
        """Return full state or delta depending on ``?since=`` param.

        * No ``since`` param  -> full state (backward compatible).
        * ``since=<version>`` -> delta if available, else full state with
          ``full_state: true``.
        """
        since_param = request.args.get("since")

        if since_param is None:
            # Legacy / initial load: return full state.
            state = read_state(app.config["STATE_PATH"])
            return jsonify(state)

        # Client wants a delta.
        try:
            since_version = int(since_param)
        except (ValueError, TypeError):
            # Bad param: fall back to full state.
            state = read_state(app.config["STATE_PATH"])
            return jsonify(state)

        # Read the latest delta file.
        delta = read_delta(app.config["DELTA_PATH"])
        if not delta:
            # No delta file yet: return full state.
            state = read_state(app.config["STATE_PATH"])
            return jsonify(state)

        current_version = delta.get("version", 0)

        if since_version >= current_version:
            # Client is already up-to-date.
            return jsonify({"version": current_version, "no_change": True})

        if since_version == current_version - 1:
            # Client is exactly one version behind: the delta file is
            # an accurate diff from their state.
            return jsonify(delta)

        # Client is multiple versions behind (or version 0 / first load):
        # return the full state so they can reset.
        state = read_state(app.config["STATE_PATH"])
        # Strip available_players to save bandwidth when client has
        # the board cached.  Include a flag so the client knows this
        # is a full-state reset.
        state_slim = {k: v for k, v in state.items() if k != "available_players"}
        state_slim["full_state"] = True
        return jsonify(state_slim)

    _register_writer_routes(app)

    return app
