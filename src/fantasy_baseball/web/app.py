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


def _load_board_cached():
    """Load the preseason draft board once per process.

    Returns ``None`` if the board is unavailable (which triggers a 501
    from ``/api/recs`` in the real-data path). Real board loading is
    deferred past Phase 5 — monkeypatched tests prove the downstream
    wiring works.
    """
    if getattr(_load_board_cached, "_board", None) is None:
        try:
            raise NotImplementedError("board loading wired in Phase 5.2")
        except NotImplementedError:
            _load_board_cached._board = None  # type: ignore[attr-defined]
    return _load_board_cached._board  # type: ignore[attr-defined]


def _build_rec_inputs(board, team, state_path):
    """Gather inputs ``rank_candidates`` needs.

    The real implementation is deferred past Phase 5 — tests monkeypatch
    this helper. In the real-data path, raising ``NotImplementedError``
    propagates up to ``/api/recs`` as a 501.
    """
    raise NotImplementedError("real _build_rec_inputs lands in Phase 5.2")


def _picks_until_next_turn(state, team):
    """Count opponent picks between now and ``team``'s next turn.

    Falls back to ``0`` when the controller can't determine it (e.g. the
    draft is already done, or state is malformed).
    """
    try:
        from fantasy_baseball.draft.draft_controller import _snake_order

        league_yaml = _load_league_yaml()
        teams_by_position = _teams_by_position(league_yaml)
        order = _snake_order(teams_by_position, num_rounds=30)
        picks_so_far = len(state.get("picks", []))
        for i in range(picks_so_far + 1, len(order)):
            if order[i] == team:
                return i - picks_so_far
        return 0
    except Exception:
        return 0


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
        from fantasy_baseball.draft import eroto_recs

        team = request.args.get("team")
        if not team:
            return jsonify({"error": "missing team parameter"}), 400
        try:
            board = _load_board_cached()
            (
                candidates,
                replacements,
                projected_standings,
                team_sds,
                adp_table,
            ) = _build_rec_inputs(board, team, app.config["STATE_PATH"])
        except NotImplementedError as e:
            return jsonify({"error": str(e)}), 501
        state = draft_controller.resume_or_init(app.config["STATE_PATH"])
        picks_until_next = _picks_until_next_turn(state, team)
        rows = eroto_recs.rank_candidates(
            candidates=candidates,
            replacements=replacements,
            team_name=team,
            projected_standings=projected_standings,
            team_sds=team_sds,
            picks_until_next_turn=picks_until_next,
            adp_table=adp_table,
        )
        return jsonify([row.__dict__ for row in rows[:10]])

    @app.get("/api/roster")
    def roster():
        team = request.args.get("team")
        if not team:
            return jsonify({"error": "missing team parameter"}), 400
        state = draft_controller.resume_or_init(app.config["STATE_PATH"])
        drafted = [
            p for p in (state.get("keepers", []) + state.get("picks", [])) if p["team"] == team
        ]
        roster_slots = _load_league_yaml().get("roster_slots", {}) or {}
        rows = [
            {"slot": p["position"], "name": p["player_name"], "replacement": False} for p in drafted
        ]
        total = sum(roster_slots.values()) - roster_slots.get("IL", 0) if roster_slots else 0
        for _ in range(max(0, total - len(rows))):
            rows.append({"slot": "?", "name": "Replacement", "replacement": True})
        return jsonify(rows)

    @app.get("/api/standings")
    def standings():
        state = draft_controller.resume_or_init(app.config["STATE_PATH"])
        cache = state.get("projected_standings_cache", {})
        rows = [
            {
                "team": team,
                "total": sum(c["point_estimate"] for c in cats.values()),
                "sd": (sum(c["sd"] ** 2 for c in cats.values())) ** 0.5,
            }
            for team, cats in cache.items()
        ]
        rows.sort(key=lambda r: r["total"], reverse=True)
        return jsonify(rows)


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
