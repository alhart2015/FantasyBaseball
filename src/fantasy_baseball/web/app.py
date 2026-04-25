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
from fantasy_baseball.draft.state import (
    StateKey,
    read_board,
    read_delta,
    read_state,
    write_state,
)

DEFAULT_STATE_PATH = Path(__file__).resolve().parents[3] / "data" / "draft_state.json"

# Flask app.config keys. Module-level constants so a typo at a read site
# is caught by mypy (against `Final[str]`) rather than silently returning
# None from app.config.get and crashing later.
CFG_STATE_PATH: str = "STATE_PATH"
CFG_BOARD_PATH: str = "BOARD_PATH"
CFG_DELTA_PATH: str = "DELTA_PATH"

# /api/recs ERoto-scores every remaining candidate, which is O(N * score_roto).
# Past pick ~50 by VAR there's effectively no chance a player ranks in the
# top-10 immediate_delta — VAR is monotone with raw stat contribution, so a
# bottom-of-board player has near-zero or negative delta vs replacement.
# Capping the candidate pool turns a 6.5s response into ~200ms.
# Board is var-sorted desc by build_draft_board, so this is just a slice.
RECS_CANDIDATE_POOL_SIZE: int = 200


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


def _resolve_keeper_factory(app: Flask):
    """Return a keeper-resolver callable backed by the on-disk board JSON.

    Looks up by normalized name with VAR tie-break — mirrors the logic
    in ``scripts/run_draft.py``'s CLI keeper matching.
    """
    from fantasy_baseball.draft.draft_controller import KeeperNotFound
    from fantasy_baseball.utils.name_utils import normalize_name

    board = read_board(app.config[CFG_BOARD_PATH])
    # Index rows by normalized name. Multiple rows per name are rare but
    # possible (namesakes) — we keep a list and tie-break at lookup time.
    by_norm: dict[str, list[dict[str, Any]]] = {}
    for row in board:
        key = row.get("name_normalized") or normalize_name(row.get("name", ""))
        by_norm.setdefault(key, []).append(row)

    def _resolver(name: str, _team: str) -> tuple[str, str, str]:
        norm = normalize_name(name)
        candidates = by_norm.get(norm, [])
        if not candidates:
            raise KeeperNotFound(f"no board match for keeper {name!r}")
        # Tie-break by var (fall back to 0 if missing).
        best = max(candidates, key=lambda r: r.get("var") or 0.0)
        return (
            best["player_id"],
            best["name"],
            best.get("best_position") or best["positions"][0],
        )

    return _resolver


def _load_board_cached(app):
    """Return the on-disk draft board, cached per-app.

    Reads from ``app.config[CFG_BOARD_PATH]`` once and stashes the result
    on the app instance. Returns ``None`` when the board file does not
    exist or is empty — callers should treat that as "no real-data path
    available yet" and fall back to a 503.
    """
    cached = getattr(app, "_draft_board_cache", "__missing__")
    if cached != "__missing__":
        return cached
    rows = read_board(app.config[CFG_BOARD_PATH])
    app._draft_board_cache = rows if rows else None
    return app._draft_board_cache


def _build_rec_inputs(app, state, league_yaml):
    """Gather the five inputs ``rank_candidates`` needs.

    Loads the cached board, normalises into ``Player`` objects, and
    delegates to :mod:`fantasy_baseball.draft.recs_integration`. Raises
    ``RuntimeError`` when no board is available so the caller can map
    that to a 503.
    """
    from fantasy_baseball.draft import recs_integration

    board = _load_board_cached(app)
    if not board:
        raise RuntimeError(
            "draft board not loaded — run scripts/run_draft.py at least once to cache the board"
        )
    return recs_integration.compute_rec_inputs(
        state,
        app.config[CFG_BOARD_PATH],
        league_yaml,
    )


def _picks_until_next_turn(state, team):
    """Count opponent picks between ``team``'s upcoming pick and the one after.

    For VOPN: if ``team`` were to take a player at their next turn, this is
    the number of opponent picks before they pick again. The dashboard
    typically queries ``team = on_the_clock``; works either way.

    Falls back to ``0`` when the controller can't determine it (draft done,
    state malformed, team unknown).
    """
    try:
        from fantasy_baseball.draft.draft_controller import _snake_order

        league_yaml = _load_league_yaml()
        teams_by_position = _teams_by_position(league_yaml)
        order = _snake_order(teams_by_position, num_rounds=30)
        picks_so_far = len(state.get(StateKey.PICKS, []))
        # First, find team's upcoming pick at or after picks_so_far.
        upcoming_idx = next((i for i in range(picks_so_far, len(order)) if order[i] == team), None)
        if upcoming_idx is None:
            return 0
        # Then find the pick AFTER that.
        next_idx = next((i for i in range(upcoming_idx + 1, len(order)) if order[i] == team), None)
        if next_idx is None:
            return 0
        # Indices in (upcoming, next) are opponent picks; team picks at next.
        return next_idx - upcoming_idx - 1
    except Exception:
        return 0


def _attach_standings_cache(
    app: Flask, state: dict[str, Any], league_yaml: dict[str, Any]
) -> dict[str, Any]:
    """Best-effort: compute projected standings and stash them on ``state``.

    Swallows errors (missing board, malformed league.yaml, etc.) so a
    pick can never fail because of standings wiring. Returns the state
    with or without ``projected_standings_cache`` populated. The next
    successful pick will refresh the cache; until then, ``/api/standings``
    just returns ``[]``.
    """
    try:
        board = _load_board_cached(app)
        if not board:
            return state
        from fantasy_baseball.draft import recs_integration

        inputs = recs_integration.compute_rec_inputs(
            state,
            app.config[CFG_BOARD_PATH],
            league_yaml,
        )
        rows = recs_integration.compute_standings_cache(inputs.projected_standings, inputs.team_sds)
        state[StateKey.PROJECTED_STANDINGS_CACHE] = recs_integration.serialize_standings_cache(rows)
    except Exception:
        logging.getLogger(__name__).exception(
            "failed to refresh projected_standings_cache; leaving stale cache"
        )
    return state


def _register_writer_routes(app):
    @app.post("/api/new-draft")
    def new_draft():
        league_yaml = _load_league_yaml()
        try:
            state = draft_controller.start_new_draft(
                league_yaml,
                resolve_keeper=_resolve_keeper_factory(app),
            )
        except draft_controller.UnresolvedKeeperError as e:
            return jsonify({"error": str(e)}), 400
        state = _attach_standings_cache(app, state, league_yaml)
        write_state(state, app.config[CFG_STATE_PATH])
        return jsonify(state)

    @app.post("/api/pick")
    def record_pick():
        body = request.get_json(silent=True) or {}
        required = ("player_id", "player_name", "position", "team")
        missing = [k for k in required if k not in body]
        if missing:
            return jsonify({"error": f"missing fields: {missing}"}), 400
        league_yaml = _load_league_yaml()
        state = draft_controller.resume_or_init(app.config[CFG_STATE_PATH])
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
        new_state = _attach_standings_cache(app, new_state, league_yaml)
        write_state(new_state, app.config[CFG_STATE_PATH])
        return jsonify(new_state)

    @app.post("/api/undo")
    def undo():
        league_yaml = _load_league_yaml()
        state = draft_controller.resume_or_init(app.config[CFG_STATE_PATH])
        new_state = draft_controller.undo_pick(
            state,
            teams_by_position=_teams_by_position(league_yaml),
        )
        new_state = _attach_standings_cache(app, new_state, league_yaml)
        write_state(new_state, app.config[CFG_STATE_PATH])
        return jsonify(new_state)

    @app.post("/api/on-the-clock")
    def override_on_the_clock():
        body = request.get_json(silent=True) or {}
        if "team" not in body:
            return jsonify({"error": "missing fields: ['team']"}), 400
        league_yaml = _load_league_yaml()
        state = draft_controller.resume_or_init(app.config[CFG_STATE_PATH])
        new_state = {**state, "on_the_clock": body["team"]}
        new_state = _attach_standings_cache(app, new_state, league_yaml)
        write_state(new_state, app.config[CFG_STATE_PATH])
        return jsonify(new_state)

    @app.get("/api/recs")
    def recs():
        from fantasy_baseball.draft import eroto_recs

        team = request.args.get("team")
        if not team:
            return jsonify({"error": "missing team parameter"}), 400
        league_yaml = _load_league_yaml()
        state = draft_controller.resume_or_init(app.config[CFG_STATE_PATH])
        try:
            inputs = _build_rec_inputs(app, state, league_yaml)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        picks_until_next = _picks_until_next_turn(state, team)
        rows = eroto_recs.rank_candidates(
            candidates=inputs.candidates[:RECS_CANDIDATE_POOL_SIZE],
            replacements=inputs.replacements,
            team_name=team,
            projected_standings=inputs.projected_standings,
            team_sds=inputs.team_sds,
            picks_until_next_turn=picks_until_next,
            adp_table=inputs.adp_table,
        )
        return jsonify([row.__dict__ for row in rows[:10]])

    @app.get("/api/meta")
    def meta():
        """League metadata the dashboard needs once on page load.

        Used to populate the Team Inspector's team dropdown (which the
        polling /api/state response doesn't carry).
        """
        league_yaml = _load_league_yaml()
        teams_by_position = _teams_by_position(league_yaml)
        return jsonify(
            {
                "teams": [teams_by_position[i] for i in sorted(teams_by_position)],
                "user_team": (league_yaml.get("league") or {}).get("team_name"),
            }
        )

    @app.get("/api/roster")
    def roster():
        team = request.args.get("team")
        if not team:
            return jsonify({"error": "missing team parameter"}), 400
        state = draft_controller.resume_or_init(app.config[CFG_STATE_PATH])
        drafted = [
            p
            for p in (state.get(StateKey.KEEPERS, []) + state.get(StateKey.PICKS, []))
            if p["team"] == team
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
        from fantasy_baseball.draft import recs_integration

        state = draft_controller.resume_or_init(app.config[CFG_STATE_PATH])
        cache = recs_integration.deserialize_standings_cache(
            state.get(StateKey.PROJECTED_STANDINGS_CACHE) or {}
        )
        # Sort on the typed dataclass so mypy sees float, not object.
        sorted_items = sorted(cache.items(), key=lambda kv: kv[1].total, reverse=True)
        return jsonify(
            [{"team": team, "total": row.total, "sd": row.total_sd} for team, row in sorted_items]
        )


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
    app.config[CFG_STATE_PATH] = state_path
    app.config[CFG_BOARD_PATH] = board_path
    app.config[CFG_DELTA_PATH] = delta_path

    # Suppress Flask/werkzeug request logging (htmx polls every 2s)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.route("/")
    def index():
        state = read_state(app.config[CFG_STATE_PATH])
        roster_slots = state.get("roster_slots", {})
        return render_template("dashboard.html", state=state, roster_slots=roster_slots)

    @app.route("/api/board")
    def api_board():
        """Return the full player board (fetched once by the client)."""
        board = read_board(app.config[CFG_BOARD_PATH])
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
            state = read_state(app.config[CFG_STATE_PATH])
            return jsonify(state)

        # Client wants a delta.
        try:
            since_version = int(since_param)
        except (ValueError, TypeError):
            # Bad param: fall back to full state.
            state = read_state(app.config[CFG_STATE_PATH])
            return jsonify(state)

        # Read the latest delta file.
        delta = read_delta(app.config[CFG_DELTA_PATH])
        if not delta:
            # No delta file yet: return full state.
            state = read_state(app.config[CFG_STATE_PATH])
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
        state = read_state(app.config[CFG_STATE_PATH])
        # Strip available_players to save bandwidth when client has
        # the board cached.  Include a flag so the client knows this
        # is a full-state reset.
        state_slim = {k: v for k, v in state.items() if k != "available_players"}
        state_slim["full_state"] = True
        return jsonify(state_slim)

    _register_writer_routes(app)

    return app
