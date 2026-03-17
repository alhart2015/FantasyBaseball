"""Draft state serialization for the live dashboard.

Converts in-memory draft objects into a JSON-serializable dict and provides
atomic file I/O to prevent the Flask thread from reading partial writes.
"""
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from fantasy_baseball.draft.tracker import DraftTracker
from fantasy_baseball.draft.balance import CategoryBalance


def serialize_state(
    tracker: DraftTracker,
    balance: CategoryBalance,
    board: pd.DataFrame,
    recommendations: list[dict],
    filled_positions: dict[str, int],
) -> dict:
    """Convert all draft objects into a JSON-serializable dict."""
    available = board[~board["name"].isin(tracker.drafted_players)]
    available_players = []
    for _, row in available.iterrows():
        player: dict = {
            "name": row["name"],
            "positions": row["positions"] if isinstance(row["positions"], list) else [row["positions"]],
            "var": round(float(row["var"]), 1),
            "player_type": row["player_type"],
        }
        if row["player_type"] == "hitter":
            player["r"] = int(row.get("r", 0))
            player["hr"] = int(row.get("hr", 0))
            player["rbi"] = int(row.get("rbi", 0))
            player["sb"] = int(row.get("sb", 0))
            player["avg"] = round(float(row.get("avg", 0)), 3)
        elif row["player_type"] == "pitcher":
            player["w"] = int(row.get("w", 0))
            player["k"] = int(row.get("k", 0))
            player["sv"] = int(row.get("sv", 0))
            player["era"] = round(float(row.get("era", 0)), 2)
            player["whip"] = round(float(row.get("whip", 0)), 2)
        available_players.append(player)

    totals = balance.get_totals()
    # Round totals for clean JSON
    rounded_totals = {}
    for cat, val in totals.items():
        if cat in ("AVG", "ERA", "WHIP"):
            rounded_totals[cat] = round(float(val), 3)
        else:
            rounded_totals[cat] = round(float(val))

    return {
        "current_pick": tracker.current_pick,
        "current_round": tracker.current_round,
        "picking_team": tracker.picking_team,
        "is_user_pick": tracker.is_user_pick,
        "picks_until_user_turn": tracker.picks_until_user_turn,
        "user_roster": list(tracker.user_roster),
        "drafted_players": list(tracker.drafted_players),
        "recommendations": [
            {
                "name": r["name"],
                "var": round(float(r["var"]), 1),
                "best_position": r["best_position"],
                "positions": r["positions"] if isinstance(r["positions"], list) else [r["positions"]],
                "need_flag": bool(r["need_flag"]),
                "note": r.get("note", ""),
            }
            for r in recommendations
        ],
        "balance": {
            "totals": rounded_totals,
            "warnings": balance.get_warnings(),
        },
        "available_players": available_players,
        "filled_positions": dict(filled_positions),
    }


def write_state(state: dict, path: Path) -> None:
    """Atomically write state dict to JSON (write tmp + rename).

    Writes to a temporary file in the same directory as ``path``, then
    renames.  On Windows ``os.replace`` is used which is atomic for
    same-volume renames.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix="draft_state_"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_state(path: Path) -> dict:
    """Read state dict from JSON. Returns empty dict on decode error or missing file."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
