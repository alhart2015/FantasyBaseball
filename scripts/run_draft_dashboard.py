#!/usr/bin/env python3
"""Launch the draft dashboard.

Web-only pick entry. All state lives in data/draft_state*.json; the
browser polls /api/state every 500ms and POSTs picks back. Legacy CLI
loop (scripts/run_draft.py) remains available until the dashboard's
real-data path is complete.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fantasy_baseball.web.app import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
BOARD_PATH = PROJECT_ROOT / "data" / "draft_state_board.json"


def rebuild_board() -> None:
    """Rebuild the preseason draft board from SQLite projections.

    Mirrors the board-construction path from scripts/run_draft.py so the
    dashboard can run without needing the CLI to have seeded the board
    first. Writes to ``data/draft_state_board.json``.
    """
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.db import get_connection
    from fantasy_baseball.draft.board import build_draft_board
    from fantasy_baseball.draft.state import serialize_board, write_board

    config = load_config(CONFIG_PATH)
    print(f"Rebuilding board from SQLite (league: {config.league_id})...")
    conn = get_connection()
    try:
        full_board = build_draft_board(
            conn=conn,
            sgp_overrides=config.sgp_overrides or None,
            roster_slots=config.roster_slots or None,
            num_teams=config.num_teams,
        )
    finally:
        conn.close()
    write_board(serialize_board(full_board), BOARD_PATH)
    print(f"  wrote {len(full_board)} rows to {BOARD_PATH.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to serve the dashboard on (default: 5050).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode with auto-reload (off by default "
        "to avoid restarts mid-draft).",
    )
    parser.add_argument(
        "--rebuild-board",
        action="store_true",
        help="Rebuild data/draft_state_board.json from SQLite projections "
        "before serving. Needed before a fresh draft (or any time "
        "projections have changed).",
    )
    args = parser.parse_args()

    if args.rebuild_board:
        rebuild_board()

    app = create_app()
    print(f"Draft dashboard: http://localhost:{args.port}")
    app.run(port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
