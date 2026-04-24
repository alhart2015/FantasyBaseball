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
    args = parser.parse_args()

    app = create_app()
    print(f"Draft dashboard: http://localhost:{args.port}")
    app.run(port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
