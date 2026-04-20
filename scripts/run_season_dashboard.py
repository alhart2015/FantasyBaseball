#!/usr/bin/env python3
"""Launch the season dashboard.

First step: sync the remote Upstash KV down to the local SQLite KV so
the dashboard reads the same state the Render app writes. Skip with
``--no-sync`` when offline or when the remote is known-empty.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fantasy_baseball.data.kv_store import is_remote
from fantasy_baseball.data.kv_sync import sync_remote_to_local
from fantasy_baseball.web.season_app import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the initial remote→local KV sync.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to serve the dashboard on (default: 5001).",
    )
    args = parser.parse_args()

    if not args.no_sync and not is_remote():
        print("Syncing remote Upstash KV → local SQLite...")
        stats = sync_remote_to_local()
        print(f"  synced: {stats.summary()}")

    app = create_app()
    print(f"Season dashboard: http://localhost:{args.port}")
    app.run(port=args.port, debug=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
