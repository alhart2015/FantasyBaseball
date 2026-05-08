"""CLI: rebuild ``hitter_windows`` from the current ``hitter_games`` + ``hitter_statcast_pa``.

Usage:
    python -m scripts.streaks.compute_windows [--db-path PATH]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/streaks/compute_windows.py` without -m.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection
from fantasy_baseball.streaks.windows import compute_windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild hitter_windows.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        n = compute_windows(conn)
    finally:
        conn.close()
    print(f"Wrote {n} rows to hitter_windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
