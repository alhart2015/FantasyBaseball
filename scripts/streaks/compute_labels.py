"""CLI: rebuild hitter_windows + thresholds + hitter_streak_labels.

Usage:
    python -m scripts.streaks.compute_labels [--db-path PATH] [--season-set 2023-2025] [--qualifying-pa 150]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/streaks/compute_labels.py` without -m.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild streaks windows + thresholds + labels.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--season-set", default="2023-2025")
    parser.add_argument("--qualifying-pa", type=int, default=150)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        n_windows = compute_windows(conn)
        n_thresholds = compute_thresholds(
            conn, season_set=args.season_set, qualifying_pa=args.qualifying_pa
        )
        n_labels = apply_labels(conn, season_set=args.season_set)
    finally:
        conn.close()
    print(
        f"windows: {n_windows} rows; "
        f"thresholds: {n_thresholds} rows for season_set={args.season_set}; "
        f"labels: {n_labels} rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
