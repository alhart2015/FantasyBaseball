"""CLI: fetch one season of game logs + Statcast PA data into the streaks DuckDB.

Usage:
    python scripts/streaks/fetch_history.py --season 2024
    python scripts/streaks/fetch_history.py --season 2025 --min-pa 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.fetch_history import fetch_season
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--min-pa", type=int, default=150)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--force-statcast",
        action="store_true",
        help="Re-fetch and upsert Statcast PAs even when dates for this season "
        "are already in the DB. Used to backfill new columns after a schema "
        "migration (INSERT OR REPLACE updates existing PK rows in place).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = get_connection(args.db_path)
    try:
        summary = fetch_season(
            season=args.season,
            conn=conn,
            min_pa=args.min_pa,
            force_statcast=args.force_statcast,
        )
    finally:
        conn.close()

    print(f"Done: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
