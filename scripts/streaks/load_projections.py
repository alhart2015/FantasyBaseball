"""CLI: load preseason projection rates for given seasons into the streaks DB.

Usage:
    python -m scripts.streaks.load_projections [--db-path PATH] [--seasons 2023 2024 2025] \\
        [--projections-root data/projections]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.projections import load_projection_rates_for_seasons
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load preseason projection rates.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2023, 2024, 2025],
        help="Seasons to load (default: 2023 2024 2025)",
    )
    parser.add_argument(
        "--projections-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "projections",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rates = load_projection_rates_for_seasons(args.projections_root, args.seasons)
    print(f"loaded {len(rates)} projection-rate rows from {args.projections_root}")

    conn = get_connection(args.db_path)
    try:
        upsert_projection_rates(conn, rates)
    finally:
        conn.close()
    print("upserted to hitter_projection_rates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
