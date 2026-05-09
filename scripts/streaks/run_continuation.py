"""CLI: rebuild continuation_rates for the Phase 3 go/no-go gate.

Assumes hitter_windows, thresholds, hitter_projection_rates, and
hitter_streak_labels are already populated (run scripts/streaks/load_projections.py
and scripts/streaks/compute_labels.py first).

Usage:
    python -m scripts.streaks.run_continuation [--db-path PATH] [--season-set 2023-2025]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.analysis.continuation import compute_continuation_rates
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild continuation_rates.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--season-set", default="2023-2025")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        n = compute_continuation_rates(conn, season_set=args.season_set)
    finally:
        conn.close()
    print(f"continuation_rates: {n} rows for season_set={args.season_set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
