"""CLI: run the Phase 2 schema migration against the local streaks DuckDB.

Usage:
    python -m scripts.streaks.migrate [--db-path PATH]

After this, re-run::

    python -m scripts.streaks.fetch_history --season 2023
    python -m scripts.streaks.fetch_history --season 2024
    python -m scripts.streaks.fetch_history --season 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/streaks/migrate.py` without -m.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.migrate import migrate_to_phase_2
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate streaks DB to Phase 2 schema.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        migrate_to_phase_2(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
