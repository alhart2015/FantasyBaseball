"""CLI: run the streaks DuckDB schema migration.

Usage:
    python -m scripts.streaks.migrate [--db-path PATH] [--phase {2,3}]

Default is ``--phase 3`` (latest). After Phase 2 migration, re-run history
fetch (``python -m scripts.streaks.fetch_history --season ...``). After
Phase 3 migration, re-run ``apply_labels`` to repopulate labels.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/streaks/migrate.py` without -m.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.migrate import migrate_to_phase_2, migrate_to_phase_3
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the streaks DuckDB schema.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--phase", type=int, choices=[2, 3], default=3)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        if args.phase == 2:
            migrate_to_phase_2(conn)
        else:
            migrate_to_phase_3(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
