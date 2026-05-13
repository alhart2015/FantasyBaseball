"""CLI: run the streaks DuckDB schema migration.

Usage:
    python -m scripts.streaks.migrate [--db-path PATH] [--phase {2,3,4,5,b}]

Default is ``--phase b`` (latest). After Phase 2 migration, re-run history
fetch (``python -m scripts.streaks.fetch_history --season ...``). After
Phase 3 migration, re-run ``apply_labels`` to repopulate labels. After
Phase 4 migration, re-run ``scripts/streaks/load_projections.py`` to
backfill the new dense-cat rate columns. After Phase 5 migration, re-run
``scripts/streaks/fetch_history.py --force-statcast --season {year}`` for
each year to populate ``launch_speed_angle`` on existing PA rows. After
Phase B migration, re-run ``refit_models_for_report`` (via the Sunday
report or dashboard refresh) so model_fits rows pick up the new
pipeline-state columns.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/streaks/migrate.py` without -m.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.migrate import (
    migrate_to_phase_2,
    migrate_to_phase_3,
    migrate_to_phase_4,
    migrate_to_phase_5,
    migrate_to_phase_b,
)
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection

_PHASE_FUNCS = {
    "2": migrate_to_phase_2,
    "3": migrate_to_phase_3,
    "4": migrate_to_phase_4,
    "5": migrate_to_phase_5,
    "b": migrate_to_phase_b,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the streaks DuckDB schema.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--phase", type=str, choices=sorted(_PHASE_FUNCS), default="b")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        _PHASE_FUNCS[args.phase](conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
