"""CLI: run a streaks DuckDB schema migration.

Usage:
    python -m scripts.streaks.migrate [--db-path PATH] --phase {2,3,5}

Additive columns now heal automatically on every DB open (``init_schema``'s
additive-drift healer), so most schema changes need no migration here. The
remaining explicit phases are destructive/structural ones the healer can't do:

- ``--phase 2`` -- full reset: drop + recreate all tables. After: re-run
  ``python -m scripts.streaks.fetch_history --season ...``.
- ``--phase 3`` -- drop + recreate ``hitter_streak_labels`` for its PK change.
  After: re-run ``apply_labels`` to repopulate labels.
- ``--phase 5`` -- drop the legacy ``barrel`` column. After: re-run
  ``scripts/streaks/fetch_history.py --force-statcast --season {year}`` per
  year to populate ``launch_speed_angle`` on existing PA rows.

Backfilling auto-healed columns: when the healer adds the dense-cat rate
columns (``r_per_pa`` / ``rbi_per_pa`` / ``avg``), re-run
``scripts/streaks/load_projections.py``; when it adds the ``model_fits``
pipeline-state columns, re-run ``refit_models_for_report`` (via the Sunday
report or dashboard refresh).
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
    migrate_to_phase_5,
)
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection

_PHASE_FUNCS = {
    "2": migrate_to_phase_2,
    "3": migrate_to_phase_3,
    "5": migrate_to_phase_5,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the streaks DuckDB schema.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--phase", type=str, choices=sorted(_PHASE_FUNCS), required=True)
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
