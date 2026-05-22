"""CLI: compact the streaks DuckDB to reclaim fragmented space.

The streaks pipeline rebuilds ``hitter_windows`` (DELETE + bulk INSERT)
on every refresh; DuckDB does not vacuum the tombstoned row groups in
place, so the file grows monotonically (~12 GiB for ~550k window rows).
This rewrites the database into a fresh, densely-packed file.

Usage:
    # Measure achievable savings without touching the original (default):
    python -m scripts.streaks.compact_db

    # Actually compact in place, keeping a .bak backup:
    python -m scripts.streaks.compact_db --replace

    # Compact in place without keeping a backup:
    python -m scripts.streaks.compact_db --replace --no-backup

Without ``--replace`` the original is left untouched and the temp file at
``<db>.compact`` is deleted after measuring (pass ``--keep-temp`` to keep
it). The source is opened READ_ONLY, so a concurrent dashboard refresh
holding the file lock makes this fail fast rather than corrupt anything.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/streaks/compact_db.py` without -m.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.maintenance import compact_database
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH


def _fmt_bytes(n: int) -> str:
    """Human-readable size, ASCII only (no non-ASCII units)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"  # unreachable; keeps type checkers happy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compact the streaks DuckDB.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Swap the compacted file in for the original (default: measure only).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="With --replace, do NOT keep a <db>.bak backup of the original.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="In measure-only mode, keep the <db>.compact temp file instead of deleting it.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = compact_database(
        args.db_path,
        replace=args.replace,
        keep_backup=not args.no_backup,
    )

    print()
    print(f"Database:    {result.db_path}")
    print(f"Size before: {_fmt_bytes(result.size_before)}")
    print(f"Size after:  {_fmt_bytes(result.size_after)}")
    print(f"Reclaimed:   {_fmt_bytes(result.bytes_saved)} ({result.pct_saved:.1f}%)")
    print("Row counts (preserved exactly):")
    for table, n in sorted(result.table_rows.items()):
        print(f"  {table:28s} {n:>12,}")

    if result.replaced:
        print("\nReplaced original with compacted file.")
        if result.backup_path is not None:
            print(f"Backup kept at {result.backup_path} -- delete it once you've confirmed.")
    else:
        compact_path = args.db_path.with_name(args.db_path.name + ".compact")
        if args.keep_temp:
            print(f"\nMeasure-only: compacted file left at {compact_path}")
        else:
            compact_path.unlink(missing_ok=True)
            print("\nMeasure-only: original untouched. Re-run with --replace to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
