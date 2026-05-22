"""Maintenance utilities for the streaks DuckDB.

Historically the pipeline rebuilt ``hitter_windows`` via
``DELETE FROM hitter_windows`` + bulk ``INSERT`` on every run. DuckDB does
not vacuum the tombstoned row groups in place, so across hundreds of
Sunday-report / dashboard refreshes (and the dev/eval scripts that
recompute windows far more often) the file accumulated thousands of
sparsely-packed "used" blocks that the free-list never reclaimed -- the
table grew to ~12 GiB for ~550k rows that pack into well under 1 GiB.
``compact_database`` reclaims that space by rewriting the whole database
into a fresh file via ``COPY FROM DATABASE`` (which repacks rows into
dense row groups), verifying every table's row count survived, and
optionally swapping the compacted file in atomically.

This is a maintenance tool, not a pipeline step -- run it manually via
``scripts/streaks/compact_db.py``. The ``hitter_windows`` write pattern
now does DROP + recreate instead of DELETE + INSERT (see
:func:`fantasy_baseball.streaks.windows._bulk_replace_hitter_windows`), so
it no longer re-accumulates dead blocks; this tool stays useful for the
one-time cleanup of already-bloated files and any residual fragmentation
from the other (smaller) tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

# Suffixes appended to the database filename for the rewritten file and the
# pre-swap backup. Both live alongside the original so the rename is on the
# same filesystem (atomic, no cross-device copy).
_COMPACT_SUFFIX = ".compact"
_BACKUP_SUFFIX = ".bak"


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of a :func:`compact_database` run.

    ``size_before`` / ``size_after`` are the main DuckDB file sizes in
    bytes (the compacted file's size, whether or not it was swapped in).
    ``table_rows`` maps each table to its verified row count. ``replaced``
    is True when the compacted file was swapped in for the original.
    ``backup_path`` is the kept pre-swap backup, or None.
    """

    db_path: Path
    size_before: int
    size_after: int
    table_rows: dict[str, int]
    replaced: bool
    backup_path: Path | None

    @property
    def bytes_saved(self) -> int:
        return self.size_before - self.size_after

    @property
    def pct_saved(self) -> float:
        if self.size_before == 0:
            return 0.0
        return 100.0 * self.bytes_saved / self.size_before


def _sql_str(value: str) -> str:
    """Quote *value* as a DuckDB single-quoted string literal.

    ATTACH / COPY FROM DATABASE statements take the path as a literal --
    prepared-statement parameters are not accepted there -- so we escape
    embedded single quotes by doubling. Windows backslashes are literal
    inside DuckDB single-quoted strings, so paths need no further escaping.
    """
    return "'" + value.replace("'", "''") + "'"


def _table_names(conn: duckdb.DuckDBPyConnection, catalog: str) -> list[str]:
    """Return the ``main``-schema table names in attached database *catalog*."""
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = ? AND table_schema = 'main' ORDER BY table_name",
        [catalog],
    ).fetchall()
    return [str(r[0]) for r in rows]


def _row_counts(conn: duckdb.DuckDBPyConnection, catalog: str, tables: list[str]) -> dict[str, int]:
    """Return ``{table: row_count}`` for *tables* in attached database *catalog*."""
    counts: dict[str, int] = {}
    for table in tables:
        # Table names come from information_schema (not user input); quote
        # the identifier defensively anyway.
        row = conn.execute(f'SELECT COUNT(*) FROM {catalog}."{table}"').fetchone()
        counts[table] = int(row[0]) if row is not None else 0
    return counts


def compact_database(
    db_path: Path | str,
    *,
    replace: bool = False,
    keep_backup: bool = True,
) -> CompactionResult:
    """Rewrite the DuckDB at *db_path* into a fresh, densely-packed file.

    Opens an in-memory coordinator connection, attaches the source
    read-only and a fresh ``<db>.compact`` target, and runs
    ``COPY FROM DATABASE`` to copy all schema + data. Verifies that every
    source table's row count matches in the target before doing anything
    destructive.

    With ``replace=False`` (default) the compacted file is left in place at
    ``<db>.compact`` and the original is untouched -- the caller inspects
    the reported sizes and decides. With ``replace=True`` the compacted
    file is swapped in for the original; if ``keep_backup`` the original is
    moved to ``<db>.bak`` first, otherwise it is deleted.

    Raises ``FileNotFoundError`` if *db_path* does not exist and
    ``RuntimeError`` if the post-copy row counts do not match (in which
    case the temp file is removed and the original is left untouched).
    The source is attached READ_ONLY, so a concurrent writer holding the
    DuckDB file lock will surface as a ``duckdb`` error from ATTACH.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No DuckDB file at {db_path}")

    compact_path = db_path.with_name(db_path.name + _COMPACT_SUFFIX)
    if compact_path.exists():
        compact_path.unlink()

    size_before = db_path.stat().st_size
    logger.info("Compacting %s (%d bytes)...", db_path, size_before)

    conn = duckdb.connect()  # in-memory coordinator; never touches db_path's lock as RW
    try:
        conn.execute(f"ATTACH {_sql_str(str(db_path))} AS src (READ_ONLY)")
        conn.execute(f"ATTACH {_sql_str(str(compact_path))} AS dst")
        tables = _table_names(conn, "src")
        conn.execute("COPY FROM DATABASE src TO dst")
        conn.execute("CHECKPOINT dst")
        src_counts = _row_counts(conn, "src", tables)
        dst_counts = _row_counts(conn, "dst", tables)
    finally:
        conn.close()

    mismatches = {
        t: (src_counts[t], dst_counts.get(t))
        for t in src_counts
        if dst_counts.get(t) != src_counts[t]
    }
    if mismatches:
        compact_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Row-count mismatch after compaction; aborting and leaving "
            f"{db_path} untouched. Mismatches (src, dst): {mismatches}"
        )

    size_after = compact_path.stat().st_size
    logger.info(
        "Compacted -> %d bytes (%.1f%% smaller); verified %d tables",
        size_after,
        100.0 * (size_before - size_after) / size_before if size_before else 0.0,
        len(tables),
    )

    backup_path: Path | None = None
    if replace:
        if keep_backup:
            backup_path = db_path.with_name(db_path.name + _BACKUP_SUFFIX)
            if backup_path.exists():
                backup_path.unlink()
            db_path.rename(backup_path)
            logger.info("Backed up original to %s", backup_path)
        else:
            db_path.unlink()
        compact_path.rename(db_path)
        logger.info("Swapped compacted file in for %s", db_path)

    return CompactionResult(
        db_path=db_path,
        size_before=size_before,
        size_after=size_after,
        table_rows=src_counts,
        replaced=replace,
        backup_path=backup_path,
    )
