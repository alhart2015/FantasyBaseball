"""Tests for streaks DuckDB compaction (:mod:`...streaks.data.maintenance`).

The historical bloat came from the ``hitter_windows`` DELETE+INSERT write
pattern -- see :func:`fantasy_baseball.streaks.windows._bulk_replace_hitter_windows`
for why it bloated and how the DROP+recreate fix avoids it. DuckDB 1.5.3 stopped
that pattern from growing the file, so the fixture here leaves free space
another way that every DuckDB version keeps until a full rewrite.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from fantasy_baseball.streaks.data import maintenance
from fantasy_baseball.streaks.data.maintenance import compact_database

_ROWS = 2000
# hash() output is effectively incompressible, so 200k rows x 2 UBIGINT
# columns occupy ~14 storage blocks (~3.5 MiB). Sequential integers would
# compress (bitpacking/delta) into a single block and leave almost nothing
# for compaction to reclaim.
_SCRATCH_ROWS = 200_000
# Minimum free blocks the fixture must leave for the compaction assertions
# to be meaningful; fail loudly here if a DuckDB upgrade changes that.
_MIN_FREE_BLOCKS = 8


def _build_bloated_db(db_path: Path) -> int:
    """Create a DuckDB at *db_path* holding free space only a rewrite reclaims.

    A large scratch table is checkpointed first, the real rows are written
    after it, then the scratch table is dropped: its freed blocks sit before
    live data, so the file can't simply be truncated. Returns the final
    on-disk size in bytes. Includes a second small table so the multi-table
    copy path is exercised.
    """
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE windows("
            "player_id INTEGER, window_end DATE, window_days INTEGER, "
            "pa INTEGER, avg DOUBLE, "
            "PRIMARY KEY(player_id, window_end, window_days))"
        )
        conn.execute("CREATE TABLE meta(k VARCHAR PRIMARY KEY, v INTEGER)")
        conn.execute("INSERT INTO meta VALUES ('schema_version', 5)")
        conn.execute(
            "CREATE TABLE scratch AS SELECT hash(i) AS a, hash(i * 7) AS b FROM range(?) tbl(i)",
            [_SCRATCH_ROWS],
        )
        conn.execute("CHECKPOINT")
        conn.execute(
            "INSERT INTO windows SELECT "
            "(i % 500)::INTEGER, DATE '2024-01-01' + (i // 500)::INTEGER, 14, "
            "(i % 30)::INTEGER, (i % 30) / 100.0 FROM range(?) tbl(i)",
            [_ROWS],
        )
        conn.execute("CHECKPOINT")
        conn.execute("DROP TABLE scratch")
        conn.execute("CHECKPOINT")
        row = conn.execute("SELECT free_blocks FROM pragma_database_size()").fetchone()
        free_blocks = int(row[0]) if row is not None else 0
    finally:
        conn.close()
    assert free_blocks >= _MIN_FREE_BLOCKS, (
        f"fixture left only {free_blocks} free blocks (need >= {_MIN_FREE_BLOCKS}); "
        f"DuckDB {duckdb.__version__} no longer bloats this way -- fix the fixture"
    )
    return db_path.stat().st_size


def _read_rows(db_path: Path, table: str) -> list[tuple]:
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return conn.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
    finally:
        conn.close()


def test_compact_reclaims_space_and_preserves_data(tmp_path: Path) -> None:
    db = tmp_path / "streaks.duckdb"
    size_before = _build_bloated_db(db)
    expected_windows = _read_rows(db, "windows")

    result = compact_database(db, replace=False)

    # Space was reclaimed.
    assert result.size_before == size_before
    assert result.size_after < result.size_before
    assert result.bytes_saved > 0
    assert result.pct_saved > 0

    # Original untouched, compacted file sits alongside it (measure-only).
    assert db.stat().st_size == size_before
    assert result.replaced is False
    assert result.backup_path is None
    compact_path = db.with_name(db.name + ".compact")
    assert compact_path.exists()

    # Row counts verified for every table.
    assert result.table_rows == {"windows": _ROWS, "meta": 1}

    # Data is byte-for-byte equivalent in the compacted file.
    assert _read_rows(compact_path, "windows") == expected_windows
    assert _read_rows(compact_path, "meta") == [("schema_version", 5)]


def test_compact_replace_keeps_backup(tmp_path: Path) -> None:
    db = tmp_path / "streaks.duckdb"
    size_before = _build_bloated_db(db)
    expected = _read_rows(db, "windows")

    result = compact_database(db, replace=True, keep_backup=True)

    assert result.replaced is True
    # Live file is now the compacted one.
    assert db.stat().st_size == result.size_after
    assert db.stat().st_size < size_before
    # Backup kept at <db>.bak with the original (bloated) size.
    backup = db.with_name(db.name + ".bak")
    assert result.backup_path == backup
    assert backup.exists()
    assert backup.stat().st_size == size_before
    # Temp file was consumed by the swap.
    assert not db.with_name(db.name + ".compact").exists()
    # Data intact in the swapped-in file.
    assert _read_rows(db, "windows") == expected


def test_compact_replace_no_backup(tmp_path: Path) -> None:
    db = tmp_path / "streaks.duckdb"
    _build_bloated_db(db)
    expected = _read_rows(db, "windows")

    result = compact_database(db, replace=True, keep_backup=False)

    assert result.replaced is True
    assert result.backup_path is None
    assert not db.with_name(db.name + ".bak").exists()
    assert not db.with_name(db.name + ".compact").exists()
    assert _read_rows(db, "windows") == expected


def test_compact_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compact_database(tmp_path / "does_not_exist.duckdb")


def test_compact_stale_temp_file_is_overwritten(tmp_path: Path) -> None:
    db = tmp_path / "streaks.duckdb"
    _build_bloated_db(db)
    # A leftover temp file from a previous aborted run must not block compaction.
    stale = db.with_name(db.name + ".compact")
    stale.write_bytes(b"garbage from an aborted run")

    result = compact_database(db, replace=False)

    assert result.size_after > 0
    assert _read_rows(stale, "windows")  # the stale bytes were replaced by a real DB


def test_compact_row_count_mismatch_aborts(tmp_path: Path, monkeypatch) -> None:
    """If post-copy verification fails, the temp file is removed and the
    original is left untouched -- no destructive swap on a bad copy."""
    db = tmp_path / "streaks.duckdb"
    size_before = _build_bloated_db(db)

    real_row_counts = maintenance._row_counts

    def _lying_counts(conn, catalog, tables):
        counts = real_row_counts(conn, catalog, tables)
        if catalog == "dst":
            # Pretend the copy lost rows.
            counts = {t: n + 1 for t, n in counts.items()}
        return counts

    monkeypatch.setattr(maintenance, "_row_counts", _lying_counts)

    with pytest.raises(RuntimeError, match="Row-count mismatch"):
        compact_database(db, replace=True)

    # Original intact, temp cleaned up, no backup created.
    assert db.stat().st_size == size_before
    assert not db.with_name(db.name + ".compact").exists()
    assert not db.with_name(db.name + ".bak").exists()
