"""Create the isolated manual-mode KV store as a consistent copy of the Yahoo baseline.

The Yahoo-free ("manual") pipeline is isolated from the normal pipeline by WHOLE
STORE, not by key prefix: every manual process exports
``FANTASY_LOCAL_KV_PATH=data/manual.db`` before the first ``fantasy_baseball``
import, so ``get_kv()`` builds ``SqliteKVStore(data/manual.db)`` and every read
and write in that process -- ``cache:*`` and the bare keys alike -- lands in that
file. This script creates that file once.

    python scripts/bootstrap_manual_kv.py                        # local.db -> manual.db
    python scripts/bootstrap_manual_kv.py --dest data/manual.db  # same, explicit
    python scripts/bootstrap_manual_kv.py --force                # overwrite existing dest

WHY A COPY AND NOT AN EMPTY STORE. ``_load_projections`` raises when
``blended_projections:hitters`` / ``:pitchers`` are missing, the frozen
``cache:positions`` blob is the most accurate Yahoo eligibility map available for
the synthesized free-agent pool, and carrying ``game_logs:*`` /
``game_log_totals:*`` over means the first MLB Stats API sync is a short
incremental from the existing watermark rather than a full-season backfill.

WHY ``Connection.backup()`` AND NOT ``shutil.copy``. ``SqliteKVStore`` opens with
``isolation_level=None`` and ``PRAGMA journal_mode=WAL``, so committed rows can
live in the ``-wal`` sidecar. A file copy of the ``.db`` alone can miss them; the
SQLite backup API takes a transactionally consistent snapshot that includes them.

WHY THE SOURCE HANDLE IS READ-ONLY. ``data/local.db`` is the Yahoo baseline, and
the whole point of manual mode is that it cannot be touched. The source is opened
with ``file:...?mode=ro`` so that even a bug in this script cannot write to it.
(SQLite may still create empty ``-shm`` / ``-wal`` sidecars next to a WAL database
opened read-only; the ``.db`` file itself is never modified, which is the property
the tests pin.)

Safety rails, all of which exit non-zero WITHOUT writing anything:
  - refuses when ``RENDER`` is set at all (stricter than ``kv_store.is_remote()``,
    which only treats ``RENDER=true`` as remote) -- this script is a local tool;
  - refuses when the destination is the source, or is any file named ``local.db``,
    or is the repo's ``data/local.db``;
  - refuses when the destination already exists unless ``--force`` is given.

Relative paths are resolved against the repository root, not the shell's working
directory, so the same command means the same thing from anywhere. Both resolved
absolute paths are printed before any check runs.

``data/manual.db`` is already ignored by the existing ``data/*.db`` rule in
``.gitignore`` -- no ``.gitignore`` edit is needed. Confirm with::

    git check-ignore -v data/manual.db
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "local.db"
DEFAULT_DEST = PROJECT_ROOT / "data" / "manual.db"

# SQLite writes these next to the database; a stale one left over from a previous
# store at the same path would be replayed into the fresh copy.
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

_TABLES = ("kv", "hash_kv")


def resolve_path(raw: str | Path) -> Path:
    """Absolute path for ``raw``, anchoring relative paths at the repo root."""
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open ``path`` read-only. Writes through this handle raise, by construction."""
    uri = "file:" + path.as_posix() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def row_counts(path: Path) -> dict[str, int]:
    """``{"kv": n, "hash_kv": m}`` for the store at ``path`` (read-only handle)."""
    conn = open_readonly(path)
    try:
        return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in _TABLES}
    finally:
        conn.close()


def content_digest(path: Path) -> str:
    """SHA-256 over every row of both tables, in a stable order.

    Row-level rather than file-level on purpose: it is the KV CONTENT that must be
    identical between source and copy (the backup rewrites page layout, so the two
    files are not byte-identical). The tests also use it on the source, before and
    after, to prove the Yahoo baseline was not mutated.
    """
    digest = hashlib.sha256()
    conn = open_readonly(path)
    try:
        for key, value, expires_at in conn.execute(
            "SELECT key, value, expires_at FROM kv ORDER BY key"
        ):
            digest.update(b"kv\x00")
            for field in (key, value, expires_at):
                digest.update(repr(field).encode("utf-8"))
                digest.update(b"\x00")
        for hash_name, field_name, value in conn.execute(
            "SELECT hash_name, field, value FROM hash_kv ORDER BY hash_name, field"
        ):
            digest.update(b"hash_kv\x00")
            for field in (hash_name, field_name, value):
                digest.update(repr(field).encode("utf-8"))
                digest.update(b"\x00")
    finally:
        conn.close()
    return digest.hexdigest()


def backup(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` via the SQLite backup API, source opened read-only."""
    source = open_readonly(src)
    try:
        target = sqlite3.connect(str(dest))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def remove_store(path: Path) -> None:
    """Delete ``path`` and any SQLite sidecars beside it."""
    path.unlink(missing_ok=True)
    for suffix in _SIDECAR_SUFFIXES:
        Path(str(path) + suffix).unlink(missing_ok=True)


def _refuse(message: str) -> int:
    print(f"REFUSING: {message}")
    print("Nothing was written.")
    return 2


def _check_guards(src: Path, dest: Path, *, force: bool) -> int:
    """Return 0 when the copy may proceed, else the exit code to return."""
    render = os.environ.get("RENDER")
    if render:
        return _refuse(
            f"RENDER is set ({render!r}). This is a local bootstrap tool; on Render the "
            "KV is Upstash and there is no SQLite store to seed."
        )
    if not src.exists():
        return _refuse(f"source does not exist: {src}")
    if dest == src:
        return _refuse(f"destination is the source: {dest}")
    if dest.name.lower() == "local.db" or dest == DEFAULT_SOURCE.resolve():
        return _refuse(
            f"destination is the Yahoo baseline store: {dest}. The manual store must be a "
            "separate file -- that separation IS the isolation mechanism."
        )
    if dest.exists() and not force:
        return _refuse(f"destination already exists: {dest}. Pass --force to overwrite it.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Store to copy FROM, opened read-only (default: data/local.db).",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Store to create (default: data/manual.db).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing destination instead of refusing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    src = resolve_path(args.source)
    dest = resolve_path(args.dest)
    print("Bootstrapping the manual-mode KV store.")
    print(f"  source (read-only): {src}")
    print(f"  destination:        {dest}")

    guard_rc = _check_guards(src, dest, force=args.force)
    if guard_rc != 0:
        return guard_rc

    before_counts = row_counts(src)
    before_digest = content_digest(src)
    print(f"  source rows: kv={before_counts['kv']} hash_kv={before_counts['hash_kv']}")

    if dest.exists():
        print("  --force: removing the existing destination and its sidecars")
        remove_store(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    backup(src, dest)

    after_counts = row_counts(src)
    after_digest = content_digest(src)
    dest_counts = row_counts(dest)
    dest_digest = content_digest(dest)

    print(f"  copy rows:   kv={dest_counts['kv']} hash_kv={dest_counts['hash_kv']}")

    if after_counts != before_counts or after_digest != before_digest:
        print("ERROR: the source changed during the copy. The Yahoo baseline may be corrupted.")
        print(f"  before: {before_counts} {before_digest}")
        print(f"  after:  {after_counts} {after_digest}")
        return 1
    print(f"  source unchanged (sha256 {before_digest[:16]}...)")

    if dest_counts != before_counts or dest_digest != before_digest:
        print("ERROR: the copy does not match the source.")
        print(f"  source: {before_counts} {before_digest}")
        print(f"  copy:   {dest_counts} {dest_digest}")
        return 1
    print("  copy matches the source, row for row.")

    print(f"Wrote {dest} ({dest.stat().st_size} bytes).")
    print("Manual-mode processes must export FANTASY_LOCAL_KV_PATH to this path")
    print("BEFORE the first fantasy_baseball import.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
