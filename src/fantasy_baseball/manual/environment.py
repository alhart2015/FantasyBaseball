"""Bind a process to the isolated manual KV store.

The manual pipeline is isolated by whole store, not by key prefix (see
``scripts/run_manual_refresh.py``'s module docstring), so binding must happen
before anything resolves a store.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import time
from pathlib import Path

from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: The isolated store the Yahoo-free pipeline reads and writes. Never
#: ``data/local.db``, which is the only copy of the pre-outage Yahoo history.
DEFAULT_MANUAL_KV_PATH = _PROJECT_ROOT / "data" / "manual.db"


def activate_manual_environment(kv_path: Path | None = None) -> Path:
    """Bind this process to the manual store; return the absolute path bound.

    Call before anything resolves a KV store: ``get_kv()`` caches on its first
    call. Sets an ABSOLUTE path -- ``kv_store`` resolves the variable against
    the working directory.

    Raises on Render, where the KV is Upstash and no variable redirects it.
    """
    from fantasy_baseball.data.kv_store import LOCAL_KV_PATH_ENV, is_remote

    if is_remote():
        raise RuntimeError(
            "activate_manual_environment() called with RENDER set. The KV there is "
            "Upstash, which no environment variable can redirect."
        )

    resolved = (kv_path if kv_path is not None else DEFAULT_MANUAL_KV_PATH).resolve()
    os.environ[LOCAL_KV_PATH_ENV] = str(resolved)

    # Imported after the path is set, not before: this pulls in the web layer,
    # and anything it touches must already see the manual binding.
    from fantasy_baseball.web.refresh_pipeline import SKIP_YAHOO_ENV

    os.environ[SKIP_YAHOO_ENV] = "1"

    from fantasy_baseball.data import kv_store

    kv_store._reset_singleton()
    return resolved


def deactivate_manual_environment() -> dict[str, str]:
    """Unbind this process from the manual store; return what was cleared.

    Clears only the store binding. ``FB_SKIP_YAHOO`` is a standalone stale-data
    switch against the ordinary ``data/local.db``
    (``docs/stale-data-refresh-runbook.md``), so it is reported but not
    removed. No-op on Render.
    """
    from fantasy_baseball.data.kv_store import LOCAL_KV_PATH_ENV, is_remote

    if is_remote():
        return {}

    cleared = {}
    if LOCAL_KV_PATH_ENV in os.environ:
        cleared[LOCAL_KV_PATH_ENV] = os.environ.pop(LOCAL_KV_PATH_ENV)
        from fantasy_baseball.data import kv_store

        kv_store._reset_singleton()
    return cleared


def surviving_manual_vars() -> dict[str, str]:
    """Manual-mode variables still set after :func:`deactivate_manual_environment`."""
    from fantasy_baseball.web.refresh_pipeline import SKIP_YAHOO_ENV

    return {k: os.environ[k] for k in (SKIP_YAHOO_ENV,) if k in os.environ}


class _NotAKVStore(Exception):
    """The file at the manual path is not a KV store."""


#: Mirrors ``scripts/run_manual_refresh.py``. That script must not import from
#: ``fantasy_baseball`` before it binds the store, so the probe cannot be
#: shared; keep the two techniques in step.
SQLITE_HEADER = b"SQLite format 3\x00"

#: Columns ``_stamp_row``'s query needs. A ``kv`` table without them is somebody
#: else's schema, not a store this code can read.
_KV_COLUMNS = frozenset({"key", "expires_at"})


def _stamp_row(kv_path: Path, *, immutable: bool) -> tuple[object | None, bool]:
    """``(stamp row or None, whether a usable kv table was visible)``.

    Checks the COLUMNS, not just the table name: a foreign file with its own
    ``kv`` table raises ``no such column: expires_at`` on the query below, and
    that lands in the read-failure branch, which forbids the one recovery.
    """
    flags = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    with contextlib.closing(
        sqlite3.connect("file:" + kv_path.as_posix() + flags, uri=True)
    ) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info('kv')")}
        if not columns >= _KV_COLUMNS:
            return None, False
        # Same liveness rule as kv_store.get(): a stamp past its expiry is not a
        # stamp. rosters.manual_store_active() reads it through get_kv(), which
        # applies that filter, and the two must not disagree.
        row = conn.execute(
            "SELECT 1 FROM kv WHERE key = ? AND (expires_at IS NULL OR expires_at >= ?) LIMIT 1",
            (MANUAL_PROVENANCE_KEY, time.time()),
        ).fetchone()
        return row, True


def _has_live_provenance(kv_path: Path) -> bool:
    """True when ``kv_path`` carries an unexpired manual stamp.

    Raises :class:`_NotAKVStore` when the file is not one. Read failures
    propagate.
    """
    # Magic bytes, not the driver's error text, which is not version-stable.
    with kv_path.open("rb") as handle:
        if handle.read(len(SQLITE_HEADER)) != SQLITE_HEADER:
            raise _NotAKVStore

    # immutable=1 leaves no trace. A plain mode=ro open of a WAL database
    # CREATES -shm/-wal sidecars beside the store and, being read-only, cannot
    # remove them on close -- and this probe runs against a file the caller may
    # be about to refuse.
    row, usable = _stamp_row(kv_path, immutable=True)
    if (not usable or row is None) and _sidecars(kv_path):
        # The schema or the stamp may live in an uncheckpointed WAL, which the
        # immutable read cannot see. This open can leave sidecars, which is why
        # it is reached only for a file that already has one.
        row, usable = _stamp_row(kv_path, immutable=False)

    if not usable:
        raise _NotAKVStore
    return row is not None


def _sidecars(kv_path: Path) -> list[Path]:
    """Existing ``-wal``/``-shm`` files beside ``kv_path``."""
    return [p for p in (Path(f"{kv_path}-wal"), Path(f"{kv_path}-shm")) if p.exists()]


def _overwritable(kv_path: Path, problem: str) -> str:
    """Refusal for a file that holds nothing this code can read."""
    return (
        f"{kv_path} exists but {problem}.\n"
        "There is no readable KV data here to serve. If it is a stray file, "
        "'python scripts/bootstrap_manual_kv.py --force' seeds it -- but that "
        "OVERWRITES it with a copy of data/local.db, so copy it somewhere safe "
        "first if it might be a damaged manual store."
    )


def _damaged(kv_path: Path, problem: str, sidecars: list[Path]) -> str:
    """Refusal for a file that was a live database, so --force is unsafe."""
    names = ", ".join(p.name for p in sidecars)
    return (
        f"{kv_path} exists but {problem}, and has SQLite sidecars beside it "
        f"({names}).\n"
        "That means it WAS a live database, so this is a damaged store rather "
        "than a stray file.\n"
        "Do NOT run 'bootstrap_manual_kv.py --force': it overwrites the "
        "destination with a copy of data/local.db. Copy the file and its "
        "sidecars somewhere safe and try 'sqlite3 manual.db .recover' first."
    )


def manual_store_refusal(kv_path: Path) -> str | None:
    """Why ``kv_path`` is not a usable manual store, or None if it is.

    Reads the file directly rather than through ``get_kv()``, which would CREATE
    it -- and an unstamped store reads as Yahoo mode.

    Three outcomes, because they need different advice: a file with no readable
    KV data (--force is the recovery), a file that was a live database (--force
    would destroy it), and one that cannot be opened at all.
    """
    # The probe builds a file: URI, which sqlite resolves against the process
    # CWD if it is relative -- a different file than the caller named.
    kv_path = kv_path.resolve()

    try:
        exists = kv_path.exists()
    except OSError as exc:
        return f"cannot stat the manual KV store at {kv_path}: {exc}"

    if not exists:
        return (
            f"the manual KV store does not exist: {kv_path}\n"
            "Create it with 'python scripts/bootstrap_manual_kv.py'."
        )

    try:
        stamped = _has_live_provenance(kv_path)
    except _NotAKVStore:
        # A crash mid-write can clobber page 1 of a REAL store, which then has
        # no SQLite header and looks exactly like a stray file. Its sidecars are
        # the evidence that it was a live database, and getting this wrong loses
        # the only copy of the hand transcription.
        sidecars = _sidecars(kv_path)
        if sidecars:
            return _damaged(kv_path, "is not a readable KV store", sidecars)
        return _overwritable(kv_path, "is not a KV store")
    except (OSError, sqlite3.Error) as exc:
        return (
            f"{kv_path} exists but could not be read ({exc}).\n"
            "Check that the file and its -wal/-shm sidecars are readable and not held "
            "by another process -- SQLite creates a -shm file even to open a WAL "
            "database read-only.\n"
            "Do NOT reach for 'bootstrap_manual_kv.py --force' to clear this: it "
            "overwrites the destination with a copy of data/local.db, and an "
            "unreadable store is not evidence that its contents are gone."
        )

    if not stamped:
        return (
            f"{kv_path} is a KV store but not a MANUAL one -- no live "
            f"'{MANUAL_PROVENANCE_KEY}' stamp.\n"
            "Refusing because an unstamped store reads as Yahoo mode, so the "
            "dashboard would serve production rosters under a manual banner.\n"
            "Seed it with 'python scripts/bootstrap_manual_kv.py --force', which "
            "OVERWRITES it with a copy of data/local.db -- check what is there first."
        )
    return None
