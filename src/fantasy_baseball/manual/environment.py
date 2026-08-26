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
    """Manual-mode variables still set after :func:`deactivate_manual_environment`.

    ``FB_SKIP_YAHOO`` is left in place deliberately, which means a plain launch
    can still be in stale-data mode with nothing on screen saying so. The
    caller reports these so the mode is never invisible.
    """
    from fantasy_baseball.web.refresh_pipeline import SKIP_YAHOO_ENV

    return {k: os.environ[k] for k in (SKIP_YAHOO_ENV,) if k in os.environ}


def manual_store_refusal(kv_path: Path) -> str | None:
    """Why ``kv_path`` is not a usable manual store, or None if it is.

    Reads the file directly rather than through ``get_kv()``, which would
    CREATE it -- an empty store carries no provenance stamp, so
    ``rosters.manual_store_active()`` reads it as Yahoo mode and the dashboard
    serves production rosters under a manual banner.

    Recovery advice is deliberately asymmetric. A MISSING store is safe to
    bootstrap. An unreadable one is NOT told to run ``--force``: that copies
    ``data/local.db`` over the destination, and a store can be unreadable for
    reasons that have nothing to do with its contents -- a read-only directory,
    a network share, or sidecars locked by another process, since SQLite must
    create a ``-shm`` file even to open a WAL database read-only.
    """
    try:
        exists = kv_path.exists()
    except OSError as exc:
        return f"cannot stat the manual KV store at {kv_path}: {exc}"

    if not exists:
        return (
            f"the manual KV store does not exist: {kv_path}\n"
            "Create it with 'python scripts/bootstrap_manual_kv.py'.\n"
            "Refusing rather than creating an empty one: an unstamped store reads as "
            "Yahoo mode, so the dashboard would serve production rosters under a "
            "manual banner."
        )

    try:
        with contextlib.closing(sqlite3.connect(f"{kv_path.as_uri()}?mode=ro", uri=True)) as conn:
            # Same liveness rule as kv_store.get(): a stamp past its expiry is
            # not a stamp. rosters.manual_store_active() reads it through
            # get_kv(), which applies that filter, and the two must not
            # disagree about whether a store is manual.
            row = conn.execute(
                "SELECT 1 FROM kv WHERE key = ? "
                "AND (expires_at IS NULL OR expires_at >= ?) LIMIT 1",
                (MANUAL_PROVENANCE_KEY, time.time()),
            ).fetchone()
    except (OSError, sqlite3.Error) as exc:
        return (
            f"{kv_path} exists but could not be read ({exc}).\n"
            "Check that the file and its -wal/-shm sidecars are readable and not "
            "held by another process -- SQLite creates a -shm file even to open a "
            "WAL database read-only.\n"
            "Do NOT reach for 'bootstrap_manual_kv.py --force' to clear this: it "
            "overwrites the destination with a copy of data/local.db, and an "
            "unreadable store is not evidence that its contents are gone."
        )

    if row is None:
        return (
            f"{kv_path} exists but is not a manual store -- no live "
            f"'{MANUAL_PROVENANCE_KEY}' stamp.\n"
            "If it is a stray file, seed it with "
            "'python scripts/bootstrap_manual_kv.py --force' (which OVERWRITES it "
            "with a copy of data/local.db -- check what is there first).\n"
            "Refusing because an unstamped store reads as Yahoo mode, so the "
            "dashboard would serve production rosters under a manual banner."
        )
    return None
