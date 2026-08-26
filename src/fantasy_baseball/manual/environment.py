"""Bind a process to the isolated manual KV store.

Isolation is by whole store rather than key prefix -- see the package docstring
-- so binding has to happen before anything resolves a store, and getting the
sequence subtly wrong is silent.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: The isolated store the Yahoo-free pipeline reads and writes. Never
#: ``data/local.db``, which is the only copy of the pre-outage Yahoo history.
DEFAULT_MANUAL_KV_PATH = _PROJECT_ROOT / "data" / "manual.db"


def activate_manual_environment(kv_path: Path | None = None) -> Path:
    """Bind this process to the manual store; return the absolute path bound.

    Call before anything resolves a KV store: ``get_kv()`` caches on its first
    call. Sets an ABSOLUTE path -- ``kv_store`` resolves the variable against
    the working directory, so a relative value creates a second, empty store.

    Does NOT verify the store is a real manual store; that is
    :func:`manual_store_refusal`, which the caller runs before touching it.

    Refuses on Render rather than binding: the KV there is Upstash, this
    variable cannot reach it, and disabling Yahoo against production is not
    something a helper should do by side effect.
    """
    from fantasy_baseball.data.kv_store import LOCAL_KV_PATH_ENV, is_remote
    from fantasy_baseball.web.refresh_pipeline import SKIP_YAHOO_ENV

    if is_remote():
        raise RuntimeError(
            "activate_manual_environment() called with RENDER set. The KV there is "
            "Upstash, which no environment variable can redirect, and the manual "
            "pipeline never runs against production."
        )

    resolved = (kv_path if kv_path is not None else DEFAULT_MANUAL_KV_PATH).resolve()
    os.environ[LOCAL_KV_PATH_ENV] = str(resolved)
    os.environ[SKIP_YAHOO_ENV] = "1"

    # Discard any singleton an earlier import already built, so the next
    # get_kv() rebinds to the variable just set.
    from fantasy_baseball.data import kv_store

    kv_store._reset_singleton()
    return resolved


def deactivate_manual_environment() -> dict[str, str]:
    """Unbind this process from the manual store; return what was cleared.

    Clears ONLY the store binding. ``FB_SKIP_YAHOO`` is deliberately left
    alone: it is a standalone stale-data switch against the ordinary
    ``data/local.db`` (``docs/stale-data-refresh-runbook.md``), and with the
    Yahoo API unavailable, clearing someone's seatbelt would re-arm live auth
    on their next refresh.

    These variables are exported into a shell, so they outlive the command that
    set them -- without an explicit clear, a later launch inherits the previous
    manual session and serves the transcription while the caller believes they
    are reading Yahoo. No-op on Render, where the KV is Upstash.
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


def manual_store_refusal(kv_path: Path) -> str | None:
    """Why ``kv_path`` is not a usable manual store, or None if it is.

    Checks the file WITHOUT opening it through ``get_kv()``, because
    ``SqliteKVStore.__init__`` creates the file and schema on open -- asking
    the question would answer it. An empty store then carries no provenance
    stamp, ``rosters.manual_store_active()`` reads it as Yahoo mode, and
    ``live_rosters()`` falls through to production Upstash. That is how
    month-stale prod rosters get spliced into a page labelled manual, and it is
    what ``bootstrap_manual_kv`` stamps the store to prevent.

    Returns the refusal text so each caller keeps its own exit code and
    recovery advice, mirroring ``kv_sync.sync_destination_refusal``.
    """
    if not kv_path.exists():
        return (
            f"the manual KV store does not exist: {kv_path}\n"
            "Create it first with 'python scripts/bootstrap_manual_kv.py'.\n"
            "Refusing rather than creating an empty one: an unstamped store reads as "
            "Yahoo mode, so the dashboard would serve production rosters under a "
            "manual banner."
        )

    import sqlite3

    from fantasy_baseball.data.cache_keys import MANUAL_PROVENANCE_KEY

    try:
        with sqlite3.connect(f"file:{kv_path}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT 1 FROM kv WHERE key = ? LIMIT 1", (MANUAL_PROVENANCE_KEY,)
            ).fetchone()
    except sqlite3.Error as exc:
        return (
            f"{kv_path} exists but could not be read as a KV store ({exc}).\n"
            "Rebuild it with 'python scripts/bootstrap_manual_kv.py --force'."
        )

    if row is None:
        return (
            f"{kv_path} exists but is not a manual store -- it carries no "
            f"'{MANUAL_PROVENANCE_KEY}' provenance stamp.\n"
            "Seed it with 'python scripts/bootstrap_manual_kv.py --force'.\n"
            "Refusing because an unstamped store reads as Yahoo mode, so the "
            "dashboard would serve production rosters under a manual banner."
        )
    return None
