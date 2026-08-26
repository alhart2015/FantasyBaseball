#!/usr/bin/env python3
"""Launch the season dashboard.

First step: sync the remote Upstash KV down to the local SQLite KV so
the dashboard reads the same state the Render app writes. Skip with
``--no-sync`` when offline or when the remote is known-empty.

Pass ``--manual`` to open the dashboard against the hand-transcribed manual
store with Yahoo disabled; it implies ``--no-sync`` and needs no environment
variables set by hand.

The resolved KV store is printed first, every time, and the startup sync
refuses to run against anything but the default ``data/local.db`` -- see
:func:`guard_sync_target` and ``docs/manual-pipeline-runbook.md``.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# The baseline path this script guards against is NOT re-derived here. It lives
# in ``kv_store`` and is read, at call time, inside
# ``kv_sync.sync_destination_refusal`` -- so this script and
# ``scripts/refresh_remote.py`` cannot drift apart, and neither can drift from
# the store ``get_kv()`` actually resolves.
from fantasy_baseball.data.kv_store import get_kv, is_remote
from fantasy_baseball.data.kv_sync import sync_destination_refusal, sync_remote_to_local
from fantasy_baseball.manual import environment as _manual_env
from fantasy_baseball.manual.seed import describe_kv_target, resolve_kv_path
from fantasy_baseball.web.season_app import create_app

#: Exit codes, matching ``scripts/run_manual_refresh.py``: 2 means "refused,
#: nothing happened", so a wrapper can tell it apart from a run that started
#: and then failed.
RC_OK = 0
RC_REFUSED = 2


def resolve_kv_target() -> tuple[Path | None, str]:
    """Return ``(absolute SQLite path or None, human description)``.

    Reports the OUTCOME -- the store the live ``get_kv()`` singleton is
    actually backed by -- rather than re-reading ``FANTASY_LOCAL_KV_PATH``
    and hoping the two agree. That is the same store
    ``sync_remote_to_local()`` resolves for its destination, which is the
    one the guard below has to reason about.

    The path is ``None`` on Render, where the KV is Upstash and has no
    local file at all.
    """
    if is_remote():
        return None, "Upstash (RENDER is set -- production store, no local file)"
    client = get_kv()
    return resolve_kv_path(client), describe_kv_target(client)


def guard_sync_target(kv_path: Path | None) -> int:
    """``RC_OK`` when the startup sync may run, else ``RC_REFUSED``.

    ``kv_sync.sync_remote_to_local()`` resolves its destination as
    ``local if local is not None else get_kv()`` and then wipes it
    UNCONDITIONALLY -- ``DELETE FROM kv; DELETE FROM hash_kv;`` -- before
    refilling it from Upstash. This script is one of the callers that
    passes ``local=None``, so the destination is whatever
    ``FANTASY_LOCAL_KV_PATH`` points at.

    That makes an exported ``FANTASY_LOCAL_KV_PATH=data/manual.db`` -- the
    isolated store the Yahoo-free pipeline writes, which
    ``scripts/run_manual_refresh.py`` sets in the same shell -- a silent
    destroy-and-replace: the hand-transcribed standings and rosters are
    deleted and the store is refilled with the last Yahoo snapshot, with
    no error and no prompt. So the sync runs only against the default
    baseline; ``--no-sync`` is how you open the dashboard against a
    manual store.

    The comparison and the wording live in ``kv_sync.sync_destination_refusal``,
    shared with ``scripts/refresh_remote.py``. This wrapper keeps the dashboard's
    own recovery advice and exit code. ``kv_path`` is the path the startup banner
    already printed, passed through so the banner and the refusal cannot name
    different stores.
    """
    refusal = sync_destination_refusal(
        kv_path,
        action="The startup sync",
        recovery=[
            "Either:",
            "  * re-run with --no-sync to open the dashboard against this store, or",
            "  * unset FANTASY_LOCAL_KV_PATH and re-run to sync the Yahoo baseline.",
        ],
    )
    if refusal is None:
        return RC_OK
    print(refusal)
    return RC_REFUSED


def guard_manual_store(bound: Path | None) -> int:
    """``RC_OK`` when ``--manual`` really bound a real, seeded manual store.

    ``bound`` is what ``activate_manual_environment`` reports it set, NOT the
    constant this compares against -- passing the constant would make the
    identity check compare a value with itself and assert nothing.
    """
    expected = _manual_env.DEFAULT_MANUAL_KV_PATH.resolve()
    if bound is None or bound != expected:
        print(
            "--manual did not bind the manual store.\n"
            f"  expected : {expected}\n"
            f"  bound    : {bound}\n"
            "Refusing rather than serving the wrong store under a manual banner."
        )
        return RC_REFUSED

    refusal = _manual_env.manual_store_refusal(expected)
    if refusal is None:
        return RC_OK
    print(refusal)
    return RC_REFUSED


def enter_manual_mode(args) -> int:
    """Bind the manual store and verify it, or refuse. ``RC_OK`` to continue.

    Everything ``--manual`` does lives here: bind, skip the sync, check the
    store. The sync is skipped rather than merely allowed to fail --
    ``sync_remote_to_local()`` wipes its destination before refilling from
    production, which against the transcription is pure loss.
    """
    try:
        bound = _manual_env.activate_manual_environment()
    except RuntimeError as exc:
        print(f"{exc}\n--manual is a local-only mode.")
        return RC_REFUSED

    args.no_sync = True
    print(f"KV store: {bound}")

    rc = guard_manual_store(bound)
    if rc != RC_OK:
        return rc
    print("Manual mode: Yahoo disabled, startup sync skipped.")
    return RC_OK


def _should_run_sync(no_sync: bool) -> bool:
    """Whether to run the remote->local KV sync on startup.

    Runs exactly once. Skipped when ``--no-sync`` was passed, on Render
    (where the Upstash KV is authoritative and there's nothing to sync
    to), or inside Flask's debug-reloader child process
    (``WERKZEUG_RUN_MAIN=true``). Without the reloader guard, ``main()``
    executes in both the reloader supervisor and the child, so the sync
    -- ~1,300+ network reads -- would fire twice per startup.
    """
    if no_sync or is_remote():
        return False
    # In the reloader child WERKZEUG_RUN_MAIN=="true"; only the supervisor syncs.
    return os.environ.get("WERKZEUG_RUN_MAIN") != "true"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the initial remote->local KV sync.",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Open the dashboard against the hand-transcribed manual store "
            "(data/manual.db) with Yahoo disabled. Implies --no-sync, because "
            "the sync would wipe that store. Equivalent to setting "
            "FANTASY_LOCAL_KV_PATH and FB_SKIP_YAHOO by hand."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to serve the dashboard on (default: 5001).",
    )
    args = parser.parse_args()

    # The store is decided here rather than at import time: nothing in src/ or
    # scripts/ resolves one at module level, and both helpers discard an
    # existing singleton, so argparse gets to decide and main() stays
    # re-entrant.
    if args.manual:
        rc = enter_manual_mode(args)
        if rc != RC_OK:
            return rc
        kv_path = _manual_env.DEFAULT_MANUAL_KV_PATH.resolve()
    else:
        cleared = _manual_env.deactivate_manual_environment()
        kv_path, kv_description = resolve_kv_target()
        print(f"KV store: {kv_description}")

        # Name what changed and what did not. Clearing silently would make a
        # shell that worked yesterday behave differently today; leaving
        # FB_SKIP_YAHOO set silently would put the run in stale-data mode with
        # nothing on screen saying so.
        if cleared:
            names = ", ".join(sorted(cleared))
            print(f"Ignored inherited {names} (no --manual); reading the Yahoo baseline.")
        for name, value in sorted(_manual_env.surviving_manual_vars().items()):
            print(f"Note: {name}={value} is still set -- Yahoo calls stay disabled.")

    if _should_run_sync(args.no_sync):
        rc = guard_sync_target(kv_path)
        if rc != RC_OK:
            return rc
        print("Syncing remote Upstash KV -> local SQLite...")
        stats = sync_remote_to_local()
        print(f"  synced: {stats.summary()}")
        # Surface freshness on startup so it's obvious how stale the
        # local KV is. Pulls from the same KV the dashboard reads from
        # (the just-synced SQLite) -- meta is written by the refresh
        # pipeline as "YYYY-MM-DD HH:MM" local time.
        from fantasy_baseball.web.season_data import read_meta

        meta = read_meta()
        last_refresh = meta.get("last_refresh") if meta else None
        if last_refresh:
            print(f"  last_refresh: {last_refresh}")
        else:
            print("  last_refresh: (none -- Upstash may be empty)")

    app = create_app()
    print(f"Season dashboard: http://localhost:{args.port}")
    app.run(port=args.port, debug=True)
    return RC_OK


if __name__ == "__main__":
    sys.exit(main())
