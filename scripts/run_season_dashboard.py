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


def guard_manual_store(kv_path: Path | None) -> int:
    """``RC_OK`` when ``--manual`` bound a real, seeded manual store.

    Checks the OUTCOME, not the flag, and checks the store rather than the
    path: `manual_store_refusal` inspects the file without opening it through
    `get_kv()`, which would create it.
    """
    expected = _manual_env.DEFAULT_MANUAL_KV_PATH.resolve()
    if kv_path is None:
        print(
            "--manual is meaningless on Render: the KV there is Upstash, which no "
            "environment variable can redirect."
        )
        return RC_REFUSED
    if kv_path != expected:
        print(
            "--manual did not bind the manual store.\n"
            f"  expected : {expected}\n"
            f"  resolved : {kv_path}\n"
            "Refusing rather than serving the wrong store under a manual banner."
        )
        return RC_REFUSED

    refusal = _manual_env.manual_store_refusal(kv_path)
    if refusal is None:
        return RC_OK
    print(refusal)
    return RC_REFUSED


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

    # Bind (or unbind) the store BEFORE anything resolves one. This can live
    # here rather than at import time because nothing in src/ or scripts/ calls
    # get_kv() at module level, and both helpers discard any singleton that
    # already exists -- so argparse gets to decide, and main() stays re-entrant.
    if args.manual:
        _manual_env.activate_manual_environment()
        cleared: dict[str, str] = {}
    else:
        cleared = _manual_env.deactivate_manual_environment()

    # --manual implies --no-sync rather than merely being compatible with it:
    # sync_remote_to_local() wipes its destination before refilling, so on the
    # manual store it is a silent destroy-and-replace. guard_sync_target()
    # would refuse anyway; making it implicit means the user never has to
    # remember the second flag to avoid an error they should never see.
    if args.manual:
        args.no_sync = True

    # Printed before any sync or serve so a terminal's mode is never
    # ambiguous: an exported FANTASY_LOCAL_KV_PATH is invisible otherwise.
    # Before resolve_kv_target(), which calls get_kv() and would CREATE the file.
    if args.manual:
        rc = guard_manual_store(_manual_env.DEFAULT_MANUAL_KV_PATH.resolve())
        if rc != RC_OK:
            return rc

    kv_path, kv_description = resolve_kv_target()
    print(f"KV store: {kv_description}")

    # Say what was taken away. Clearing an inherited manual binding is the right
    # default, but doing it silently would make a shell that "worked yesterday"
    # behave differently today with nothing on screen to explain it.
    if cleared:
        names = ", ".join(sorted(cleared))
        print(f"Ignored inherited {names} (no --manual); reading the Yahoo baseline.")

    # The early sys.argv sniff above is what actually bound the store; this
    # asserts the outcome matches the parsed flag, so a future refactor that
    # moves or breaks that sniff fails loudly here instead of serving the Yahoo
    # baseline while the banner says "manual".
    if args.manual:
        print("Manual mode: Yahoo disabled, startup sync skipped.")

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
