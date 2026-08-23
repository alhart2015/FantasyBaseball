#!/usr/bin/env python3
"""Launch the season dashboard.

First step: sync the remote Upstash KV down to the local SQLite KV so
the dashboard reads the same state the Render app writes. Skip with
``--no-sync`` when offline or when the remote is known-empty.

The resolved KV store is printed first, every time, and the startup sync
refuses to run against anything but the default ``data/local.db`` -- see
:func:`guard_sync_target` and ``docs/manual-pipeline-runbook.md``.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ``_DEFAULT_LOCAL_DB`` is the single definition of the Yahoo baseline store's
# location; ``fantasy_baseball.manual.seed`` imports it for the same reason.
# Re-deriving the path here would let the two drift, and the refusal below is
# only worth having if it names the same file ``kv_store`` resolves.
from fantasy_baseball.data.kv_store import _DEFAULT_LOCAL_DB, get_kv, is_remote
from fantasy_baseball.data.kv_sync import sync_remote_to_local
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
    """
    baseline = _DEFAULT_LOCAL_DB.resolve()
    if kv_path == baseline:
        return RC_OK

    target = str(kv_path) if kv_path is not None else "a store with no local file"
    print("")
    print(f"REFUSING TO SYNC: the resolved KV store is {target},")
    print(f"not the Yahoo baseline {baseline}.")
    print(
        "The startup sync WIPES its destination (DELETE FROM kv; DELETE FROM hash_kv;) "
        "and refills it from Upstash, so syncing here would destroy this store -- most "
        "likely the hand-transcribed manual store written by "
        "scripts/run_manual_refresh.py."
    )
    print("")
    print("Either:")
    print("  * re-run with --no-sync to open the dashboard against this store, or")
    print("  * unset FANTASY_LOCAL_KV_PATH and re-run to sync the Yahoo baseline.")
    print("See docs/manual-pipeline-runbook.md.")
    print("No sync ran and nothing was deleted.")
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
        "--port",
        type=int,
        default=5001,
        help="Port to serve the dashboard on (default: 5001).",
    )
    args = parser.parse_args()

    # Printed before any sync or serve so a terminal's mode is never
    # ambiguous: an exported FANTASY_LOCAL_KV_PATH is invisible otherwise.
    kv_path, kv_description = resolve_kv_target()
    print(f"KV store: {kv_description}")

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
