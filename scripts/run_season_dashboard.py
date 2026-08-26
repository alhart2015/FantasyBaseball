#!/usr/bin/env python3
"""Launch the season dashboard.

First step: sync the remote Upstash KV down to the local SQLite KV so
the dashboard reads the same state the Render app writes. Skip with
``--no-sync`` when offline or when the remote is known-empty.

Pass ``--manual`` to open the dashboard against the hand-transcribed manual
store with Yahoo disabled; it implies ``--no-sync`` and needs no environment
variables set by hand.

A launch without ``--manual`` clears an inherited ``FANTASY_LOCAL_KV_PATH``,
so the sync resolves the baseline. The refusal in :func:`guard_sync_target` is
the backstop behind that. See ``docs/manual-pipeline-runbook.md``.
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
from fantasy_baseball.web.refresh_pipeline import skip_yahoo_requested
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

    The sync wipes its destination -- ``DELETE FROM kv; DELETE FROM hash_kv;``
    -- before refilling from Upstash, and this script passes ``local=None``, so
    the destination is whatever ``FANTASY_LOCAL_KV_PATH`` points at.

    This runs on every syncing launch; it is the REFUSAL that ``main()`` can no
    longer trigger, because the non-manual branch clears the variable first and
    ``--manual`` forces ``--no-sync``. Kept as a backstop against an edit that
    relaxes either -- do not mistake the dead branch for a dead call.

    ``kv_path`` is the path the banner already printed, so the two cannot name
    different stores. The comparison lives in
    ``kv_sync.sync_destination_refusal``.
    """
    refusal = sync_destination_refusal(
        kv_path,
        action="The startup sync",
        # Not "--manual": that binds DEFAULT_MANUAL_KV_PATH unconditionally, and
        # the only way to reach this refusal is FANTASY_LOCAL_KV_PATH naming
        # some OTHER store -- which --manual would not open.
        recovery=[
            "Unset FANTASY_LOCAL_KV_PATH and re-run to sync the Yahoo baseline.",
            "To open the hand-transcribed store instead, re-run with --manual.",
        ],
    )
    if refusal is None:
        return RC_OK
    print(refusal)
    return RC_REFUSED


def enter_manual_mode(args) -> int:
    """Bind the manual store and check it exists and is seeded.

    Runs before :func:`resolve_kv_target`, which would CREATE the file.
    """
    try:
        bound = _manual_env.activate_manual_environment()
    except RuntimeError as exc:
        # activate_manual_environment raises its Render refusal, but also
        # imports the web layer, and a RuntimeError from anywhere in that chain
        # would otherwise be reported as a Render problem it has nothing to do
        # with. Only claim Render when Render is actually set.
        if is_remote():
            print(f"{exc}\n--manual is a local-only mode.")
        else:
            print(f"--manual could not bind the manual store: {exc}")
        return RC_REFUSED

    args.no_sync = True
    refusal = _manual_env.manual_store_refusal(bound)
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

    # Decided here rather than at import: nothing in src/ or scripts/ resolves
    # a store at module level, so argparse gets to choose and main() stays
    # re-entrant.
    cleared: dict[str, str] = {}
    cleared: dict[str, str] = {}
    if args.manual:
        rc = enter_manual_mode(args)
        if rc != RC_OK:
            return rc
    else:
        cleared = _manual_env.deactivate_manual_environment()

    # After the branch above, so it reports the store that was actually
    # resolved rather than the one either path intended.
    kv_path, kv_description = resolve_kv_target()
    print(f"KV store: {kv_description}")

    if args.manual:
        print("Manual mode: Yahoo disabled, startup sync skipped.")
    else:
        # Name what changed and what did not: clearing silently would make a
        # shell that worked yesterday behave differently today, and a surviving
        # FB_SKIP_YAHOO would put the run in stale-data mode invisibly.
        if cleared:
            names = ", ".join(sorted(cleared))
            print(f"Ignored inherited {names} (no --manual); reading the Yahoo baseline.")
        # Only FB_SKIP_YAHOO can survive, and its VALUE decides: FB_SKIP_YAHOO=0
        # is set but off, so presence alone guarantees nothing.
        state = (
            "Yahoo calls disabled" if skip_yahoo_requested() else "not a value that disables Yahoo"
        )
        for name, value in sorted(_manual_env.surviving_manual_vars().items()):
            print(f"Note: inherited {name}={value} -- {state}.")

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
