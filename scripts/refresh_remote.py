#!/usr/bin/env python3
"""Run the season dashboard refresh locally, writing to remote Upstash.

Normal flow: Render runs ``run_full_refresh`` on its schedule and writes
to Upstash. Sometimes we want to trigger a refresh out-of-band (before
a cron fires, while iterating on pipeline code, or because Render is
asleep). This script does exactly that -- and then pulls the fresh
state back down so local dashboards see it too.

Steps:
  1. Set ``RENDER=true`` in-process so ``get_kv()`` resolves to
     Upstash (the env gate is the whole point of the redesign; don't
     subvert it, set it).
  2. Run ``run_full_refresh`` exactly as Render would.
  3. Sync Upstash -> local SQLite so ``run_season_dashboard.py``
     reflects the new data without a second round trip.

Upstash credentials must be in the environment or ``.env`` -- the
dotenv loader in ``kv_store`` picks them up automatically.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


#: Exit codes, matching scripts/run_season_dashboard.py and
#: scripts/run_manual_refresh.py: 2 means "refused, nothing happened".
RC_REFUSED = 2


def _sync_destination_refusal() -> str | None:
    """Refusal message when the sync-back would wipe a non-baseline store, else None.

    Step 3 calls ``sync_remote_to_local(remote=remote)`` with no ``local=``, so the
    destination is whatever ``get_kv()`` resolves -- i.e. whatever
    ``FANTASY_LOCAL_KV_PATH`` points at. The sync then runs
    ``DELETE FROM kv; DELETE FROM hash_kv;`` on that store before refilling it.

    ``scripts/run_manual_refresh.py`` exports ``FANTASY_LOCAL_KV_PATH`` to isolate
    the Yahoo-free manual store, so running this script from that same shell
    silently destroys the hand-transcribed standings and rosters and refills them
    with the last Yahoo snapshot -- no error, and a store that still looks
    populated afterwards.

    The check lives here rather than in ``kv_sync`` because
    ``FANTASY_LOCAL_KV_PATH`` is ALSO how the test suite isolates its KV:
    ``tests/test_data/test_kv_sync.py::test_default_local_is_get_kv`` pins the
    library contract that the default destination is simply whatever ``get_kv()``
    returns. Narrowing that contract library-side breaks legitimate callers -- the
    hazard is specific to operator-facing entry points, so the guard belongs in
    them.

    Steps 1 and 2 have already written to prod Upstash by the time this runs; that
    is the intended effect of the script and is not undone. Only the local
    sync-back is refused.
    """
    from fantasy_baseball.data.kv_store import _DEFAULT_LOCAL_DB, get_kv
    from fantasy_baseball.manual.seed import resolve_kv_path

    resolved = resolve_kv_path(get_kv())
    baseline = _DEFAULT_LOCAL_DB.resolve()
    if resolved == baseline:
        return None

    target = str(resolved) if resolved is not None else "a store with no local file"
    lines = [
        "",
        f"REFUSING TO SYNC BACK: the resolved local KV store is {target},",
        f"not the Yahoo baseline {baseline}.",
        "The sync-back WIPES its destination (DELETE FROM kv; DELETE FROM "
        "hash_kv;) and refills it from Upstash, so it would destroy this store "
        "-- most likely the hand-transcribed manual store written by "
        "scripts/run_manual_refresh.py.",
        "",
        "The remote refresh already completed; only the local sync-back was skipped.",
        "Unset FANTASY_LOCAL_KV_PATH and re-run if you want the baseline synced.",
        "See docs/manual-pipeline-runbook.md.",
    ]
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-yahoo",
        action="store_true",
        help=(
            "Run in stale-data mode: skip every step that pulls from Yahoo "
            "(auth, rosters, standings, free agents, transactions) and reuse "
            "the last persisted league state. Everything that depends only on "
            "MLB game logs and projections still recomputes. Use while Yahoo "
            "auth is broken. Equivalent to FB_SKIP_YAHOO=1."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Must flip the gate BEFORE importing the pipeline: import-time
    # module state (e.g. cached singletons) reads RENDER once.
    os.environ["RENDER"] = "true"

    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
    from fantasy_baseball.data.kv_sync import sync_remote_to_local
    from fantasy_baseball.web.refresh_pipeline import run_full_refresh

    # In case anything has already cached a local singleton during
    # import, clear it so the first post-flip get_kv() rebuilds as
    # Upstash.
    kv_store._reset_singleton()

    if args.skip_yahoo:
        print("Running refresh against remote Upstash (stale-data mode: Yahoo steps skipped)...")
    else:
        print("Running refresh against remote Upstash...")
    # Pass None when the flag is absent so FB_SKIP_YAHOO still applies.
    run_full_refresh(skip_yahoo=True if args.skip_yahoo else None)
    print("Refresh complete.")

    # Archive a trimmed snapshot of the ROS projection vintage this refresh
    # used, so the in-season playing-time residual can be calibrated later
    # (projected-vs-realized). Lives here rather than in the deployed pipeline
    # because ROS is fetched manually -- only a manual refresh should add a
    # snapshot; Render's cron would otherwise re-archive stale ROS. RENDER is
    # still "true" here, so get_kv() / read_cache resolve to Upstash.
    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.data.redis_store import write_ros_projection_snapshot
    from fantasy_baseball.data.ros_pipeline import parse_snapshot_date
    from fantasy_baseball.web.season_data import read_cache_with_meta

    # Best-effort side-car: a failure here must NOT abort the remote->local
    # sync below (the dashboard's whole point), so swallow-and-log.
    try:
        ros_blob, ros_meta = read_cache_with_meta(CacheKey.ROS_PROJECTIONS)
        # Normalize to a clean ISO key so it matches the other weekly histories;
        # a hand-staged vintage may be "YYYY-MM-DD-manual" (see parse_snapshot_date).
        vintage = parse_snapshot_date(ros_meta.get("_ros_snapshot_date") or "")
        if ros_blob and vintage:
            write_ros_projection_snapshot(get_kv(), ros_blob, vintage.isoformat())
            print(f"Archived ROS projection snapshot for {vintage.isoformat()}.")
        else:
            print("No ROS projection snapshot archived (missing blob or snapshot date).")
    except Exception as exc:
        print(f"WARNING: ROS snapshot archive failed ({type(exc).__name__}: {exc}); continuing.")

    # Sync back down. We need a handle to remote Upstash explicitly
    # (since get_kv() is now returning Upstash in this process, but
    # the sync's local target must be SQLite -- so we flip RENDER off
    # and re-resolve).
    remote = build_explicit_upstash_kv()
    os.environ["RENDER"] = "false"
    kv_store._reset_singleton()

    refusal = _sync_destination_refusal()
    if refusal is not None:
        print(refusal)
        return RC_REFUSED

    print("Syncing remote -> local SQLite...")
    stats = sync_remote_to_local(remote=remote)
    print(f"  synced: {stats.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
