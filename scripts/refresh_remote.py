#!/usr/bin/env python3
"""Run the season dashboard refresh locally, writing to remote Upstash.

Normal flow: Render runs ``run_full_refresh`` on its schedule and writes
to Upstash. Sometimes we want to trigger a refresh out-of-band (before
a cron fires, while iterating on pipeline code, or because Render is
asleep). This script does exactly that — and then pulls the fresh
state back down so local dashboards see it too.

Steps:
  1. Set ``RENDER=true`` in-process so ``get_kv()`` resolves to
     Upstash (the env gate is the whole point of the redesign; don't
     subvert it, set it).
  2. Run ``run_full_refresh`` exactly as Render would.
  3. Sync Upstash → local SQLite so ``run_season_dashboard.py``
     reflects the new data without a second round trip.

Upstash credentials must be in the environment or ``.env`` — the
dotenv loader in ``kv_store`` picks them up automatically.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
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

    print("Running refresh against remote Upstash...")
    run_full_refresh()
    print("Refresh complete.")

    # Sync back down. We need a handle to remote Upstash explicitly
    # (since get_kv() is now returning Upstash in this process, but
    # the sync's local target must be SQLite — so we flip RENDER off
    # and re-resolve).
    remote = build_explicit_upstash_kv()
    os.environ["RENDER"] = "false"
    kv_store._reset_singleton()

    print("Syncing remote → local SQLite...")
    stats = sync_remote_to_local(remote=remote)
    print(f"  synced: {stats.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
