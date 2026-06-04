#!/usr/bin/env python3
"""Guided ingest of FanGraphs one-click member exports -> prod Upstash ROS.

FanGraphs supports only one-click member exports (no scraping/API/web query).
This walks you through exporting the 5 systems x {hitters, pitchers} in your
browser, stages each freshly-downloaded CSV into today's snapshot dir, then
blends and pushes to prod (the same RENDER-flip tail as the manual restore).

Usage:
    python scripts/ingest_ros_export.py                 # ~/Downloads, push to prod
    python scripts/ingest_ros_export.py --source D:/dl  # custom download dir
    python scripts/ingest_ros_export.py --no-push       # stage only (dry run)
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _push_to_prod(season_year: int, systems: list[str]) -> None:
    """Blend today's staged snapshot and write it to prod Upstash, then verify.

    Only ``systems`` (the fully-staged ones) are blended -- a skipped or
    half-staged system is excluded entirely, so it can't leak one side (e.g.
    hitters-only) into the blend. Weights are filtered to match; blend_projections
    re-normalizes them.

    Mirrors scripts/refresh_remote.py: flip RENDER on BEFORE importing the
    pipeline so get_kv() resolves to Upstash.
    """
    import json

    os.environ["RENDER"] = "true"

    from fantasy_baseball.config import load_config
    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv, get_kv
    from fantasy_baseball.data.redis_store import get_latest_roster_names
    from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros

    kv_store._reset_singleton()
    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    projections_dir = PROJECT_ROOT / "data" / "projections"
    roster_names = get_latest_roster_names(get_kv())
    weights = {s: config.projection_weights[s] for s in systems}

    print(f"Blending {len(systems)} complete system(s) -> prod Upstash...")
    ros_h, ros_p = blend_and_cache_ros(
        projections_dir,
        systems,
        weights,
        roster_names,
        season_year,
        progress_cb=lambda m: print(f"  {m}") if not m.startswith("QUALITY") else None,
    )
    print(f"Persisted {len(ros_h)} ROS hitters + {len(ros_p)} ROS pitchers to prod")

    remote = build_explicit_upstash_kv()
    for key in ("cache:ros_projections", "cache:full_season_projections"):
        obj = json.loads(remote.get(key))
        meta = obj.get("_meta", {})
        data = obj.get("_data", obj)
        print(
            f"{key}: snapshot={meta.get('_ros_snapshot_date')} "
            f"hitters={len(data.get('hitters', []))} pitchers={len(data.get('pitchers', []))}"
        )


def main() -> int:
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.ros_export_ingest import run_guided_ingest
    from fantasy_baseball.utils.time_utils import local_today

    parser = argparse.ArgumentParser(description="Ingest FanGraphs one-click exports to prod ROS.")
    parser.add_argument("--source", default=str(Path.home() / "Downloads"), help="download dir")
    parser.add_argument("--season", type=int, default=None, help="season year (default: config)")
    parser.add_argument("--no-push", action="store_true", help="stage only; do not push to prod")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    season = args.season or config.season_year
    dest_dir = (
        PROJECT_ROOT
        / "data"
        / "projections"
        / str(season)
        / "rest_of_season"
        / local_today().isoformat()
    )
    source_dir = Path(args.source)

    print(f"Exports source: {source_dir}")
    print(f"Staging into:   {dest_dir}\n")
    result = run_guided_ingest(
        config.projection_systems,
        source_dir,
        dest_dir,
        prompt_fn=input,
        output_fn=print,
        now_fn=time.time,
    )

    if result.aborted:
        print("\nAborted -- nothing pushed; last-good prod ROS unchanged.")
        return 1
    complete = result.complete_systems(config.projection_systems)
    if not complete:
        print("\nNo complete systems staged -- not pushing; last-good prod ROS unchanged.")
        return 1
    print(f"\nComplete systems: {', '.join(complete)}")
    if result.skipped_systems:
        print(f"Skipped: {', '.join(sorted(result.skipped_systems))}")
    if args.no_push:
        print("--no-push set; staged only, prod unchanged.")
        return 0

    _push_to_prod(season, complete)
    print("\nDone. (Run scripts/refresh_remote.py to propagate into dashboard standings.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
