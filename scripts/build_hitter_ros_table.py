"""Build the hitter ROS training table from the pitch-data store (#403).

Reads data/pitch_data/ (fill it with scripts/fetch_pitch_data.py) and writes one row per
(hitter, season, as-of week) to data/hitter_ros/table.parquet. Takes about a minute.

Usage:
    python scripts/build_hitter_ros_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.table import build_options, build_table

DEFAULT_STORE = PROJECT_ROOT / "data" / "pitch_data"
DEFAULT_OUT = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = build_table(args.store)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    args.out.with_suffix(".json").write_text(json.dumps(build_options(), indent=2))

    print(f"{len(df)} rows x {len(df.columns)} columns -> {args.out}")
    print("\nseason  rows  hitters  weeks")
    for season, g in df.groupby("season"):
        print(f"{season}  {len(g):>6}  {g.player_id.nunique():>7}  {g.week.nunique():>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
