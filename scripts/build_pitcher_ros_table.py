"""Build the pitcher ROS training table from the pitch-data store (#465).

Reads data/pitch_data/ (fill it with scripts/fetch_pitch_data.py; needs the pitching,
lineups and schedule files) and writes one row per (pitcher, season, as-of week) to
data/pitcher_ros/table.parquet.

Usage:
    python scripts/build_pitcher_ros_table.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.pitcher_ros.table import build_table

DEFAULT_STORE = PROJECT_ROOT / "data" / "pitch_data"
DEFAULT_OUT = PROJECT_ROOT / "data" / "pitcher_ros" / "table.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = build_table(args.store)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"{len(df)} rows x {len(df.columns)} columns -> {args.out}")
    print("\nseason  rows  pitchers  weeks  complete")
    for season, g in df.groupby("season"):
        print(
            f"{season}  {len(g):>6}  {g.player_id.nunique():>8}  {g.week.nunique():>5}"
            f"  {bool(g.season_complete.all())!s:>8}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
