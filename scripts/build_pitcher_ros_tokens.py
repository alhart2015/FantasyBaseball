"""Build the pitcher net's appearance tokens (#465): data/pitcher_ros/app_tokens.parquet.

One row per regular-season appearance, from the pitch-data store's box scores and
pitch tracking (see pitcher_ros/sequence.py). The pitcher net's default transformer
(train_pitcher_ros.py) reads it. Rebuild whenever the store gains games.

Usage:
    python scripts/build_pitcher_ros_tokens.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.pitcher_ros.sequence import TOKEN_FEATURES, build_appearance_tokens

DEFAULT_STORE = PROJECT_ROOT / "data" / "pitch_data"
DEFAULT_OUT = PROJECT_ROOT / "data" / "pitcher_ros" / "app_tokens.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tokens = build_appearance_tokens(args.store)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokens.to_parquet(args.out, index=False)
    print(f"{len(tokens)} appearances x {len(TOKEN_FEATURES)} features -> {args.out}")
    print("\nseason  appearances  pitchers  tracked")
    for season, g in tokens.groupby("season"):
        print(f"{season}  {len(g):>11}  {g.player_id.nunique():>8}  {g.tracked.mean():>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
