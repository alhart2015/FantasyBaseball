"""Build the pitch tokens for pretraining the hitter ROS encoder (#415).

Reads data/pitch_data/ and writes one row per regular-season pitch a hitter saw to
data/hitter_ros/pitch_tokens.parquet (see hitter_ros/pitch_tokens.py). ~8.3M rows.

Usage:
    python scripts/build_hitter_ros_pitch_tokens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.pitch_tokens import OUTCOMES, build_pitch_tokens

STORE = PROJECT_ROOT / "data" / "pitch_data"
OUT = PROJECT_ROOT / "data" / "hitter_ros" / "pitch_tokens.parquet"


def main() -> int:
    tokens = build_pitch_tokens(STORE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tokens.to_parquet(OUT, index=False)
    print(f"{len(tokens)} pitches, {tokens['player_id'].nunique()} hitters -> {OUT}")
    print("\nseason  pitches")
    for season, n in tokens.groupby("season").size().items():
        print(f"{season}  {n}")
    shares = tokens["outcome"].value_counts(normalize=True).sort_index()
    print(
        "\noutcome shares: "
        + ", ".join(f"{o} {shares.get(i, 0):.3f}" for i, o in enumerate(OUTCOMES))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
