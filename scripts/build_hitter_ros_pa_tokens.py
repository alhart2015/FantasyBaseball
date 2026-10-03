"""Build the plate-appearance tokens for sequence runs of the hitter ROS net (#414).

Reads data/pitch_data/ and writes one row per regular-season PA to
data/hitter_ros/pa_tokens.parquet (see hitter_ros/sequence.py for the features).

Usage:
    python scripts/build_hitter_ros_pa_tokens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.sequence import RESULT_CLASSES, build_pa_tokens

STORE = PROJECT_ROOT / "data" / "pitch_data"
OUT = PROJECT_ROOT / "data" / "hitter_ros" / "pa_tokens.parquet"


def main() -> int:
    tokens = build_pa_tokens(STORE)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tokens.to_parquet(OUT, index=False)
    print(f"{len(tokens)} PAs, {tokens['player_id'].nunique()} hitters -> {OUT}")
    print("\nseason  PAs")
    for season, n in tokens.groupby("season").size().items():
        print(f"{season}  {n}")
    shares = tokens[[f"res_{c}" for c in RESULT_CLASSES]].mean()
    print("\nresult shares: " + ", ".join(f"{c} {shares[f'res_{c}']:.3f}" for c in RESULT_CLASSES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
