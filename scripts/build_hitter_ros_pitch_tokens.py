"""Build the pitch tokens for pretraining the hitter ROS encoder (#415).

Reads data/pitch_data/ and writes one row per regular-season pitch a hitter saw to
data/hitter_ros/pitch_tokens.parquet (see hitter_ros/pitch_tokens.py). ~8.3M rows.
With --zone fixed, pitch heights are measured in one fixed strike zone for every season
(#433) and the file is pitch_tokens_fixed.parquet.

Usage:
    python scripts/build_hitter_ros_pitch_tokens.py
    python scripts/build_hitter_ros_pitch_tokens.py --zone fixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.features import ZONES
from fantasy_baseball.hitter_ros.pitch_tokens import OUTCOMES, build_pitch_tokens

STORE = PROJECT_ROOT / "data" / "pitch_data"
OUT_DIR = PROJECT_ROOT / "data" / "hitter_ros"


def token_path(zone: str) -> Path:
    return OUT_DIR / (
        "pitch_tokens.parquet" if zone == "statcast" else f"pitch_tokens_{zone}.parquet"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--zone",
        choices=list(ZONES),
        default="statcast",
        help="strike zone pitch heights are measured in (#433)",
    )
    args = parser.parse_args()
    tokens = build_pitch_tokens(STORE, zone=args.zone)
    out = token_path(args.zone)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens.to_parquet(out, index=False)
    print(f"{len(tokens)} pitches, {tokens['player_id'].nunique()} hitters -> {out}")
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
