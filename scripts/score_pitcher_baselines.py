"""Score FanGraphs' pitcher projections and the simple baselines on actuals (#465).

The bar a pitcher model has to clear: preseason (whole-season actuals) and every dated
mid-season snapshot, MSE on K, W, SV, ERA and WHIP. Writes scored_preseason.parquet,
scored_snapshots.parquet and scores.md to data/pitcher_ros/baselines/ and prints the
summary. Needs the pitching lines in the pitch-data store
(scripts/fetch_pitch_data.py --only pitching).

Usage:
    python scripts/score_pitcher_baselines.py
    python scripts/score_pitcher_baselines.py --seasons 2024 2025 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import league_denominators
from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.stats import load_games

STORE = PROJECT_ROOT / "data" / "pitch_data"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
LEAGUE = PROJECT_ROOT / "config" / "league.yaml"
OUT = PROJECT_ROOT / "data" / "pitcher_ros" / "baselines"
# The seasons with FanGraphs pitcher projections on disk.
DEFAULT_SEASONS = (2022, 2023, 2024, 2025, 2026)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    games = load_games(STORE)
    denoms = league_denominators(LEAGUE)
    pre_parts, snap_parts = [], []
    for season in sorted(args.seasons):
        pre = backtest.preseason(games, season, PROJECTIONS, denoms)
        if pre is None:
            print(f"{season}: no games played yet, skipped")
            continue
        pre_parts.append(pre)
        snap = backtest.snapshots(games, season, PROJECTIONS, denoms)
        if snap is not None:
            snap_parts.append(snap)
    pre_all = pd.concat(pre_parts, ignore_index=True) if pre_parts else None
    snap_all = pd.concat(snap_parts, ignore_index=True) if snap_parts else None

    args.out.mkdir(parents=True, exist_ok=True)
    for frame, name in ((pre_all, "scored_preseason"), (snap_all, "scored_snapshots")):
        path = args.out / f"{name}.parquet"
        if frame is not None:
            frame.to_parquet(path)
        else:
            path.unlink(missing_ok=True)
    md = ["### Pitcher projections vs. actuals", "", *backtest.summarize(pre_all, snap_all)]
    (args.out / "scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
