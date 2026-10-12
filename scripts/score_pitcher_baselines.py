"""Score FanGraphs' pitcher projections and the simple baselines on actuals (#465).

The bar a pitcher model has to clear: preseason (whole-season actuals) and every dated
mid-season snapshot, MSE on K, W, SV, ERA and WHIP. Writes the scored frames
(scored_preseason/_snapshots.parquet, and the *_pooled ones behind the pooled tables)
and scores.md to data/pitcher_ros/baselines/ and prints the summary. A season that
can't be scored is reported and the rest still are; the exit code is then 1. Needs
the pitching lines in the pitch-data store (scripts/fetch_pitch_data.py --only pitching).

Usage:
    python scripts/score_pitcher_baselines.py
    python scripts/score_pitcher_baselines.py --seasons 2024 2025 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scores = backtest.score_seasons(
        load_games(STORE), args.seasons, PROJECTIONS, league_denominators(LEAGUE)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    frames = {
        "scored_preseason": scores.pre,
        "scored_snapshots": scores.snap,
        "scored_preseason_pooled": scores.pre_pooled,
        "scored_snapshots_pooled": scores.snap_pooled,
    }
    for name, frame in frames.items():
        path = args.out / f"{name}.parquet"
        if frame is not None:
            frame.to_parquet(path)
        else:
            path.unlink(missing_ok=True)
    md = ["### Pitcher projections vs. actuals", "", *backtest.summarize(scores)]
    (args.out / "scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if scores.failures else 0


if __name__ == "__main__":
    sys.exit(main())
