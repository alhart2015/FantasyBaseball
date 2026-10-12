"""Train the pitcher rest-of-season net and score it against FanGraphs (#465).

For each test season: train on every complete season before it, predict its rows, then
score the predictions as ``ours`` beside FanGraphs and the baselines (preseason and the
mid-season snapshots, pitcher_ros.backtest). Writes predictions.parquet, config.json,
info.json, the scored frames and scores.md to data/pitcher_ros/runs/<name>/.

Needs the training table (scripts/build_pitcher_ros_table.py) and torch (see
scripts/train_hitter_ros.py).

Usage:
    python scripts/train_pitcher_ros.py --name 001-plain
    python scripts/train_pitcher_ros.py --name 001-plain-s1 --seed 1
    python scripts/train_pitcher_ros.py --name quick --test-seasons 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import league_denominators
from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.features import input_frame, target_frame, weight_frame
from fantasy_baseball.pitcher_ros.model import fit_season, plain_config
from fantasy_baseball.pitcher_ros.stats import load_games

TABLE = PROJECT_ROOT / "data" / "pitcher_ros" / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
LEAGUE = PROJECT_ROOT / "config" / "league.yaml"
RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
# The seasons with FanGraphs pitcher projections on disk.
DEFAULT_TEST_SEASONS = (2022, 2023, 2024, 2025, 2026)
OURS = "ours"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="run folder under data/pitcher_ros/runs/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-seasons", type=int, nargs="+", default=list(DEFAULT_TEST_SEASONS))
    parser.add_argument("--table", type=Path, default=TABLE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    config = plain_config(seed=args.seed)
    (run / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    table = pd.read_parquet(args.table)
    x, y, w = input_frame(table), target_frame(table), weight_frame(table)
    parts, infos = [], []
    for season in sorted(args.test_seasons):
        logging.info("test season %s", season)
        preds, info = fit_season(table, x, y, w, season, config)
        parts.append(preds)
        infos.append(info)
    preds = pd.concat(parts, ignore_index=True)
    preds.to_parquet(run / "predictions.parquet", index=False)
    (run / "info.json").write_text(json.dumps(infos, indent=2))

    scores = backtest.score_seasons(
        load_games(STORE),
        args.test_seasons,
        PROJECTIONS,
        league_denominators(LEAGUE),
        ours={OURS: preds},
    )
    for frame, name in (
        (scores.pre, "scored_preseason"),
        (scores.snap, "scored_snapshots"),
        (scores.pre_pooled, "scored_preseason_pooled"),
        (scores.snap_pooled, "scored_snapshots_pooled"),
    ):
        if frame is not None:
            frame.to_parquet(run / f"{name}.parquet")
    md = [f"### Scores for pitcher run `{args.name}`", "", *backtest.summarize(scores)]
    (run / "scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if scores.failures else 0


if __name__ == "__main__":
    sys.exit(main())
