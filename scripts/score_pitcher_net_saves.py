"""Score the pitcher net with its SV taken from the saves model (#474).

``ours`` is a pitcher net run's predictions (scripts/train_pitcher_ros.py); ``ours_sv``
is the same with SV from the saves model's ``mix`` (the average of FanGraphs' share and
our usage share, each x our team projection; scripts/compare_pitcher_saves.py), as a
rate over FanGraphs' projected batters faced. Both are scored beside FanGraphs and the
baselines (pitcher_ros.backtest), on the seasons the saves units cover. Writes
saves-scores.md into the run's folder and prints it.

Needs the run and the saves units for the seed (units-pre-s<seed>.parquet,
units-mid-s<seed>.parquet in data/pitcher_ros/saves/, from compare_pitcher_saves.py).

Usage:
    python scripts/score_pitcher_net_saves.py --run 001-plain-s0 --saves-seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import league_denominators
from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.saves import net_with_saves
from fantasy_baseball.pitcher_ros.stats import load_games

STORE = PROJECT_ROOT / "data" / "pitch_data"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
LEAGUE = PROJECT_ROOT / "config" / "league.yaml"
RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
SAVES = PROJECT_ROOT / "data" / "pitcher_ros" / "saves"
METHOD = "mix"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder under data/pitcher_ros/runs/")
    parser.add_argument("--saves-seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = RUNS / args.run
    paths = {
        "predictions": run / "predictions.parquet",
        "pre": SAVES / f"units-pre-s{args.saves_seed}.parquet",
        "mid": SAVES / f"units-mid-s{args.saves_seed}.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        parser.error(f"missing {missing}; run train_pitcher_ros.py / compare_pitcher_saves.py")
    preds = pd.read_parquet(paths["predictions"])
    pre, mid = pd.read_parquet(paths["pre"]), pd.read_parquet(paths["mid"])
    if "bf_fg" not in pre.columns:
        parser.error(f"{paths['pre']} predates bf_fg; rerun scripts/compare_pitcher_saves.py")

    seasons = sorted(set(preds["season"]) & set(pre["season"]))
    preds = preds[preds["season"].isin(seasons)]
    scores = backtest.score_seasons(
        load_games(STORE),
        seasons,
        PROJECTIONS,
        league_denominators(LEAGUE),
        ours={"ours": preds, "ours_sv": net_with_saves(preds, pre, mid, METHOD)},
    )
    md = [
        f"### Pitcher run `{args.run}` with SV from the saves model (saves seed {args.saves_seed})",
        "",
        f"`ours_sv` = `ours` with SV from the saves model's `{METHOD}`; every other stat "
        "is the same.",
        "",
        *backtest.summarize(scores),
    ]
    (run / f"saves-scores-s{args.saves_seed}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if scores.failures else 0


if __name__ == "__main__":
    sys.exit(main())
