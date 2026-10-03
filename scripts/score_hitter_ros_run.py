"""Re-score saved hitter ROS runs against FanGraphs and the simple baselines (#410).

Reads data/hitter_ros/runs/<name>/predictions.parquet, scores it (preseason and the
mid-season snapshots), writes scored_preseason.parquet, scored_snapshots.parquet and
scores.md into the run folder, and prints the summary. With several runs (e.g. the same
settings under different seeds), also prints the range of our MAE across them.

Usage:
    python scripts/score_hitter_ros_run.py 001-baseline-mlp
    python scripts/score_hitter_ros_run.py 001-baseline-mlp 001-seed1 001-seed2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros import backtest
from fantasy_baseball.hitter_ros.evaluate import mae_table
from fantasy_baseball.hitter_ros.features import TARGETS

TABLE = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"


def score_run(table: pd.DataFrame, run: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    preds = pd.read_parquet(run / "predictions.parquet")
    pre_all, snap_all = backtest.score_predictions(table, preds, PROJECTIONS, STORE)
    backtest.write_scores(run, pre_all, snap_all)
    md = [f"### Scores for run `{run.name}`", "", *backtest.summarize(pre_all, snap_all)]
    (run / "scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return pre_all, snap_all


def seed_range(results: dict[str, tuple[pd.DataFrame | None, pd.DataFrame | None]]) -> None:
    """Min / max of our MAE across runs: preseason mean over seasons, and each snapshot."""
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for name, (pre, snap) in results.items():
        if pre is not None:
            means = backtest.mean_over_seasons(pre)
            for s in TARGETS:
                rows.setdefault(("preseason mean", s), {})[name] = means.loc[backtest.OURS, s]
        if snap is not None:
            for snapshot, g in snap.groupby("snapshot"):
                t = mae_table(g)
                for s in TARGETS:
                    rows.setdefault((snapshot, s), {})[name] = t.loc[backtest.OURS, s]
    if not rows:
        print("\nNo run produced scores; nothing to compare across runs.")
        return
    df = pd.DataFrame(rows).T
    cells = df.min(axis=1).map("{:.2f}".format) + " - " + df.max(axis=1).map("{:.2f}".format)
    summary = cells.unstack()[list(TARGETS)]
    print("\n### Our MAE across runs (min - max)\n")
    print(backtest.to_markdown(summary))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    table = pd.read_parquet(TABLE)
    results = {name: score_run(table, RUNS / name) for name in args.runs}
    if len(results) > 1:
        seed_range(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
