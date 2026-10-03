"""Compare hitter ROS runs side by side: training behavior and scores (#404).

For each run in data/hitter_ros/runs/<name>/:

* best_epoch, val_loss: the early-stopping epoch and the best validation loss, averaged
  over test seasons. Validation loss compares across runs with the same --seed only:
  the seed picks the validation players. A best epoch of 0-1 means the net overfits
  almost at once.
* Preseason: our MAE per stat (mean over test seasons), and the gap to the FanGraphs
  blend (negative = ours better).
* Mid-season: our gap to the blend, averaged over the snapshots between --from and --to.

Usage:
    python scripts/compare_hitter_ros_runs.py 001-baseline-mlp 002a-lr3e-4 002b-lr1e-4
    python scripts/compare_hitter_ros_runs.py --glob "002*" --include 001-baseline-mlp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import BLEND, OURS, mean_over_seasons, to_markdown
from fantasy_baseball.hitter_ros.features import TARGETS

RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"


def run_row(run: Path, snap_from: str, snap_to: str) -> dict[str, object]:
    meta = json.loads((run / "config.json").read_text())
    seasons = meta["seasons"]
    row: dict[str, object] = {
        "best_epoch": float(np.mean([s["best_epoch"] for s in seasons])),
        "val_loss": float(np.mean([min(s["val_loss"]) for s in seasons])),
    }
    pre_path = run / "scored_preseason.parquet"
    if pre_path.exists():
        means = mean_over_seasons(pd.read_parquet(pre_path))
        for s in TARGETS:
            row[f"pre_{s}"] = means.loc[OURS, s]
        for s in TARGETS:
            row[f"pre_gap_{s}"] = means.loc[OURS, s] - means.loc[BLEND, s]
    snap_path = run / "scored_snapshots.parquet"
    if snap_path.exists():
        snap = pd.read_parquet(snap_path)
        snap = snap[snap["snapshot"].between(snap_from, snap_to)]
        per = snap.groupby(["snapshot", "system", "stat"])["abs_err"].mean().unstack("system")
        gap = (per[OURS] - per[BLEND]).groupby(level="stat").mean()
        for s in TARGETS:
            row[f"mid_gap_{s}"] = gap[s]
    return row


def compare(names: list[str], snap_from: str, snap_to: str) -> pd.DataFrame:
    return pd.DataFrame({n: run_row(RUNS / n, snap_from, snap_to) for n in names}).T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--glob", help="also include runs whose folder name matches")
    parser.add_argument("--include", nargs="*", default=[], help="extra runs, listed first")
    parser.add_argument("--from", dest="snap_from", default="2026-06-01")
    parser.add_argument("--to", dest="snap_to", default="2026-07-31")
    args = parser.parse_args()
    names = [*args.include, *args.runs]
    if args.glob:
        names += sorted(p.name for p in RUNS.glob(args.glob) if p.is_dir())
    names = list(dict.fromkeys(names))
    unfinished = [n for n in names if not (RUNS / n / "config.json").exists()]
    for n in unfinished:
        print(f"skipping {n}: no config.json yet (still training, or not a run)")
    names = [n for n in names if n not in unfinished]
    if not names:
        parser.error("no finished run to compare")
    df = compare(names, args.snap_from, args.snap_to)
    groups = [
        ("Training", ["best_epoch", "val_loss"]),
        ("Preseason MAE (ours)", [f"pre_{s}" for s in TARGETS]),
        (
            "Preseason gap to FanGraphs blend (negative = ours better)",
            [f"pre_gap_{s}" for s in TARGETS],
        ),
        (
            f"Mid-season gap to blend, snapshots {args.snap_from} to {args.snap_to}",
            [f"mid_gap_{s}" for s in TARGETS],
        ),
    ]
    for title, cols in groups:
        present = [c for c in cols if c in df.columns]
        if present:
            print(f"\n**{title}**\n")
            table = df[present].astype(float)
            table.columns = [
                c.split("_")[-1] if c not in ("best_epoch", "val_loss") else c for c in present
            ]
            print(to_markdown(table, digits=3 if title == "Training" else 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
