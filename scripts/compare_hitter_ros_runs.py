"""Compare hitter ROS runs side by side: training behavior and scores (#404).

For each finished run in data/hitter_ros/runs/<name>/ (one with a summary.md, which
train_hitter_ros.py writes last):

* seed, seasons: what the run was trained and scored on.
* best_epoch, val_loss: the early-stopping epoch and the best validation loss, averaged
  over test seasons. Validation loss compares across runs with the same seed only (the
  seed picks the validation players); the script warns when seeds are mixed.
* Preseason: our MAE per stat (mean over test seasons), and the gap to the FanGraphs
  blend (negative = ours better), over the seasons where both were scored.
* Mid-season: our gap to the blend, averaged over the snapshots where both were scored
  (all of them, or those between --from and --to), with how many snapshots that was.

Usage:
    python scripts/compare_hitter_ros_runs.py 001-baseline-mlp 002a-lr3e-4 002b-lr1e-4
    python scripts/compare_hitter_ros_runs.py --glob "002*" --include 001-baseline-mlp
    python scripts/compare_hitter_ros_runs.py --glob "002*" --from 2026-06-01 --to 2026-07-31
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
from fantasy_baseball.hitter_ros.evaluate import mae_table
from fantasy_baseball.hitter_ros.features import TARGETS

RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
FINISHED_MARKER = "summary.md"


def _both_scored(scored: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Rows of the ``unit`` values (seasons or snapshots) where ours and the blend both
    were scored, so the two are always averaged over the same set."""
    has = scored.groupby(unit)["system"].agg(lambda s: {OURS, BLEND} <= set(s))
    return scored[scored[unit].isin(has.index[has])]


def run_row(run: Path, snap_from: str | None, snap_to: str | None) -> dict[str, object]:
    meta = json.loads((run / "config.json").read_text())
    seasons = meta["seasons"]
    row: dict[str, object] = {
        "seed": meta["config"]["seed"],
        "seasons": ",".join(str(s["test_season"]) for s in seasons),
        "best_epoch": float(np.mean([s["best_epoch"] for s in seasons])),
        "val_loss": float(np.mean([min(s["val_loss"]) for s in seasons])),
    }
    pre_path = run / "scored_preseason.parquet"
    if pre_path.exists():
        pre = pd.read_parquet(pre_path)
        means = mean_over_seasons(pre[pre["system"] == OURS])
        for s in TARGETS:
            row[f"pre_{s}"] = means.loc[OURS, s]
        both = _both_scored(pre, "season")
        if not both.empty:
            paired = mean_over_seasons(both)
            for s in TARGETS:
                row[f"pre_gap_{s}"] = paired.loc[OURS, s] - paired.loc[BLEND, s]
    snap_path = run / "scored_snapshots.parquet"
    if snap_path.exists():
        snap = pd.read_parquet(snap_path)
        if snap_from:
            snap = snap[snap["snapshot"] >= snap_from]
        if snap_to:
            snap = snap[snap["snapshot"] <= snap_to]
        snap = _both_scored(snap, "snapshot") if not snap.empty else snap
        gaps = [
            mae_table(g).loc[OURS, list(TARGETS)] - mae_table(g).loc[BLEND, list(TARGETS)]
            for _, g in snap.groupby("snapshot")
        ]
        row["mid_snapshots"] = len(gaps)
        if gaps:
            mean_gap = pd.concat(gaps, axis=1).mean(axis=1)
            for s in TARGETS:
                row[f"mid_gap_{s}"] = mean_gap[s]
    return row


def compare(names: list[str], snap_from: str | None, snap_to: str | None) -> pd.DataFrame:
    return pd.DataFrame({n: run_row(RUNS / n, snap_from, snap_to) for n in names}).T


def warnings_for(df: pd.DataFrame) -> list[str]:
    out = []
    if df["seed"].nunique() > 1:
        out.append(
            "seeds differ: val_loss is on different validation players, so compare it "
            "only between runs with the same seed"
        )
    if df["seasons"].nunique() > 1:
        out.append("test seasons differ: preseason means average different years")
    if "mid_snapshots" in df.columns and df["mid_snapshots"].nunique() > 1:
        out.append("runs cover different numbers of snapshots in the window")
    return out


# (title, [(column, label)], digits)
def _groups(window: str) -> list[tuple[str, list[tuple[str, str]], int]]:
    return [
        ("Run", [("seed", "seed"), ("seasons", "test seasons")], 0),
        ("Training", [("best_epoch", "best epoch"), ("val_loss", "val loss")], 3),
        ("Preseason MAE (ours)", [(f"pre_{s}", s) for s in TARGETS], 2),
        (
            "Preseason gap to FanGraphs blend (negative = ours better)",
            [(f"pre_gap_{s}", s) for s in TARGETS],
            2,
        ),
        (
            f"Mid-season gap to blend, {window}",
            [("mid_snapshots", "snapshots"), *((f"mid_gap_{s}", s) for s in TARGETS)],
            2,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--glob", help="also include runs whose folder name matches")
    parser.add_argument("--include", nargs="*", default=[], help="extra runs, listed first")
    parser.add_argument("--from", dest="snap_from", help="first snapshot date, YYYY-MM-DD")
    parser.add_argument("--to", dest="snap_to", help="last snapshot date, YYYY-MM-DD")
    args = parser.parse_args()
    names = [*args.include, *args.runs]
    if args.glob:
        names += sorted(p.name for p in RUNS.glob(args.glob) if p.is_dir())
    names = list(dict.fromkeys(names))
    unfinished = [n for n in names if not (RUNS / n / FINISHED_MARKER).exists()]
    for n in unfinished:
        print(f"skipping {n}: no {FINISHED_MARKER} yet (still training or scoring, or not a run)")
    names = [n for n in names if n not in unfinished]
    if not names:
        parser.error("no finished run to compare")

    df = compare(names, args.snap_from, args.snap_to)
    for w in warnings_for(df):
        print(f"WARNING: {w}")
    window = (
        f"snapshots {args.snap_from or 'start'} to {args.snap_to or 'end'}"
        if args.snap_from or args.snap_to
        else "all snapshots"
    )
    for title, cols, digits in _groups(window):
        present = [(c, label) for c, label in cols if c in df.columns]
        if not present:
            continue
        table = df[[c for c, _ in present]].rename(columns=dict(present))
        if digits:
            numeric = [label for c, label in present if c != "mid_snapshots"]
            table[numeric] = table[numeric].astype(float)
        print(f"\n**{title}**\n")
        print(to_markdown(table, digits=digits or 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
