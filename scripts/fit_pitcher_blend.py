"""Fit and score the final blend of ours with FanGraphs, per stat (#499).

Reads a pitcher run's pooled scored frames (scripts/train_pitcher_ros.py) and, per
stat, finds the weight on ours (pitcher_ros/final.py: ``w x ours + (1 - w) x fg_blend``)
with the lowest MSE:

* Preseason: the weights for each season are picked on the other seasons, and the
  blend is scored with those, so no weight is picked on the rows it is scored on.
* Mid-season: the snapshots are split at July 1; each half's weights are picked on the
  other half. One season of snapshots only, so this is the weaker of the two.

Prints, and writes to blend-scores.md in the run's folder: the weights fitted on all
rows next to final.WEIGHTS (the ones in use), each held-out set of weights, and the
blend paired against ours and against FanGraphs (all pitchers and fantasy-relevant).

Usage:
    python scripts/fit_pitcher_blend.py --run 035-saves-net
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.pitcher_ros.evaluate import mse_bootstrap
from fantasy_baseball.pitcher_ros.final import (
    FG,
    MIDSEASON,
    OURS,
    PRESEASON,
    WEIGHTS,
    best_weights,
    blended_scored,
    held_out_weights,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.projection_eval import sure_text

RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
# Mid-season snapshots before this date are the first half.
HALF = "07-01"


def _weights_lines(fitted: pd.Series, held: pd.Series, span: str, group: str) -> list[str]:
    md = [
        "",
        f"Weight on ours ({span}; 1 = all ours, 0 = all FanGraphs):",
        "",
        "| stat | in use | fitted on all | " + " | ".join(f"without {g}" for g in held.index.unique(group)) + " |",
        "|---" * (3 + held.index.unique(group).size) + "|",
    ]  # fmt: skip
    for stat in TARGETS:
        each = " | ".join(f"{held[(g, stat)]:.2f}" for g in held.index.unique(group))
        md.append(f"| {stat} | {WEIGHTS[span][stat]:.2f} | {fitted[stat]:.2f} | {each} |")
    return md


def _paired_lines(blended: pd.DataFrame, title: str) -> list[str]:
    md = ["", f"**{title}** (MSE, lower is better; how sure the blend is better)", ""]
    md += ["| group | stat | blend | ours | how sure | FanGraphs | how sure |", "|---" * 7 + "|"]
    relevant = blended["relevant"].fillna(False).astype(bool)
    for label, sub in (("all", blended), ("fantasy-relevant", blended[relevant])):
        vs_ours = mse_bootstrap(sub, "final", OURS)
        vs_fg = mse_bootstrap(sub, "final", FG)
        for stat in TARGETS:
            o, f = vs_ours.loc[stat], vs_fg.loc[stat]
            md.append(
                f"| {label} | {stat} | {o['a']:.4g} | {o['b']:.4g} | {sure_text(o['sure'])} "
                f"| {f['b']:.4g} | {sure_text(f['sure'])} |"
            )
    return md


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder under data/pitcher_ros/runs/")
    args = parser.parse_args()
    run = RUNS / args.run
    paths = {s: run / f"scored_{s}_pooled.parquet" for s in ("preseason", "snapshots")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        parser.error(f"missing {missing}; train and score the run first")

    md = [f"### Pitcher run `{args.run}`: ours blended with FanGraphs (pitcher_ros/final.py)"]
    pre = pd.read_parquet(paths["preseason"])
    held = held_out_weights(pre, pre["season"])
    md += _weights_lines(best_weights(pre), held, PRESEASON, "season")
    md += _paired_lines(blended_scored(pre, held), "Preseason, weights picked without the season")

    snap = pd.read_parquet(paths["snapshots"])
    snap = snap.assign(
        half=pd.Series("second", index=snap.index).where(snap["snapshot"].str[5:] >= HALF, "first")
    )
    held = held_out_weights(snap, snap["half"])
    md += _weights_lines(best_weights(snap), held, MIDSEASON, "half")
    md += _paired_lines(
        blended_scored(snap, held), f"Mid-season, weights picked on the other half (split {HALF})"
    )
    (run / "blend-scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
