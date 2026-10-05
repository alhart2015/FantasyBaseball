"""Compare hitter ROS runs side by side: training behavior and scores (#404).

For each finished run in data/hitter_ros/runs/<name>/ (one with a summary.md, which
train_hitter_ros.py writes last):

* seed, seasons: what the run was trained and scored on.
* best_epoch, val_loss: the early-stopping epoch and the best validation loss, averaged
  over test seasons. Validation loss compares across runs with the same seed and
  weighting only (the seed picks the validation players, the weighting weights their
  rows); the script warns when either is mixed.
* Main score first (#424): gap-weighted pairwise accuracy in % (higher is better), then
  plain pairwise accuracy, raw MAE and level-free MAE (lower is better). The pairwise
  scores need runs scored after #424 (re-score older ones with score_hitter_ros_run.py).
* Preseason: our score per stat (mean over test seasons), and the gap to the FanGraphs
  blend, over the seasons where both were scored.
* Mid-season: our gap to the blend, averaged over the snapshots where both were scored
  (all of them, or those between --from and --to), with how many snapshots that was.
* Vets and rookies (#433): the main-score gap to the blend for each group on its own
  (pairs only inside a group). Runs scored before the vet/rookie tag lack these rows;
  re-score them with score_hitter_ros_run.py.

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

from fantasy_baseball.hitter_ros.backtest import (
    BLEND,
    GROUPS,
    OURS,
    mean_over_seasons,
    to_markdown,
)
from fantasy_baseball.hitter_ros.evaluate import order_scores, order_table
from fantasy_baseball.hitter_ros.features import TARGETS

RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
FINISHED_MARKER = "summary.md"


def _both_scored(scored: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Rows of the ``unit`` values (seasons or snapshots) where ours and the blend both
    were scored, so the two are always averaged over the same set."""
    has = scored.groupby(unit)["system"].agg(lambda s: {OURS, BLEND} <= set(s))
    return scored[scored[unit].isin(has.index[has])]


def _scores(scored: pd.DataFrame, unit: str) -> dict[str, pd.DataFrame]:
    """Systems x stats, averaged over seasons or snapshots: raw MAE, and (in frames
    scored after #424) level-free MAE and gap-weighted and plain pairwise accuracy."""
    out = {"": mean_over_seasons(scored, unit=unit)}
    if "lf_err" in scored.columns:
        out["lf_"] = mean_over_seasons(scored, "lf_err", unit)
        per_unit = order_scores(scored)
        out["pairw_"] = order_table(scored, "pairwise_w", per_unit)
        out["pair_"] = order_table(scored, "pairwise", per_unit)
    return out


def _group_gaps(row: dict[str, object], both: pd.DataFrame, prefix: str) -> None:
    """Main-score gap to the blend for vets and rookies separately (#433), into ``row``
    as ``{prefix}_{group}_pairw_gap_{stat}``. Nothing for frames without the tag."""
    if "group" not in both.columns or "lf_err" not in both.columns:
        return
    for group in GROUPS:
        sub = both[both["group"] == group]
        if sub.empty:
            continue
        pairw = order_table(sub, "pairwise_w")
        for s in TARGETS:
            row[f"{prefix}_{group}_pairw_gap_{s}"] = pairw.loc[OURS, s] - pairw.loc[BLEND, s]


def run_row(run: Path, snap_from: str | None, snap_to: str | None) -> dict[str, object]:
    meta = json.loads((run / "config.json").read_text())
    seasons = meta["seasons"]
    row: dict[str, object] = {
        "seed": meta["config"]["seed"],
        # Runs from before #424 have no loss setting: they all used MSE.
        "loss": meta["config"].get("loss", "mse"),
        # Runs from before the weighting setting all used PA weighting. A --split run's
        # val_loss averages two models, each over its own rows.
        "weighting": meta["config"].get("weighting", "pa")
        # head_balance was an option in early #422 runs (015c); pre_mid replaced it.
        + ("+head_balance" if meta["config"].get("head_balance") else "")
        + ("+split" if meta["config"].get("split") else "")
        # A horizons run's val_loss averages every horizon's outputs (weighted), not
        # just the rest-of-season ones, so it doesn't compare with other runs' (#419).
        + ("+horizons" if meta["config"].get("horizons") else ""),
        # A --split run lists each season twice (a preseason and a mid-season model).
        "seasons": ",".join(dict.fromkeys(str(s["test_season"]) for s in seasons)),
        "best_epoch": float(np.mean([s["best_epoch"] for s in seasons])),
        "val_loss": float(np.mean([min(s["val_loss"]) for s in seasons])),
    }
    pre_path = run / "scored_preseason.parquet"
    if pre_path.exists():
        pre = pd.read_parquet(pre_path)
        for kind, means in _scores(pre[pre["system"] == OURS], "season").items():
            for s in TARGETS:
                row[f"pre_{kind}{s}"] = means.loc[OURS, s]
        both = _both_scored(pre, "season")
        if not both.empty:
            for kind, paired in _scores(both, "season").items():
                for s in TARGETS:
                    row[f"pre_{kind}gap_{s}"] = paired.loc[OURS, s] - paired.loc[BLEND, s]
            _group_gaps(row, both, "pre")
    snap_path = run / "scored_snapshots.parquet"
    if snap_path.exists():
        snap = pd.read_parquet(snap_path)
        if snap_from:
            snap = snap[snap["snapshot"] >= snap_from]
        if snap_to:
            snap = snap[snap["snapshot"] <= snap_to]
        snap = _both_scored(snap, "snapshot") if not snap.empty else snap
        row["mid_snapshots"] = snap["snapshot"].nunique()
        if not snap.empty:
            for kind, means in _scores(snap, "snapshot").items():
                for s in TARGETS:
                    row[f"mid_{kind}gap_{s}"] = means.loc[OURS, s] - means.loc[BLEND, s]
            _group_gaps(row, snap, "mid")
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
    if df["loss"].nunique() > 1:
        out.append(
            "losses differ: val_loss is a squared error for mse and a pairwise logistic "
            "loss for rank, so it does not compare across them"
        )
    if df["weighting"].nunique() > 1:
        out.append(
            "weightings differ: val_loss weights rows differently (and a split run "
            "averages two models, a horizons run its short-horizon outputs too), so it "
            "does not compare across them"
        )
    if df["seasons"].nunique() > 1:
        out.append("test seasons differ: preseason means average different years")
    if "mid_snapshots" in df.columns and df["mid_snapshots"].nunique() > 1:
        out.append("runs cover different numbers of snapshots in the window")
    return out


# (title, [(column, label)], digits)
def _groups(window: str) -> list[tuple[str, list[tuple[str, str]], int]]:
    def cols(prefix: str) -> list[tuple[str, str]]:
        return [(f"{prefix}{s}", s) for s in TARGETS]

    return [
        (
            "Run",
            [
                ("seed", "seed"),
                ("loss", "loss"),
                ("weighting", "weighting"),
                ("seasons", "test seasons"),
            ],
            0,
        ),
        ("Training", [("best_epoch", "best epoch"), ("val_loss", "val loss")], 3),
        ("MAIN: preseason gap-weighted pairwise % (ours)", cols("pre_pairw_"), 2),
        (
            "MAIN: preseason gap-weighted pairwise gap to blend (positive = ours better)",
            cols("pre_pairw_gap_"),
            2,
        ),
        (
            f"MAIN: mid-season gap-weighted pairwise gap to blend, {window}",
            [("mid_snapshots", "snapshots"), *cols("mid_pairw_gap_")],
            2,
        ),
        *(
            (
                f"MAIN, {group}s only: {when} gap-weighted pairwise gap to blend"
                + (f", {window}" if prefix == "mid" else ""),
                cols(f"{prefix}_{group}_pairw_gap_"),
                2,
            )
            for prefix, when in (("pre", "preseason"), ("mid", "mid-season"))
            for group in GROUPS
        ),
        ("Preseason pairwise % (ours)", cols("pre_pair_"), 2),
        ("Preseason pairwise gap to blend (positive = ours better)", cols("pre_pair_gap_"), 2),
        (f"Mid-season pairwise gap to blend, {window}", cols("mid_pair_gap_"), 2),
        ("Preseason raw MAE (ours)", cols("pre_"), 2),
        ("Preseason raw MAE gap to blend (negative = ours better)", cols("pre_gap_"), 2),
        (
            f"Mid-season raw MAE gap to blend, {window}",
            [("mid_snapshots", "snapshots"), *cols("mid_gap_")],
            2,
        ),
        ("Preseason level-free MAE (ours)", cols("pre_lf_"), 2),
        ("Preseason level-free gap to blend (negative = ours better)", cols("pre_lf_gap_"), 2),
        (f"Mid-season level-free gap to blend, {window}", cols("mid_lf_gap_"), 2),
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
