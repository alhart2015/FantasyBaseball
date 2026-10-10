"""Compare hitter ROS runs side by side: training behavior and scores (#404).

For each finished run in data/hitter_ros/runs/<name>/ (one with a summary.md, which
train_hitter_ros.py writes last):

* seed, seasons: what the run was trained and scored on.
* SB inputs (#451): "box" when SB came from a second net on box-score inputs; then the
  training columns add that net's best epoch and validation loss.
* best_epoch, val_loss: the early-stopping epoch and the best validation loss, averaged
  over test seasons. Validation loss compares across runs with the same seed and
  weighting only (the seed picks the validation players, the weighting weights their
  rows); the script warns when either is mixed.
* Main score first: MSE (``evaluate``; lower is better), pooled over every scored
  hitter-season or hitter-snapshot. Then gap-weighted and plain pairwise accuracy in %
  (higher is better; the main score before #433), raw MAE and level-free MAE (lower is
  better). The pairwise scores need runs scored after #424 (re-score older ones with
  score_hitter_ros_run.py).
* Preseason: our score per stat (mean over test seasons), and the gap to the FanGraphs
  blend, over the seasons where both were scored.
* Mid-season: our gap to the blend, averaged over the snapshots where both were scored
  (all of them, or those between --from and --to), with how many snapshots that was.
* How sure the MSE and pairwise gaps are (evaluate.sure, from resampling the scored hitters,
  each redrawn once across every season or snapshot): +0.95 = 95% sure ours is better
  than the blend, -0.90 = 90% sure it is worse, 0 = no lean; +/-0.95 or beyond is real.
  Only the luck of which hitters were scored, not seed swings.
* Vets and rookies (#433): the MSE and pairwise gaps to the blend for each group on its
  own (pairs only inside a group). Runs scored before the vet/rookie tag lack these rows;
  re-score them with score_hitter_ros_run.py.
* Fantasy-relevant hitters (#442): the MSE and pairwise gaps to the blend over only the top
  ``RELEVANT_TOP`` hitters by FanGraphs' projected or by actual fantasy value (pairs
  only among them). Runs scored before the tag lack these rows; re-score them.

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
    RELEVANT_TOP,
    mean_over_seasons,
    to_markdown,
)
from fantasy_baseball.hitter_ros.evaluate import (
    mse_bootstrap,
    mse_table,
    order_scores,
    order_table,
    pairwise_bootstrap,
)
from fantasy_baseball.hitter_ros.features import TARGETS

RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
FINISHED_MARKER = "summary.md"


def _both_scored(scored: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Rows of the ``unit`` values (seasons or snapshots) where ours and the blend both
    were scored, so the two are always averaged over the same set."""
    has = scored.groupby(unit)["system"].agg(lambda s: {OURS, BLEND} <= set(s))
    return scored[scored[unit].isin(has.index[has])]


def _scores(scored: pd.DataFrame, unit: str) -> dict[str, pd.DataFrame]:
    """Systems x stats: MSE pooled over every scored player-unit, and averaged over
    seasons or snapshots: raw MAE, and (in frames scored after #424) level-free MAE and
    gap-weighted and plain pairwise accuracy."""
    out = {"": mean_over_seasons(scored, unit=unit), "mse_": mse_table(scored)}
    if "lf_err" in scored.columns:
        out["lf_"] = mean_over_seasons(scored, "lf_err", unit)
        per_unit = order_scores(scored)
        out["pairw_"] = order_table(scored, "pairwise_w", per_unit)
        out["pair_"] = order_table(scored, "pairwise", per_unit)
    return out


def _luck(row: dict[str, object], both: pd.DataFrame, prefix: str) -> None:
    """How sure ours is better (+) or worse (-) than the blend on MSE and on gap-weighted
    pairwise, into ``row`` as ``{prefix}_mse_sure_{stat}`` and ``{prefix}_pairw_sure_{stat}``.
    No pairwise for frames scored before #424."""
    _mse_gap(row, both, prefix, gap=False)
    if "lf_err" not in both.columns:
        return
    b = pairwise_bootstrap(both, OURS, BLEND)
    for s in TARGETS:
        row[f"{prefix}_pairw_sure_{s}"] = b.loc[s, "sure"]


def _mse_gap(row: dict[str, object], sub: pd.DataFrame, prefix: str, *, gap: bool = True) -> None:
    """Ours-minus-blend MSE over ``sub`` (negative = ours better) and how sure, into
    ``row`` as ``{prefix}_mse_gap_{stat}`` (when ``gap``) and ``{prefix}_mse_sure_{stat}``."""
    m = mse_bootstrap(sub, OURS, BLEND)
    for s in TARGETS:
        if gap:
            row[f"{prefix}_mse_gap_{s}"] = m.loc[s, "diff"]
        row[f"{prefix}_mse_sure_{s}"] = m.loc[s, "sure"]


def _group_gaps(row: dict[str, object], both: pd.DataFrame, prefix: str) -> None:
    """MSE gap (with how sure) and pairwise gap to the blend for vets and rookies
    separately (#433), into ``row`` as ``{prefix}_{group}_mse_gap_{stat}``,
    ``{prefix}_{group}_mse_sure_{stat}`` and ``{prefix}_{group}_pairw_gap_{stat}``.
    Nothing for frames without the tag."""
    if "group" not in both.columns:
        return
    for group in GROUPS:
        sub = both[both["group"] == group]
        if sub.empty:
            continue
        _mse_gap(row, sub, f"{prefix}_{group}")
        pairw = order_table(sub, "pairwise_w")
        for s in TARGETS:
            row[f"{prefix}_{group}_pairw_gap_{s}"] = pairw.loc[OURS, s] - pairw.loc[BLEND, s]


def _relevant_gaps(row: dict[str, object], both: pd.DataFrame, prefix: str) -> None:
    """MSE gap (with how sure) and pairwise gap to the blend over the fantasy-relevant
    hitters only (#442), into ``row`` as ``{prefix}_top_mse_gap_{stat}``,
    ``{prefix}_top_mse_sure_{stat}`` and ``{prefix}_top_pairw_gap_{stat}``. Nothing for
    frames without the tag."""
    if "relevant" not in both.columns:
        return
    sub = both[both["relevant"].fillna(False).astype(bool)]
    if sub.empty:
        return
    _mse_gap(row, sub, f"{prefix}_top")
    pairw = order_table(sub, "pairwise_w")
    for s in TARGETS:
        row[f"{prefix}_top_pairw_gap_{s}"] = pairw.loc[OURS, s] - pairw.loc[BLEND, s]


def run_row(run: Path, snap_from: str | None, snap_to: str | None) -> dict[str, object]:
    meta = json.loads((run / "config.json").read_text())
    seasons = meta["seasons"]
    row: dict[str, object] = {
        "seed": meta["config"]["seed"],
        # Runs from before #424 have no loss setting: they all used MSE.
        "loss": meta["config"].get("loss", "mse")
        # A binomial AVG loss (#433) is a deviance, not a squared error, in val_loss.
        + ("+avg_binomial" if meta["config"].get("avg_loss", "mse") == "binomial" else "")
        # AVG's pieces (#433) add binomial-deviance columns to val_loss.
        + (
            f"+avg_pieces_{meta['config']['avg_pieces']}"
            if meta["config"].get("avg_pieces", "none") != "none"
            else ""
        )
        # SB's pieces (#413) add Poisson and binomial deviances to val_loss.
        + (
            f"+sb_pieces_{meta['config']['sb_pieces']}"
            if meta["config"].get("sb_pieces", "none") != "none"
            else ""
        ),
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
        # Where SB comes from (#451): the main net, or a second net on box-score inputs.
        # Runs from before the setting all used the main net.
        "sb_inputs": meta["config"].get("sb_inputs", "full"),
    }
    if all("sb_net" in s for s in seasons):  # the SB net's own training, when there is one
        row["sb_best_epoch"] = float(np.mean([s["sb_net"]["best_epoch"] for s in seasons]))
        row["sb_val_loss"] = float(np.mean([min(s["sb_net"]["val_loss"]) for s in seasons]))
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
            _luck(row, both, "pre")
            _group_gaps(row, both, "pre")
            _relevant_gaps(row, both, "pre")
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
            _luck(row, snap, "mid")
            _group_gaps(row, snap, "mid")
            _relevant_gaps(row, snap, "mid")
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
            "losses differ: val_loss is a squared error for mse, a pairwise logistic "
            "loss for rank and a binomial deviance for AVG with +avg_binomial (and for "
            "AVG's pieces with +avg_pieces; +sb_pieces adds Poisson and binomial "
            "deviances for SB's pieces), so it does not compare across them"
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
                ("sb_inputs", "SB inputs"),
                ("seasons", "test seasons"),
            ],
            0,
        ),
        (
            "Training",
            [
                ("best_epoch", "best epoch"),
                ("val_loss", "val loss"),
                ("sb_best_epoch", "SB net best epoch"),
                ("sb_val_loss", "SB net val loss"),
            ],
            3,
        ),
        ("MAIN: preseason MSE (ours)", cols("pre_mse_"), 2),
        (
            "MAIN: preseason MSE gap to blend (negative = ours better)",
            cols("pre_mse_gap_"),
            2,
        ),
        (
            "MAIN: preseason MSE, how sure ours is better (+) or worse (-) than blend "
            "(0.95+ = real)",
            cols("pre_mse_sure_"),
            2,
        ),
        (
            f"MAIN: mid-season MSE gap to blend (negative = ours better), {window}",
            [("mid_snapshots", "snapshots"), *cols("mid_mse_gap_")],
            2,
        ),
        (
            "MAIN: mid-season MSE, how sure ours is better (+) or worse (-) than blend "
            f"(0.95+ = real), {window}",
            cols("mid_mse_sure_"),
            2,
        ),
        *(
            (
                f"MAIN, {who}: {when} MSE {what}" + (f", {window}" if prefix == "mid" else ""),
                cols(f"{prefix}_{key}_mse_{kind}_"),
                2,
            )
            for key, who in (
                ("top", f"top {RELEVANT_TOP} fantasy hitters only"),
                *((g, f"{g}s only") for g in GROUPS),
            )
            for prefix, when in (("pre", "preseason"), ("mid", "mid-season"))
            for kind, what in (
                ("gap", "gap to blend (negative = ours better)"),
                ("sure", "how sure ours is better (+) or worse (-) than blend (0.95+ = real)"),
            )
        ),
        ("Preseason gap-weighted pairwise % (ours)", cols("pre_pairw_"), 2),
        (
            "Preseason gap-weighted pairwise gap to blend (positive = ours better)",
            cols("pre_pairw_gap_"),
            2,
        ),
        (
            "Preseason gap-weighted pairwise, how sure ours is better (+) or worse (-) "
            "than blend (0.95+ = real)",
            cols("pre_pairw_sure_"),
            2,
        ),
        (
            f"Mid-season gap-weighted pairwise gap to blend, {window}",
            [("mid_snapshots", "snapshots"), *cols("mid_pairw_gap_")],
            2,
        ),
        (
            "Mid-season gap-weighted pairwise, how sure ours is better (+) or worse (-) "
            f"than blend (0.95+ = real), {window}",
            cols("mid_pairw_sure_"),
            2,
        ),
        *(
            (
                f"Top {RELEVANT_TOP} fantasy hitters only: {when} gap-weighted "
                "pairwise gap to blend" + (f", {window}" if prefix == "mid" else ""),
                cols(f"{prefix}_top_pairw_gap_"),
                2,
            )
            for prefix, when in (("pre", "preseason"), ("mid", "mid-season"))
        ),
        *(
            (
                f"{group.capitalize()}s only: {when} gap-weighted pairwise gap to blend"
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
