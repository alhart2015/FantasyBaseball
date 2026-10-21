"""Fit and score the final blend of ours with FanGraphs, per stat (#499).

Reads a pitcher run's pooled scored frames (scripts/train_pitcher_ros.py) and, per
stat, finds the weight on ours (pitcher_ros/final.py: ``w x ours + (1 - w) x fg_blend``)
with the lowest MSE, for preseason and mid-season (whichever the run has scored).

The blend is checked by holding out a season: each season's weights are picked on the
other seasons and the blend is scored with those, so no weight is picked on the rows
it is scored on. A span with one season only (mid-season, until a second season of
FanGraphs rest-of-season files is stored) can't be checked that way: its weights are
fitted and scored on the same rows, and the report says it is not validated. Splitting
one season's snapshots by date would not help, since every snapshot's actuals run to
the end of that season and so overlap.

Prints, and writes to blend-scores.md in the run's folder: the weights fitted on all
rows next to final.WEIGHTS (the ones in use), each held-out set of weights, and the
blend paired against ours and against FanGraphs (all pitchers and fantasy-relevant).
Warns when the run has fewer seeds than the default, or when its FanGraphs systems
differ from the ones final.WEIGHTS were fitted with (final.SYSTEMS).

Usage:
    python scripts/fit_pitcher_blend.py --run 035-saves-net
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.blend_fit import FG, OURS, best_weights, blended_scored, held_out_weights
from fantasy_baseball.pitcher_ros.backtest import fangraphs_systems
from fantasy_baseball.pitcher_ros.evaluate import mse_bootstrap
from fantasy_baseball.pitcher_ros.final import (
    MIDSEASON,
    PRESEASON,
    SCORING,
    SYSTEMS,
    WEIGHTS,
)
from fantasy_baseball.pitcher_ros.model import DEFAULT_SEEDS
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.projection_eval import sure_text

RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
# Scored frame per span.
FILES = {PRESEASON: "scored_preseason_pooled.parquet", MIDSEASON: "scored_snapshots_pooled.parquet"}
TITLES = {PRESEASON: "Preseason", MIDSEASON: "Mid-season"}


def _weights_lines(fitted: pd.Series, held: pd.Series | None, span: str) -> list[str]:
    seasons = list(held.index.unique("season")) if held is not None else []
    md = [
        "",
        f"Weight on ours ({span}; 1 = all ours, 0 = all FanGraphs):",
        "",
        "| stat | in use | fitted on all |" + "".join(f" without {g} |" for g in seasons),
        "|---" * (3 + len(seasons)) + "|",
    ]  # fmt: skip
    for stat in TARGETS:
        each = "".join(f" {held[(g, stat)]:.2f} |" for g in seasons) if held is not None else ""
        md.append(f"| {stat} | {WEIGHTS[span][stat]:.2f} | {fitted[stat]:.2f} |{each}")
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


def span_lines(scored: pd.DataFrame, span: str) -> list[str]:
    """The weights and paired scores for one span's scored frame."""
    md = ["", f"#### {TITLES[span]}"]
    used = fangraphs_systems(scored, [OURS])
    if set(used) != set(SYSTEMS[span]):
        md += [
            "",
            f"**Warning:** FanGraphs here = {used}, but the {span} weights in use were "
            f"fitted with {list(SYSTEMS[span])} (final.SYSTEMS). Update both together.",
        ]
    fitted = best_weights(scored, SCORING)
    seasons = scored["season"].unique()
    if len(seasons) >= 2:
        held = held_out_weights(scored, scored["season"], SCORING)
        md += _weights_lines(fitted, held, span)
        md += _paired_lines(
            blended_scored(scored, held), f"{TITLES[span]}, weights picked without the season"
        )
    else:
        md += [
            "",
            f"**Not validated:** one season only ({seasons[0]}), so the weights are fitted "
            "and scored on the same rows, and the gains below are optimistic.",
        ]
        md += _weights_lines(fitted, None, span)
        md += _paired_lines(
            blended_scored(scored, fitted), f"{TITLES[span]}, weights fitted on these rows"
        )
    return md


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder under data/pitcher_ros/runs/")
    args = parser.parse_args()
    run = RUNS / args.run
    paths = {span: run / name for span, name in FILES.items() if (run / name).exists()}
    if not paths:
        parser.error(f"no {sorted(FILES.values())} in {run}; train and score the run first")

    md = [f"### Pitcher run `{args.run}`: ours blended with FanGraphs (pitcher_ros/final.py)"]
    seeds = json.loads((run / "config.json").read_text()).get("seeds", [])
    if len(seeds) < len(DEFAULT_SEEDS):
        md += [
            "",
            f"**Warning:** ours here averages {len(seeds)} seed(s) {seeds}, fewer than the "
            f"default {len(DEFAULT_SEEDS)}; don't put these weights in use.",
        ]
    for span in FILES:
        if span in paths:
            md += span_lines(pd.read_parquet(paths[span]), span)
        else:
            md += ["", f"#### {TITLES[span]}", "", f"Skipped: no {FILES[span]} in the run."]
    (run / "blend-scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
