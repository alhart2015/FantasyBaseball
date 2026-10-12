"""Experiment: score the pitcher net with its SV taken from the saves model (#474).

``ours`` is a pitcher net run's predictions (scripts/train_pitcher_ros.py), and stays
our projection (the net's own SV; see pitcher_ros/saves.py). ``ours_<method>`` is the
same with SV from one of the saves model's projections (scripts/compare_pitcher_saves.py),
as a rate over FanGraphs' projected batters faced (the playing time projections run
on): ``usage`` (our usage share x our team projection; no FanGraphs saves or shares in
it) and ``mix`` (also FanGraphs' share: a blend with FanGraphs, not ours). All are
scored in one pass beside FanGraphs and the baselines (pitcher_ros.backtest), on the
same pitchers and the seasons the saves units cover, plus each swap's SV paired
against ``ours``. A pitcher with no saves units (FanGraphs doesn't project him) keeps
the net's SV. Writes saves-scores-s<seed>.md into the run's folder and prints it.

Needs the run and the saves units for the seed (units-pre-s<seed>.parquet,
units-mid-s<seed>.parquet in data/pitcher_ros/saves/, from compare_pitcher_saves.py).
The seed is the run's (its config.json); another one is refused. Every snapshot the
backtest scores must have saves units, or the swaps would carry stale SV there.

Usage:
    python scripts/score_pitcher_net_saves.py --run 001-plain-s0
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

from fantasy_baseball.hitter_ros.backtest import league_denominators, sure_text
from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.evaluate import mse_bootstrap
from fantasy_baseball.pitcher_ros.saves import net_with_saves
from fantasy_baseball.pitcher_ros.stats import load_games

STORE = PROJECT_ROOT / "data" / "pitch_data"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
LEAGUE = PROJECT_ROOT / "config" / "league.yaml"
RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
SAVES = PROJECT_ROOT / "data" / "pitcher_ros" / "saves"
# The saves units' columns swapped in (compare_pitcher_saves.METHODS).
METHODS = ("usage", "mix")


def _paired_lines(frame: pd.DataFrame | None, title: str) -> list[str]:
    """Each swap's SV MSE minus the net's own, on all scored pitchers and the
    fantasy-relevant ones, with a 95% interval and how sure."""
    if frame is None:
        return []
    md = [
        "",
        f"**{title}: SV swapped in vs the net's own SV** (a minus b; negative = a better)",
        "",
    ]
    for label, sub in (("all", frame), ("fantasy-relevant", frame[frame["relevant"] == True])):  # noqa: E712
        for m in METHODS:
            r = mse_bootstrap(sub, f"ours_{m}", "ours").loc["sv"]
            md.append(
                f"* {label}, ours_{m} - ours: {r['a']:.2f} vs {r['b']:.2f}, "
                f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] {sure_text(r['sure'])}"
            )
    return md


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run folder under data/pitcher_ros/runs/")
    parser.add_argument(
        "--saves-seed", type=int, default=None, help="default: the run's seed (config.json)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = RUNS / args.run
    if not (run / "config.json").exists():
        parser.error(f"{run / 'config.json'} not found; train the run first")
    seed = json.loads((run / "config.json").read_text())["seed"]
    if args.saves_seed is not None and args.saves_seed != seed:
        parser.error(f"--saves-seed {args.saves_seed} is not the run's seed ({seed})")
    paths = {
        "predictions": run / "predictions.parquet",
        "pre": SAVES / f"units-pre-s{seed}.parquet",
        "mid": SAVES / f"units-mid-s{seed}.parquet",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        parser.error(f"missing {missing}; run train_pitcher_ros.py / compare_pitcher_saves.py")
    preds = pd.read_parquet(paths["predictions"])
    pre, mid = pd.read_parquet(paths["pre"]), pd.read_parquet(paths["mid"])
    for name, units in (("pre", pre), ("mid", mid)):
        absent = {"bf_fg", *METHODS} - set(units.columns)
        if absent:
            parser.error(f"{paths[name]} has no {sorted(absent)}; rerun compare_pitcher_saves.py")

    seasons = sorted(set(preds["season"]) & set(pre["season"]))
    preds = preds[preds["season"].isin(seasons)]
    ours = {"ours": preds, **{f"ours_{m}": net_with_saves(preds, pre, mid, m) for m in METHODS}}
    scores = backtest.score_seasons(
        load_games(STORE), seasons, PROJECTIONS, league_denominators(LEAGUE), ours=ours
    )
    if scores.snap is not None:
        uncovered = set(scores.snap["snapshot"]) - set(mid["snapshot"])
        if uncovered:
            parser.error(
                f"snapshots with no saves units: {sorted(uncovered)}; rerun "
                "scripts/compare_pitcher_saves.py"
            )
    md = [
        f"### Pitcher run `{args.run}` with SV from the saves model (seed {seed})",
        "",
        "`ours_<method>` = `ours` with SV from the saves model's `<method>`; every other "
        "stat is the same.",
        *_paired_lines(scores.pre_pooled, "Preseason, all seasons"),
        *_paired_lines(scores.snap_pooled, "All snapshots"),
        "",
        *backtest.summarize(scores),
    ]
    (run / f"saves-scores-s{seed}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if scores.failures else 0


if __name__ == "__main__":
    sys.exit(main())
