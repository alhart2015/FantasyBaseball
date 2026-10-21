"""Fit and score the draft board's hitter blend: ours with the board's FanGraphs blend,
per stat (hitter_ros/final.py ``WEIGHTS``, #459).

Seasons --seasons, preseason. Rows: hitters with 300+ actual PA (as the hitter
backtest scores preseason) that ours and the board's blend both project.

* ours: --run's seeds' week-0 predictions averaged (each season predicted by nets
  trained only on earlier seasons). --run must be the held-out twin of the run the
  board uses (scripts/build_our_hitters.py): same config, test seasons --seasons.
* board: data.projections.blend_projections of the league's systems on file that season
  at the league's weights -- what the board ranks -- as published, as rates
  (draft.our_hitters.fg_rates). A hitter on two blend rows (one MLBAM id) is dropped.

Weights: per stat, picked leaving each season out (blend_fit.held_out_weights) and
scored with those, so no weight is picked on the rows it is scored on; also fit on
every season, next to hitter_ros.final.WEIGHTS. Prints the blend's MSE against the
board blend and against ours, with how sure (0-100).

    python scripts/fit_board_hitter_blend.py --run 459-base
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd
from backtest_hitter_ranges import CACHE

from fantasy_baseball import blend_fit
from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.draft.our_hitters import fg_rates
from fantasy_baseball.hitter_ros import final
from fantasy_baseball.hitter_ros.evaluate import mse_bootstrap
from fantasy_baseball.hitter_ros.features import TARGETS, rates_from_counts
from fantasy_baseball.pt_model import range_inputs as ri

CONFIG = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
MIN_PA = 300


def board_fg(season: int) -> tuple[pd.DataFrame, list[str]]:
    """The board's FanGraphs hitter blend for ``season`` as rates by MLBAM id, and its
    systems."""
    config = load_config(CONFIG)
    folder = PROJECTIONS / str(season)
    systems = [x for x in config.projection_systems if list(folder.glob(f"{x}-hitters*.csv"))]
    if not systems:
        raise SystemExit(f"no league projection system on file for {season} in {folder}")
    fg, _, _ = blend_projections(
        folder, systems, {x: config.projection_weights.get(x, 1.0) for x in systems}
    )
    fg = fg[fg["mlbam_id"].notna()]
    fg = fg[~fg["mlbam_id"].duplicated(keep=False)]
    fg = fg[(fg["pa"] > 0) & (fg["ab"] > 0)]
    rates = fg_rates(fg).set_index(pd.Index(fg["mlbam_id"].astype(int), name="player_id"))
    return rates, systems


def scored_rows(seasons: list[int], run: str, seeds: list[int]) -> pd.DataFrame:
    """Scored-frame rows (hitter_ros.evaluate's shape): systems ``ours`` and ``fg_blend``
    (the board's), per hitter, season and stat."""
    actuals = ri.load_seasons(CACHE, seasons)
    rates = ri.load_rates(RUNS / f"{run}-s{s}" for s in seeds)
    rates = rates.rename(columns=lambda c: c.removeprefix("p_"))
    parts = []
    for season in seasons:
        if season not in rates.index.get_level_values("season"):
            raise SystemExit(f"{run} has no {season} preseason rows")
        ours = rates.xs(season, level="season")
        fg, systems = board_fg(season)
        a = actuals[season]
        a = a[a["pa"] >= MIN_PA]
        ids = a.index.intersection(ours.index).intersection(fg.index)
        a = a.loc[ids]
        actual = rates_from_counts(a)
        for name, frame in ((blend_fit.OURS, ours), (blend_fit.FG, fg)):
            for stat in TARGETS:
                parts.append(
                    pd.DataFrame(
                        {
                            "player_id": ids,
                            "season": season,
                            "system": name,
                            "stat": stat,
                            "projected": frame.loc[ids, stat].to_numpy(dtype=float),
                            "actual": actual[stat].to_numpy(dtype=float),
                            "pa": a["pa"].to_numpy(dtype=float),
                            "ab": a["ab"].to_numpy(dtype=float),
                        }
                    )
                )
        print(f"{season}: {len(ids)} hitters, board blend of {', '.join(systems)}")
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--run", default="459-base", help="seeds at <run>-s<N>")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    args = parser.parse_args()
    scored = scored_rows(sorted(args.seasons), args.run, args.seeds)
    held = blend_fit.held_out_weights(scored, scored["season"], final.SCORING)
    print("\nweight on ours, each season's picked on the others:")
    print(held.unstack("stat")[list(TARGETS)].to_string())
    fitted = blend_fit.best_weights(scored, final.SCORING)
    print("\nfit on every season vs hitter_ros.final.WEIGHTS:")
    for stat in TARGETS:
        print(f"  {stat:4s} {fitted[stat]:.2f}  (in use {final.WEIGHTS[stat]:.2f})")
    blend = blend_fit.blended_scored(scored, held.rename_axis(["season", "stat"]))
    for a, b in (("final", blend_fit.FG), ("final", blend_fit.OURS)):
        r = mse_bootstrap(blend, a, b)
        print(f"\n{a} vs {b} (MSE; sure = how sure {a} is lower):")
        for stat, row in r.iterrows():
            print(f"  {stat:4s} {row['a']:.4g} vs {row['b']:.4g}  ({row['sure'] * 100:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
