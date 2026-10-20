"""Fit and score the draft board's pitcher blend: ours (age-corrected) with the board's
FanGraphs blend, per stat (pitcher_ros/final.py ``BOARD``).

Seasons --seasons, preseason. Rows: pitchers with 100+ actual batters faced (as the
pitcher backtest scores) that ours and the board's blend both project.

* ours: --run's week-0 predictions (each season predicted by a net trained only on
  earlier seasons), starters' rates age-corrected (pitcher_ros.aging) by a fit on the
  other seasons (role from Steamer's GS / G, as scripts/backtest_pitcher_aging.py).
* board: data.projections.blend_projections of the league's systems on file that season
  at the league's weights -- what the board ranks -- as published, its K, W, SV per
  batter faced (its systems' TBF, blended: draft.our_pitchers.batters_faced).

Weights: per stat, picked leaving each season out (final.held_out_weights) and scored
with those, so no weight is picked on the rows it is scored on; also fit on every
season, next to final.WEIGHTS[BOARD]. Prints the blend's MSE against the board blend and
against ours with how sure (0-100).

    python scripts/fit_board_blend.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
from backtest_hitter_ranges import CACHE, pt_file
from backtest_pitcher_aging import RUNS, aging_rows, births

from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.draft.our_pitchers import batters_faced
from fantasy_baseball.pitcher_ros import aging, final
from fantasy_baseball.pitcher_ros.evaluate import mse_bootstrap
from fantasy_baseball.pt_model import range_inputs as ri

CONFIG = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
STATS = ("k", "w", "sv", "era", "whip")
MIN_BF = 100


def board_fg(season: int) -> tuple[pd.DataFrame, list[str]]:
    """The board's FanGraphs blend for ``season`` as rates by MLBAM id, and its systems."""
    config = load_config(CONFIG)
    folder = PROJECTIONS / str(season)
    systems = [x for x in config.projection_systems if list(folder.glob(f"{x}-pitchers*.csv"))]
    if not systems:
        raise SystemExit(f"no league projection system on file for {season} in {folder}")
    _, fg, _ = blend_projections(
        folder, systems, {x: config.projection_weights.get(x, 1.0) for x in systems}
    )
    fg = fg[fg["mlbam_id"].notna()].drop_duplicates("mlbam_id")
    fg.index = pd.Index(fg["mlbam_id"].astype(int), name="player_id")
    bf = batters_faced(fg)
    rates = pd.DataFrame(
        {
            "k": fg["k"] / bf,
            "w": fg["w"] / bf,
            "sv": fg["sv"] / bf,
            "era": fg["era"],
            "whip": fg["whip"],
        }
    )
    return rates, systems


def scored_rows(seasons: list[int], run: str) -> pd.DataFrame:
    """Scored-frame rows (pitcher_ros.evaluate's shape): systems ``ours`` (age-corrected),
    ``ours_raw`` and ``fg_blend`` (the board's), per pitcher, season and stat."""
    pitching = ri.load_pitching_seasons(CACHE, seasons)
    preds = ri.load_pitcher_rates(RUNS / run).rename(columns=lambda c: c.removeprefix("p_"))
    preds = preds[list(STATS)]
    birth = births()
    arows = aging_rows(seasons, run, "steamer", birth)
    parts = []
    for season in seasons:
        fg, systems = board_fg(season)
        raw = preds.xs(season, level="season")
        pt = ri.load_pitcher_pt(pt_file("steamer", season, "pitchers"))
        fit = aging.fit_rates(arows[arows["season"] != season])
        aged = aging.adjust_rates(
            raw, ri.season_age(birth, season), aging.role(pt["gs_share"]), fit
        )
        a = pitching[season]
        a = a[a["bf"] >= MIN_BF]
        ids = a.index.intersection(raw.index).intersection(fg.index)
        a = a.loc[ids]
        with np.errstate(divide="ignore", invalid="ignore"):
            actual = {
                "k": a["k"] / a["bf"],
                "w": a["w"] / a["bf"],
                "sv": a["sv"] / a["bf"],
                "era": 9 * a["er"] / a["ip"],
                "whip": (a["h"] + a["bb"]) / a["ip"],
            }
        for name, frame in ((final.OURS, aged), ("ours_raw", raw), (final.FG, fg)):
            for stat in STATS:
                parts.append(
                    pd.DataFrame(
                        {
                            "player_id": ids,
                            "season": season,
                            "system": name,
                            "stat": stat,
                            "projected": frame.loc[ids, stat].to_numpy(dtype=float),
                            "actual": actual[stat].to_numpy(dtype=float),
                            "bf": a["bf"].to_numpy(dtype=float),
                            "ip": a["ip"].to_numpy(dtype=float),
                        }
                    )
                )
        print(f"{season}: {len(ids)} pitchers, board blend of {', '.join(systems)}")
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--run", default="035-saves-net")
    args = parser.parse_args()
    scored = scored_rows(sorted(args.seasons), args.run)
    pair = scored[scored["system"].isin([final.OURS, final.FG])]
    held = final.held_out_weights(pair, pair["season"])
    print("\nweight on ours, each season's picked on the others:")
    print(held.unstack("stat")[list(STATS)].to_string())
    fitted = final.best_weights(pair)
    print("\nfit on every season vs final.WEIGHTS[BOARD]:")
    for stat in STATS:
        print(f"  {stat:4s} {fitted[stat]:.2f}  (in use {final.WEIGHTS[final.BOARD][stat]:.2f})")
    blend = final.blended_scored(pair, held.rename_axis(["season", "stat"]))
    both = pd.concat([blend, scored[scored["system"] == "ours_raw"]], ignore_index=True)
    for a, b in (("final", final.FG), ("final", final.OURS), (final.OURS, "ours_raw")):
        r = mse_bootstrap(both, a, b)
        print(f"\n{a} vs {b} (MSE; sure = how sure {a} is lower):")
        for stat, row in r.iterrows():
            print(f"  {stat:4s} {row['a']:.4g} vs {row['b']:.4g}  ({row['sure'] * 100:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
