"""Fit and score the draft board's pitcher blend: ours (age-corrected) with the board's
FanGraphs blend, per stat (pitcher_ros/final.py ``BOARD``).

Seasons --seasons, preseason. Rows: every pitcher the board would blend -- the board's
FanGraphs blend projects him for --min-proj-bf+ batters faced and ours projects him --
who then pitched (picked by projection, not by how his season went; MSE is on totals
over actual BF and innings-weighted, so a short season weighs little).

* ours: --run's week-0 predictions (each season predicted by a net trained only on
  earlier seasons), starters' rates age-corrected (pitcher_ros.aging) by a fit on the
  other seasons (role from Steamer's GS / G, as scripts/backtest_pitcher_aging.py).
* board: draft.our_players.board_blend -- the league's systems on file that season at
  the league's weights, exactly as the board's blend is built -- as published, as rates
  (draft.our_pitchers.fg_rates: K, W, SV per its systems' TBF, blended), matched to
  MLBAM ids the way the board matches them (our_players.season_mlbam).

Weights: per stat, picked leaving each season out (blend_fit.held_out_weights) and scored
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

from fantasy_baseball import blend_fit
from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection
from fantasy_baseball.draft.our_pitchers import batters_faced, fg_rates
from fantasy_baseball.draft.our_players import board_blend, match, season_mlbam
from fantasy_baseball.pitcher_ros import aging, final
from fantasy_baseball.pitcher_ros.evaluate import mse_bootstrap
from fantasy_baseball.pt_model import range_inputs as ri

CONFIG = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
STATS = ("k", "w", "sv", "era", "whip")


def scored_rows(seasons: list[int], run: str, min_proj_bf: float) -> pd.DataFrame:
    """Scored-frame rows (pitcher_ros.evaluate's shape): systems ``ours`` (age-corrected),
    ``ours_raw`` and ``fg_blend`` (the board's), per pitcher, season and stat."""
    pitching = ri.load_pitching_seasons(CACHE, seasons)
    preds = ri.load_pitcher_rates(RUNS / run).rename(columns=lambda c: c.removeprefix("p_"))
    preds = preds[list(STATS)]
    birth = births()
    arows = aging_rows(seasons, run, "steamer", birth)
    config = load_config(CONFIG)
    conn = get_connection()
    parts = []
    for season in seasons:
        _, pitchers, systems = board_blend(
            PROJECTIONS / str(season), config.projection_systems, config.projection_weights
        )
        raw = preds.xs(season, level="season")
        usable = batters_faced(pitchers) >= min_proj_bf
        rows, key = match(pitchers, raw.index, season_mlbam(conn, season, "pitcher"), usable)
        fg = fg_rates(pitchers.loc[rows]).set_axis(key)
        pt = ri.load_pitcher_pt(pt_file("steamer", season, "pitchers"))
        fit = aging.fit_rates(arows[arows["season"] != season])
        aged = aging.adjust_rates(
            raw, ri.season_age(birth, season), aging.role(pt["gs_share"]), fit
        )
        a = pitching[season].reindex(key)
        ids = key[((a["bf"] > 0) & (a["ip"] > 0)).to_numpy()]
        a = a.loc[ids]
        with np.errstate(divide="ignore", invalid="ignore"):
            actual = {
                "k": a["k"] / a["bf"],
                "w": a["w"] / a["bf"],
                "sv": a["sv"] / a["bf"],
                "era": 9 * a["er"] / a["ip"],
                "whip": (a["h"] + a["bb"]) / a["ip"],
            }
        for name, frame in ((blend_fit.OURS, aged), ("ours_raw", raw), (blend_fit.FG, fg)):
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
        print(
            f"{season}: {len(ids)} pitchers ({len(key) - len(ids)} more never pitched), "
            f"board blend of {', '.join(systems)}"
        )
    conn.close()
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--run", default="035-saves-net")
    parser.add_argument("--min-proj-bf", type=float, default=100.0)
    args = parser.parse_args()
    scored = scored_rows(sorted(args.seasons), args.run, args.min_proj_bf)
    pair = scored[scored["system"].isin([blend_fit.OURS, blend_fit.FG])]
    held = blend_fit.held_out_weights(pair, pair["season"], final.SCORING)
    print("\nweight on ours, each season's picked on the others:")
    print(held.unstack("stat")[list(STATS)].to_string())
    fitted = blend_fit.best_weights(pair, final.SCORING)
    print("\nfit on every season vs final.WEIGHTS[BOARD]:")
    for stat in STATS:
        print(f"  {stat:4s} {fitted[stat]:.2f}  (in use {final.WEIGHTS[final.BOARD][stat]:.2f})")
    blend = blend_fit.blended_scored(pair, held.rename_axis(["season", "stat"]))
    both = pd.concat([blend, scored[scored["system"] == "ours_raw"]], ignore_index=True)
    for a, b in (
        ("final", blend_fit.FG),
        ("final", blend_fit.OURS),
        (blend_fit.OURS, "ours_raw"),
    ):
        r = mse_bootstrap(both, a, b)
        print(f"\n{a} vs {b} (MSE; sure = how sure {a} is lower):")
        for stat, row in r.iterrows():
            print(f"  {stat:4s} {row['a']:.4g} vs {row['b']:.4g}  ({row['sure'] * 100:.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
