"""Outcome ranges for the draft board's lines (pt_model.ranges, #505): P10/P50/P90 of
each stat per player, around what the board shows -- the board's mixed rates (ours
with the board's FanGraphs blend: draft.our_hitters / our_pitchers) and the FanGraphs
blend's playing time (his rank in its PA / IP).

Fit on past seasons built the same way: each training season's board blend
(draft.our_players.board_blend, the league's systems on file) mixed with ours held out
(hitters --hitter-run's seeds; pitchers --pitcher-run, starters age-corrected by a fit
leaving the season out), at the weights in use (hitter_ros.final.WEIGHTS,
pitcher_ros.final.WEIGHTS[BOARD]). Then predict --season from its board blend and
ours (data/<type>_ros/ours-<season>.parquet; a type without it gets the FanGraphs
blend's rates, as the board does).

--backtest instead scores the method on the training seasons, each left out: how often
the real season landed above P90 and below P10 (want ~10% each).

Writes data/board_ranges/<season>.parquet (player_id = MLBAM id); the board picks it
up on its next build (python scripts/run_draft_dashboard.py --rebuild-board).

    python scripts/build_board_ranges.py --season 2027
    python scripts/build_board_ranges.py --season 2027 --backtest
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
from backtest_hitter_ranges import CACHE, PEOPLE
from backtest_pitcher_aging import aging_rows, births

from fantasy_baseball.config import load_config
from fantasy_baseball.draft import our_hitters, our_pitchers
from fantasy_baseball.draft.board_ranges import ranges_path
from fantasy_baseball.draft.our_players import board_blend, fg_to_mlbam, load_ours
from fantasy_baseball.hitter_ros import final as hitter_final
from fantasy_baseball.pitcher_ros import aging
from fantasy_baseball.pitcher_ros import final as pitcher_final
from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.range_specs import HITTER, RELIEVER, STARTER
from fantasy_baseball.pt_model.ranges import backtest, fit, pt_model

CONFIG = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
HITTER_RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
PITCHER_RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
LEVELS = (0.1, 0.5, 0.9)
ROLES = (STARTER, RELIEVER)
# Scored in --backtest where the projection ranks him at this much volume.
REGULAR = {"hitter": 400.0, "starter": 120.0, "reliever": 45.0}
# A rate stat is scored where the season had this many trials (AB, IP).
RATE_MIN = {"avg": 100.0, "era": 30.0, "whip": 30.0}


def by_mlbam(frame: pd.DataFrame) -> pd.DataFrame:
    """A FanGraphs blend indexed by MLBAM id (its own fg_id -> MLBAM pairs, an id that
    maps two ways dropped)."""
    ids = frame["fg_id"].astype(str).map(fg_to_mlbam(frame[["fg_id", "mlbam_id"]]))
    out = frame[ids.notna()].copy()
    out.index = pd.Index(ids[ids.notna()].astype(int).to_numpy(), name="player_id")
    return out[~out.index.duplicated(keep=False)]


def season_inputs(
    season: int, ours_hit: pd.DataFrame | None, ours_pit: pd.DataFrame | None, config
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Per type: (the board blend's playing time, the board's mixed rates as ``p_``
    columns), both by MLBAM id."""
    hitters, pitchers, _ = board_blend(
        PROJECTIONS / str(season), config.projection_systems, config.projection_weights
    )
    hit = by_mlbam(hitters)
    hit = hit[(hit["pa"].astype(float) > 0) & (hit["ab"].astype(float) > 0)]
    hit_pt = hit[["pa", "ab"]].astype(float)
    hit_fg = our_hitters.fg_rates(hit)
    hit_rates = hit_fg if ours_hit is None else hitter_final.board_rates(ours_hit, hit_fg)
    pit = by_mlbam(pitchers)
    bf = our_pitchers.batters_faced(pit)
    pit = pit[(pit["ip"].astype(float) > 0) & (bf > 0)]
    games = pit["g"].astype(float)
    pit_pt = pd.DataFrame(
        {
            "ip": pit["ip"].astype(float),
            "bf": our_pitchers.batters_faced(pit),
            "gs_share": pit["gs"].astype(float) / games.where(games > 0),
        }
    )
    pit_fg = our_pitchers.fg_rates(pit)
    pit_rates = pit_fg if ours_pit is None else pitcher_final.board_rates(ours_pit, pit_fg)

    def tagged(rates: pd.DataFrame) -> pd.DataFrame:
        r = rates.rename(columns=lambda c: f"p_{c}")
        r.index = pd.MultiIndex.from_arrays(
            [r.index, np.full(len(r), season)], names=["player_id", "season"]
        )
        return r

    return {"hitter": (hit_pt, tagged(hit_rates)), "pitcher": (pit_pt, tagged(pit_rates))}


def held_out_ours(
    seasons: list[int], hitter_run: str, seeds: list[int], pitcher_run: str
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Per training season: ours held out (hitters' seeds averaged; pitchers' starters
    age-corrected by a fit on the other training seasons), by MLBAM id."""
    hit = ri.load_rates(HITTER_RUNS / f"{hitter_run}-s{s}" for s in seeds)
    pit = ri.load_pitcher_rates(PITCHER_RUNS / pitcher_run)
    birth = births()
    arows = aging_rows(seasons, pitcher_run, "steamer", birth)
    out = {}
    for season in seasons:
        h = hit.xs(season, level="season").rename(columns=lambda c: c.removeprefix("p_"))
        p = pit.xs(season, level="season").rename(columns=lambda c: c.removeprefix("p_"))
        roles = ri.load_pitcher_pt(
            sorted((PROJECTIONS / str(season)).glob("steamer-pitchers*.csv"))[0]
        )["gs_share"]
        fitted = aging.fit_rates(arows[arows["season"] != season])
        p = aging.adjust_rates(p, ri.season_age(birth, season), aging.role(roles), fitted)
        out[season] = (h, p)
    return out


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--train-seasons", type=int, nargs="+")
    parser.add_argument("--hitter-run", default="459-base")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--pitcher-run", default="035-saves-net")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()
    train = sorted(args.train_seasons or range(2022, args.season))
    if args.season in train or len(train) < 3:
        parser.error("--train-seasons: 3+ seasons, not --season")
    config = load_config(CONFIG)

    ours = held_out_ours(train, args.hitter_run, args.seeds, args.pitcher_run)
    inputs = {s: season_inputs(s, *ours[s], config) for s in train}
    years = set(ri.history_seasons([*train, args.season])) | set(train)
    hitting = ri.load_seasons(CACHE, years)
    pitching = ri.load_pitching_seasons(CACHE, years)
    people = pd.read_csv(PEOPLE, usecols=["id", "birthDate", "mlbDebutDate"])
    info = ri.player_info(people)

    def rows_for(spec, season, pt, rates, with_actuals):
        if spec is HITTER:
            return ri.build_rows(season, pt, rates, hitting, info, with_actuals=with_actuals)[0]
        return ri.build_pitcher_rows(
            season, spec, pt, rates, pitching, info, with_actuals=with_actuals
        )[0]

    specs = (("hitter", HITTER), ("pitcher", STARTER), ("pitcher", RELIEVER))
    fits = {}
    for kind, spec in specs:
        rows = pd.concat([rows_for(spec, s, *inputs[s][kind], True) for s in train])
        volume = (
            ri.actual_pa(hitting, train) if spec is HITTER else ri.actual_ip(pitching, train, spec)
        )
        fits[spec.name] = (rows, volume)

    rng = np.random.default_rng(args.seed)
    if args.backtest:
        print(f"Held out by season, {train}. Share of real seasons above P90 / below P10:")
        for kind, spec in specs:
            rows, volume = fits[spec.name]
            scored = backtest(rows, volume, pt_model, rng, spec, (0.1, 0.9), args.draws)
            name = spec.name if kind == "pitcher" else "hitter"
            # Picked by projection (the ranked volume), not by how the season went; a
            # rate stat also needs enough real trials to be a rate.
            regular = scored[scored["ranked"] >= REGULAR[name]]
            regular = regular[regular["trials"] >= regular["stat"].map(RATE_MIN).fillna(0)]
            print(f"\n  {name}s (projected {REGULAR[name]:g}+ {spec.volume}):")
            for stat, g in regular.groupby("stat", sort=False):
                hi = (g["actual"] > g["q90"]).mean()
                lo = (g["actual"] < g["q10"]).mean()
                print(f"    {stat:5s} above P90 {hi:5.1%}  below P10 {lo:5.1%}  ({len(g)})")
        return 0

    ours_hit = load_ours("hitter", args.season)
    ours_pit = load_ours("pitcher", args.season)
    target = season_inputs(
        args.season,
        None if ours_hit is None else ours_hit,
        None if ours_pit is None else ours_pit,
        config,
    )
    parts = []
    for kind, spec in specs:
        rows, volume = fits[spec.name]
        ranges = fit(rows, volume, pt_model, rng, spec)
        new = rows_for(spec, args.season, *target[kind], False)
        ranked, sims = ranges.predict(new, LEVELS, rng, args.draws)
        parts.append(
            pd.DataFrame(
                {
                    "player_id": ranked.index,
                    "player_type": kind,
                    "role": spec.name,
                    **{
                        f"{c}_p{round(lv * 100)}": sims[c][:, k]
                        for c in (spec.volume, *spec.stat_names)
                        for k, lv in enumerate(LEVELS)
                    },
                }
            )
        )
        print(f"{args.season} {spec.name}: {len(ranked)} ranges")
    out = pd.concat(parts, ignore_index=True)
    path = ranges_path(args.season)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
