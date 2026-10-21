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

Each simulated whole season is also scored in SGP the way the board scores a line (its
denominators and the pool's replacement rates: draft.board_ranges.season_sgp), giving
sgp_p10/p50/p90 and sgp_mean; the board turns them into P90 VAR and risk-adjusted VAR
(board.with_range_var). --backtest also scores sgp_mean as a forecast against the board's
line.

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
from backtest_hitter_ranges import CACHE, PEOPLE, pt_file
from backtest_pitcher_aging import aging_rows, births

from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection, get_positions
from fantasy_baseball.draft import our_hitters, our_pitchers
from fantasy_baseball.draft.board import ScaleInputs, build_board_from_frames
from fantasy_baseball.draft.board_ranges import ranges_path, season_sgp
from fantasy_baseball.draft.our_players import board_blend, load_ours, season_mlbam
from fantasy_baseball.hitter_ros import final as hitter_final
from fantasy_baseball.pitcher_ros import aging
from fantasy_baseball.pitcher_ros import final as pitcher_final
from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.range_report import sure_better
from fantasy_baseball.pt_model.range_specs import HITTER, RELIEVER, STARTER
from fantasy_baseball.pt_model.ranges import actual_totals, backtest, fit, pt_model, quantiles

CONFIG = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
HITTER_RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
PITCHER_RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
LEVELS = (0.1, 0.5, 0.9)
# Scored in --backtest where the projection ranks him at this much volume.
REGULAR = {"hitter": 400.0, "starter": 120.0, "reliever": 45.0}
# A rate stat is scored where the season had this many trials (AB, IP).
RATE_MIN = {"avg": 100.0, "era": 30.0, "whip": 30.0}


def by_mlbam(frame: pd.DataFrame, mlbam: pd.Series) -> pd.DataFrame:
    """A FanGraphs blend indexed by MLBAM id, matched the way the board matches it
    (``mlbam``: ``our_players.season_mlbam``, an id that maps two ways dropped)."""
    if mlbam.empty:
        raise SystemExit(
            "data/fantasy.db has no raw projection ids for a season the ranges read: "
            "rebuild it (python scripts/build_db.py)"
        )
    ids = frame["fg_id"].astype(str).map(mlbam)
    out = frame[ids.notna()].copy()
    out.index = pd.Index(ids[ids.notna()].astype(int).to_numpy(), name="player_id")
    return out[~out.index.duplicated(keep=False)]


def season_inputs(
    season: int, ours_hit: pd.DataFrame | None, ours_pit: pd.DataFrame | None, config, conn
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Per type: (the board blend's playing time, the board's mixed rates as ``p_``
    columns), both by MLBAM id."""
    hitters, pitchers, _ = board_blend(
        PROJECTIONS / str(season), config.projection_systems, config.projection_weights
    )
    hit = by_mlbam(hitters, season_mlbam(conn, season, "hitter"))
    hit = hit[(hit["pa"].astype(float) > 0) & (hit["ab"].astype(float) > 0)]
    hit_pt = hit[["pa", "ab"]].astype(float)
    hit_fg = our_hitters.fg_rates(hit)
    hit_rates = hit_fg if ours_hit is None else hitter_final.board_rates(ours_hit, hit_fg)
    pit = by_mlbam(pitchers, season_mlbam(conn, season, "pitcher"))
    bf = our_pitchers.batters_faced(pit)
    usable = (pit["ip"].astype(float) > 0) & (bf > 0)
    pit, bf = pit[usable], bf[usable]
    games = pit["g"].astype(float)
    pit_pt = pd.DataFrame(
        {
            "ip": pit["ip"].astype(float),
            "bf": bf,
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


def actual_season(rows: pd.DataFrame, spec) -> dict[str, np.ndarray]:
    """``rows``' real seasons in ``ranges.draw``'s shape (one draw each): counting
    totals, rate stats' rates and their real trials, for ``season_sgp``."""
    totals = actual_totals(rows, spec)
    # A rate with no trials is NaN; at 0 it weighs nothing in SGP.
    out = {s.name: np.nan_to_num(totals[s.name])[:, None] for s in spec.stats}
    for s in spec.stats:
        if not s.counting:
            out[f"{s.name}_trials"] = rows[f"a_{s.trials}"].to_numpy(dtype=float)[:, None]
    return out


def line_season(rows: pd.DataFrame, volume: np.ndarray, spec) -> dict[str, np.ndarray]:
    """The board's own line for ``rows`` in ``ranges.draw``'s shape (one draw each): the
    projected rates at the board blend's ``volume`` (PA / IP), for ``season_sgp``."""
    trials = {spec.volume: volume}
    trials |= {t: volume * rows[col].to_numpy(dtype=float) for t, col in spec.trials.items()}
    out = {}
    for st in spec.stats:
        rate = rows[f"p_{st.name}"].to_numpy(dtype=float)
        n = trials[st.trials]
        out[st.name] = (rate * n if st.counting else rate)[:, None]
        if not st.counting:
            out[f"{st.name}_trials"] = n[:, None]
    return out


def board_scale(season: int, ours_hit, ours_pit, config, conn) -> ScaleInputs:
    """The board's scale for ``season``, built as build_draft_board builds it (its
    blend mixed with ``ours``)."""
    hitters, pitchers, _ = board_blend(
        PROJECTIONS / str(season), config.projection_systems, config.projection_weights
    )
    if ours_hit is not None:
        hitters = our_hitters.blend_board_hitters(
            hitters, ours_hit, season_mlbam(conn, season, "hitter")
        )
    if ours_pit is not None:
        pitchers = our_pitchers.blend_board_pitchers(
            pitchers, ours_pit, season_mlbam(conn, season, "pitcher")
        )
    _, scale = build_board_from_frames(
        hitters,
        pitchers,
        get_positions(conn),
        config.roster_slots or None,
        config.num_teams,
        sgp_overrides=config.sgp_overrides,
    )
    return scale


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
        roles = ri.load_pitcher_pt(pt_file("steamer", season, "pitchers"))["gs_share"]
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
    conn = get_connection()
    inputs = {s: season_inputs(s, *ours[s], config, conn) for s in train}
    scales = {s: board_scale(s, *ours[s], config, conn) for s in train} if args.backtest else {}
    if not args.backtest:
        target_ours = (load_ours("hitter", args.season), load_ours("pitcher", args.season))
        target = season_inputs(args.season, *target_ours, config, conn)
        # The board's scale (SGP denominators, the pool's replacement rates, team
        # volumes), so a season's SGP is the board's.
        scale = board_scale(args.season, *target_ours, config, conn)
    conn.close()
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
        print("\nWhole-season SGP (the board's P90 VAR = P90 SGP over replacement):")
        points = []
        for kind, spec in specs:
            rows, volume = fits[spec.name]
            name = spec.name if kind == "pitcher" else "hitter"
            hit = lo_n = n = 0
            for season in train:
                test = rows[rows["season"] == season]
                ranges = fit(rows[rows["season"] != season], volume, pt_model, rng, spec)
                ranked, draws = ranges.draws(test, rng, args.draws)
                sims = season_sgp(draws, kind, scales[season])
                q10, q90 = np.quantile(sims, (0.1, 0.9), axis=1)
                actual = season_sgp(actual_season(test, spec), kind, scales[season])[:, 0]
                pt = inputs[season][kind][0][spec.volume].reindex(test.index).to_numpy(dtype=float)
                line = season_sgp(line_season(test, pt, spec), kind, scales[season])[:, 0]
                keep = ranked[spec.rank_col].to_numpy(dtype=float) >= REGULAR[name]
                hit += int((actual[keep] > q90[keep]).sum())
                lo_n += int((actual[keep] < q10[keep]).sum())
                n += int(keep.sum())
                points.append(
                    pd.DataFrame(
                        {
                            "who": name,
                            "player_id": test.index.to_numpy()[keep],
                            "actual": actual[keep],
                            "line": line[keep],
                            "mean": sims.mean(axis=1)[keep],
                        }
                    )
                )
            print(f"  {name}s: above P90 {hit / n:5.1%}  below P10 {lo_n / n:5.1%}  ({n})")
        # Risk-adjusted VAR = the mean simulated season's SGP over replacement: is that a
        # better forecast of the real season than the board's line (FanGraphs' PT)?
        print("\nOne number per player, vs the real season's SGP (bias = forecast - actual):")
        # Resampling players, each with all his seasons, so one player's five seasons are
        # not five independent players.
        boot = np.random.default_rng(args.seed)
        for name, g in pd.concat(points, ignore_index=True).groupby("who", sort=False):
            miss = {c: g[c] - g["actual"] for c in ("line", "mean")}
            sq = {c: m**2 for c, m in miss.items()}
            sure = sure_better(sq["mean"], sq["line"], g["player_id"], boot)
            print(
                f"  {name}s ({len(g)}): board line MSE {sq['line'].mean():6.2f} "
                f"(bias {miss['line'].mean():+.2f}) | mean season MSE {sq['mean'].mean():6.2f} "
                f"(bias {miss['mean'].mean():+.2f}) | sure mean is better {sure:.0f}"
            )
        return 0

    parts = []
    for kind, spec in specs:
        rows, volume = fits[spec.name]
        ranges = fit(rows, volume, pt_model, rng, spec)
        new = rows_for(spec, args.season, *target[kind], False)
        ranked, draws = ranges.draws(new, rng, args.draws)
        sims = quantiles(draws, ranked, LEVELS, spec)
        seasons = season_sgp(draws, kind, scale)
        sims["sgp"] = np.quantile(seasons, LEVELS, axis=1).T
        parts.append(
            pd.DataFrame(
                {
                    "player_id": ranked.index,
                    "player_type": kind,
                    "role": spec.name,
                    **{
                        f"{c}_p{round(lv * 100)}": sims[c][:, k]
                        for c in (spec.volume, *spec.stat_names, "sgp")
                        for k, lv in enumerate(LEVELS)
                    },
                    # The risk-adjusted value: the average season, injuries and all.
                    "sgp_mean": seasons.mean(axis=1),
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
