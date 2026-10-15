"""Train the pitcher rest-of-season net and score it against FanGraphs (#465).

For each seed and test season: train on every complete season before it and predict
its rows. ``ours`` is the seeds' predictions averaged (model.average_seeds; one seed
alone is noisy), scored beside FanGraphs and the baselines (preseason and the
mid-season snapshots, pitcher_ros.backtest). Writes predictions.parquet (the average),
predictions-s<N>.parquet (each seed), config.json, info.json, the scored frames and
scores.md to data/pitcher_ros/runs/<name>/.

Needs the training table (scripts/build_pitcher_ros_table.py), torch (see
scripts/train_hitter_ros.py), for the default transformer over each pitcher's recent
appearances (model.DEFAULT_SEQ), the appearance tokens
(scripts/build_pitcher_ros_tokens.py), and for the default park inputs
(model.DEFAULT_PARKS), the park file (scripts/build_pitcher_ros_parks.py --name p3),
and for the default stuff inputs (model.DEFAULT_STUFF), the stuff file
(scripts/build_pitcher_stuff.py --features-file s2-located --run s2-located).
--seq none drops only the sequence encoder (parks and the relative target stay on).

Usage:
    python scripts/train_pitcher_ros.py --name 003-avg          # seeds 0-4, averaged
    python scripts/train_pitcher_ros.py --name quick --seeds 0 --test-seasons 2026
    python scripts/train_pitcher_ros.py --name plain --seq none   # no sequence
    python scripts/train_pitcher_ros.py --name noparks --parks none   # no park inputs
    python scripts/train_pitcher_ros.py --name nopieces --no-era-pieces   # 5 outputs only
    python scripts/train_pitcher_ros.py --name nostuff --stuff none   # no stuff inputs
    python scripts/train_pitcher_ros.py --name allstuff --stuff-rates all   # all 60
    python scripts/train_pitcher_ros.py --name 001-plain --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import league_denominators
from fantasy_baseball.hitter_ros.net import RELATIVE_TARGETS, device
from fantasy_baseball.hitter_ros.sequence import SequenceBatcher
from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.features import PER_BF_WEIGHTS, answers, input_frame
from fantasy_baseball.pitcher_ros.model import (
    DEFAULT_ERA_PIECES,
    DEFAULT_PARKS,
    DEFAULT_RELATIVE_TARGET,
    DEFAULT_SEEDS,
    DEFAULT_SEQ,
    DEFAULT_STUFF,
    DEFAULT_STUFF_RATES,
    applied_settings,
    average_seeds,
    check_league_forecast,
    check_week_zero,
    fit_season,
    league_rates,
    net_config,
)
from fantasy_baseball.pitcher_ros.parks import load_park_inputs
from fantasy_baseball.pitcher_ros.sequence import TOKEN_FEATURES, TOKEN_ORDER
from fantasy_baseball.pitcher_ros.stats import load_games
from fantasy_baseball.pitcher_ros.stuff_features import STUFF_RATE_SETS, load_stuff_inputs

TABLE = PROJECT_ROOT / "data" / "pitcher_ros" / "table.parquet"
TOKENS = PROJECT_ROOT / "data" / "pitcher_ros" / "app_tokens.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
LEAGUE = PROJECT_ROOT / "config" / "league.yaml"
RUNS = PROJECT_ROOT / "data" / "pitcher_ros" / "runs"
# The seasons with FanGraphs pitcher projections on disk.
DEFAULT_TEST_SEASONS = (2022, 2023, 2024, 2025, 2026)
OURS = "ours"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="run folder under data/pitcher_ros/runs/")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="seeds to train; ours is their average",
    )
    parser.add_argument("--test-seasons", type=int, nargs="+", default=list(DEFAULT_TEST_SEASONS))
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument(
        "--tokens",
        type=Path,
        default=TOKENS,
        help="appearance tokens (scripts/build_pitcher_ros_tokens.py), with --seq",
    )
    parser.add_argument(
        "--per-bf-weight",
        choices=PER_BF_WEIGHTS,
        default="bf",
        help="K/W/SV loss weight: batters faced, or its square (what the scorer charges)",
    )
    parser.add_argument(
        "--relative-target",
        choices=RELATIVE_TARGETS,
        default=DEFAULT_RELATIVE_TARGET,
        help="learn rates relative to the league (model.fit_season, DEFAULT_RELATIVE_TARGET)",
    )
    parser.add_argument(
        "--seq",
        choices=["none", "gru", "transformer"],
        default=DEFAULT_SEQ["seq"],
        help="encoder over each pitcher's recent appearances (model.DEFAULT_SEQ); "
        "none: the plain net",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ["seq_len"],
        help="appearances each row reads, newest first",
    )
    parser.add_argument("--seq-dim", type=int, default=DEFAULT_SEQ["seq_dim"])
    parser.add_argument("--seq-layers", type=int, default=DEFAULT_SEQ["seq_layers"])
    parser.add_argument(
        "--parks",
        default=DEFAULT_PARKS,
        help="park inputs from data/pitcher_ros/parks_<name>.parquet (build them with "
        "scripts/build_pitcher_ros_parks.py --name <name>; none = off; model.DEFAULT_PARKS)",
    )
    parser.add_argument(
        "--era-pieces",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ERA_PIECES,
        help="also predict ERA's pieces (BB, HBP, HR per batter faced) as extra outputs "
        "(model.DEFAULT_ERA_PIECES)",
    )
    parser.add_argument(
        "--stuff",
        default=DEFAULT_STUFF,
        help="stuff inputs from data/pitcher_ros/stuff_<name>.parquet (build them with "
        "scripts/build_pitcher_stuff.py --features-file <name>; none = off; "
        "model.DEFAULT_STUFF)",
    )
    parser.add_argument(
        "--stuff-rates",
        choices=sorted(STUFF_RATE_SETS),
        default=DEFAULT_STUFF_RATES,
        help="which stuff rates to read (stuff_features.STUFF_RATE_SETS; "
        "model.DEFAULT_STUFF_RATES)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace a run with this name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seasons = sorted(set(args.test_seasons))
    seeds = sorted(set(args.seeds))

    run = RUNS / args.name
    if run.exists() and any(run.iterdir()) and not args.overwrite:
        parser.error(f"{run} already has a run; pick another --name or pass --overwrite")
    if args.seq != "none" and not args.tokens.exists():
        parser.error(f"{args.tokens} is missing; run scripts/build_pitcher_ros_tokens.py")
    configs = {
        seed: net_config(
            seed=seed,
            relative_target=args.relative_target,
            seq=args.seq,
            seq_len=args.seq_len,
            seq_dim=args.seq_dim,
            seq_layers=args.seq_layers,
            parks=args.parks,
        )
        for seed in seeds
    }
    table = pd.read_parquet(args.table)
    try:
        check_week_zero(table)
    except ValueError as err:
        parser.error(f"{args.table}: {err}")
    league = league_rates(table) if args.relative_target != "none" else None
    if league is not None:
        # Checked here, before an old run of this name is deleted (fit_season checks too).
        for season in seasons:
            try:
                check_league_forecast(table, league, season)
            except ValueError as err:
                parser.error(f"--test-seasons: {err}")
    x = input_frame(table)
    parks_build = None
    if args.parks != "none":
        try:
            park_inputs, parks_build = load_park_inputs(table, args.table.parent, args.parks)
        except ValueError as err:
            parser.error(str(err))
        x = pd.concat([x, park_inputs], axis=1)
        logging.info("added %d park inputs (%s)", park_inputs.shape[1], args.parks)
    stuff_build = None
    if args.stuff != "none":
        try:
            stuff_inputs, stuff_build = load_stuff_inputs(
                table, args.table.parent, args.stuff, args.stuff_rates
            )
        except ValueError as err:
            parser.error(str(err))
        x = pd.concat([x, stuff_inputs], axis=1)
        logging.info("added %d stuff inputs (%s)", stuff_inputs.shape[1], args.stuff)
    y, w = answers(table, args.per_bf_weight, era_pieces=args.era_pieces)
    games = load_games(STORE)
    batcher = None
    if args.seq != "none":
        tokens = pd.read_parquet(args.tokens)
        # Tokens built before the store gained games would silently leave a row's most
        # recent appearances out of its sequence.
        stale = pd.to_datetime(tokens["game_date"]).max() < games["game_date"].max()
        if stale:
            parser.error(
                f"{args.tokens} is older than {STORE}; rerun scripts/build_pitcher_ros_tokens.py"
            )
        batcher = SequenceBatcher(
            tokens,
            table,
            args.seq_len,
            device(),
            features=TOKEN_FEATURES,
            order=TOKEN_ORDER,
        )
    # Everything has loaded: only now clear an old run, so a failed start leaves it.
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    settings = {
        **{k: v for k, v in applied_settings(configs[seeds[0]]).items() if k != "seed"},
        "seeds": seeds,
        "per_bf_weight": args.per_bf_weight,
        "parks_build": parks_build,
        "stuff_build": stuff_build,
        "stuff_rates": args.stuff_rates if args.stuff != "none" else None,
        "era_pieces": args.era_pieces,
    }
    (run / "config.json").write_text(json.dumps(settings, indent=2))

    by_seed, infos = [], []
    for seed, config in configs.items():
        parts = []
        for season in seasons:
            logging.info("seed %s, test season %s", seed, season)
            preds, info = fit_season(table, x, y, w, season, config, league, batcher)
            parts.append(preds)
            infos.append({"seed": seed, **info})
        seed_preds = pd.concat(parts, ignore_index=True)
        seed_preds.to_parquet(run / f"predictions-s{seed}.parquet", index=False)
        by_seed.append(seed_preds)
    preds = average_seeds(by_seed)
    preds.to_parquet(run / "predictions.parquet", index=False)
    (run / "info.json").write_text(json.dumps(infos, indent=2))

    scores = backtest.score_seasons(
        games,
        seasons,
        PROJECTIONS,
        league_denominators(LEAGUE),
        ours={OURS: preds},
    )
    for frame, name in (
        (scores.pre, "scored_preseason"),
        (scores.snap, "scored_snapshots"),
        (scores.pre_pooled, "scored_preseason_pooled"),
        (scores.snap_pooled, "scored_snapshots_pooled"),
    ):
        if frame is not None:
            frame.to_parquet(run / f"{name}.parquet")
    md = [
        f"### Scores for pitcher run `{args.name}` "
        + (f"(seeds {seeds}, averaged)" if len(seeds) > 1 else f"(one seed, {seeds[0]})"),
        "",
        *backtest.summarize(scores),
    ]
    (run / "scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if scores.failures else 0


if __name__ == "__main__":
    sys.exit(main())
