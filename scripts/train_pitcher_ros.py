"""Train the pitcher rest-of-season net and score it against FanGraphs (#465).

For each seed and test season: train on every complete season before it and predict
its rows. ``ours`` is the seeds' predictions averaged (model.average_seeds; one seed
alone is noisy), scored beside FanGraphs and the baselines (preseason and the
mid-season snapshots, pitcher_ros.backtest). Writes predictions.parquet (the average),
predictions-s<N>.parquet (each seed), config.json, info.json, the scored frames and
scores.md to data/pitcher_ros/runs/<name>/.

--saves-net (on by default, model.DEFAULT_SAVES_NET) trains a second net per seed and
test season, reading his history only as the appearance sequence (no per-window rates
of his own, no stuff, probe or fine-tuning inputs; parks as the main net), and takes
ours' saves outputs from it (model.swap_saves). predictions-s<N>.parquet stay the main
net's; predictions-saves-s<N>.parquet and predictions-saves.parquet are the saves net's.
It doubles the training time; --no-saves-net skips it.

--predict-only skips the scoring: for a season with nothing to score yet (next season's
preseason rows, once its schedule is in the store and the table is rebuilt). The run
then has the predictions, config.json, info.json and a short summary.md only.

Needs the training table (scripts/build_pitcher_ros_table.py), torch (see
scripts/train_hitter_ros.py), for the default transformer over each pitcher's recent
appearances (model.DEFAULT_SEQ), the appearance tokens
(scripts/build_pitcher_ros_tokens.py), and for the default park inputs
(model.DEFAULT_PARKS), the park file (scripts/build_pitcher_ros_parks.py --name p3),
and for the default stuff inputs (model.DEFAULT_STUFF), the stuff file
(scripts/build_pitcher_stuff.py --features-file s2-located --run s2-located).
--seq none drops only the sequence encoder (parks and the relative target stay on).
--probes <run> adds the pitcher probe features (#492) from data/pitcher_ros/probes_<run>.parquet
(scripts/build_pitcher_ros_probes.py --run <run>); off by default. Probes from a run
pretrained on one half (--half) need --train-half for the other half.
--finetune <run> joins the pitcher pretraining run's pitch encoder (model T for test
season T) to the net and trains it on the projection loss (pitcher_ros/finetune.py,
#492); needs data/pitcher_ros/pitch_tokens.parquet (build_hitter_ros_pitch_tokens.py
--side pitcher), the token file the run was pretrained on. Off by default.
--train-half 0|1 trains on that half of the pitchers only (pretrain.player_half); every
test row is still predicted. A --finetune run pretrained on one half (--half) must be
fine-tuned on the other: then the encoder has never read the pitchers it trains on.
--finetune-cross <run> adds a run pretrained on the other half: every pitcher trains,
each read by the encoder of the half he is not in (finetune.cross_encoders).

Usage:
    python scripts/train_pitcher_ros.py --name 003-avg          # seeds 0-4, averaged
    python scripts/train_pitcher_ros.py --name quick --seeds 0 --test-seasons 2026
    python scripts/train_pitcher_ros.py --name plain --seq none   # no sequence
    python scripts/train_pitcher_ros.py --name noparks --parks none   # no park inputs
    python scripts/train_pitcher_ros.py --name nopieces --no-era-pieces   # 5 outputs only
    python scripts/train_pitcher_ros.py --name nohz --no-horizons   # no 7- / 30-day outputs
    python scripts/train_pitcher_ros.py --name nostuff --stuff none   # no stuff inputs
    python scripts/train_pitcher_ros.py --name one --no-saves-net   # main net's own saves
    python scripts/train_pitcher_ros.py --name proj-2027 --test-seasons 2027 --predict-only
    python scripts/train_pitcher_ros.py --name allstuff --stuff-rates all   # all 60
    python scripts/train_pitcher_ros.py --name probes --probes q001   # + probe features
    python scripts/train_pitcher_ros.py --name ft --finetune q001 --micro-batch 256
    python scripts/train_pitcher_ros.py --name ftB --finetune q002 --micro-batch 256 --train-half 1
    python scripts/train_pitcher_ros.py --name ftX --finetune q002 --finetune-cross q003 \
        --micro-batch 256
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

from fantasy_baseball.hitter_ros.backtest import league_denominators, unplayed_seasons
from fantasy_baseball.hitter_ros.net import RELATIVE_TARGETS, build_model, device
from fantasy_baseball.hitter_ros.pitch_tokens import recorded_token_options, token_side
from fantasy_baseball.hitter_ros.pretrain import (
    PitchStore,
    player_half,
    run_token_options,
    tokens_fingerprint,
)
from fantasy_baseball.hitter_ros.sequence import SequenceBatcher
from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.features import PER_BF_WEIGHTS, answers, input_frame
from fantasy_baseball.pitcher_ros.finetune import (
    FinetuneBatcher,
    FinetuneConfig,
    PretrainRun,
    check_cross,
    check_length,
    cross_encoders,
    model_maker,
)
from fantasy_baseball.pitcher_ros.horizons import horizon_summary, score_horizons
from fantasy_baseball.pitcher_ros.model import (
    DEFAULT_ERA_PIECES,
    DEFAULT_HORIZONS,
    DEFAULT_PARKS,
    DEFAULT_RELATIVE_TARGET,
    DEFAULT_SAVES_NET,
    DEFAULT_SEEDS,
    DEFAULT_SEQ,
    DEFAULT_STUFF,
    DEFAULT_STUFF_RATES,
    HORIZON_WEIGHTS,
    applied_settings,
    average_seeds,
    check_league_forecast,
    check_week_zero,
    fit_season,
    league_rates,
    net_config,
    swap_saves,
)
from fantasy_baseball.pitcher_ros.parks import load_park_inputs
from fantasy_baseball.pitcher_ros.probes import load_probe_inputs
from fantasy_baseball.pitcher_ros.sequence import TOKEN_FEATURES, TOKEN_ORDER
from fantasy_baseball.pitcher_ros.stats import load_games
from fantasy_baseball.pitcher_ros.stuff_features import STUFF_RATE_SETS, load_stuff_inputs
from fantasy_baseball.pitcher_ros.table import HORIZON_DAYS

TABLE = PROJECT_ROOT / "data" / "pitcher_ros" / "table.parquet"
TOKENS = PROJECT_ROOT / "data" / "pitcher_ros" / "app_tokens.parquet"
PITCH_TOKENS = PROJECT_ROOT / "data" / "pitcher_ros" / "pitch_tokens.parquet"
PRETRAIN = PROJECT_ROOT / "data" / "pitcher_ros" / "pretrain"
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
    parser.add_argument(
        "--probes",
        default="none",
        help="probe features from data/pitcher_ros/probes_<run>.parquet (build them with "
        "scripts/build_pitcher_ros_probes.py --run <run>; none = off)",
    )
    d_ft = FinetuneConfig(run="")
    parser.add_argument(
        "--finetune",
        default="none",
        help="pitcher pretraining run whose encoder joins the net and is fine-tuned "
        "(data/pitcher_ros/pretrain/<run>; none = off)",
    )
    parser.add_argument(
        "--finetune-len",
        type=int,
        default=d_ft.length,
        help="pitches of history each row's encoder reads",
    )
    parser.add_argument(
        "--finetune-lr-scale",
        type=float,
        default=d_ft.lr_scale,
        help="the encoder's learning rate as a share of --lr's",
    )
    parser.add_argument(
        "--finetune-dim", type=int, default=d_ft.out_dim, help="encoder summary width"
    )
    parser.add_argument(
        "--finetune-cross",
        help="a run pretrained on the other half of the pitchers than --finetune's: "
        "cross-fit, every pitcher read by the encoder that never saw him",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=0,
        help="rows per forward pass within a batch (0 = the whole batch); "
        "fine-tuning needs one, e.g. 256",
    )
    parser.add_argument(
        "--horizons",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HORIZONS,
        help="also predict the next 7 and 30 days (table.HORIZON_DAYS) as extra outputs "
        "(model.HORIZON_WEIGHTS; model.DEFAULT_HORIZONS)",
    )
    parser.add_argument(
        "--no-own-windows",
        action="store_true",
        help="leave out his own big-league rates per window (features.input_frame "
        "own_windows): his history only as the appearance sequence; needs --seq and "
        "--stuff none (the stuff inputs are per window too)",
    )
    parser.add_argument(
        "--saves-net",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVES_NET,
        help="take the saves outputs from a second net that reads his history only as "
        "the appearance sequence (model.DEFAULT_SAVES_NET); needs --seq",
    )
    parser.add_argument(
        "--no-role-windows",
        action="store_true",
        help="leave out only his role rates per window (features.ROLE_RATES: saves, "
        "holds, games finished, start share...): his role from the sequence; needs --seq",
    )
    parser.add_argument(
        "--train-half",
        type=int,
        choices=(0, 1),
        help="train on this half of the pitchers only (pretrain.player_half; default all)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace a run with this name")
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="write the predictions without scoring them (a season not played yet)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seasons = sorted(set(args.test_seasons))
    seeds = sorted(set(args.seeds))

    run = RUNS / args.name
    if run.exists() and any(run.iterdir()) and not args.overwrite:
        parser.error(f"{run} already has a run; pick another --name or pass --overwrite")
    if args.no_own_windows and args.seq == "none":
        parser.error("--no-own-windows reads his history from the sequence: it needs --seq")
    if args.no_role_windows and args.seq == "none":
        parser.error("--no-role-windows reads his role from the sequence: it needs --seq")
    if args.saves_net and args.seq == "none":
        parser.error(
            "--saves-net reads his history from the sequence: it needs --seq "
            "(or pass --no-saves-net)"
        )
    if args.no_own_windows and args.stuff != "none":
        # The stuff inputs are his own rates per window too (stuff_features.WINDOWS).
        parser.error("--no-own-windows: the stuff inputs are his own windows too; add --stuff none")
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
            micro_batch=args.micro_batch,
        )
        for seed in seeds
    }
    finetune = pretrain_run = cross_run = None
    if args.finetune_cross is not None and args.finetune == "none":
        parser.error("--finetune-cross needs --finetune")
    if args.finetune != "none":
        if args.seq == "none":
            parser.error("--finetune joins the appearance encoder; it needs --seq")
        if args.micro_batch == 0:
            parser.error("--finetune needs --micro-batch (e.g. 256): a whole batch won't fit")
        try:
            finetune = FinetuneConfig(
                run=args.finetune,
                length=args.finetune_len,
                lr_scale=args.finetune_lr_scale,
                out_dim=args.finetune_dim,
                cross=args.finetune_cross,
            )
            pretrain_run = PretrainRun(PRETRAIN, args.finetune)
            check_length(finetune, pretrain_run)
            if args.finetune_cross is not None:
                cross_run = PretrainRun(PRETRAIN, args.finetune_cross)
                check_cross(pretrain_run, cross_run)
        except ValueError as err:
            parser.error(f"--finetune: {err}")
        encoder_half = pretrain_run.meta.get("half")
        if cross_run is not None:
            if args.train_half is not None:
                parser.error("--finetune-cross trains every pitcher; drop --train-half")
        elif encoder_half is not None and args.train_half != 1 - encoder_half:
            # Training on pitchers the encoder read would bring back the leak the split is for.
            parser.error(
                f"--finetune {args.finetune} was pretrained on half {encoder_half}: "
                f"pass --train-half {1 - encoder_half} (or cross-fit, --finetune-cross)"
            )
        for r in (pretrain_run, cross_run):
            if r is None:
                continue
            missing = [s for s in seasons if s not in r.meta["seasons"]]
            if missing:
                parser.error(f"pretraining run {r.dir.name} has no model for {missing}")
        if not PITCH_TOKENS.exists():
            parser.error(
                f"{PITCH_TOKENS} is missing; run scripts/build_hitter_ros_pitch_tokens.py "
                "--side pitcher"
            )
        # The fingerprint (checked once the file is read) can't tell the sides apart.
        recorded = recorded_token_options(PITCH_TOKENS)
        if token_side(recorded) != "pitcher":
            parser.error(f"{PITCH_TOKENS} holds {token_side(recorded)}-side tokens")
        for r in (pretrain_run, cross_run):
            if r is not None and run_token_options(r.meta) != recorded:
                parser.error(
                    f"{PITCH_TOKENS} was built with {recorded}; {r.dir.name} was pretrained "
                    f"on {run_token_options(r.meta)}"
                )
    table = pd.read_parquet(args.table)
    try:
        check_week_zero(table)
    except ValueError as err:
        parser.error(f"{args.table}: {err}")
    absent = sorted(set(seasons) - set(table["season"].unique()))
    if absent:
        parser.error(
            f"{args.table} has no rows for {absent}; for next season, store its schedule "
            "(fetch_pitch_data.py) and rebuild the table (build_pitcher_ros_table.py)"
        )
    unplayed = unplayed_seasons(table, seasons, count="ros_bf")
    if unplayed and not args.predict_only:
        parser.error(f"no games played yet in {unplayed} to score; add --predict-only")
    train_on = None
    if args.train_half is not None:
        train_on = pd.Series(player_half(table["player_id"]) == args.train_half, index=table.index)
    horizons = HORIZON_DAYS if args.horizons else ()
    absent_answers = [f"ros_d{n}_bf" for n in horizons if f"ros_d{n}_bf" not in table.columns]
    if absent_answers:
        parser.error(
            f"{args.table} predates the short-horizon answers {absent_answers}; rebuild it "
            "with scripts/build_pitcher_ros_table.py"
        )
    league = league_rates(table, horizons) if args.relative_target != "none" else None
    if league is not None:
        # Checked here, before an old run of this name is deleted (fit_season checks too).
        for season in seasons:
            try:
                check_league_forecast(table, league, season)
            except ValueError as err:
                parser.error(f"--test-seasons: {err}")
    x = input_frame(
        table, own_windows=not args.no_own_windows, role_windows=not args.no_role_windows
    )
    x_saves = input_frame(table, own_windows=False) if args.saves_net else None
    parks_build = None
    if args.parks != "none":
        try:
            park_inputs, parks_build = load_park_inputs(table, args.table.parent, args.parks)
        except ValueError as err:
            parser.error(str(err))
        x = pd.concat([x, park_inputs], axis=1)
        if x_saves is not None:
            x_saves = pd.concat([x_saves, park_inputs], axis=1)
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
    probes_build = None
    if args.probes != "none":
        try:
            probe_inputs, probes_build = load_probe_inputs(table, args.table.parent, args.probes)
        except ValueError as err:
            parser.error(str(err))
        probe_half = probes_build.get("half")
        if probe_half is not None and args.train_half != 1 - probe_half:
            # As --finetune: training on pitchers the encoder read brings the leak back.
            parser.error(
                f"probes_{args.probes} came from a run pretrained on half {probe_half}: "
                f"pass --train-half {1 - probe_half}"
            )
        x = pd.concat([x, probe_inputs], axis=1)
        logging.info("added %d probe inputs (%s)", probe_inputs.shape[1], args.probes)
    y, w = answers(table, args.per_bf_weight, era_pieces=args.era_pieces, horizons=horizons)
    games = load_games(STORE)
    if stuff_build is not None:
        # Stuff inputs read from pitches older than the store would leave a row's newest
        # pitches out of its this-season window, with nothing else to flag it.
        through = stuff_build.get("pitches_through")
        if through is None or pd.Timestamp(str(through)) < games["game_date"].max():
            parser.error(
                f"stuff_{args.stuff}.parquet read pitches through {through}, the store has "
                f"games through {games['game_date'].max().date()}; rerun "
                "scripts/build_pitcher_stuff.py --build, then --features-file "
                f"{args.stuff} --run <stuff run>"
            )
    if probes_build is not None:
        # Probes read from tokens older than the store would leave a row's newest pitches
        # out of its history.
        through = probes_build.get("pitches_through")
        if through is None or pd.Timestamp(str(through)) < games["game_date"].max():
            parser.error(
                f"probes_{args.probes}.parquet read pitches through {through}, the store has "
                f"games through {games['game_date'].max().date()}; rerun "
                "scripts/build_hitter_ros_pitch_tokens.py --side pitcher, pretrain, and "
                f"scripts/build_pitcher_ros_probes.py --run {args.probes}"
            )
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
    # The saves net reads the plain appearance sequence, never a fine-tuned encoder.
    saves_batcher = batcher
    pitch_store = None
    if finetune is not None and pretrain_run is not None and batcher is not None:
        pitch_tokens = pd.read_parquet(PITCH_TOKENS)
        if tokens_fingerprint(pitch_tokens) != pretrain_run.meta["tokens"]:
            parser.error(f"{PITCH_TOKENS} is not the token file {args.finetune} was pretrained on")
        # As the appearance tokens: an older file would leave a row's newest pitches out.
        if pd.to_datetime(pitch_tokens["game_date"]).max() < games["game_date"].max():
            parser.error(
                f"{PITCH_TOKENS} is older than {STORE}; rebuild it, pretrain a new run and "
                "fine-tune that"
            )
        # In the precision the run was pretrained in (check_cross: both runs alike).
        pitch_store = PitchStore(pitch_tokens, device(), pretrain_run.dtype)
        del pitch_tokens
        encoder = (
            cross_encoders(table["player_id"], pretrain_run) if cross_run is not None else None
        )
        batcher = FinetuneBatcher(batcher, pitch_store, table, finetune.length, encoder)
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
        "probes_build": probes_build,
        "finetune": finetune.to_dict() if finetune is not None else None,
        "train_half": args.train_half,
        "own_windows": not args.no_own_windows,
        "role_windows": not args.no_role_windows,
        "era_pieces": args.era_pieces,
        "horizons": list(horizons),
        "horizon_weights": HORIZON_WEIGHTS if horizons else None,
        "saves_net": args.saves_net,
    }
    (run / "config.json").write_text(json.dumps(settings, indent=2))

    infos: list[dict] = []

    def train_net(net: str, x_net: pd.DataFrame, batcher_net, fine_tuned: bool) -> pd.DataFrame:
        """Every seed x test season of one net, written per seed; the seeds' average."""
        by_seed = []
        tag = "" if net == "main" else f"-{net}"
        for seed, config in configs.items():
            parts = []
            for season in seasons:
                logging.info("%s net, seed %s, test season %s", net, seed, season)
                make = (
                    model_maker(pretrain_run, season, finetune, device(), cross_run)
                    if fine_tuned and finetune is not None and pretrain_run is not None
                    else build_model
                )
                preds, info = fit_season(
                    table, x_net, y, w, season, config, league, batcher_net, make, train_on
                )
                parts.append(preds)
                infos.append({"net": net, "seed": seed, **info})
            seed_preds = pd.concat(parts, ignore_index=True)
            seed_preds.to_parquet(run / f"predictions{tag}-s{seed}.parquet", index=False)
            by_seed.append(seed_preds)
        return average_seeds(by_seed)

    preds = train_net("main", x, batcher, fine_tuned=True)
    if x_saves is not None:
        saves_preds = train_net("saves", x_saves, saves_batcher, fine_tuned=False)
        saves_preds.to_parquet(run / "predictions-saves.parquet", index=False)
        preds = swap_saves(preds, saves_preds)
    preds.to_parquet(run / "predictions.parquet", index=False)
    (run / "info.json").write_text(json.dumps(infos, indent=2))
    if args.predict_only:
        epochs = ", ".join(
            f"{i['net']} s{i['seed']} {i['test_season']}: {i['best_epoch']}" for i in infos
        )
        md = [
            f"### Pitcher run `{args.name}` (predictions only, not scored)",
            "",
            f"Config: `{json.dumps(settings)}`",
            "",
            f"Best epoch per seed and test season: {epochs}. Inputs: {infos[0]['n_features']}.",
        ]
        (run / "summary.md").write_text("\n".join(md) + "\n")
        print("\n".join(md))
        return 0

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
    horizon_scores = score_horizons(table, preds)
    if horizon_scores is not None:
        horizon_scores.to_parquet(run / "scored_horizons.parquet")
        md += ["", *horizon_summary(horizon_scores)]
    (run / "scores.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 1 if scores.failures else 0


if __name__ == "__main__":
    sys.exit(main())
