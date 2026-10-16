"""Pretrain the pitch models for the hitter ROS net: one per season, walk-forward (#415).

For each season S: pretrain on every pitch from seasons before S (predict the next
pitch's outcome). Anything read off model S about a row from season S is then out of
sample for that row -- training rows as well as test rows. Saves under
data/hitter_ros/pretrain/<name>/<S>/:

    model.pt       PretrainModel weights (encoder + prediction head)
    metrics.json   config, train/val CE per epoch, count-only baseline CE, minutes

and <name>/run.json listing the seasons and config. With --side pitcher, the sequences
are the pitches each pitcher threw (#492): tokens and runs under data/pitcher_ros/. With --overwrite the whole <name>
folder is replaced, but only after the inputs are found; each season is written to a
temporary folder and moved into place when it finishes, so a crash never leaves a
half-written season behind. With --add, --seasons are added to an existing run with its
own settings and strike zone, and only if the token file is exactly the one it was
pretrained on (the same fingerprint: e.g. next season's model for a run pretrained after
the last season was complete). A run pretrained on an older token file can't be
extended: its seasons' models would have read different pitches. Pretrain a new run.
With --half 0|1, only that half of the players' pitches are read (``pretrain.player_half``;
recorded in run.json): a net fine-tuned on the other half's rows never sees players the
encoder memorized (#492). --add keeps the run's half.

Setup (once): python scripts/build_hitter_ros_pitch_tokens.py [--side pitcher]
Usage:
    python scripts/pretrain_hitter_ros.py --name p002
    python scripts/pretrain_hitter_ros.py --name smoke --seasons 2026 --max-epochs 1
    python scripts/pretrain_hitter_ros.py --name p005 --zone fixed
    python scripts/pretrain_hitter_ros.py --name p004 --add --seasons 2027
    python scripts/pretrain_hitter_ros.py --side pitcher --name q001 --seasons 2016 2026
    python scripts/pretrain_hitter_ros.py --side pitcher --name q002 --half 0 --seasons 2022 2026
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.features import ZONES
from fantasy_baseball.hitter_ros.net import device
from fantasy_baseball.hitter_ros.pitch_tokens import (
    SIDES,
    recorded_token_options,
    token_path,
    token_side,
)
from fantasy_baseball.hitter_ros.pretrain import (
    PitchStore,
    PretrainConfig,
    player_half,
    pretrain,
    run_token_options,
    tokens_fingerprint,
)

TOKEN_DIR = PROJECT_ROOT / "data" / "hitter_ros"
PRETRAIN = PROJECT_ROOT / "data" / "hitter_ros" / "pretrain"
# --side pitcher (#492): its token file, and where its pretraining runs go.
PITCHER_TOKEN_DIR = PROJECT_ROOT / "data" / "pitcher_ros"
PITCHER_PRETRAIN = PROJECT_ROOT / "data" / "pitcher_ros" / "pretrain"

logger = logging.getLogger("pretrain_hitter_ros")

_INT_FIELDS = (
    "dim",
    "layers",
    "heads",
    "window",
    "batch_size",
    "warmup_steps",
    "max_epochs",
    "patience",
    "seed",
)
_FLOAT_FIELDS = ("lr", "weight_decay", "val_frac", "dropout")


def main() -> int:
    d = PretrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--side",
        choices=sorted(SIDES),
        default="hitter",
        help="whose pitch sequences to pretrain on (build_hitter_ros_pitch_tokens --side)",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="seasons to pretrain a model for (default: every season with an earlier one)",
    )
    # Settings default to None (= PretrainConfig's default), so --add can tell a setting
    # that was passed from one that wasn't.
    for field in _INT_FIELDS:
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=int, help=f"default {getattr(d, field)}"
        )
    for field in _FLOAT_FIELDS:
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=float, help=f"default {getattr(d, field)}"
        )
    parser.add_argument(
        "--zone",
        choices=list(ZONES),
        help="the token file's strike zone (build_hitter_ros_pitch_tokens --zone, #433); "
        "default statcast, or with --add the run's",
    )
    parser.add_argument("--no-amp", action="store_true", help="float32 instead of bfloat16")
    parser.add_argument(
        "--half",
        type=int,
        choices=(0, 1),
        help="read only this half of the players' pitches (pretrain.player_half); "
        "default all, or with --add the run's",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--add",
        action="store_true",
        help="add --seasons to the existing run <name>, with its own settings",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    pitcher = args.side == "pitcher"
    token_dir = PITCHER_TOKEN_DIR if pitcher else TOKEN_DIR
    out_root = (PITCHER_PRETRAIN if pitcher else PRETRAIN) / args.name
    settings = (*_INT_FIELDS, *_FLOAT_FIELDS)
    added_to: dict | None = None
    zone = args.zone or "statcast"
    if args.add:
        if args.overwrite:
            parser.error("--add keeps the run's seasons; it can't also --overwrite")
        if not args.seasons:
            parser.error("--add needs --seasons")
        if not (out_root / "run.json").exists():
            parser.error(f"{out_root} has no run.json to add to")
        added_to = json.loads((out_root / "run.json").read_text())
        run_config = added_to["config"]
        known = {f.name for f in dataclasses.fields(PretrainConfig)}
        if set(run_config) != known:
            parser.error(
                f"{args.name}'s settings don't match today's PretrainConfig "
                f"(missing {sorted(known - set(run_config))}, "
                f"unknown {sorted(set(run_config) - known)}); pretrain a new run"
            )
        changed = [
            f
            for f in settings
            if getattr(args, f) is not None and getattr(args, f) != run_config[f]
        ] + (["no_amp"] if args.no_amp and run_config["amp"] else [])
        zone = run_token_options(added_to)["zone"]
        if args.zone is not None and args.zone != zone:
            changed.append("zone")
        if args.half is not None and args.half != added_to.get("half"):
            changed.append("half")
        if changed:
            parser.error(f"--add uses the run's own settings; drop {', '.join(changed)}")
        already = sorted(set(args.seasons) & set(added_to["seasons"]))
        if already:
            parser.error(f"{args.name} already has {already}")
        # A crash can leave a season's folder (or its temporary one) behind: the save at
        # the end would fail only after the whole pretraining.
        leftover = [
            str(path)
            for s in args.seasons
            for path in (out_root / str(s), out_root / f".{s}.tmp")
            if path.exists()
        ]
        if leftover:
            parser.error(f"remove what an earlier run left behind first: {leftover}")
    try:
        if added_to is not None:
            config = PretrainConfig(**added_to["config"])
        else:
            values = {
                f: getattr(args, f) if getattr(args, f) is not None else getattr(d, f)
                for f in settings
            }
            config = PretrainConfig(**values, amp=not args.no_amp)
    except (TypeError, ValueError) as err:
        parser.error(str(err))
    tokens_path = token_path(token_dir, zone)
    if not tokens_path.exists():
        parser.error(
            f"{tokens_path} is missing; run scripts/build_hitter_ros_pitch_tokens.py "
            f"--zone {zone} --side {args.side}"
        )
    # Checked before the (large) token file is read.
    recorded = recorded_token_options(tokens_path)
    if token_side(recorded) != args.side:
        parser.error(f"{tokens_path} holds {token_side(recorded)}-side tokens, not {args.side}")
    if added_to is not None and run_token_options(added_to) != recorded:
        parser.error(f"{tokens_path}'s build options differ from {args.name}'s")
    if added_to is None and out_root.exists() and not args.overwrite:
        parser.error(f"{out_root} exists; pick another --name or pass --overwrite")

    tokens = pd.read_parquet(tokens_path)
    all_seasons = sorted(int(s) for s in tokens["season"].unique())
    seasons = args.seasons or all_seasons[1:]
    too_early = [s for s in seasons if s <= all_seasons[0]]
    if too_early:
        parser.error(f"no earlier season to pretrain on for {too_early}")
    # The whole file's fingerprint: what fine-tuning feeds the run's models is checked
    # against it.
    fingerprint = tokens_fingerprint(tokens)
    half = added_to.get("half") if added_to is not None else args.half
    if half is not None:
        tokens = tokens[player_half(tokens["player_id"]) == half].reset_index(drop=True)
        logger.info("half %d: %d players' pitches", half, tokens["player_id"].nunique())
    store = PitchStore(tokens, device(), torch.bfloat16 if config.amp else torch.float32)
    del tokens

    if added_to is not None:
        # The run's other seasons were pretrained on one token file: the new ones must be too.
        if added_to.get("tokens") != fingerprint:
            parser.error(f"{tokens_path} is not the token file {args.name} was pretrained on")
        run_meta = added_to
        # Adding seasons doesn't finish a run that stopped early.
        finished = bool(added_to.get("complete"))
        run_meta["complete"] = False
    else:
        # Every input is loaded and checked; only now replace an old run of this name.
        if out_root.exists():
            shutil.rmtree(out_root)
        out_root.mkdir(parents=True)
        run_meta = {
            "config": config.to_dict(),
            "tokens": fingerprint,
            "token_options": recorded,
            "half": half,
            "seasons": [],
            "complete": False,
        }
        finished = True
    for season in seasons:
        t0 = time.time()
        result = pretrain(store, before_season=season, config=config)
        minutes = (time.time() - t0) / 60
        tmp = out_root / f".{season}.tmp"
        tmp.mkdir()
        torch.save(result.model.state_dict(), tmp / "model.pt")
        meta = {
            "config": config.to_dict(),
            "season": season,
            "trained_on": f"seasons before {season}",
            "best_epoch": result.best_epoch,
            "train_ce": result.train_ce,
            "val_ce": result.val_ce,
            "baseline_ce": result.baseline_ce,
            "minutes": round(minutes, 1),
        }
        (tmp / "metrics.json").write_text(json.dumps(meta, indent=2))
        tmp.rename(out_root / str(season))
        run_meta["seasons"].append(season)  # type: ignore[union-attr]
        (out_root / "run.json").write_text(json.dumps(run_meta, indent=2))
        print(
            f"{season}: best val CE {min(result.val_ce):.4f} vs count-only "
            f"{result.baseline_ce:.4f} (epoch {result.best_epoch}, {minutes:.0f} min)"
        )
    run_meta["complete"] = finished
    (out_root / "run.json").write_text(json.dumps(run_meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
