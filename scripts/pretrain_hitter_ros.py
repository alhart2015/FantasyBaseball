"""Pretrain the pitch models for the hitter ROS net: one per season, walk-forward (#415).

For each season S: pretrain on every pitch from seasons before S (predict the next
pitch's outcome). Anything read off model S about a row from season S is then out of
sample for that row -- training rows as well as test rows. Saves under
data/hitter_ros/pretrain/<name>/<S>/:

    model.pt       PretrainModel weights (encoder + prediction head)
    metrics.json   config, train/val CE per epoch, count-only baseline CE, minutes

and <name>/run.json listing the seasons and config. With --overwrite the whole <name>
folder is replaced, but only after the inputs are found; each season is written to a
temporary folder and moved into place when it finishes, so a crash never leaves a
half-written season behind.

Setup (once): python scripts/build_hitter_ros_pitch_tokens.py
Usage:
    python scripts/pretrain_hitter_ros.py --name p002
    python scripts/pretrain_hitter_ros.py --name smoke --seasons 2026 --max-epochs 1
"""

from __future__ import annotations

import argparse
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

from fantasy_baseball.hitter_ros.net import device
from fantasy_baseball.hitter_ros.pretrain import (
    PitchStore,
    PretrainConfig,
    pretrain,
    tokens_fingerprint,
)

TOKENS = PROJECT_ROOT / "data" / "hitter_ros" / "pitch_tokens.parquet"
PRETRAIN = PROJECT_ROOT / "data" / "hitter_ros" / "pretrain"

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
        "--seasons",
        type=int,
        nargs="+",
        help="seasons to pretrain a model for (default: every season with an earlier one)",
    )
    for field in _INT_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", type=int, default=getattr(d, field))
    for field in _FLOAT_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", type=float, default=getattr(d, field))
    parser.add_argument("--no-amp", action="store_true", help="float32 instead of bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    try:
        values = {f: getattr(args, f) for f in (*_INT_FIELDS, *_FLOAT_FIELDS)}
        config = PretrainConfig(**values, amp=not args.no_amp)
    except ValueError as err:
        parser.error(str(err))
    if not TOKENS.exists():
        parser.error(f"{TOKENS} is missing; run scripts/build_hitter_ros_pitch_tokens.py")
    out_root = PRETRAIN / args.name
    if out_root.exists() and not args.overwrite:
        parser.error(f"{out_root} exists; pick another --name or pass --overwrite")

    tokens = pd.read_parquet(TOKENS)
    all_seasons = sorted(int(s) for s in tokens["season"].unique())
    seasons = args.seasons or all_seasons[1:]
    too_early = [s for s in seasons if s <= all_seasons[0]]
    if too_early:
        parser.error(f"no earlier season to pretrain on for {too_early}")
    store = PitchStore(tokens, device(), torch.bfloat16 if config.amp else torch.float32)
    fingerprint = tokens_fingerprint(tokens)
    del tokens

    # Every input is loaded and checked; only now replace an old run of this name.
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    run_meta = {
        "config": config.to_dict(),
        "tokens": fingerprint,
        "seasons": [],
        "complete": False,
    }
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
    run_meta["complete"] = True
    (out_root / "run.json").write_text(json.dumps(run_meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
