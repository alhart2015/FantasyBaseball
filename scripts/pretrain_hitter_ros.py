"""Pretrain the pitch encoder for the hitter ROS net, walk-forward (#415).

For each test season T: pretrain on every pitch from seasons before T (predict the next
pitch's outcome), then embed every training-table row from seasons <= T (the hitter's
last --window pitches before its as-of date). Saves under
data/hitter_ros/pretrain/<name>/<T>/:

    encoder.pt        encoder weights
    embeddings.npy    float16 [table rows, 2*dim + 1]; zeros for seasons after T
    metrics.json      config, epochs, train/val CE per epoch, count-only baseline CE

Then train the ROS net on top: train_hitter_ros.py --pretrained <name>.

Setup (once): python scripts/build_hitter_ros_pitch_tokens.py
Usage:
    python scripts/pretrain_hitter_ros.py --name p001
    python scripts/pretrain_hitter_ros.py --name p001 --test-seasons 2026 --max-epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.net import device
from fantasy_baseball.hitter_ros.pretrain import (
    PitchStore,
    PretrainConfig,
    embed_rows,
    pretrain,
    table_fingerprint,
)

TABLE = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"
TOKENS = PROJECT_ROOT / "data" / "hitter_ros" / "pitch_tokens.parquet"
PRETRAIN = PROJECT_ROOT / "data" / "hitter_ros" / "pretrain"

logger = logging.getLogger("pretrain_hitter_ros")


def main() -> int:
    d = PretrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--test-seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026]
    )
    for field in (
        "dim",
        "layers",
        "heads",
        "window",
        "batch_size",
        "warmup_steps",
        "max_epochs",
        "patience",
        "seed",
    ):
        parser.add_argument(f"--{field.replace('_', '-')}", type=int, default=getattr(d, field))
    for field in ("lr", "weight_decay", "val_frac", "dropout"):
        parser.add_argument(f"--{field.replace('_', '-')}", type=float, default=getattr(d, field))
    parser.add_argument("--no-amp", action="store_true", help="float32 instead of bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    try:
        config = PretrainConfig(
            **{k: getattr(args, k) for k in d.to_dict() if k != "amp"}, amp=not args.no_amp
        )
    except ValueError as err:
        parser.error(str(err))
    if not TOKENS.exists():
        parser.error(f"{TOKENS} is missing; run scripts/build_hitter_ros_pitch_tokens.py")
    out_root = PRETRAIN / args.name
    if out_root.exists() and not args.overwrite:
        parser.error(f"{out_root} exists; pick another --name or pass --overwrite")

    table = pd.read_parquet(TABLE)
    store = PitchStore(pd.read_parquet(TOKENS), device())
    for season in args.test_seasons:
        out = out_root / str(season)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        result = pretrain(store, before_season=season, config=config)
        minutes = (time.time() - t0) / 60
        rows = table["season"] <= season
        emb = np.zeros((len(table), 2 * config.dim + 1), dtype=np.float16)
        emb[rows.to_numpy()] = embed_rows(
            result.model.encoder, store, table.loc[rows, ["player_id", "as_of"]], config
        ).astype(np.float16)
        np.save(out / "embeddings.npy", emb)
        torch.save(result.model.encoder.state_dict(), out / "encoder.pt")
        meta = {
            "config": config.to_dict(),
            "test_season": season,
            "table_fingerprint": table_fingerprint(table),
            "best_epoch": result.best_epoch,
            "train_ce": result.train_ce,
            "val_ce": result.val_ce,
            "baseline_ce": result.baseline_ce,
            "minutes": round(minutes, 1),
        }
        (out / "metrics.json").write_text(json.dumps(meta, indent=2))
        print(
            f"{season}: best val CE {min(result.val_ce):.4f} vs count-only {result.baseline_ce:.4f} "
            f"(epoch {result.best_epoch}, {minutes:.0f} min)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
