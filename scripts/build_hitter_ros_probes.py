"""Build probe features for every training-table row from a pretraining run (#417).

For each season S in the table, load model S of the pretraining run (trained on
seasons before S only) and read probe features for every row of season S (see
hitter_ros/probes.py). Seasons without a model (the store's first season) get NaN, and so do the contact-
quality features of models pretrained only before Statcast's contact classes.
Writes data/hitter_ros/probes_<run>.parquet: player_id, season, week + the features.
Then train with:  python scripts/train_hitter_ros.py --probes <run> ...

The token file must be the one the run was pretrained on (p003: pitch_tokens.parquet).

Usage:
    python scripts/build_hitter_ros_probes.py --run p003
    python scripts/build_hitter_ros_probes.py --run p003 --seasons 2024   # quick check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.net import device
from fantasy_baseball.hitter_ros.pretrain import PitchStore, PretrainConfig, PretrainModel
from fantasy_baseball.hitter_ros.probes import (
    PROBE_FEATURES,
    blank_unseen_contact,
    first_contact_season,
    probe_features,
)

TABLE = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"
TOKENS = PROJECT_ROOT / "data" / "hitter_ros" / "pitch_tokens.parquet"
PRETRAIN = PROJECT_ROOT / "data" / "hitter_ros" / "pretrain"
OUT_DIR = PROJECT_ROOT / "data" / "hitter_ros"

logger = logging.getLogger("build_hitter_ros_probes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="pretraining run name, e.g. p003")
    parser.add_argument("--tokens", type=Path, default=TOKENS)
    parser.add_argument("--seasons", type=int, nargs="*", help="only these (default: all)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    run_dir = PRETRAIN / args.run
    if not (run_dir / "run.json").exists():
        parser.error(f"{run_dir} has no run.json; is the pretraining run name right?")
    config = PretrainConfig(**json.loads((run_dir / "run.json").read_text())["config"])
    table = pd.read_parquet(TABLE, columns=["player_id", "season", "week", "as_of"])
    seasons = sorted(args.seasons or table["season"].unique())
    tokens = pd.read_parquet(args.tokens)
    first_contact = first_contact_season(tokens)
    store = PitchStore(tokens, device(), torch.bfloat16 if config.amp else torch.float32)
    del tokens

    parts = []
    for s in seasons:
        rows = table[table["season"] == s]
        path = run_dir / str(s) / "model.pt"
        if not path.exists():
            logger.info("season %s: no model in %s, features left blank", s, args.run)
            continue
        t0 = time.time()
        model = PretrainModel(config).to(device())
        model.load_state_dict(torch.load(path, map_location=device()))
        feats = probe_features(
            model,
            store,
            rows["player_id"].to_numpy(),
            rows["as_of"],
            window=config.window,
            batch_size=args.batch_size,
            amp=config.amp,
        )
        parts.append(pd.concat([rows.reset_index(drop=True), feats], axis=1))
        logger.info(
            "season %s: %d rows, %d with history, %.0fs",
            s,
            len(rows),
            int(feats.iloc[:, 0].notna().sum()),
            time.time() - t0,
        )
    keys = ["player_id", "season", "week"]
    rows = table[table["season"].isin(seasons)][keys]
    found = pd.concat(parts, ignore_index=True)[[*keys, *PROBE_FEATURES]] if parts else None
    # Every table row of the seasons asked for; blank where a season had no model.
    out = rows.merge(found, on=keys, how="left") if found is not None else rows
    out = out.reindex(columns=[*keys, *PROBE_FEATURES])
    # Models pretrained only before contact classes existed read contact as exactly 0.
    out = blank_unseen_contact(out, first_contact)
    logger.info(
        "contact-quality features blank through %s (model trained before it)", first_contact
    )
    suffix = "" if args.seasons is None else "_partial"
    path = OUT_DIR / f"probes_{args.run}{suffix}.parquet"
    out.to_parquet(path)
    logger.info("wrote %s (%d rows)", path, len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
