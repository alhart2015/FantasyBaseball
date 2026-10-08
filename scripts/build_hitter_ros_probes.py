"""Build probe features for every training-table row from a pretraining run (#417).

For each season S in the table, load model S of the pretraining run (trained on
seasons before S only) and read probe features for every row of season S (see
hitter_ros/probes.py), standardized against a reference group known before the season
(last season's hitters, read on Opening Day). Seasons without a model (the store's first season) get NaN, and so do the contact-
quality features of models pretrained only before Statcast's contact classes.
Writes data/hitter_ros/probes_<run>.parquet: player_id, season, week, as_of + the features.
Then train with:  python scripts/train_hitter_ros.py --probes <run> ...

The token file must be the one the run was pretrained on (p004: pitch_tokens.parquet;
p005: pitch_tokens_fixed.parquet, the fixed strike zone of #433).

Usage:
    python scripts/build_hitter_ros_probes.py --run p003
    python scripts/build_hitter_ros_probes.py --run p003 --seasons 2024   # quick check
    python scripts/build_hitter_ros_probes.py --run p005 \
        --tokens data/hitter_ros/pitch_tokens_fixed.parquet
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
    PretrainModel,
    same_tokens,
)
from fantasy_baseball.hitter_ros.probes import (
    CONTACT_FEATURES,
    PROBE_FEATURES,
    PROBE_KEYS,
    first_contact_season,
    probe_features,
    probe_path,
    reference_cohort,
    side_prefix,
    standardize,
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
    parser.add_argument("--seasons", type=int, nargs="+", help="only these (default: all)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    run_dir = PRETRAIN / args.run
    if not (run_dir / "run.json").exists():
        parser.error(f"{run_dir} has no run.json; is the pretraining run name right?")
    run_meta = json.loads((run_dir / "run.json").read_text())
    config = PretrainConfig(**run_meta["config"])
    table = pd.read_parquet(TABLE, columns=[*PROBE_KEYS, "as_of"])
    seasons = sorted(args.seasons or table["season"].unique())
    tokens = pd.read_parquet(args.tokens)
    if "tokens" in run_meta:
        if not same_tokens(run_meta["tokens"], tokens):
            parser.error(f"{args.tokens} is not the token file {args.run} was pretrained on")
    else:
        logger.warning(
            "%s predates token fingerprints; can't check %s is its token file",
            args.run,
            args.tokens.name,
        )
    first_contact = first_contact_season(tokens)
    store = PitchStore(tokens, device(), torch.bfloat16 if config.amp else torch.float32)
    del tokens
    prefix = side_prefix(store)

    def read(model: PretrainModel, players: np.ndarray, as_of: pd.Series) -> pd.DataFrame:
        return probe_features(
            model,
            store,
            players,
            as_of,
            window=config.window,
            batch_size=args.batch_size,
            amp=config.amp,
            prefix=prefix,
        )

    parts = []
    for s in seasons:
        rows = table[table["season"] == s].reset_index(drop=True)
        path = run_dir / str(s) / "model.pt"
        if not path.exists():
            logger.info("season %s: no model in %s, features left blank", s, args.run)
            continue
        t0 = time.time()
        model = PretrainModel(config).to(device())
        model.load_state_dict(torch.load(path, map_location=device()))
        feats = read(model, rows["player_id"].to_numpy(), rows["as_of"])
        # Reference group, known before the season: last season's hitters, on Opening Day.
        opening_day = rows.loc[rows["week"] == 0, "as_of"].min()
        ref_players = reference_cohort(table, s)
        reference = read(model, ref_players, pd.Series([opening_day] * len(ref_players)))
        if s <= first_contact:  # model pretrained before contact classes: reads 0
            feats[list(CONTACT_FEATURES)] = np.nan
            reference[list(CONTACT_FEATURES)] = np.nan
        parts.append(pd.concat([rows, standardize(feats, reference)], axis=1))
        logger.info(
            "season %s: %d rows, %d with history, reference %d hitters, %.0fs",
            s,
            len(rows),
            int(feats.iloc[:, 0].notna().sum()),
            int(reference.iloc[:, 0].notna().sum()),
            time.time() - t0,
        )
    rows = table[table["season"].isin(seasons)]
    found = pd.concat(parts, ignore_index=True) if parts else None
    # Every table row of the seasons asked for; blank where a season had no model.
    out = (
        rows.merge(found.drop(columns="as_of"), on=PROBE_KEYS, how="left")
        if found is not None
        else rows
    )
    out = out.reindex(columns=[*PROBE_KEYS, "as_of", *PROBE_FEATURES])
    logger.info(
        "contact-quality features blank through %s (model trained before it)", first_contact
    )
    run = args.run if not args.seasons else f"{args.run}_partial"
    path = probe_path(OUT_DIR, run)
    out.to_parquet(path)
    logger.info("wrote %s (%d rows)", path, len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
