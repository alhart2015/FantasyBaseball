"""Build pitcher probe features for every pitcher-table row from a pretraining run (#492).

For each season S in the table, load model S of the pitcher-side pretraining run
(``pretrain_hitter_ros.py --side pitcher``; trained on seasons before S only) and read
probe features for every row of season S (see pitcher_ros/probes.py), standardized
against a reference group known before the season (last season's pitchers, read on
Opening Day). Seasons without a model get NaN, and so do the contact-quality features
of models pretrained only before Statcast's contact classes.
Writes data/pitcher_ros/probes_<run>.parquet: player_id, season, week, as_of + the
features (with --seasons, probes_<run>_partial.parquet). Then train with:
    python scripts/train_pitcher_ros.py --probes <run> ...

The token file must be the one the run was pretrained on, with Savant's strike zone:
the probe pitches' heights (hitter_ros.probes.LOCATIONS) are in it.

Usage:
    python scripts/build_pitcher_ros_probes.py --run q001
    python scripts/build_pitcher_ros_probes.py --run q001 --seasons 2026   # quick check
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
from fantasy_baseball.hitter_ros.pitch_tokens import recorded_token_options, token_options
from fantasy_baseball.hitter_ros.pretrain import (
    PitchStore,
    PretrainConfig,
    PretrainModel,
    run_token_options,
    tokens_fingerprint,
)
from fantasy_baseball.hitter_ros.probes import first_contact_season, standardize
from fantasy_baseball.pitcher_ros.probes import (
    CONTACT_FEATURES,
    PROBE_FEATURES,
    PROBE_KEYS,
    probe_features,
    probe_path,
    reference_cohort,
    throwing_hand,
)

ROOT = PROJECT_ROOT / "data" / "pitcher_ros"
TABLE = ROOT / "table.parquet"
TOKENS = ROOT / "pitch_tokens.parquet"
PRETRAIN = ROOT / "pretrain"

logger = logging.getLogger("build_pitcher_ros_probes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="pitcher pretraining run, e.g. q001")
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
    trained_on, given = run_token_options(run_meta), recorded_token_options(args.tokens)
    if given != trained_on:
        parser.error(
            f"{args.tokens} was built with {given}; {args.run} was pretrained on {trained_on}"
        )
    if trained_on != token_options("statcast"):
        parser.error(
            f"probe pitch heights are in Savant's strike zone; {args.run} has {trained_on}"
        )
    table = pd.read_parquet(TABLE, columns=[*PROBE_KEYS, "as_of"])
    seasons = sorted(args.seasons or table["season"].unique())
    tokens = pd.read_parquet(args.tokens)
    if run_meta["tokens"] != tokens_fingerprint(tokens):
        parser.error(f"{args.tokens} is not the token file {args.run} was pretrained on")
    first_contact = first_contact_season(tokens)
    # The last game read, for train_pitcher_ros.py to refuse a file older than the store.
    pitches_through = str(pd.to_datetime(tokens["game_date"]).max().date())
    store = PitchStore(tokens, device(), torch.bfloat16 if config.amp else torch.float32)
    del tokens
    hand_sums = throwing_hand(store)

    def read(model: PretrainModel, players: np.ndarray, as_of: pd.Series) -> pd.DataFrame:
        return probe_features(
            model,
            store,
            players,
            as_of,
            window=config.window,
            batch_size=args.batch_size,
            amp=config.amp,
            hand_sums=hand_sums,
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
        # Reference group, known before the season: last season's pitchers, on Opening Day.
        opening_day = rows.loc[rows["week"] == 0, "as_of"].min()
        ref_players = reference_cohort(table, s)
        reference = read(model, ref_players, pd.Series([opening_day] * len(ref_players)))
        if s <= first_contact:  # model pretrained before contact classes: reads 0
            feats[list(CONTACT_FEATURES)] = np.nan
            reference[list(CONTACT_FEATURES)] = np.nan
        parts.append(pd.concat([rows, standardize(feats, reference)], axis=1))
        logger.info(
            "season %s: %d rows, %d with history, reference %d pitchers, %.0fs",
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
    run = args.run if not args.seasons else f"{args.run}_partial"
    path = probe_path(ROOT, run)
    out.to_parquet(path)
    options = {
        "run": args.run,
        "seasons": [int(s) for s in seasons],
        "tokens": run_meta["tokens"],
        "pitches_through": pitches_through,
    }
    path.with_suffix(".json").write_text(json.dumps(options, indent=2))
    logger.info("wrote %s (%d rows)", path, len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
