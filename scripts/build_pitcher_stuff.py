"""Per-pitch stuff models for the pitcher net (#489, stage 1): build, train, check.

1. ``--build`` writes every regular-season pitch's stuff inputs and labels to
   data/pitcher_ros/stuff_pitches.parquet (pitcher_ros/stuff.py).
3. ``--features-file <name> --run <run>`` reads every pitcher-table row's pitches with
   that run's season models into data/pitcher_ros/stuff_<name>.parquet
   (pitcher_ros/stuff_features.py), for train_pitcher_ros.py --stuff <name>.
2. For each ``--seasons`` S: train on pitches before S only, score on season S against a
   pitch-type-only guess (when S has been played), and save the model to data/pitcher_ros/stuff/<name>/model-<S>.pt
   (with scores.json). ``--features`` picks the input set: stuff (the pitch only) or
   located (also location and count).

Usage:
    python scripts/build_pitcher_stuff.py --build
    python scripts/build_pitcher_stuff.py --name s1 --features stuff --seasons 2020 2023 2026
    python scripts/build_pitcher_stuff.py --name s2-stuff --features stuff --seasons $(seq 2009 2027)
    python scripts/build_pitcher_stuff.py --features-file s2-stuff --run s2-stuff
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.features import ROW_KEYS
from fantasy_baseball.hitter_ros.net import device
from fantasy_baseball.pitcher_ros.stuff import (
    FEATURE_SETS,
    StuffConfig,
    build_stuff_pitches,
    evaluate,
    fit_stuff,
    load_stuff,
    pitch_tensors,
    stuff_path,
    type_baseline,
)
from fantasy_baseball.pitcher_ros.stuff_features import (
    STUFF_INPUTS,
    build_stuff_features,
    stuff_features_path,
)

ROOT = PROJECT_ROOT / "data" / "pitcher_ros"
TABLE = ROOT / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="(re)build the per-pitch frame")
    parser.add_argument("--name", help="run folder under data/pitcher_ros/stuff/")
    parser.add_argument("--features", choices=sorted(FEATURE_SETS), default="stuff")
    parser.add_argument("--seasons", type=int, nargs="+", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--features-file",
        help="write the pitcher table's stuff inputs to data/pitcher_ros/stuff_<name>.parquet "
        "from the models of --run",
    )
    parser.add_argument("--run", help="stuff model run to read with --features-file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    path = stuff_path(ROOT)
    if args.build:
        frame = build_stuff_pitches(STORE)
        frame.to_parquet(path, index=False)
        logging.info("wrote %s: %d pitches", path, len(frame))
    if args.features_file:
        if not args.run:
            parser.error("--features-file needs --run")
        write_features(args.features_file, args.run, path)
    if not args.seasons:
        return 0
    if not args.name:
        parser.error("--seasons needs --name")
    if not path.exists():
        parser.error(f"{path} is missing; run with --build first")
    pitches = pd.read_parquet(path)
    features = FEATURE_SETS[args.features]
    config = StuffConfig(seed=args.seed)
    run = ROOT / "stuff" / args.name
    run.mkdir(parents=True, exist_ok=True)
    scores_path = run / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    for season in sorted(set(args.seasons)):
        train = pitches[pitches["season"] < season]
        test = pitches[pitches["season"] == season]
        if train.empty:
            parser.error(f"season {season}: no pitches before it to train on")
        logging.info("season %d: training on %d pitches", season, len(train))
        fit = fit_stuff(train, features, config)
        # A season not played yet (next season's preseason) has a model but no score.
        if not test.empty:
            model_scores = evaluate(fit.model, *pitch_tensors(test, features, device()))
            base = type_baseline(train, test)
            names = ("outcome_ce", "bb_ce", "xwoba_se")
            scores[str(season)] = {
                "pitches": len(test),
                "best_epoch": fit.best_epoch,
                "model": dict(zip(names, model_scores, strict=True)),
                "type_only": dict(zip(names, base, strict=True)),
            }
            logging.info("season %d: %s", season, scores[str(season)])
        torch.save(
            {
                "state_dict": fit.model.state_dict(),
                "features": list(features),
                "config": config.to_dict(),
                "season": season,
            },
            run / f"model-{season}.pt",
        )
        scores_path.write_text(json.dumps(scores, indent=2))
    (run / "config.json").write_text(
        json.dumps({"features": args.features, "config": config.to_dict()}, indent=2)
    )
    return 0


def write_features(name: str, run_name: str, path: Path) -> None:
    """The pitcher table's stuff inputs from every season model of ``run_name``."""
    run = ROOT / "stuff" / run_name
    models = {
        int(m.stem.removeprefix("model-")): load_stuff(m) for m in sorted(run.glob("model-*.pt"))
    }
    if not models:
        raise SystemExit(f"{run} has no models; train them with --name {run_name} --seasons ...")
    table = pd.read_parquet(TABLE, columns=[*ROW_KEYS, "as_of"])
    pitches = pd.read_parquet(path)
    features = build_stuff_features(table, pitches, models)
    out = stuff_features_path(ROOT, name)
    features.to_parquet(out, index=False)
    options = {"run": run_name, **json.loads((run / "config.json").read_text())}
    out.with_suffix(".json").write_text(json.dumps(options, indent=2))
    known = features[STUFF_INPUTS].notna().mean()
    logging.info("wrote %s: %d rows, %d inputs", out, len(features), len(STUFF_INPUTS))
    logging.info("share known: %s", known.filter(like="whiff_swing").round(3).to_dict())


if __name__ == "__main__":
    sys.exit(main())
