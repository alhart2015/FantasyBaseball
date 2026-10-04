"""Compare pretraining runs fairly: every run's model S predicts every pitch of season S.

Each run's own validation CE isn't comparable across runs (different held-out hitters,
and pre-2015 balls in play are a single easier class). Here model S of each run -- which
never saw season S -- is scored on exactly the same pitches.

Usage (one --run per pretraining run, with the token file it was trained on):
    python scripts/compare_pretrain_hitter_ros.py \
        --run p002=data/hitter_ros/pitch_tokens_from2015.parquet \
        --run p003=data/hitter_ros/pitch_tokens.parquet --seasons 2016 2020 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.net import device
from fantasy_baseball.hitter_ros.pretrain import (
    PitchStore,
    PretrainConfig,
    PretrainModel,
    count_baseline_ce,
    season_ce,
)

PRETRAIN = PROJECT_ROOT / "data" / "hitter_ros" / "pretrain"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, help="NAME=TOKENS_PARQUET")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    args = parser.parse_args()

    results: dict[str, dict[int, float]] = {}
    counts: dict[int, int] = {}
    baseline: dict[int, float] = {}
    for spec in args.run:
        name, tokens_path = spec.split("=", 1)
        meta = json.loads((PRETRAIN / name / "run.json").read_text())
        config = PretrainConfig(**meta["config"])
        tokens = pd.read_parquet(tokens_path)
        for s in args.seasons:
            if s not in baseline:
                t = tokens[tokens["season"] == s]
                baseline[s] = count_baseline_ce(t)
        store = PitchStore(tokens, device(), torch.bfloat16 if config.amp else torch.float32)
        del tokens
        for s in args.seasons:
            model = PretrainModel(config).to(device())
            model.load_state_dict(torch.load(PRETRAIN / name / str(s) / "model.pt"))
            ce, n = season_ce(model, store, s, config)
            if s in counts and counts[s] != n:
                raise SystemExit(f"season {s}: {name} scored {n} pitches, earlier run {counts[s]}")
            counts[s] = n
            results.setdefault(name, {})[s] = ce
            print(f"{name} {s}: CE {ce:.4f} on {n} pitches")
        del store
        torch.cuda.empty_cache()
    names = list(results)
    print("\nseason  " + "  ".join(f"{n:>8}" for n in names) + "  count-only")
    for s in args.seasons:
        row = "  ".join(f"{results[n][s]:8.4f}" for n in names)
        print(f"{s}  {row}  {baseline[s]:10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
