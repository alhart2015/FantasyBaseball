"""Next season's hitter outcome ranges (pt_model.ranges, #505): P10/P50/P90 of PA, R,
HR, RBI, SB and AVG per hitter.

Fit on the past seasons as scripts/backtest_hitter_ranges.py scores it (--pt-system
for playing time, --train-run rates), then predict --season from the same kind of PT
source -- --pt-file (his rank in its projected PA) for a projection system, his PA the
season before for last-season -- and the --rates-run seeds' preseason rows.

A --pt-file should be a one-year-ahead projection with a depth chart's playing time
(FanGraphs Depth Charts or Steamer for --season), the kind the training ranks came
from. ZiPS isn't one: it gives full playing time to minor leaguers (130 of its top 390
for 2027), which pushes every regular down the ranks.

Writes data/hitter_ranges/<season>-<name>.parquet and prints the top 5 per stat.

    python scripts/hitter_ranges.py --season 2027 --name steamer \\
        --pt-file data/projections/2027/steamer-hitters.csv --rates-run proj2027seq
    python scripts/hitter_ranges.py --season 2027 --name last --pt-system last-season \\
        --rates-run proj2027seq --train-seasons 2023 2024 2025 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
from backtest_hitter_ranges import (
    LAST_SEASON,
    OUT,
    PEOPLE,
    RUNS,
    load_inputs,
    pt_projection,
    training_rows,
)

from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.ranges import STATS, QuantilePT, fit

LEVELS = (0.1, 0.5, 0.9)
TOP = 5
# AVG leaders: a median of this many PA (the batting-title line).
QUALIFIED_PA = 502


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--pt-system",
        default="steamer",
        help=f"training PT source; {LAST_SEASON!r} for last season's PA",
    )
    parser.add_argument(
        "--pt-file", type=Path, help="the --season PT projection (not with last-season)"
    )
    parser.add_argument("--rates-run", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--train-seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--train-run", default="414-seq")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.season in args.train_seasons:
        parser.error("--season can't be a training season")
    if (args.pt_system == LAST_SEASON) == (args.pt_file is not None):
        parser.error("pass --pt-file exactly when --pt-system is a projection system")

    train, train_pa = training_rows(
        sorted(args.train_seasons), args.pt_system, args.train_run, args.seeds
    )
    rng = np.random.default_rng(args.seed)
    ranges = fit(train, train_pa, QuantilePT, rng)

    loaded, info = load_inputs([args.season], args.pt_system, with_actuals=False)
    rows, dropped = ri.build_rows(
        args.season,
        pt_projection(args.pt_system, args.season, loaded)
        if args.pt_file is None
        else ri.load_pt(args.pt_file),
        ri.load_rates(RUNS / f"{args.rates_run}-s{s}" for s in args.seeds),
        loaded,
        info,
        with_actuals=False,
    )
    names = pd.read_csv(PEOPLE, usecols=["id", "fullName"]).set_index("id")["fullName"]
    print(f"{args.season}: {len(rows)} hitters ({dropped} ranked but without our rates, left out)")
    ranked, sims = ranges.predict(rows, LEVELS, rng, args.draws)
    out = pd.DataFrame(
        {
            "player_id": ranked.index,
            "name": names.reindex(ranked.index).to_numpy(),
            "season": args.season,
            "age": ranked["age"].to_numpy(),
            "proj_rank": ranked["proj_rank"].to_numpy(),
            "rank_pa": ranked["rank_pa"].to_numpy(),
            **{
                f"{c}_p{round(lv * 100)}": sims[c][:, k]
                for c in ("pa", *STATS)
                for k, lv in enumerate(LEVELS)
            },
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.season}-{args.name}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path}")
    for c in STATS:
        pool = out[out["pa_p50"] >= QUALIFIED_PA] if c == "avg" else out
        top = pool.nlargest(TOP, f"{c}_p50")
        print(f"\n{c.upper()}: median (P10-P90)")
        for _, r in top.iterrows():
            fmt = "{:.3f}" if c == "avg" else "{:.0f}"
            print(
                f"  {r['name']}: {fmt.format(r[f'{c}_p50'])} "
                f"({fmt.format(r[f'{c}_p10'])}-{fmt.format(r[f'{c}_p90'])})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
