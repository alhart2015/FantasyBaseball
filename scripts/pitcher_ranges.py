"""Next season's pitcher outcome ranges (pt_model.ranges, #505): P10/P50/P90 of IP, K,
W, SV, ERA and WHIP per pitcher, starters and relievers each fit within the role.

Fit on the past seasons as scripts/backtest_pitcher_ranges.py scores them (--pt-system
for playing time and role, --train-run rates), then predict --season from the same kind
of PT source -- --pt-file for a projection system (its projected IP and GS / G), last
season's line for last-season -- and --rates-run's preseason rows. Each role's PT range
is the one its spec says (range_specs: starters one-size, relievers player-specific).

Writes data/pitcher_ranges/<season>-<name>.parquet and prints the top 5 per stat.

    python scripts/pitcher_ranges.py --season 2027 --name last --pt-system last-season
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
from backtest_hitter_ranges import LAST_SEASON, PEOPLE
from backtest_pitcher_ranges import (
    OUT,
    ROLES,
    RUNS,
    load_inputs,
    pt_projection,
    training_rows,
)

from fantasy_baseball.pt_model import range_inputs as ri
from fantasy_baseball.pt_model.ranges import fit, pt_model

LEVELS = (0.1, 0.5, 0.9)
TOP = 5
# ERA and WHIP leaders: a median of this many IP.
QUALIFIED_IP = 150.0
LOWER_IS_BETTER = ("era", "whip")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--pt-system",
        default="steamer",
        help=f"training PT source; {LAST_SEASON!r} for last season's line",
    )
    parser.add_argument(
        "--pt-file", type=Path, help="the --season PT projection (not with last-season)"
    )
    parser.add_argument("--rates-run", default="proj-2027-sv")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--train-run", default="035-saves-net")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.season in args.train_seasons:
        parser.error("--season can't be a training season")
    if (args.pt_system == LAST_SEASON) == (args.pt_file is not None):
        parser.error("pass --pt-file exactly when --pt-system is a projection system")

    train = training_rows(sorted(args.train_seasons), args.pt_system, args.train_run)
    loaded, info = load_inputs([args.season], args.pt_system, with_actuals=False)
    pt = (
        pt_projection(args.pt_system, args.season, loaded)
        if args.pt_file is None
        else ri.load_pitcher_pt(args.pt_file)
    )
    rates = ri.load_pitcher_rates(RUNS / args.rates_run)
    names = pd.read_csv(PEOPLE, usecols=["id", "fullName"]).set_index("id")["fullName"]
    rng = np.random.default_rng(args.seed)
    parts = []
    for spec in ROLES:
        rows, ip = train[spec.name]
        ranges = fit(rows, ip, pt_model, rng, spec)
        new, dropped = ri.build_pitcher_rows(
            args.season, spec, pt, rates, loaded, info, with_actuals=False
        )
        print(f"{args.season} {spec.name}s: {len(new)} ({dropped} ranked but without our rates)")
        ranked, sims = ranges.predict(new, LEVELS, rng, args.draws)
        parts.append(
            pd.DataFrame(
                {
                    "player_id": ranked.index,
                    "name": names.reindex(ranked.index).to_numpy(),
                    "role": spec.name,
                    "season": args.season,
                    "age": ranked["age"].to_numpy(),
                    "proj_rank": ranked["proj_rank"].to_numpy(),
                    "rank_ip": ranked["rank_ip"].to_numpy(),
                    **{
                        f"{c}_p{round(lv * 100)}": sims[c][:, k]
                        for c in ("ip", *spec.stat_names)
                        for k, lv in enumerate(LEVELS)
                    },
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.season}-{args.name}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path}")
    for c in ROLES[0].stat_names:
        if c in LOWER_IS_BETTER:
            top = out[out["ip_p50"] >= QUALIFIED_IP].nsmallest(TOP, f"{c}_p50")
        else:
            top = out.nlargest(TOP, f"{c}_p50")
        print(f"\n{c.upper()}: median (P10-P90)")
        fmt = "{:.2f}" if c in LOWER_IS_BETTER else "{:.0f}"
        for _, r in top.iterrows():
            print(
                f"  {r['name']} ({r['role']}): {fmt.format(r[f'{c}_p50'])} "
                f"({fmt.format(r[f'{c}_p10'])}-{fmt.format(r[f'{c}_p90'])})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
