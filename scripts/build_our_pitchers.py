"""Write our pitcher projection for a season, as the draft board blends it
(pitcher_ros/final.py ``board_rates``): the --run's preseason rates (K, W, SV per batter
faced; ERA, WHIP), starters' age-corrected (pitcher_ros.aging, #503), fit on the seasons
before --season with --aging-run's held-out projections and role from --pt-system's
preseason files (GS / G; the same system's --season file gives this season's roles).

Writes data/pitcher_ros/ours-<season>.parquet (player_id = MLBAM id) and a .json beside
it saying what made it. Rebuild the draft board after (python
scripts/run_draft_dashboard.py --rebuild-board).

    python scripts/build_our_pitchers.py --season 2027 --run proj-2027-sv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest_hitter_ranges import pt_file
from backtest_pitcher_aging import RUNS, aging_rows, births

from fantasy_baseball.draft.our_pitchers import RATES
from fantasy_baseball.draft.our_players import ours_path
from fantasy_baseball.pitcher_ros import aging
from fantasy_baseball.pt_model import range_inputs as ri

FIRST_AGING_SEASON = 2022


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--run", required=True, help="the pitcher run with --season's predictions")
    parser.add_argument("--aging-run", default="035-saves-net")
    parser.add_argument("--pt-system", default="steamer", help="role (GS / G) source")
    parser.add_argument("--aging-seasons", type=int, nargs="+")
    args = parser.parse_args()
    seasons = sorted(args.aging_seasons or range(FIRST_AGING_SEASON, args.season))
    if len(seasons) < 2 or seasons[-1] >= args.season:
        parser.error(f"--aging-seasons: want 2+ seasons, all before --season {args.season}")

    rates = ri.load_pitcher_rates(RUNS / args.run)
    if args.season not in rates.index.get_level_values("season"):
        parser.error(f"{args.run} has no {args.season} preseason rows")
    ours = rates.xs(args.season, level="season").rename(columns=lambda c: c.removeprefix("p_"))
    birth = births()
    fit = aging.fit_rates(aging_rows(seasons, args.aging_run, args.pt_system, birth))
    pt = ri.load_pitcher_pt(pt_file(args.pt_system, args.season, "pitchers"))
    aged = aging.adjust_rates(
        ours[list(RATES)], ri.season_age(birth, args.season), aging.role(pt["gs_share"]), fit
    )
    path = ours_path("pitcher", args.season)
    aged.rename_axis("player_id").reset_index().to_parquet(path, index=False)
    meta = {
        "season": args.season,
        "run": args.run,
        "aging": {
            "run": args.aging_run,
            "seasons": seasons,
            "role_from": args.pt_system,
            "fit": aging.describe(fit, sorted(aging.CORRECTED)),
        },
        "pitchers": len(aged),
        "made_by": "scripts/build_our_pitchers.py",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path} ({len(aged)} pitchers)")
    print("\n".join("  " + x for x in meta["aging"]["fit"]))  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
