"""Write our hitter projection for a season, as the draft board blends it
(hitter_ros/final.py ``board_rates``, #459): the --run seeds' preseason rates (R, HR,
RBI, SB per PA; AVG), averaged over the seeds.

Writes data/hitter_ros/ours-<season>.parquet (player_id = MLBAM id) and a .json beside
it saying what made it. Rebuild the draft board after (python
scripts/run_draft_dashboard.py --rebuild-board).

    python scripts/build_our_hitters.py --season 2027 --run proj2027seq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.draft.our_players import ours_path
from fantasy_baseball.hitter_ros.features import TARGETS
from fantasy_baseball.pt_model import range_inputs as ri

RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--run", required=True, help="seeds at data/hitter_ros/runs/<run>-s<N>")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    args = parser.parse_args()
    rates = ri.load_rates(RUNS / f"{args.run}-s{s}" for s in args.seeds)
    if args.season not in rates.index.get_level_values("season"):
        parser.error(f"{args.run} has no {args.season} preseason rows")
    ours = rates.xs(args.season, level="season").rename(columns=lambda c: c.removeprefix("p_"))
    ours = ours[list(TARGETS)]
    path = ours_path("hitter", args.season)
    ours.rename_axis("player_id").reset_index().to_parquet(path, index=False)
    meta = {
        "season": args.season,
        "run": args.run,
        "seeds": args.seeds,
        "hitters": len(ours),
        "made_by": "scripts/build_our_hitters.py",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path} ({len(ours)} hitters, {args.run} seeds {args.seeds})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
