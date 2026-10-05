"""Print the minor-league level factors (#435): what a line at each level is worth in MLB.

Grades every level against MLB with hitter_ros/milb_grade.py and prints markdown tables
for the issue: the factors over all seasons, the factors a backtest of each test season
would use (only the seasons before it), and the spread of league environments.

Usage:
    python scripts/grade_milb_levels.py
    python scripts/grade_milb_levels.py --test-seasons 2016 2022 2026 --window 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.backtest import to_markdown
from fantasy_baseball.hitter_ros.milb_grade import (
    factor_table,
    league_rates,
    level_factors,
    load_milb_lines,
    load_mlb_lines,
    player_levels,
    relative_lines,
)
from fantasy_baseball.pitch_data.milb import MILB_LEVELS
from fantasy_baseball.pitch_data.store import connect

STORE = PROJECT_ROOT / "data" / "pitch_data"


def _pairs_as_n(factors: pd.DataFrame) -> pd.DataFrame:
    """Level names for the index; the pair count as ``n`` so it prints as a whole number."""
    return factor_table(factors).rename(columns={"pairs": "n"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-seasons", type=int, nargs="+", default=[2016, 2022, 2026])
    parser.add_argument("--window", type=int, default=7, help="seasons before the test season")
    args = parser.parse_args()

    conn = connect(STORE)
    lines = pd.concat([load_milb_lines(conn), load_mlb_lines(conn)], ignore_index=True)
    levels = player_levels(relative_lines(lines))
    first, last = int(lines["season"].min()), int(lines["season"].max())

    print("n = player pairs in the level's own step (to MLB for AAA and AA, else up a level)\n")
    print(f"### Factors to MLB, all seasons ({first}-{last})\n")
    print(to_markdown(_pairs_as_n(level_factors(levels, (first, last))), digits=3))
    for test in args.test_seasons:
        window = (test - args.window, test - 1)
        print(f"\n### Factors a {test} backtest uses ({window[0]}-{window[1]})\n")
        print(to_markdown(_pairs_as_n(level_factors(levels, window)), digits=3))

    env = league_rates(lines).reset_index()
    env = env[env["sport_id"] != 1]
    spread = env.groupby("sport_id")[["avg", "hr"]].agg(["min", "max"])
    spread.columns = [f"{s} {agg}" for s, agg in spread.columns]
    print("\n### League environments by level (min / max over league-seasons)\n")
    print(to_markdown(spread.rename(index=MILB_LEVELS), digits=3))

    ages = levels[levels["sport_id"] != 1].groupby("sport_id")["age_vs_level"].describe()
    print("\n### Age vs. level (player age minus the level's PA-weighted mean age)\n")
    print(to_markdown(ages[["count", "mean", "std", "min", "max"]].rename(index=MILB_LEVELS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
