"""Fetch prospect pedigree into the pitch-data store (#433): MLB Pipeline rankings and
June draft picks (see pitch_data/pedigree.py). Seasons and years on disk are skipped
unless --refresh (or, for rankings, the season was fetched while current and is over).

Usage:
    python scripts/fetch_pedigree.py
    python scripts/fetch_pedigree.py --rankings 2026 2026 --refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.pitch_data.pedigree import (
    FIRST_RANKINGS_SEASON,
    fetch_draft_year,
    fetch_rankings_season,
)
from fantasy_baseball.utils.time_utils import local_today

DEFAULT_ROOT = PROJECT_ROOT / "data" / "pitch_data"
# The oldest hitters in the table were drafted in the 1990s.
FIRST_DRAFT_YEAR = 1990


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    this_year = local_today().year
    parser.add_argument(
        "--rankings", nargs=2, type=int, default=[FIRST_RANKINGS_SEASON, this_year],
        metavar=("FIRST", "LAST"), help="seasons of prospect lists",
    )  # fmt: skip
    parser.add_argument(
        "--draft", nargs=2, type=int, default=[FIRST_DRAFT_YEAR, this_year],
        metavar=("FIRST", "LAST"), help="draft years",
    )  # fmt: skip
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--refresh", action="store_true", help="re-fetch what is on disk")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    failures = []
    for season in range(args.rankings[0], args.rankings[1] + 1):
        try:
            n = fetch_rankings_season(args.root, season, refresh=args.refresh)
            logging.info("rankings %s: %s players", season, n)
        except Exception:
            logging.exception("rankings %s: failed", season)
            failures.append(f"rankings {season}")
    for year in range(args.draft[0], args.draft[1] + 1):
        try:
            n = fetch_draft_year(args.root, year, refresh=args.refresh)
            logging.info("draft %s: %s picks", year, n)
        except Exception:
            logging.exception("draft %s: failed", year)
            failures.append(f"draft {year}")
    if failures:
        logging.warning("incomplete, re-run to retry: %s", "; ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
