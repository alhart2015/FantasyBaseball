"""Build the historical playing-time panel for the PT projection model (#291, #290).

Fetches MLB Stats API season leaderboards for a year range, plus the per-player
covariates (birth date, primary position, debut date), and writes the hitter panel:
one row per (player, season) over each player's observed career span, with absent
seasons represented explicitly as NaN rather than zero.

Pitching seasons are fetched and cached too -- the pull is the same request and
paying for it now means the pitcher model (#290 step 3) starts with warm data -- but
only the hitter panel is shaped here.

Raw pulls land in `data/cache/keeper_skills/` alongside the fielding pulls
`scripts/keeper_rankings.py` already caches there, because that is where
`fetch_mlb_season` writes; one shared raw cache beats a better-named second copy.
Completed seasons never expire (`fetch_or_cache` has no default max_age), so a
re-run is free -- but the LIVE season's cache is a point-in-time snapshot and needs
--refresh to move.

Usage:
    python scripts/build_pt_panel.py
    python scripts/build_pt_panel.py --start 2010 --end 2026
    python scripts/build_pt_panel.py --refresh          # re-fetch the live season
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.keepers.mlb_stats import fetch_mlb_people, fetch_mlb_season
from fantasy_baseball.pt_model.panel import build_hitter_panel

RAW_DIR = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
PANEL_DIR = PROJECT_ROOT / "data" / "playing_time"
DEFAULT_START = 2010

logger = logging.getLogger(__name__)


def _live_seasons(years: range) -> list[int]:
    """Years that may still be in progress.

    Anything at or past the current calendar year. Deliberately conservative: run in
    November and the finished season is still flagged partial, which costs one season
    of training data and cannot cause a live season to be trained on as complete.
    """
    return [y for y in years if y >= dt.date.today().year]


def _fetch_seasons(years: range, group: str, *, refresh: list[int]) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for year in years:
        path = RAW_DIR / f"mlb_{group}_{year}.csv"
        if year in refresh and path.exists():
            logger.info("refresh: dropping cached %s", path.name)
            path.unlink()
        frames[year] = fetch_mlb_season(RAW_DIR, year, group)
        logger.info("%s %d: %d rows", group, year, len(frames[year]))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=dt.date.today().year)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-fetch the live season(s); completed seasons are immutable and kept",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.end < args.start:
        parser.error(f"--end {args.end} precedes --start {args.start}")

    years = range(args.start, args.end + 1)
    partial = _live_seasons(years)
    refresh = partial if args.refresh else []

    hitting = _fetch_seasons(years, "hitting", refresh=refresh)
    _fetch_seasons(years, "pitching", refresh=refresh)

    ids = sorted({int(i) for f in hitting.values() for i in f["player.id"].dropna()})
    logger.info("distinct hitters %d-%d: %d", args.start, args.end, len(ids))
    people = fetch_mlb_people(RAW_DIR, ids, f"{args.start}_{args.end}")

    panel = build_hitter_panel(hitting, people, partial_seasons=partial)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    out = PANEL_DIR / f"hitter_pt_panel_{args.start}_{args.end}.csv"
    panel.to_csv(out, index=False)

    observed = int(panel["observed"].sum())
    logger.info("")
    logger.info("wrote %s", out.relative_to(PROJECT_ROOT))
    logger.info("  rows          %d", len(panel))
    logger.info("  observed      %d", observed)
    logger.info("  absent (NaN)  %d", len(panel) - observed)
    logger.info("  players       %d", panel["mlbam_id"].nunique())
    logger.info("  partial-year  %d", int(panel["partial_season"].sum()))
    logger.info("  no birth date %d", int(panel["age"].isna().sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
