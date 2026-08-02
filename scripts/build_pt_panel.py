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
from fantasy_baseball.pt_model.panel import build_hitter_panel, build_pitcher_panel

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


# A regular season is comfortably finished by this date, so a cache file written
# before it cannot hold a complete season.
_SEASON_OVER = (11, 1)
_OPENING_DAY_MONTH = 4


def _default_end(today: dt.date | None = None) -> int:
    """Last season that could plausibly have data.

    January-to-March the current calendar year has no leaderboard at all, and keeper
    work happens precisely in that window -- so defaulting to `today().year` made the
    documented bare invocation fail exactly when the tool matters most. Before opening
    day, fall back to the previous season.
    """
    today = today or dt.date.today()
    return today.year if today.month >= _OPENING_DAY_MONTH else today.year - 1


def _captured_while_live(year: int, written_on: dt.date) -> bool:
    """Was a cache file for `year` written before that season finished?

    This is the question the calendar cannot answer on its own. A season fetched in
    August is a two-thirds snapshot, but by the NEXT run it is no longer "live", so a
    check against today's date would serve it from cache and train it as a completed
    year -- a fabricated league-wide collapse season that deflates every volume
    coefficient. Comparing the file's own vintage to the season's end is the only
    signal that survives into later runs.
    """
    return written_on < dt.date(year, *_SEASON_OVER)


def _stale_live_caches(
    years: range, groups: tuple[str, ...] = ("hitting", "pitching")
) -> list[int]:
    """Seasons whose cached leaderboard was captured before that season ended."""
    stale = set()
    for year in years:
        for group in groups:
            path = RAW_DIR / f"mlb_{group}_{year}.csv"
            if not path.exists():
                continue
            written = dt.date.fromtimestamp(path.stat().st_mtime)
            if _captured_while_live(year, written):
                stale.add(year)
    return sorted(stale)


def _fetch_seasons(years: range, group: str, *, refresh: list[int]) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for year in years:
        path = RAW_DIR / f"mlb_{group}_{year}.csv"
        keep = None
        if year in refresh and path.exists():
            # Move aside rather than delete: the fetch can fail (statsapi down, rate
            # limited) and live seasons now refresh on EVERY run, so unlinking first
            # would destroy the only local snapshot and leave the panel unbuildable.
            # `fetch_keeper_skills.py` takes the same precaution for the same reason.
            keep = path.with_suffix(".csv.prev")
            logger.info("refresh: setting aside cached %s", path.name)
            path.replace(keep)
        try:
            frames[year] = fetch_mlb_season(RAW_DIR, year, group)
        except Exception:
            if keep is not None:
                logger.warning("fetch failed for %s %d; restoring the cached copy", group, year)
                keep.replace(path)
            raise
        if keep is not None:
            keep.unlink(missing_ok=True)
        logger.info("%s %d: %d rows", group, year, len(frames[year]))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=_default_end())
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

    # A season cached WHILE LIVE is a partial snapshot, and by the next run it is no
    # longer in `partial` -- so it would be served from cache and trained as a COMPLETE
    # year. The previous guard unioned in `_live_seasons`, which is what `partial`
    # already is, so it was a no-op for exactly the case it described. Ask the cache
    # files their own vintage instead.
    stale = _stale_live_caches(years)
    if stale:
        logger.info("re-fetching %s: cached while the season was still live", stale)
    refresh = sorted(set(refresh) | set(stale))
    hitting = _fetch_seasons(years, "hitting", refresh=refresh)
    pitching = _fetch_seasons(years, "pitching", refresh=refresh)

    # Ids from BOTH leaderboards. Taking them from hitting alone left 13% of pitcher
    # seasons with no birth date and therefore no age -- under the universal DH most
    # pitchers never take a plate appearance, so they simply are not in the hitting
    # pull. Age is a real feature of the curve, so those rows predicted NaN.
    ids = sorted(
        {int(i) for f in hitting.values() for i in f["player.id"].dropna()}
        | {int(i) for f in pitching.values() for i in f["player.id"].dropna()}
    )
    logger.info("distinct players %d-%d: %d", args.start, args.end, len(ids))
    # --refresh must INVALIDATE the people cache, not key around it. A static suffix
    # only re-fetched once and then froze a second copy, so every later refresh missed
    # players who had debuted since -- leaving them without a birth date, hence without
    # an age, hence unscoreable. Unlinking mirrors what _fetch_seasons does.
    tag = f"all_{args.start}_{args.end}"
    # The people cache must follow the SAME invalidation rule as the leaderboards. Tying
    # it to --refresh alone meant a bare rerun re-fetched the live season, discovered
    # players who had debuted since, then served them from a frozen people file with no
    # birth date -- hence no age, hence a NaN forecast. That is the ageless-player bug
    # this branch has now fixed twice; here it would have returned on the DEFAULT path.
    if refresh:
        (RAW_DIR / f"mlb_people_{tag}.csv").unlink(missing_ok=True)
    people = fetch_mlb_people(RAW_DIR, ids, tag)

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for label, seasons, builder in (
        ("hitter", hitting, build_hitter_panel),
        ("pitcher", pitching, build_pitcher_panel),
    ):
        panel = builder(seasons, people, partial_seasons=partial)
        out = PANEL_DIR / f"{label}_pt_panel_{args.start}_{args.end}.csv"
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
