"""Fetch and derive season-to-date true-talent stats for keeper ranking.

Pulls the raw Savant and Baseball Reference leaderboards, derives the skill
rates, and writes two CSVs indexed by MLBAM id:

    hitters:  pa, barrel_pct, barrel_pa_pct, xwoba, xba, wrc_plus
    pitchers: ip, era_minus, fip, k_pct, swstr_pct, whiff_pct, csw_pct

Raw pulls are cached per-year under the output directory, so a re-run costs
nothing until you delete them -- pass --refresh to force a new fetch. The first
run is dominated by the pitch-level Statcast pull behind CSW% (~1 minute).

Park factors come from the FanGraphs-coded `Team` column in the most recent ROS
projection snapshot, which is the only MLBAM->team bridge in the repo; BBRef's
own `Tm` is an ambiguous city name. Players absent from that snapshot fall back
to neutral. Pass --no-park to skip the adjustment entirely.

Because that bridge is a snapshot of CURRENT teams, a traded player has his whole
season corrected by his new park -- a hitter who spent half a year in Coors before
a July move to Seattle is charged Seattle's factor throughout. Rankings that turn
on park-extreme mid-season trades should not lean on `wrc_plus`/`era_minus` alone.

Usage:
    python scripts/fetch_keeper_skills.py
    python scripts/fetch_keeper_skills.py --year 2026 --refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.park_factors import get_park_factor
from fantasy_baseball.data.ros_pipeline import latest_ros_snapshot
from fantasy_baseball.keepers.actuals import index_by_mlbam
from fantasy_baseball.keepers.bref import (
    fetch_bref_batting,
    fetch_bref_pitching,
)
from fantasy_baseball.keepers.savant import (
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_pitch_mix,
)
from fantasy_baseball.keepers.skills import (
    normalize_hitter_skills,
    normalize_pitcher_skills,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"

# PARK_FACTORS carries `ops` and `k`; there is no runs factor, so the pitcher
# side uses `ops` as a proxy for the same inflated offensive environment. See
# that module's docstring for what this costs a quantitative consumer.
_PARK_STAT = "ops"


def _first_per_id(frame: pd.DataFrame, id_col: str, column: str) -> pd.Series:
    """`column` keyed by MLBAM id, first row winning a duplicate."""
    rows = index_by_mlbam(frame, id_col)
    return rows.loc[~rows.index.duplicated(keep="first"), column]


def build_park_factors(year: int, kind: str) -> pd.Series | None:
    """Map MLBAM id -> park multiplier via the newest ROS projection snapshot.

    Returns None when no snapshot exists, which leaves the adjustment neutral
    rather than silently dropping every player from the join.

    All five systems agree on `Team` but cover different player sets -- on the
    2026-07-27 pitcher snapshot their union reaches 760 of 761 join targets where
    steamer alone reaches 720 -- so every file is read for coverage, not
    consensus, and the first row per id wins. A `usecols` mismatch means the file
    is not a projection export; skip it rather than fail, as `summary.crosswalk`
    does.
    """
    newest = latest_ros_snapshot(PROJECTIONS_DIR, year)
    if newest is None:
        return None
    snapshot = newest[0]
    frames = []
    for path in sorted(snapshot.glob(f"*-{kind}.csv")):
        try:
            frames.append(pd.read_csv(path, encoding="utf-8-sig", usecols=["MLBAMID", "Team"]))
        except (ValueError, FileNotFoundError):
            continue
    if not frames:
        return None
    teams = _first_per_id(pd.concat(frames, ignore_index=True), "MLBAMID", "Team")
    factors = teams.map(lambda team: get_park_factor(str(team), _PARK_STAT))
    return factors.rename("park_factor")


def with_names(skills: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Prepend the player name from a BBRef frame, so the output is readable and
    joinable to the board without a second lookup. Unmatched ids keep a blank.

    `bref` repairs the double-encoded accents at ingest, so no name handling is
    needed here.
    """
    names = _first_per_id(source, "mlbID", "Name").astype(str)
    out = skills.copy()
    out.insert(0, "name", names.reindex(out.index).fillna(""))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, help="defaults to config season_year")
    parser.add_argument("--out", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached raw pulls and refetch"
    )
    parser.add_argument("--no-park", action="store_true", help="skip park adjustment")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(CONFIG_PATH)
    if args.year is None:
        args.year = config.season_year

    raw_dir = args.out / f"raw_{args.year}"
    # --refresh treats every cache as stale rather than deleting it up front: a
    # failed pull then leaves the good cache in place instead of having already
    # destroyed it (and the ~1-minute Statcast pull with it). An empty dict
    # leaves each fetcher on its own default.
    refresh = {"max_age": timedelta(0)} if args.refresh else {}

    print(f"Fetching {args.year} raw leaderboards (cache: {raw_dir})...")
    expected = fetch_batter_expected(raw_dir, args.year, **refresh)
    barrels = fetch_batter_barrels(raw_dir, args.year, **refresh)
    batting = fetch_bref_batting(raw_dir, args.year, **refresh)
    pitching = fetch_bref_pitching(raw_dir, args.year, **refresh)
    print("  pulling pitch-level Statcast for CSW% (~1 min on a cold cache)...")
    pitch_mix = fetch_pitcher_pitch_mix(raw_dir, args.year, **refresh)
    print(
        f"  savant expected={len(expected)} barrels={len(barrels)} "
        f"pitch_mix={len(pitch_mix)} bref batting={len(batting)} pitching={len(pitching)}"
    )

    hitter_park = None if args.no_park else build_park_factors(args.year, "hitters")
    pitcher_park = None if args.no_park else build_park_factors(args.year, "pitchers")
    # Check BOTH: a snapshot can land one side and 403 the other, which would
    # park-correct wrc_plus while leaving every era_minus raw, with no signal.
    if not args.no_park:
        for label, factors in (("hitters", hitter_park), ("pitchers", pitcher_park)):
            if factors is None:
                print(f"  WARNING: no {label} team bridge; that side is park-neutral")

    hitters = with_names(
        normalize_hitter_skills(expected, barrels, batting, park_factor=hitter_park), batting
    )
    pitchers = with_names(
        normalize_pitcher_skills(pitching, pitch_mix, park_factor=pitcher_park), pitching
    )

    args.out.mkdir(parents=True, exist_ok=True)
    hitter_path = args.out / f"hitter_skills_{args.year}.csv"
    pitcher_path = args.out / f"pitcher_skills_{args.year}.csv"
    hitters.to_csv(hitter_path)
    pitchers.to_csv(pitcher_path)

    print(f"\nWrote {len(hitters)} hitters -> {hitter_path}")
    print(hitters.loc[hitters["pa"] >= 300].describe().round(3).to_string())
    print(f"\nWrote {len(pitchers)} pitchers -> {pitcher_path}")
    print(pitchers.loc[pitchers["ip"] >= 50].describe().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
