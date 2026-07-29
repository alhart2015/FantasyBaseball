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

Usage:
    python scripts/fetch_keeper_skills.py
    python scripts/fetch_keeper_skills.py --year 2026 --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.park_factors import get_park_factor
from fantasy_baseball.data.ros_pipeline import parse_snapshot_date
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


def latest_ros_dir(year: int) -> Path | None:
    """Most recent ROS projection snapshot directory for `year`, if any.

    Selects by PARSED date, not by name: a raw string sort would let an undatable
    helper dir sort above the dated ones and shadow a fresh snapshot. Shares
    `parse_snapshot_date` with the refresh pipeline so the two cannot disagree
    about which snapshot is newest.
    """
    ros_root = PROJECTIONS_DIR / str(year) / "rest_of_season"
    if not ros_root.is_dir():
        return None
    dated = [
        (p, d)
        for p in ros_root.iterdir()
        if p.is_dir() and (d := parse_snapshot_date(p.name)) is not None
    ]
    return max(dated, key=lambda pair: pair[1])[0] if dated else None


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
    snapshot = latest_ros_dir(year)
    if snapshot is None:
        return None
    frames = []
    for path in sorted(snapshot.glob(f"*-{kind}.csv")):
        try:
            frames.append(pd.read_csv(path, encoding="utf-8-sig", usecols=["MLBAMID", "Team"]))
        except (ValueError, FileNotFoundError):
            continue
    if not frames:
        return None
    rows = index_by_mlbam(pd.concat(frames, ignore_index=True), "MLBAMID")
    teams = rows.loc[~rows.index.duplicated(keep="first"), "Team"]
    return teams.map(lambda team: get_park_factor(str(team), _PARK_STAT)).rename("park_factor")


def with_names(skills: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Prepend the player name from a BBRef frame, so the output is readable and
    joinable to the board without a second lookup. Unmatched ids keep a blank.

    `bref` repairs the double-encoded accents at ingest, so no name handling is
    needed here.
    """
    rows = index_by_mlbam(source, "mlbID")
    names = rows.loc[~rows.index.duplicated(keep="first"), "Name"].astype(str)
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
    args.year = args.year or load_config(CONFIG_PATH).season_year

    raw_dir = args.out / f"raw_{args.year}"
    if args.refresh and raw_dir.exists():
        for stale in raw_dir.glob("*.csv"):
            stale.unlink()

    print(f"Fetching {args.year} raw leaderboards (cache: {raw_dir})...")
    expected = fetch_batter_expected(raw_dir, args.year)
    barrels = fetch_batter_barrels(raw_dir, args.year)
    batting = fetch_bref_batting(raw_dir, args.year)
    pitching = fetch_bref_pitching(raw_dir, args.year)
    print("  pulling pitch-level Statcast for CSW% (~1 min on a cold cache)...")
    pitch_mix = fetch_pitcher_pitch_mix(raw_dir, args.year)
    print(
        f"  savant expected={len(expected)} barrels={len(barrels)} "
        f"pitch_mix={len(pitch_mix)} bref batting={len(batting)} pitching={len(pitching)}"
    )

    hitter_park = None if args.no_park else build_park_factors(args.year, "hitters")
    pitcher_park = None if args.no_park else build_park_factors(args.year, "pitchers")
    if not args.no_park and hitter_park is None:
        print("  WARNING: no ROS snapshot found; park adjustment is neutral")

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
