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
import codecs
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.data.park_factors import (
    NEUTRAL_FACTOR,
    PARK_FACTORS,
)
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

DEFAULT_YEAR = 2026
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"

# park_factors keys OPS and K separately; a hitter's rate stats track the OPS
# factor and a pitcher's run prevention tracks it inversely (the same inflated
# offensive environment), so both sides deflate by the same multiplier.
_PARK_STAT = "ops"


def latest_ros_dir(year: int) -> Path | None:
    """Most recent ROS projection snapshot directory for `year`, if any."""
    ros_root = PROJECTIONS_DIR / str(year) / "rest_of_season"
    if not ros_root.is_dir():
        return None
    dated = sorted(d for d in ros_root.iterdir() if d.is_dir())
    return dated[-1] if dated else None


def build_park_factors(year: int, kind: str) -> pd.Series | None:
    """Map MLBAM id -> park multiplier via the newest ROS projection snapshot.

    Returns None when no snapshot exists, which leaves the adjustment neutral
    rather than silently dropping every player from the join.
    """
    snapshot = latest_ros_dir(year)
    if snapshot is None:
        return None
    frames = [
        pd.read_csv(path)
        for path in sorted(snapshot.glob(f"*-{kind}.csv"))
        if path.stat().st_size > 0
    ]
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "MLBAMID" not in combined.columns or "Team" not in combined.columns:
        return None
    rows = combined.loc[combined["MLBAMID"].notna(), ["MLBAMID", "Team"]].copy()
    # Systems disagree on nothing here -- Team is the same across all five -- so
    # the first row per player is as good as any.
    rows = rows.drop_duplicates(subset="MLBAMID", keep="first")
    factors = rows["Team"].map(lambda team: PARK_FACTORS.get(str(team), NEUTRAL_FACTOR)[_PARK_STAT])
    return pd.Series(factors.to_numpy(), index=rows["MLBAMID"].astype(int), name="park_factor")


def unescape_name(name: str) -> str:
    """Repair BBRef's double-encoded accents.

    ~100 of the 1369 names arrive with the UTF-8 bytes spelled out as literal
    backslash escapes -- "Acu\\xc3\\xb1a" as 17 characters, not "Acuna" with a
    tilde. Left alone it survives accent-stripped normalization as a distinct
    name and silently fails to match the board. Anything that does not round-trip
    is returned untouched.
    """
    try:
        return codecs.decode(name, "unicode_escape").encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def with_names(skills: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Prepend the player name from a BBRef frame, so the output is readable and
    joinable to the board without a second lookup. Unmatched ids keep a blank."""
    rows = source.loc[source["mlbID"].notna(), ["mlbID", "Name"]].drop_duplicates(
        subset="mlbID", keep="first"
    )
    names = pd.Series(
        [unescape_name(str(n)) for n in rows["Name"]],
        index=rows["mlbID"].astype(int).to_numpy(),
    )
    out = skills.copy()
    out.insert(0, "name", names.reindex(out.index).fillna(""))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--out", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached raw pulls and refetch"
    )
    parser.add_argument("--no-park", action="store_true", help="skip park adjustment")
    args = parser.parse_args()

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
