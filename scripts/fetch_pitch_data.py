"""Backfill / refresh the raw pitch-level store for the batted-ball model (#399, #400).

Pulls every Statcast pitch, the Savant sprint-speed leaderboard, and MLB box-score
batting order into parquet under data/pitch_data/. Safe to stop and re-run: finished
weekly chunks and finished seasons are skipped, so a re-run only fetches what is
missing or still changing. A full 2015-2026 backfill takes several hours, almost all
of it the pitch pull.

Usage:
    python scripts/fetch_pitch_data.py --start 2015 --end 2026
    python scripts/fetch_pitch_data.py --start 2026 --end 2026 --only lineups sprint
    python scripts/fetch_pitch_data.py --summary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.data.mlb_game_logs import _fetch_season_games
from fantasy_baseball.pitch_data.store import (
    connect,
    fetch_lineups_season,
    fetch_pitches_season,
    fetch_sprint_speed_season,
)
from fantasy_baseball.utils.time_utils import local_today

DEFAULT_ROOT = PROJECT_ROOT / "data" / "pitch_data"
KINDS = ("pitches", "lineups", "sprint")


def _dir_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*.parquet")) / 1e6


def print_summary(root: Path) -> None:
    """Per-season row counts and on-disk size."""
    conn = connect(root)
    views = {r[0] for r in conn.execute("SELECT view_name FROM duckdb_views()").fetchall()}
    if "pitches" in views:
        print("season  pitches   game_pks  bip  bip_with_ev   disk_MB")
        for season, n, games, bip, ev in conn.execute(
            """
            SELECT season, count(*), count(DISTINCT game_pk),
                   count(*) FILTER (WHERE type = 'X'),
                   count(launch_speed) FILTER (WHERE type = 'X')
            FROM pitches WHERE game_type = 'R' GROUP BY season ORDER BY season
            """
        ).fetchall():
            mb = _dir_mb(root / "pitches" / f"season={season}")
            pct = ev / bip if bip else 0.0
            print(f"{season}  {n:>8}  {games:>8}  {bip:>6}  {pct:>10.1%}  {mb:>8.1f}")
    if "lineups" in views:
        print("\nseason  lineup_rows  games  starters_per_game")
        for season, n, games, starters in conn.execute(
            """
            SELECT year(CAST(game_date AS DATE)), count(*), count(DISTINCT game_pk),
                   count(*) FILTER (WHERE sub_index = 0)
            FROM lineups GROUP BY 1 ORDER BY 1
            """
        ).fetchall():
            print(f"{season}  {n:>11}  {games:>5}  {starters / games:>17.2f}")
    if "sprint_speed" in views:
        print("\nseason  sprint_rows")
        for season, n in conn.execute(
            "SELECT season, count(*) FROM sprint_speed GROUP BY 1 ORDER BY 1"
        ).fetchall():
            print(f"{season}  {n:>11}")
    print(f"\ntotal on disk: {_dir_mb(root):.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=local_today().year)
    parser.add_argument("--only", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary", action="store_true", help="print what is on disk and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.summary:
        print_summary(args.root)
        return 0

    today = local_today()
    failures: list[str] = []
    for season in range(args.start, args.end + 1):
        # One season's failure must not stop the rest of a multi-season backfill;
        # everything that failed is retried by the next run.
        try:
            games = _fetch_season_games(season)
        except Exception:
            logging.exception("schedule %s: fetch failed", season)
            failures.append(f"{season} schedule")
            continue
        if "lineups" in args.only:
            try:
                fetch_lineups_season(args.root, season, games=games)
            except Exception:
                logging.exception("lineups %s: failed", season)
                failures.append(f"{season} lineups")
        if "sprint" in args.only:
            try:
                fetch_sprint_speed_season(args.root, season, games=games)
            except Exception:
                logging.exception("sprint speed %s: failed", season)
                failures.append(f"{season} sprint speed")
        if "pitches" in args.only:
            result = fetch_pitches_season(args.root, season, today, games=games)
            logging.info("pitches %s: %s", season, result)
            if result["incomplete"]:
                failures.append(f"{season} pitches ({result['incomplete']} chunks)")

    if failures:
        logging.warning("incomplete, re-run to retry: %s", "; ".join(failures))
    print_summary(args.root)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
