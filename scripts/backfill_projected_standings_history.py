#!/usr/bin/env python3
"""One-time backfill of `projected_standings_history`.

For every roster snapshot date in `weekly_rosters_history`, build a
`ProjectedStandings` using TODAY'S per-player full-season projection
applied to the historical roster. Writes one snapshot per date to
`projected_standings_history`.

Math note: today's per-player full_season_projection (= ROS + YTD) is
constant in expectation under a constant-rate assumption, so
"extrapolating today's ROS backwards" is equivalent to applying today's
full_season unchanged. Going forward, real per-day snapshots from the
refresh pipeline will carry projection-drift signal that this backfill
cannot.

Idempotent — safe to re-run; same-date entries are overwritten.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data import kv_store
from fantasy_baseball.data.projections import hydrate_roster_entries
from fantasy_baseball.data.redis_store import (
    write_projected_standings_snapshot,
)
from fantasy_baseball.models.league import League
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.web.season_routes import load_projections

logger = logging.getLogger(__name__)


def main(season_year: int | None = None) -> int:
    """Run the backfill. Returns 0 on success.

    ``season_year`` overrides the config value (test seam).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    year = season_year if season_year is not None else config.season_year

    # Look up get_kv on the module at call time so unittest.mock.patch on
    # ``fantasy_baseball.data.kv_store.get_kv`` takes effect (binding the
    # name at import time would freeze the original function).
    client = kv_store.get_kv()

    logger.info("Loading projections...")
    hitters_proj, pitchers_proj, ros_h, ros_p = load_projections()
    have_ros = not ros_h.empty and not ros_p.empty
    full_h = ros_h if have_ros else None
    full_p = ros_p if have_ros else None

    logger.info("Loading roster + standings history from Redis...")
    league = League.from_redis(year)

    rosters_by_date: dict[str, dict[str, object]] = {}
    for team in league.teams:
        for roster in team.rosters:
            snap_iso = roster.effective_date.isoformat()
            rosters_by_date.setdefault(snap_iso, {})[team.name] = roster

    if not rosters_by_date:
        logger.warning("No roster history found for season %s — nothing to backfill.", year)
        return 0

    logger.info("Building %d projected snapshots...", len(rosters_by_date))
    for snap_iso in sorted(rosters_by_date):
        team_rosters: dict[str, list] = {}
        for team_name, roster in rosters_by_date[snap_iso].items():
            hydrated = hydrate_roster_entries(
                roster,
                hitters_proj,
                pitchers_proj,
                full_hitters_proj=full_h,
                full_pitchers_proj=full_p,
                context=f"backfill:{snap_iso}:{team_name}",
            )
            if hydrated:
                team_rosters[team_name] = hydrated

        if not team_rosters:
            logger.info("  %s — no rosters resolved, skipping", snap_iso)
            continue

        projected = ProjectedStandings.from_rosters(
            team_rosters, effective_date=date.fromisoformat(snap_iso)
        )
        write_projected_standings_snapshot(client, projected)
        logger.info("  %s — wrote %d teams", snap_iso, len(projected.entries))

    logger.info("Backfill complete.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="Override config season_year (default: read from league.yaml).",
    )
    args = parser.parse_args()
    sys.exit(main(season_year=args.season_year))
