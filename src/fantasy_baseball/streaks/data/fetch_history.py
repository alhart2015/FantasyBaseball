"""Orchestrator: fetch one season of game logs + Statcast PA data into DuckDB.

Idempotent: skips player-seasons already present in `hitter_games`. Skips
dates already present in `hitter_statcast_pa` by adjusting the requested
Statcast window.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import duckdb
import requests

from fantasy_baseball.streaks.data.game_logs import (
    fetch_hitter_season_game_logs,
    pa_identity_gap,
)
from fantasy_baseball.streaks.data.load import (
    existing_player_seasons,
    existing_statcast_dates,
    upsert_hitter_games,
    upsert_statcast_pa,
)
from fantasy_baseball.streaks.data.qualified_hitters import fetch_qualified_hitters
from fantasy_baseball.streaks.data.statcast import fetch_statcast_pa_for_date_range

logger = logging.getLogger(__name__)

# MLB regular season + early postseason; safe envelope for Statcast pulls.
_SEASON_START_MMDD = (3, 15)
_SEASON_END_MMDD = (11, 15)


def fetch_season(
    season: int,
    conn: duckdb.DuckDBPyConnection,
    min_pa: int = 150,
    *,
    force_statcast: bool = False,
) -> dict[str, Any]:
    """Fetch and load one season of game logs + Statcast PA data.

    Returns a summary dict with row counts and a ``pa_identity_violations``
    count (games where ``pa != ab + bb + hbp + sf + sh + ci`` — logged at
    WARNING but never raised, so a single bad row can't kill a 2-hour fetch).

    When ``force_statcast`` is True, re-fetches Statcast PAs for the season
    even if dates are already loaded. ``upsert_statcast_pa`` uses INSERT OR
    REPLACE so existing PK rows get updated in place — used to backfill new
    columns (e.g. ``launch_speed_angle`` after the Phase 5 migration).
    """
    qualified = fetch_qualified_hitters(season=season, min_pa=min_pa)
    logger.info("Season %s: %d qualified hitters", season, len(qualified))

    already = existing_player_seasons(conn)
    to_fetch = [q for q in qualified if (q.player_id, season) not in already]
    logger.info("Season %s: %d new players to fetch", season, len(to_fetch))

    game_log_rows = 0
    pa_identity_violations = 0
    for i, player in enumerate(to_fetch):
        try:
            games = fetch_hitter_season_game_logs(
                player_id=player.player_id,
                name=player.name,
                team=player.team,
                season=season,
            )
            for g in games:
                gap = pa_identity_gap(g)
                if gap != 0:
                    pa_identity_violations += 1
                    logger.warning(
                        "PA identity gap of %d for %s (%s) on %s game_pk=%d",
                        gap,
                        g.name,
                        g.player_id,
                        g.date.isoformat(),
                        g.game_pk,
                    )
            upsert_hitter_games(conn, games)
            game_log_rows += len(games)
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.warning(
                "Game log fetch failed for %s (%s): %s",
                player.name,
                player.player_id,
                e,
            )
        if (i + 1) % 25 == 0:
            logger.info("  fetched %d/%d game logs", i + 1, len(to_fetch))

    start = date(season, *_SEASON_START_MMDD)
    end = date(season, *_SEASON_END_MMDD)
    loaded_dates = existing_statcast_dates(conn)
    # If we've already loaded any dates in this season, skip Statcast (a partial
    # load is rare; we treat it as an all-or-nothing per-season pull for simplicity).
    statcast_rows = 0
    season_dates_loaded = {d for d in loaded_dates if d.year == season}
    if force_statcast or not season_dates_loaded:
        if force_statcast and season_dates_loaded:
            logger.info(
                "Season %s: --force-statcast set; re-fetching despite %d loaded dates",
                season,
                len(season_dates_loaded),
            )
        statcast_pa = fetch_statcast_pa_for_date_range(start, end)
        upsert_statcast_pa(conn, statcast_pa)
        statcast_rows = len(statcast_pa)
    else:
        logger.info(
            "Season %s: %d Statcast dates already loaded, skipping Statcast pull",
            season,
            len(season_dates_loaded),
        )

    return {
        "season": season,
        "players_attempted": len(to_fetch),
        "game_log_rows": game_log_rows,
        "pa_identity_violations": pa_identity_violations,
        "statcast_rows": statcast_rows,
    }
