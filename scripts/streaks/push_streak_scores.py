#!/usr/bin/env python3
"""Recompute streak scores locally and push them to remote Upstash.

Render never runs the streaks pipeline (duckdb is a [dev] extra and
``streaks.duckdb`` is gitignored), so ``CacheKey.STREAK_SCORES`` in
Upstash is authored exclusively from a developer machine. The deployed
dashboard just reads that one cache entry.

This is the standalone version of the streak step inside the season
refresh (:meth:`RefreshRun._compute_streaks`): it runs the same
``compute_streak_report`` -> ``serialize_report`` -> write-local +
push-remote sequence, without the rest of the (heavy) refresh. Use it
to repair remote streak data out-of-band -- e.g. after a test run with
``RENDER=true`` against the real ``.env`` creds clobbers the prod key
with a fixture payload.

Upstash credentials come from the environment or ``.env`` (the
``kv_store`` dotenv loader populates ``os.environ`` via setdefault).

Usage::

    python scripts/streaks/push_streak_scores.py
    python scripts/streaks/push_streak_scores.py --skip-fetch   # use cached DB
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger("streaks.push_streak_scores")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute streak scores locally and push to remote Upstash."
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the game-log + Statcast pull (use cached DB data).",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Refit models even if model_fits is recent (default: reuse when <=14 days).",
    )
    args = parser.parse_args(argv)

    # Windows stdout is cp1252; the report objects can carry non-ASCII
    # player names. Reconfigure so a stray glyph in a log line can't crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Ensure .env Upstash creds land in os.environ before the remote-push
    # guard reads them (it checks os.environ directly, before any kv build).
    from fantasy_baseball.data import kv_store

    kv_store._load_dotenv_if_present()

    import os

    if not (
        os.environ.get("UPSTASH_REDIS_REST_URL") and os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    ):
        logger.error(
            "Upstash creds not found in environment or .env "
            "(UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN). Cannot push to remote."
        )
        return 1

    from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.streaks.dashboard import serialize_report
    from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection
    from fantasy_baseball.streaks.pipeline import compute_streak_report
    from fantasy_baseball.web.refresh_pipeline import _push_streak_scores_to_remote
    from fantasy_baseball.web.season_data import write_cache

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    logger.info(
        "Computing streak scores for league %d (%s), season %d",
        config.league_id,
        config.team_name,
        config.season_year,
    )

    session = get_yahoo_session()
    league = get_league(session, config.league_id, config.game_code)

    conn = get_connection(DEFAULT_DB_PATH)
    try:
        report = compute_streak_report(
            conn,
            league=league,
            team_name=config.team_name,
            league_id=config.league_id,
            projections_root=PROJECT_ROOT / "data" / "projections",
            scoring_season=config.season_year,
            top_n_fas=50,  # match the dashboard refresh (RefreshRun._compute_streaks).
            force_refit=args.force_refit,
            skip_fetch=args.skip_fetch,
        )
    finally:
        conn.close()

    payload = serialize_report(report)

    # Mirror _compute_streaks: write local cache (RENDER unset -> SQLite) and
    # push the same envelope to remote Upstash so Render reads fresh data.
    write_cache(CacheKey.STREAK_SCORES, payload)
    _push_streak_scores_to_remote(payload)

    logger.info(
        "Pushed streak scores: report_date=%s, %d roster row(s), %d FA row(s)",
        report.report_date.isoformat(),
        len(report.roster_rows),
        len(report.fa_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
