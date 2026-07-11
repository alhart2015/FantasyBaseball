"""Send the daily summary email. Run as a Render cron after the morning refresh.

Freshness gate: only sends if the META cache entry was written today (keyed on
the provenance ``_written_at`` timestamp, not the free-text ``last_refresh``
payload) -- else exits non-zero so the cron surfaces "the refresh didn't run".
Writes the standings snapshot only after a successful send.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from fantasy_baseball.config import LeagueConfig, load_config
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.summary.assemble import build_daily_summary, refresh_is_fresh
from fantasy_baseball.summary.render import render_html, render_text, subject_line
from fantasy_baseball.summary.send import send_email
from fantasy_baseball.utils.time_utils import local_today
from fantasy_baseball.web.season_data import read_cache, read_cache_with_meta, write_cache

logger = logging.getLogger(__name__)


def _write_snapshot(written_at: str | None) -> None:
    """Persist the post-send standings snapshot for tomorrow's delta baseline."""
    standings = read_cache(CacheKey.STANDINGS)
    if standings is None:
        logger.warning("no STANDINGS to snapshot; skipping snapshot write")
        return
    write_cache(
        CacheKey.STANDINGS_SNAPSHOT,
        {"written_at": written_at, "standings": standings},
    )


def run_summary(
    config: LeagueConfig,
    projections_root: Path,
    *,
    api_key: str,
    league: object,
    team_key: str,
    today: date | None = None,
) -> int:
    """Freshness-gate, assemble, render, send, snapshot. Returns an exit code."""
    today = today or local_today()

    # Two-part gate in one check: liveness (META present) + freshness (written
    # today). read_cache_with_meta returns (None, {}) on a total miss, so an
    # absent META has no _written_at and fails refresh_is_fresh -- both the
    # "KV empty" and "refresh didn't run today" cases skip the send.
    meta_data, envelope = read_cache_with_meta(CacheKey.META)
    written_at = envelope.get("_written_at") if isinstance(envelope, dict) else None
    if meta_data is None or not refresh_is_fresh(written_at, today):
        logger.error(
            "refresh not fresh (meta_present=%s, written_at=%r, today=%s); skipping send",
            meta_data is not None,
            written_at,
            today,
        )
        return 1

    summary = build_daily_summary(
        config, projections_root, today=today, league=league, team_key=team_key
    )
    recipients = config.summary.get("recipients") or []
    from_address = config.summary.get("from_address") or ""
    if not recipients or not from_address:
        logger.error("summary.recipients / summary.from_address not configured")
        return 2

    try:
        send_email(
            api_key=api_key,
            from_address=from_address,
            recipients=recipients,
            subject=subject_line(summary),
            html=render_html(summary),
            text=render_text(summary),
        )
    except Exception:
        logger.exception("send failed; not advancing standings snapshot")
        return 3

    # Snapshot is written ONLY after a successful send, so a failed run never
    # corrupts tomorrow's delta baseline.
    _write_snapshot(written_at)
    logger.info("daily summary sent to %s", recipients)
    return 0


def main() -> int:
    # Read Upstash, not local SQLite. Set at runtime (not module scope) so that
    # importing this module has no global side effect -- a module-level mutation
    # would leak RENDER=true into other tests sharing an xdist worker process and
    # silently flip their get_kv() to remote. No import above touches KV at import
    # time; get_kv() is only called at runtime, downstream of here.
    os.environ["RENDER"] = "true"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
    from fantasy_baseball.lineup.yahoo_roster import fetch_teams, find_user_team_key

    config = load_config(_PROJECT_ROOT / "config" / "league.yaml")
    api_key = os.environ.get("RESEND_API_KEY", "")
    if not api_key:
        logger.error("RESEND_API_KEY not set")
        return 2

    session = get_yahoo_session()
    league = get_league(session, config.league_id, config.game_code)
    teams = fetch_teams(league)
    team_key = find_user_team_key(teams, config.team_name)

    projections_root = _PROJECT_ROOT / "data" / "projections"
    return run_summary(config, projections_root, api_key=api_key, league=league, team_key=team_key)


if __name__ == "__main__":
    raise SystemExit(main())
