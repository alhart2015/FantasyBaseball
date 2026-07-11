"""Assemble a DailySummary from the refreshed KV plus a live Yahoo injury fetch.

Each section builder is wrapped so one failure degrades to an empty section (and
a section_errors entry) rather than aborting the email. The send/skip decision
is a separate up-front check on META freshness (refresh_is_fresh), applied by the
orchestrator before this runs.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from fantasy_baseball.config import LeagueConfig
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.lineup.yahoo_roster import fetch_injuries, fetch_roster
from fantasy_baseball.summary.builders import (
    build_injuries,
    build_last_night,
    build_lineup_moves,
    build_probables,
    build_standings_delta,
    build_streaks,
)
from fantasy_baseball.summary.crosswalk import build_typed_name_to_mlbam
from fantasy_baseball.summary.models import DailySummary, StandingsDelta
from fantasy_baseball.utils.time_utils import local_today
from fantasy_baseball.web.season_data import (
    read_cache,
    read_cache_dict,
    read_cache_list,
)

logger = logging.getLogger(__name__)

# fetch_injuries / fetch_roster are module-level (not imported inside the
# function) so tests can monkeypatch them, and so ruff sees them used.


def refresh_is_fresh(meta: dict[str, Any], today: date) -> bool:
    """True iff META.last_refresh is from ``today`` (local calendar date)."""
    raw = meta.get("last_refresh")
    if not raw:
        return False
    try:
        parsed = datetime.strptime(str(raw), "%Y-%m-%d %H:%M").date()
    except ValueError:
        return False
    return parsed == today


def build_daily_summary(
    config: LeagueConfig,
    projections_root: Path,
    *,
    today: date | None = None,
    league: Any,
    team_key: str,
) -> DailySummary:
    """Assemble every section. One failing builder => empty section + a note."""
    today = today or local_today()
    yesterday = today - timedelta(days=1)
    season = config.season_year
    client = get_kv()
    section_errors: list[str] = []

    def _guard(name: str, fn: Any, fallback: Any) -> Any:
        try:
            return fn()
        except Exception:  # degrade one section, keep the email
            logger.exception("summary builder %s failed", name)
            section_errors.append(name)
            return fallback

    xmap = _guard(
        "crosswalk", lambda: build_typed_name_to_mlbam(projections_root, season=season), {}
    )
    roster = _guard("roster", lambda: fetch_roster(league, team_key), [])

    last_night, unmatched = _guard(
        "build_last_night",
        lambda: build_last_night(roster, xmap, client, season, yesterday),
        ([], []),
    )
    streaks = _guard(
        "build_streaks", lambda: build_streaks(read_cache_dict(CacheKey.STREAK_SCORES)), []
    )
    lineup_moves = _guard(
        "build_lineup_moves",
        lambda: build_lineup_moves(read_cache_dict(CacheKey.LINEUP_OPTIMAL)),
        [],
    )
    injuries = _guard(
        "build_injuries", lambda: build_injuries(fetch_injuries(league, team_key)), []
    )
    probables = _guard(
        "build_probables", lambda: build_probables(read_cache_list(CacheKey.PROBABLE_STARTERS)), []
    )
    standings_delta = _guard(
        "build_standings_delta",
        lambda: build_standings_delta(
            cast("dict[str, Any] | None", read_cache(CacheKey.STANDINGS)),
            cast("dict[str, Any] | None", read_cache(CacheKey.STANDINGS_SNAPSHOT)),
            config.team_name,
        ),
        StandingsDelta(is_first_run=True, user_team_name=config.team_name),
    )

    return DailySummary(
        as_of=yesterday,
        last_night=last_night,
        unmatched=unmatched,
        streaks=streaks,
        standings_delta=standings_delta,
        lineup_moves=lineup_moves,
        injuries=injuries,
        probables=probables,
        section_errors=section_errors,
    )
