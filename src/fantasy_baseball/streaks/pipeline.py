"""End-to-end orchestration for the hot-streaks pipeline.

Wraps the DB-refresh sequence (fetch logs/statcast, upsert projection
rates, recompute windows/thresholds/labels), the refit-or-load model
decision, the Yahoo fetch, and ``build_report`` into a single function
called by both the Sunday CLI and the dashboard refresh pipeline.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from fantasy_baseball.lineup.yahoo_roster import (
    fetch_free_agents,
    fetch_roster,
    fetch_teams,
    find_user_team_key,
)
from fantasy_baseball.streaks.data.fetch_history import fetch_season
from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.projections import (
    load_projection_rates_for_seasons,
)
from fantasy_baseball.streaks.inference import (
    load_models_from_fits,
    refit_models_for_report,
)
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.reports.sunday import (
    Report,
    YahooHitter,
    build_name_to_mlbam_map,
    build_report,
)
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows
from fantasy_baseball.utils.time_utils import local_today

logger = logging.getLogger("streaks.pipeline")


_DEFAULT_MAX_FIT_AGE_DAYS = 14

# Hitter positions to scan for free agents. Pitcher streaks are out of
# scope for Phase 5.
_HITTER_FA_POSITIONS: tuple[str, ...] = ("C", "1B", "2B", "3B", "SS", "OF", "Util")


def _should_refit(conn: duckdb.DuckDBPyConnection, *, max_age_days: int, force: bool) -> bool:
    """Return True iff models should be refit rather than loaded.

    True when ``force`` is set, when ``model_fits`` is empty, or when
    the most recent ``fit_timestamp`` is older than ``max_age_days``.
    """
    if force:
        return True
    row = conn.execute("SELECT MAX(fit_timestamp) FROM model_fits").fetchone()
    if row is None or row[0] is None:
        return True
    most_recent: datetime = row[0]
    if most_recent.tzinfo is None:
        most_recent = most_recent.replace(tzinfo=UTC)
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    return bool(most_recent < cutoff)


def _normalize_position(p: str) -> str:
    """Yahoo returns mixed-case position strings ("Util" vs "UTIL"); the
    streaks report only cares about hitter-vs-pitcher distinction, so
    upper-casing is enough.
    """
    return p.upper()


def _to_yahoo_hitter(entry: dict[str, Any]) -> YahooHitter:
    positions = tuple(_normalize_position(p) for p in entry.get("positions", []))
    return YahooHitter(
        name=entry["name"],
        positions=positions,
        yahoo_id=str(entry.get("player_id", "")),
        status=entry.get("status", "") or "",
    )


def _fetch_yahoo_hitters(
    league: Any, *, team_name: str
) -> tuple[list[YahooHitter], list[YahooHitter]]:
    """Fetch the user's roster + dedup'd FAs across hitter positions.

    Identical to ``scripts/streaks/run_sunday_report.py::_fetch_yahoo_data``.
    Lifted here so dashboard refresh and the Sunday CLI share one
    implementation. The CLI will be refactored in Task 5 to delegate.
    """
    teams = fetch_teams(league)
    user_team_key = find_user_team_key(teams, team_name)
    roster_raw = fetch_roster(league, user_team_key)
    roster_hitters = [_to_yahoo_hitter(p) for p in roster_raw]

    def _fetch_one(pos: str) -> list[dict[str, Any]]:
        try:
            return fetch_free_agents(league, pos, count=50)
        except Exception:
            logger.exception("Free agent fetch failed at position %s; continuing", pos)
            return []

    seen: set[str] = set()
    fa_hitters: list[YahooHitter] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fa_raw in pool.map(_fetch_one, _HITTER_FA_POSITIONS):
            for fa in fa_raw:
                key = fa["name"].lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                fa_hitters.append(_to_yahoo_hitter(fa))
    return roster_hitters, fa_hitters


def _refresh_streaks_db(
    conn: duckdb.DuckDBPyConnection,
    *,
    season: int,
    season_set_train: str,
    projections_root: Path,
    skip_fetch: bool,
) -> None:
    """Steps 1-4 of the pipeline: fetch -> load projections -> windows -> labels."""
    if not skip_fetch:
        logger.info("Fetching %d game logs + Statcast (incremental)...", season)
        summary = fetch_season(season=season, conn=conn)
        logger.info("fetch_season summary: %s", summary)
    else:
        logger.info("--skip-fetch set; using cached game logs + Statcast")

    logger.info("Loading %d projection rates from %s...", season, projections_root)
    rates = load_projection_rates_for_seasons(projections_root, [season])
    upsert_projection_rates(conn, rates)

    logger.info("Recomputing hitter_windows...")
    n_windows = compute_windows(conn)
    logger.info("  wrote %d window rows", n_windows)

    logger.info("Recomputing thresholds and labels on %s...", season_set_train)
    compute_thresholds(conn, season_set=season_set_train)
    n_labels = apply_labels(conn, season_set=season_set_train)
    logger.info("  wrote %d label rows", n_labels)


def compute_streak_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    league: Any,
    team_name: str,
    league_id: int,
    projections_root: Path,
    scoring_season: int,
    season_set_train: str = "2023-2025",
    window_days: int = 14,
    top_n_fas: int = 50,
    force_refit: bool = False,
    skip_fetch: bool = False,
    max_fit_age_days: int = _DEFAULT_MAX_FIT_AGE_DAYS,
    today: date | None = None,
) -> Report:
    """End-to-end streak report orchestration.

    Runs DB refresh, refit-or-load models, Yahoo fetch, score, return.
    ``league`` is an opaque Yahoo league handle (e.g. the value returned
    by ``yahoo_auth.get_league``); typed as ``Any`` to keep this module
    testable without importing yahoo_fantasy_api.
    """
    _refresh_streaks_db(
        conn,
        season=scoring_season,
        season_set_train=season_set_train,
        projections_root=projections_root,
        skip_fetch=skip_fetch,
    )

    if _should_refit(conn, max_age_days=max_fit_age_days, force=force_refit):
        logger.info("Refitting models on %s...", season_set_train)
        models = refit_models_for_report(
            conn, season_set_train=season_set_train, window_days=window_days
        )
    else:
        logger.info("Reusing models from model_fits")
        models = load_models_from_fits(conn)

    roster_hitters, fa_hitters = _fetch_yahoo_hitters(league, team_name=team_name)
    logger.info(
        "Yahoo fetch complete: %d roster, %d FAs (deduped)",
        len(roster_hitters),
        len(fa_hitters),
    )

    name_to_mlbam = build_name_to_mlbam_map(projections_root, season=scoring_season)
    if not name_to_mlbam:
        raise RuntimeError(
            f"No name->mlbam mappings built - check that {projections_root}/"
            f"{scoring_season}/ contains hitter CSVs with Name + MLBAMID columns."
        )

    return build_report(
        conn,
        league_config_team_name=team_name,
        league_config_league_id=league_id,
        models=models,
        roster_hitters=roster_hitters,
        fa_hitters=fa_hitters,
        name_to_mlbam=name_to_mlbam,
        today=today or local_today(),
        season_set_train=season_set_train,
        scoring_season=scoring_season,
        window_days=window_days,
        top_n_fas=top_n_fas,
    )
