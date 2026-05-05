"""Pitcher matchup quality adjustments based on opponent team batting stats."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import statsapi

from fantasy_baseball.data.mlb_schedule import normalize_team_abbrev
from fantasy_baseball.utils.time_utils import local_today

logger = logging.getLogger(__name__)


def normalize_team_batting_stats(raw_stats: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Convert raw MLB API team batting data to {abbrev: {ops, k_pct}}.

    Args:
        raw_stats: List of dicts with keys: abbreviation, ops (str),
                   strikeouts (int), plate_appearances (int).
    Returns:
        Dict keyed by team abbreviation with float ops and k_pct values.
    """
    result = {}
    for team in raw_stats:
        abbrev = team["abbreviation"]
        pa = team["plate_appearances"]
        k_pct = team["strikeouts"] / pa if pa > 0 else 0.0
        result[abbrev] = {
            "ops": float(team["ops"]),
            "k_pct": k_pct,
        }
    return result


DEFAULT_DAMPENING = 0.5


def calculate_matchup_factors(
    team_stats: dict[str, dict[str, float]],
    dampening: float = DEFAULT_DAMPENING,
) -> dict[str, dict[str, float]]:
    """Compute matchup adjustment factors for each team relative to league average.

    For each team, produces:
      - era_whip_factor: multiplier for pitcher ERA/WHIP (>1 = harder matchup)
      - k_factor: multiplier for pitcher K (>1 = more Ks expected)

    Deviations from league average are dampened by the dampening parameter
    (0.5 = half the raw deviation applied).
    """
    if not team_stats:
        return {}

    ops_values = [t["ops"] for t in team_stats.values()]
    k_values = [t["k_pct"] for t in team_stats.values()]
    avg_ops = sum(ops_values) / len(ops_values)
    avg_k = sum(k_values) / len(k_values)

    factors = {}
    for abbrev, stats in team_stats.items():
        if avg_ops > 0:
            ops_dev = (stats["ops"] - avg_ops) / avg_ops
            era_whip_factor = 1.0 + dampening * ops_dev
        else:
            era_whip_factor = 1.0

        if avg_k > 0:
            k_dev = (stats["k_pct"] - avg_k) / avg_k
            k_factor = 1.0 + dampening * k_dev
        else:
            k_factor = 1.0

        factors[abbrev] = {
            "era_whip_factor": era_whip_factor,
            "k_factor": k_factor,
        }
    return factors


def adjust_pitcher_projection(
    pitcher: pd.Series,
    factors: dict[str, float] | list[dict[str, float]],
) -> pd.Series:
    """Adjust a pitcher's projected stats based on matchup factors.

    Args:
        pitcher: Pitcher projection Series with era, whip, k, w, sv, ip, er, bb, h_allowed.
        factors: Single matchup factor dict, or list of dicts for multi-start
                 pitchers (factors are averaged).

    Returns:
        Copy of pitcher with adjusted era, whip, k, er, bb, h_allowed.
        w and sv are left unchanged.
    """
    if isinstance(factors, list):
        era_whip = sum(f["era_whip_factor"] for f in factors) / len(factors)
        k_fac = sum(f["k_factor"] for f in factors) / len(factors)
    else:
        era_whip = factors["era_whip_factor"]
        k_fac = factors["k_factor"]

    adjusted = pitcher.copy()
    adjusted["era"] = pitcher["era"] * era_whip
    adjusted["whip"] = pitcher["whip"] * era_whip
    adjusted["k"] = pitcher["k"] * k_fac
    adjusted["er"] = pitcher.get("er", 0) * era_whip
    adjusted["bb"] = pitcher.get("bb", 0) * era_whip
    adjusted["h_allowed"] = pitcher.get("h_allowed", 0) * era_whip

    return adjusted


def fetch_team_batting_stats(season: int | None = None) -> dict[str, dict[str, float]]:
    """Fetch all 30 MLB teams' hitting stats via per-team API calls.

    Gets the team list from statsapi, then fetches season hitting stats
    for each team individually. Normalizes abbreviations to FanGraphs format.

    If the requested season returns no data (e.g. pre-season before games
    have been played), automatically retries with the previous season.

    Args:
        season: MLB season year. Defaults to the current year if None.

    Returns:
        Result of normalize_team_batting_stats: {abbrev: {ops, k_pct}}.
    """
    if season is None:
        season = local_today().year

    # Fetch teams list once and reuse for fallback
    teams_response = statsapi.get("teams", {"sportId": 1})
    teams = teams_response.get("teams", [])

    stats = _fetch_team_batting_stats_for_season(season, teams)
    if not stats and season == local_today().year:
        logger.info(
            "No %d stats available (pre-season?); falling back to %d",
            season,
            season - 1,
        )
        stats = _fetch_team_batting_stats_for_season(season - 1, teams)
    return stats


def _fetch_team_batting_stats_for_season(
    season: int,
    teams: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Fetch team batting stats for a specific season.

    Args:
        season: MLB season year.
        teams: MLB teams list (from statsapi "teams" endpoint).

    Returns:
        Result of normalize_team_batting_stats: {abbrev: {ops, k_pct}}.
    """

    def _fetch_one_team(team):
        team_id = team["id"]
        mlb_abbrev = team["abbreviation"]
        abbrev = normalize_team_abbrev(mlb_abbrev)
        try:
            stats_response = statsapi.get(
                "team_stats",
                {
                    "teamId": team_id,
                    "stats": "season",
                    "group": "hitting",
                    "season": season,
                },
            )
            splits = stats_response.get("stats", [{}])[0].get("splits", [])
            if not splits:
                logger.warning("No hitting splits for team %s (id=%s)", abbrev, team_id)
                return None
            stat = splits[0]["stat"]
            return {
                "abbreviation": abbrev,
                "ops": stat["ops"],
                "strikeouts": stat["strikeOuts"],
                "plate_appearances": stat["plateAppearances"],
            }
        except Exception:
            logger.warning("Failed to fetch stats for team %s (id=%s), skipping", abbrev, team_id)
            return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        raw_stats = [r for r in pool.map(_fetch_one_team, teams) if r is not None]

    return normalize_team_batting_stats(raw_stats)


def save_batting_stats_cache(stats: dict[str, dict[str, float]], path: Path) -> None:
    """Save batting stats to a JSON cache file with a fetch timestamp.

    Args:
        stats: Normalized team batting stats dict {abbrev: {ops, k_pct}}.
        path: Destination file path.
    """
    payload = {
        "stats": stats,
        "fetched_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.debug("Saved batting stats cache to %s", path)


def load_batting_stats_cache(path: Path) -> dict[str, dict[str, float]] | None:
    """Load batting stats from a JSON cache file.

    Returns None if the file is missing or was fetched more than 24 hours ago.

    Args:
        path: Cache file path.

    Returns:
        Normalized team batting stats dict, or None if cache is absent/stale.
    """
    path = Path(path)
    if not path.exists():
        return None

    with open(path) as f:
        payload = json.load(f)

    fetched_at_str = payload.get("fetched_at")
    if fetched_at_str is None:
        return None

    fetched_at = datetime.fromisoformat(fetched_at_str)
    # Ensure timezone-aware comparison
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=UTC)

    age_hours = (datetime.now(tz=UTC) - fetched_at).total_seconds() / 3600
    if age_hours > 24:
        logger.debug("Batting stats cache is stale (%.1fh old); ignoring", age_hours)
        return None

    return cast("dict[str, dict[str, float]] | None", payload.get("stats"))


def get_team_batting_stats(
    cache_path: Path,
    season: int | None = None,
) -> dict[str, dict[str, float]]:
    """Return team batting stats, using cache when fresh or fetching live.

    Tries to load from the cache first. If the cache is missing or stale,
    fetches from the MLB Stats API and saves the result to the cache.

    Args:
        cache_path: Path to the JSON cache file.
        season: MLB season year. Defaults to the current year if None.

    Returns:
        Normalized team batting stats dict {abbrev: {ops, k_pct}}.
    """
    cached = load_batting_stats_cache(cache_path)
    if cached is not None:
        logger.debug("Using cached batting stats from %s", cache_path)
        return cached

    logger.info("Fetching live team batting stats (season=%s)", season)
    stats = fetch_team_batting_stats(season=season)
    save_batting_stats_cache(stats, cache_path)
    return stats


def get_probable_starters(
    pitcher_roster: list[Any],
    schedule: dict[str, Any],
    matchup_factors: dict[str, dict[str, float]] | None = None,
    team_stats: dict[str, dict[str, float]] | None = None,
    today: date | None = None,
    window_start: date | None = None,
    window_end: date | None = None,
) -> list[dict[str, Any]]:
    """Build per-pitcher rollups of upcoming starts in the scoring week.

    Combines MLB-announced probables with rotation projections. Each
    rollup row carries:
        pitcher, starts, days, opponents, matchup_quality (worst-case),
        matchups (list of per-start StartEntry dicts including ``announced``).

    Args:
        pitcher_roster: roster pitchers (must have .name and .team).
        schedule: result of get_week_schedule() containing probable_pitchers.
        matchup_factors: result of calculate_matchup_factors().
        team_stats: raw team batting stats {abbrev: {ops, k_pct}}.
        today: cutoff for anchor lookup. Defaults to local_today().
        window_start, window_end: scoring-week bounds. Default to the
            schedule dict's start_date/end_date.
    """
    from fantasy_baseball.lineup.upcoming_starts import (
        build_team_game_index,
        compose_pitcher_entries,
    )

    if not schedule or not schedule.get("probable_pitchers"):
        return []

    pps = schedule["probable_pitchers"]

    if today is None:
        today = local_today()
    if window_start is None:
        window_start = date.fromisoformat(schedule["start_date"])
    if window_end is None:
        window_end = date.fromisoformat(schedule["end_date"])

    matchup_factors = matchup_factors or {}
    team_stats = team_stats or {}

    if team_stats:
        ops_ranked = sorted(team_stats.items(), key=lambda x: x[1]["ops"], reverse=True)
        k_ranked = sorted(team_stats.items(), key=lambda x: x[1]["k_pct"])
        ops_rank_map = {abbrev: i + 1 for i, (abbrev, _) in enumerate(ops_ranked)}
        k_rank_map = {abbrev: i + 1 for i, (abbrev, _) in enumerate(k_ranked)}
    else:
        ops_rank_map = {}
        k_rank_map = {}

    rollups: list[dict[str, Any]] = []
    for pitcher in pitcher_roster:
        team_abbrev = getattr(pitcher, "team", "") or ""
        if not team_abbrev:
            continue

        team_games = build_team_game_index(pps, team_abbrev)
        if not team_games:
            continue

        entries = compose_pitcher_entries(
            pitcher.name,
            team_games,
            today=today,
            window_start=window_start,
            window_end=window_end,
            matchup_factors=matchup_factors,
            team_stats=team_stats,
            ops_rank_map=ops_rank_map,
            k_rank_map=k_rank_map,
        )
        if not entries:
            continue

        matchups: list[dict[str, Any]] = [
            {
                "date": e.date,
                "day": e.day,
                "opponent": e.opponent,
                "indicator": e.indicator,
                "announced": e.announced,
                "matchup_quality": e.matchup_quality,
                "detail": e.detail,
            }
            for e in entries
        ]
        # Worst-of rollup quality: Tough > Fair > Great
        if any(m["matchup_quality"] == "Tough" for m in matchups):
            roll_quality = "Tough"
        elif any(m["matchup_quality"] == "Fair" for m in matchups):
            roll_quality = "Fair"
        else:
            roll_quality = "Great"

        rollups.append(
            {
                "pitcher": pitcher.name,
                "starts": len(matchups),
                "days": ", ".join(m["day"] for m in matchups),
                "opponents": ", ".join(f"{m['indicator']} {m['opponent']}" for m in matchups),
                "matchup_quality": roll_quality,
                "matchups": matchups,
            }
        )

    rollups.sort(key=lambda s: (-s["starts"], s["pitcher"]))
    return rollups
