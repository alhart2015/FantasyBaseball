"""Pitcher matchup quality adjustments based on opponent team batting stats."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import statsapi

from fantasy_baseball.data.mlb_schedule import normalize_team_abbrev

logger = logging.getLogger(__name__)


def normalize_team_batting_stats(raw_stats: list[dict]) -> dict[str, dict]:
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
    team_stats: dict[str, dict],
    dampening: float = DEFAULT_DAMPENING,
) -> dict[str, dict]:
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
    factors: dict | list[dict],
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


def fetch_team_batting_stats(season: int | None = None) -> dict[str, dict]:
    """Fetch all 30 MLB teams' hitting stats via per-team API calls.

    Gets the team list from statsapi, then fetches season hitting stats
    for each team individually. Normalizes abbreviations to FanGraphs format.

    Args:
        season: MLB season year. Defaults to the current year if None.

    Returns:
        Result of normalize_team_batting_stats: {abbrev: {ops, k_pct}}.
    """
    if season is None:
        season = datetime.now().year

    teams_response = statsapi.get("teams", {"sportId": 1})
    teams = teams_response.get("teams", [])

    raw_stats = []
    for team in teams:
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
                continue
            stat = splits[0]["stat"]
            raw_stats.append(
                {
                    "abbreviation": abbrev,
                    "ops": stat["ops"],
                    "strikeouts": stat["strikeOuts"],
                    "plate_appearances": stat["plateAppearances"],
                }
            )
        except Exception:
            logger.exception("Failed to fetch stats for team %s (id=%s)", abbrev, team_id)

    return normalize_team_batting_stats(raw_stats)


def save_batting_stats_cache(stats: dict[str, dict], path: Path) -> None:
    """Save batting stats to a JSON cache file with a fetch timestamp.

    Args:
        stats: Normalized team batting stats dict {abbrev: {ops, k_pct}}.
        path: Destination file path.
    """
    payload = {
        "stats": stats,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.debug("Saved batting stats cache to %s", path)


def load_batting_stats_cache(path: Path) -> dict[str, dict] | None:
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
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    age_hours = (datetime.now(tz=timezone.utc) - fetched_at).total_seconds() / 3600
    if age_hours > 24:
        logger.debug("Batting stats cache is stale (%.1fh old); ignoring", age_hours)
        return None

    return payload.get("stats")


def get_team_batting_stats(
    cache_path: Path,
    season: int | None = None,
) -> dict[str, dict]:
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
