"""Pitcher matchup quality adjustments based on opponent team batting stats."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, cast

from fantasy_baseball.utils.time_utils import local_today
from pathlib import Path

import pandas as pd
import statsapi

from fantasy_baseball.data.mlb_schedule import normalize_team_abbrev
from fantasy_baseball.utils.name_utils import normalize_name

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
            season, season - 1,
        )
        stats = _fetch_team_batting_stats_for_season(season - 1, teams)
    return stats


def _fetch_team_batting_stats_for_season(
    season: int, teams: list[dict],
) -> dict[str, dict]:
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

    return cast("dict[str, dict[Any, Any]] | None", payload.get("stats"))


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


def get_probable_starters(
    pitcher_roster: list,
    schedule: dict,
    matchup_factors: dict[str, dict] | None = None,
    team_stats: dict[str, dict] | None = None,
) -> list[dict]:
    """Cross-reference roster pitchers with the weekly schedule.

    Returns structured data for each start found, sorted by date.
    Each entry has: pitcher, date, day, opponent, indicator (@/vs),
    matchup_quality (Great/Fair/Tough), starts (count for that pitcher),
    and detail (ops, ops_rank, k_pct, k_rank) for expandable UI rows.

    Args:
        pitcher_roster: List of player dicts/Series with "name" key.
        schedule: Result of get_week_schedule() with "probable_pitchers".
        matchup_factors: Result of calculate_matchup_factors(). Used for
            quality badges. If None, falls back to raw OPS thresholds.
        team_stats: Raw team batting stats {abbrev: {ops, k_pct}}.
            Used for detail data (OPS rank, K% rank).
    """
    if not schedule or not schedule.get("probable_pitchers"):
        return []

    roster_names = {normalize_name(p.name) for p in pitcher_roster}

    # Pre-compute rankings for detail display
    if team_stats:
        ops_ranked = sorted(team_stats.items(), key=lambda x: x[1]["ops"], reverse=True)
        k_ranked = sorted(team_stats.items(), key=lambda x: x[1]["k_pct"], reverse=False)
        ops_rank_map = {abbrev: i + 1 for i, (abbrev, _) in enumerate(ops_ranked)}
        k_rank_map = {abbrev: i + 1 for i, (abbrev, _) in enumerate(k_ranked)}
    else:
        ops_rank_map = {}
        k_rank_map = {}

    # Collect starts per pitcher
    pitcher_starts: dict[str, list[dict]] = {}

    for game in schedule["probable_pitchers"]:
        for side in ("away", "home"):
            pitcher_name = game.get(f"{side}_pitcher", "TBD")
            if not pitcher_name or pitcher_name == "TBD":
                continue
            if normalize_name(pitcher_name) not in roster_names:
                continue

            opponent = game["home_team"] if side == "away" else game["away_team"]
            indicator = "@" if side == "away" else "vs"

            # Parse day name from date
            try:
                from datetime import datetime as dt
                day = dt.strptime(game["date"], "%Y-%m-%d").strftime("%a")
            except (ValueError, KeyError):
                day = "?"

            # Matchup quality badge
            if matchup_factors and opponent in matchup_factors:
                f = matchup_factors[opponent]["era_whip_factor"]
                if f <= 0.93:
                    quality = "Great"
                elif f >= 1.03:
                    quality = "Tough"
                else:
                    quality = "Fair"
            elif team_stats and opponent in team_stats:
                avg_ops = sum(s["ops"] for s in team_stats.values()) / len(team_stats)
                ops = team_stats[opponent]["ops"]
                if ops < avg_ops * 0.95:
                    quality = "Great"
                elif ops > avg_ops * 1.05:
                    quality = "Tough"
                else:
                    quality = "Fair"
            else:
                quality = "Fair"

            # Detail data for expandable rows
            opp_stats = team_stats.get(opponent, {}) if team_stats else {}
            detail = {
                "ops": round(opp_stats.get("ops", 0.0), 3),
                "ops_rank": ops_rank_map.get(opponent, 0),
                "k_pct": round(opp_stats.get("k_pct", 0.0) * 100, 1)
                         if opp_stats.get("k_pct", 0) < 1
                         else round(opp_stats.get("k_pct", 0.0), 1),
                "k_rank": k_rank_map.get(opponent, 0),
            }

            start_entry = {
                "date": game.get("date", ""),
                "day": day,
                "opponent": opponent,
                "indicator": indicator,
                "matchup_quality": quality,
                "detail": detail,
            }

            if pitcher_name not in pitcher_starts:
                pitcher_starts[pitcher_name] = []
            pitcher_starts[pitcher_name].append(start_entry)

    # Flatten into per-pitcher summaries
    result: list[dict[str, Any]] = []
    for pitcher_name, starts in pitcher_starts.items():
        starts.sort(key=lambda s: s["date"])
        result.append({
            "pitcher": pitcher_name,
            "starts": len(starts),
            "days": ", ".join(s["day"] for s in starts),
            "opponents": ", ".join(f"{s['indicator']} {s['opponent']}" for s in starts),
            "matchup_quality": (
                "Tough" if any(s["matchup_quality"] == "Tough" for s in starts)
                else "Fair" if any(s["matchup_quality"] == "Fair" for s in starts)
                else "Great"
            ),
            "matchups": starts,
        })

    result.sort(key=lambda s: (-s["starts"], s["pitcher"]))
    return result
