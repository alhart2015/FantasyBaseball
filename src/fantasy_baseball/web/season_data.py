"""Cache management and data assembly for the season dashboard."""

import json
import logging
import os
import tempfile
import threading
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

log = logging.getLogger(__name__)

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.standings import CategoryStats, StandingsEntry, StandingsSnapshot
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES, HITTER_PROJ_KEYS, IL_STATUSES, INVERSE_STATS as INVERSE_CATS,
    PITCHER_PROJ_KEYS,
)
from fantasy_baseball.utils.positions import PITCHER_POSITIONS

if TYPE_CHECKING:
    import pandas as pd

    from fantasy_baseball.models.player import Player

_opponent_cache: dict = {}
OPPONENT_CACHE_TTL_SECONDS = 900  # 15 minutes


def clear_opponent_cache() -> None:
    """Clear the opponent lineup in-memory cache (called on full refresh)."""
    _opponent_cache.clear()


_redis_client = None
_redis_initialized = False
_redis_lock = threading.Lock()


def _get_redis():
    """Lazy Upstash Redis client. Returns None if not configured."""
    global _redis_client, _redis_initialized
    if _redis_initialized:
        return _redis_client
    with _redis_lock:
        if _redis_initialized:
            return _redis_client
        url = os.environ.get("UPSTASH_REDIS_REST_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
        if url and token:
            from upstash_redis import Redis
            _redis_client = Redis(url=url, token=token)
        _redis_initialized = True
    return _redis_client


CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


def _standings_to_snapshot(
    standings: list[dict],
    effective_date: date | None = None,
) -> StandingsSnapshot:
    """Convert a raw standings list[dict] to a StandingsSnapshot.

    Used at the boundary between Yahoo fetch / cache read and typed
    consumers (calculate_leverage, format_standings_for_display).
    """
    return StandingsSnapshot(
        effective_date=effective_date or date.min,
        entries=[
            StandingsEntry(
                team_name=t["name"],
                team_key=t.get("team_key", ""),
                rank=t.get("rank", 0),
                stats=CategoryStats.from_dict(t.get("stats", {})),
            )
            for t in standings
        ],
    )


class CacheKey(StrEnum):
    """Canonical names of every cached payload.

    Typos on member access (e.g. ``CacheKey.LEVARAGE``) raise
    ``AttributeError`` at import time instead of silently reading or writing
    the wrong cache entry.
    """

    STANDINGS = "standings"
    ROSTER = "roster"
    PROJECTIONS = "projections"
    LINEUP_OPTIMAL = "lineup_optimal"
    PROBABLE_STARTERS = "probable_starters"
    MONTE_CARLO = "monte_carlo"
    META = "meta"
    RANKINGS = "rankings"
    ROSTER_AUDIT = "roster_audit"
    SPOE = "spoe"
    OPP_ROSTERS = "opp_rosters"
    LEVERAGE = "leverage"
    PENDING_MOVES = "pending_moves"
    TRANSACTION_ANALYZER = "transaction_analyzer"
    TRANSACTIONS = "transactions"
    ROS_PROJECTIONS = "ros_projections"
    POSITIONS = "positions"


def redis_key(key: "CacheKey") -> str:
    """Return the Redis key for a cache entry (``cache:<name>``)."""
    return f"cache:{key}"


CACHE_FILES: dict[CacheKey, str] = {
    CacheKey.STANDINGS: "standings.json",
    CacheKey.ROSTER: "roster.json",
    CacheKey.PROJECTIONS: "projections.json",
    CacheKey.LINEUP_OPTIMAL: "lineup_optimal.json",
    CacheKey.PROBABLE_STARTERS: "probable_starters.json",
    CacheKey.MONTE_CARLO: "monte_carlo.json",
    CacheKey.META: "meta.json",
    CacheKey.RANKINGS: "rankings.json",
    CacheKey.ROSTER_AUDIT: "roster_audit.json",
    CacheKey.SPOE: "spoe.json",
    CacheKey.OPP_ROSTERS: "opp_rosters.json",
    CacheKey.LEVERAGE: "leverage.json",
    CacheKey.PENDING_MOVES: "pending_moves.json",
    CacheKey.TRANSACTION_ANALYZER: "transaction_analyzer.json",
    CacheKey.TRANSACTIONS: "transactions.json",
    CacheKey.ROS_PROJECTIONS: "ros_projections.json",
    CacheKey.POSITIONS: "positions.json",
}


def read_cache(key: CacheKey, cache_dir: Path = CACHE_DIR) -> dict | list | None:
    """Read a cached JSON file. Falls back to Redis on local miss."""
    path = cache_dir / CACHE_FILES[key]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback to Redis (only for default cache dir)
    if cache_dir != CACHE_DIR:
        return None

    redis = _get_redis()
    if not redis:
        return None

    try:
        raw = redis.get(redis_key(key))
        if raw is None:
            return None
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[redis] read_cache({key}) corrupt data, treating as miss")
        return None
    except Exception as e:
        print(f"[redis] read_cache({key}) failed: {e}")
        return None

    # Write back to local disk for subsequent fast reads
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[redis] local write-back for {key} failed: {e}")

    return data


def write_cache(key: CacheKey, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
    """Atomically write a cached JSON file (tmpfile + rename), with Redis write-through."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_FILES[key]
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    # Write-through to Redis (only for default cache dir)
    if cache_dir == CACHE_DIR:
        redis = _get_redis()
        if redis:
            try:
                redis.set(redis_key(key), json.dumps(data))
            except Exception as e:
                print(f"[redis] write_cache({key}) failed: {e}")


def read_meta(cache_dir: Path = CACHE_DIR) -> dict:
    """Read cache metadata (last refresh time, week, etc.). Returns empty dict if missing."""
    return read_cache(CacheKey.META, cache_dir) or {}


def _load_game_log_totals(season_year: int) -> tuple[dict, dict]:
    """Load aggregated game log totals from Redis, keyed by normalized name.

    Returns (hitter_logs, pitcher_logs) where each is {normalized_name: {stat: value}}.
    The season_year parameter is accepted for signature compatibility but unused —
    Redis keys are not year-partitioned (current season only).
    """
    from fantasy_baseball.data.redis_store import get_default_client, get_game_log_totals
    from fantasy_baseball.utils.name_utils import normalize_name

    client = get_default_client()
    raw_h = get_game_log_totals(client, "hitters")
    raw_p = get_game_log_totals(client, "pitchers")

    hitter_logs: dict[str, dict] = {}
    for _mid, entry in raw_h.items():
        name = entry.get("name") or ""
        if not name:
            continue
        norm = normalize_name(name)
        hitter_logs[norm] = {
            k: entry.get(k, 0) or 0 for k in ("pa", "ab", "h", "r", "hr", "rbi", "sb")
        }

    pitcher_logs: dict[str, dict] = {}
    for _mid, entry in raw_p.items():
        name = entry.get("name") or ""
        if not name:
            continue
        norm = normalize_name(name)
        pitcher_logs[norm] = {
            k: entry.get(k, 0) or 0 for k in ("ip", "k", "w", "sv", "er", "bb", "h_allowed")
        }

    return hitter_logs, pitcher_logs


def format_standings_for_display(
    standings: StandingsSnapshot, user_team_name: str,
    *, team_sds: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Transform standings snapshot into display-ready structure with roto points and color codes.

    Args:
        standings: StandingsSnapshot with typed StandingsEntry objects.
        user_team_name: The authenticated user's team name for highlighting.
        team_sds: Per-team per-category standard deviations for ERoto scoring.
            When provided, ``score_roto`` uses Gaussian pairwise win
            probabilities instead of deterministic rank-based scoring.

    Returns:
        {"teams": [...]} where each team has roto_points, is_user flag, color_classes, and rank.
    """
    if not standings.entries:
        return {"teams": []}

    # CategoryStats defaults (0.0 for counting, 99.0 for ERA/WHIP)
    # handle early-season missing data — no _fill_stat_defaults needed.
    all_stats = {e.team_name: e.stats for e in standings.entries}
    roto = score_roto(all_stats, team_sds=team_sds)

    cat_ranks = _compute_category_ranks(standings)

    teams = []
    for entry in standings.entries:
        name = entry.team_name
        is_user = name == user_team_name
        roto_pts = roto[name]

        color_classes = {}
        for cat in ALL_CATEGORIES:
            rank = cat_ranks[cat][name]
            if rank <= 2:
                color_classes[cat] = "rank-top"
            elif rank <= 4:
                color_classes[cat] = "rank-high"
            elif rank <= 6:
                color_classes[cat] = "rank-mid"
            elif rank <= 8:
                color_classes[cat] = "rank-low"
            else:
                color_classes[cat] = "rank-bottom"

        teams.append({
            "name": name,
            "team_key": entry.team_key,
            "stats": entry.stats,
            "roto_points": roto_pts,
            "is_user": is_user,
            "color_classes": color_classes,
            "sds": team_sds.get(name, {}) if team_sds else {},
        })

    teams.sort(key=lambda t: t["roto_points"]["total"], reverse=True)

    for i, t in enumerate(teams):
        t["rank"] = i + 1

    return {"teams": teams}


def get_teams_list(
    standings: list[dict], user_team_name: str
) -> dict:
    """Build a team list for the opponent selector dropdown.

    Args:
        standings: Raw standings cache (list of team dicts with name, team_key, rank).
        user_team_name: The user's team name for flagging.

    Returns:
        {"teams": [...], "user_team_key": str | None}
    """
    if not standings:
        return {"teams": [], "user_team_key": None}

    user_team_key = None
    teams = []
    for t in standings:
        is_user = t["name"] == user_team_name
        if is_user:
            user_team_key = t.get("team_key", "")
        teams.append({
            "name": t["name"],
            "team_key": t.get("team_key", ""),
            "rank": t.get("rank", 0),
            "is_user": is_user,
        })

    teams.sort(key=lambda t: t["rank"])
    return {"teams": teams, "user_team_key": user_team_key}


def build_opponent_lineup(
    roster: list[dict],
    opponent_name: str,
    hitters_proj: "pd.DataFrame",
    pitchers_proj: "pd.DataFrame",
    rest_of_season_hitters: "pd.DataFrame",
    rest_of_season_pitchers: "pd.DataFrame",
    season_year: int,
) -> dict:
    """Build a fully enriched opponent lineup (projections, pace, SGP).

    Args:
        roster: Raw roster from fetch_roster().
        opponent_name: Opponent team name (used for logging/context).
        hitters_proj: Blended hitter projections (with _name_norm column).
        pitchers_proj: Blended pitcher projections (with _name_norm column).
        rest_of_season_hitters: ROS hitter projections (may be empty DataFrame).
        rest_of_season_pitchers: ROS pitcher projections (may be empty DataFrame).
        season_year: Season year for game log lookup.

    Returns:
        Dict with "hitters" and "pitchers" lists, each entry containing
        projection stats, pace data, and per-player SGP.
    """
    from fantasy_baseball.analysis.pace import compute_overall_pace, compute_player_pace
    from fantasy_baseball.data.projections import match_roster_to_projections
    from fantasy_baseball.utils.name_utils import normalize_name

    # Match roster to projections
    matched = match_roster_to_projections(
        roster, hitters_proj, pitchers_proj,
        context=f"opp-lineup:{opponent_name}",
    )

    # ROS projection lookup
    has_rest_of_season = not rest_of_season_hitters.empty or not rest_of_season_pitchers.empty
    if has_rest_of_season:
        rest_of_season_matched = match_roster_to_projections(
            roster, rest_of_season_hitters, rest_of_season_pitchers,
            context=f"opp-lineup:{opponent_name}:ros",
        )
        rest_of_season_lookup = {normalize_name(p.name): p for p in rest_of_season_matched}
    else:
        rest_of_season_lookup = {}

    # Load game log totals for pace
    hitter_logs, pitcher_logs = _load_game_log_totals(season_year)

    # Build enriched entries
    matched_names = set()
    enriched = []
    for player in matched:
        if player.rest_of_season is not None:
            player.rest_of_season.compute_sgp()
        norm = normalize_name(player.name)
        matched_names.add(norm)

        entry = player.to_flat_dict()
        entry.setdefault("sgp", 0.0)

        # ROS projection tooltip data
        rest_of_season_entry = rest_of_season_lookup.get(norm)
        if rest_of_season_entry and rest_of_season_entry.rest_of_season:
            if player.player_type == PlayerType.HITTER:
                entry["rest_of_season"] = {k: getattr(rest_of_season_entry.rest_of_season, k, 0) for k in ["r", "hr", "rbi", "sb", "avg"]}
            else:
                entry["rest_of_season"] = {k: getattr(rest_of_season_entry.rest_of_season, k, 0) for k in ["w", "k", "sv", "era", "whip"]}

        # Pace data
        ptype = player.player_type
        if ptype == PlayerType.HITTER:
            actuals = hitter_logs.get(norm, {})
        else:
            actuals = pitcher_logs.get(norm, {})
        proj_keys = HITTER_PROJ_KEYS if ptype == PlayerType.HITTER else PITCHER_PROJ_KEYS
        projected = {k: getattr(player.rest_of_season, k, 0) if player.rest_of_season else 0 for k in proj_keys}
        entry["pace"] = compute_player_pace(actuals, projected, ptype)
        entry["overall_pace"] = compute_overall_pace(entry["pace"])

        enriched.append(entry)

    # Include unmatched players
    for raw_player in roster:
        if normalize_name(raw_player["name"]) not in matched_names:
            entry = dict(raw_player)
            entry["sgp"] = 0.0
            entry["pace"] = {}
            entry["overall_pace"] = compute_overall_pace(entry["pace"])
            enriched.append(entry)

    # Split into hitters and pitchers
    hitters = []
    pitchers = []
    for p in enriched:
        pos = p.get("selected_position", "BN")
        p["is_bench"] = pos in ("BN", "IL", "DL")
        is_pitcher = pos in PITCHER_POSITIONS or (
            pos == "BN" and set(p.get("positions", [])).issubset(
                PITCHER_POSITIONS | {"BN"}
            )
        )
        if is_pitcher:
            pitchers.append(p)
        else:
            hitters.append(p)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h.get("selected_position", ""), 99),
                                -h.get("sgp", 0)))
    pitchers.sort(key=lambda p: (p.get("selected_position", "") in ("BN", "IL", "DL"),
                                 -p.get("sgp", 0)))

    return {
        "hitters": hitters,
        "pitchers": pitchers,
        "hitter_totals": _compute_team_totals_pace(hitters, "hitter", opponent_name),
        "pitcher_totals": _compute_team_totals_pace(pitchers, "pitcher", opponent_name),
    }


def format_monte_carlo_for_display(
    mc_data: dict, user_team_name: str
) -> dict:
    """Format Monte Carlo results for template display.

    Returns dict with:
      - teams: list sorted by median_pts desc, each with median_pts, p10, p90,
               first_pct, top3_pct, is_user
      - category_risk: list of dicts with cat, median_pts, p10, p90,
                       top3_pct, bot3_pct, risk_class
    """
    if not mc_data or "team_results" not in mc_data:
        return {"teams": [], "category_risk": []}

    teams = []
    for name, res in mc_data["team_results"].items():
        teams.append({
            "name": name,
            "median_pts": res["median_pts"],
            "p10": res["p10"],
            "p90": res["p90"],
            "first_pct": res["first_pct"],
            "top3_pct": res["top3_pct"],
            "is_user": name == user_team_name,
        })
    teams.sort(key=lambda t: t["median_pts"], reverse=True)

    risk = []
    for cat, data in mc_data.get("category_risk", {}).items():
        if data["top3_pct"] >= 50:
            risk_class = "cat-top"
        elif data["bot3_pct"] >= 30:
            risk_class = "cat-bottom"
        else:
            risk_class = ""
        risk.append({
            "cat": cat,
            "median_pts": data["median_pts"],
            "p10": data["p10"],
            "p90": data["p90"],
            "top3_pct": data["top3_pct"],
            "bot3_pct": data["bot3_pct"],
            "risk_class": risk_class,
        })

    return {"teams": teams, "category_risk": risk}


HITTER_SLOTS_ORDER = ["C", "1B", "2B", "3B", "SS", "IF", "OF", "OF", "OF", "OF",
                       "UTIL", "UTIL", "BN", "IL"]


def _compute_team_totals_pace(
    players: list[dict],
    player_type: str,
    team_name: str | None = None,
) -> dict:
    """Build a team totals row with pace highlighting.

    Actuals come from Yahoo standings (the source of truth for team totals —
    correctly accounts for players added/dropped mid-season). Expected values
    are PA/IP-weighted averages of individual player projections.
    """
    from fantasy_baseball.analysis.pace import _z_to_color, STAT_VARIANCE

    active = [p for p in players if not p.get("is_bench", False)]

    # Look up team stats from standings
    if team_name is None:
        meta = read_meta() or {}
        team_name = meta.get("team_name", "")
    standings = read_cache(CacheKey.STANDINGS) or []
    team_stats: dict = {}
    for t in standings:
        if t.get("name") == team_name:
            team_stats = t.get("stats", {})
            break

    if player_type == "hitter":
        all_cats = ["PA", "R", "HR", "RBI", "SB", "AVG"]
        counting_cats = ["R", "HR", "RBI", "SB"]
        rate_cats = {"AVG": ("h", False)}
        opp_cat = "PA"
    else:
        all_cats = ["IP", "W", "K", "SV", "ERA", "WHIP"]
        counting_cats = ["W", "K", "SV"]
        rate_cats = {"ERA": ("er", True), "WHIP": ("h_allowed", True)}
        opp_cat = "IP"

    totals: dict = {}

    # Opportunity stat (PA / IP) — from standings
    totals[opp_cat] = {"actual": team_stats.get(opp_cat, 0), "color_class": "stat-neutral"}

    # Counting stats — actuals from standings, expected from player pace sums
    for cat in counting_cats:
        actual = team_stats.get(cat, 0)
        expected = sum(
            p.get("pace", {}).get(cat, {}).get("expected", 0) or 0
            for p in active
        )
        if expected > 0:
            ratio = actual / expected
            variance = STAT_VARIANCE.get(cat.lower(), 0.0)
            z = (ratio - 1.0) / variance if variance > 0 else 0.0
        else:
            z = 0.0
        totals[cat] = {
            "actual": actual,
            "expected": round(expected, 1),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
        }

    # Rate stats — actuals from standings, expected as IP/PA-weighted proj avg
    for rate_cat, (component, is_inverse) in rate_cats.items():
        actual_val = team_stats.get(rate_cat, 0.0)
        opp_key = "IP" if player_type != "hitter" else "PA"
        proj_vals = [
            (p.get("pace", {}).get(rate_cat, {}).get("expected", 0),
             p.get("pace", {}).get(opp_key, {}).get("actual", 0))
            for p in active if p.get("pace", {}).get(rate_cat, {}).get("expected")
        ]
        weighted = sum(v * opp for v, opp in proj_vals)
        total_opp = sum(opp for _, opp in proj_vals)
        expected_val = weighted / total_opp if total_opp > 0 else 0.0

        if expected_val > 0 and actual_val > 0:
            variance = STAT_VARIANCE.get(component, 0.0)
            z = (actual_val - expected_val) / (variance * expected_val) if variance > 0 else 0.0
            if is_inverse:
                z = -z
        else:
            z = 0.0

        fmt_precision = 3 if rate_cat == "AVG" else 2
        totals[rate_cat] = {
            "actual": round(actual_val, fmt_precision),
            "expected": round(expected_val, fmt_precision),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
        }

    return totals


def format_lineup_for_display(
    roster: list[dict], optimal: dict | None
) -> dict:
    """Format roster + optimizer output for the lineup template."""
    from fantasy_baseball.analysis.pace import compute_overall_pace
    from fantasy_baseball.models.player import Player

    hitters = []
    pitchers = []

    # Name -> roto_delta lookup built from optimizer output. Starters get a
    # delta; bench/IL players are absent (rendered as "—").
    roto_delta_by_name: dict[str, float] = {}
    if optimal:
        for a in optimal.get("hitter_lineup", []) or []:
            if "name" in a and "roto_delta" in a:
                roto_delta_by_name[a["name"]] = a["roto_delta"]
        for s in optimal.get("pitcher_starters", []) or []:
            if "name" in s and "roto_delta" in s:
                roto_delta_by_name[s["name"]] = s["roto_delta"]

    for p in roster:
        player = Player.from_dict(p)
        pos = player.selected_position or "BN"
        is_pitcher = pos in PITCHER_POSITIONS or (
            pos == "BN" and set(player.positions).issubset(PITCHER_POSITIONS | {"BN"})
        )

        ros_sgp = None
        if player.rest_of_season is not None:
            ros_sgp = (
                player.rest_of_season.sgp
                if player.rest_of_season.sgp is not None
                else player.rest_of_season.compute_sgp()
            )

        entry = {
            "name": player.name,
            "positions": player.positions,
            "selected_position": pos,
            "player_id": player.yahoo_id or "",
            "status": player.status,
            "sgp": ros_sgp,
            "delta_roto": roto_delta_by_name.get(player.name),
            "games": p.get("games_this_week", 0),
            "is_bench": pos in ("BN", "IL", "DL"),
            "is_il": "IL" in player.status or pos == "IL",
            "pace": player.pace or {},
            "overall_pace": compute_overall_pace(player.pace),
            "rank": player.rank.to_dict(),
            "preseason": player.preseason.to_dict() if player.preseason else None,
        }
        # Flatten ROS stats for template tooltip (h[rest_of_season_key] access pattern)
        if player.rest_of_season is not None:
            entry.update(player.rest_of_season.to_dict())
        # Preserve ros_sgp after the flatten (to_dict omits sgp if None, but
        # may overwrite with its own computed value — keep ours for display).
        entry["sgp"] = ros_sgp

        if is_pitcher:
            pitchers.append(entry)
        else:
            hitters.append(entry)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h["selected_position"], 99), -(h["sgp"] or 0)))
    pitchers.sort(key=lambda p: (p["is_bench"], -(p["sgp"] or 0)))

    moves = optimal.get("moves", []) if optimal else []

    return {
        "hitters": hitters,
        "pitchers": pitchers,
        "hitter_totals": _compute_team_totals_pace(hitters, "hitter"),
        "pitcher_totals": _compute_team_totals_pace(pitchers, "pitcher"),
        "is_optimal": len(moves) == 0,
        "moves": moves,
    }


def run_optimize() -> dict:
    """Re-run lineup optimizer from cached data. Returns moves list."""
    optimal = read_cache(CacheKey.LINEUP_OPTIMAL)
    if optimal:
        return {"moves": optimal.get("moves", []), "is_optimal": len(optimal.get("moves", [])) == 0}
    return {"moves": [], "is_optimal": True}



def compute_comparison_standings(
    roster_player_name: str,
    other_player: "Player",
    user_roster: "list[Player]",
    projected_standings: list[dict],
    user_team_name: str,
    *,
    roster_player_projection: "Player | None" = None,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Compute before/after roto standings for a player swap.

    Uses the cached ``projected_standings`` as the single source of
    truth for team stats (built once during the refresh pipeline).
    The swap delta is applied via :func:`apply_swap_delta` rather than
    recomputing from the roster — this guarantees the "before" totals
    match the standings page exactly.

    When ``roster_player_projection`` is provided, its ROS stats are
    used for the dropped player's contribution instead of the roster
    cache entry.  This keeps the delta consistent with the browse page
    (which reads from ``ros_projections``).

    When ``team_sds`` is provided, ``score_roto`` uses EV-based pairwise
    Gaussian scoring so the comparison matches the roster audit for the
    same swap.

    Returns dict with before/after stats and roto, or {"error": ...}.
    """
    from fantasy_baseball.scoring import score_roto
    from fantasy_baseball.trades.evaluate import (
        apply_swap_delta, find_player_by_name, player_rest_of_season_stats,
    )

    dropped = find_player_by_name(roster_player_name, user_roster)
    if dropped is None:
        return {"error": f"Player '{roster_player_name}' not found on roster"}

    loses_ros = player_rest_of_season_stats(roster_player_projection or dropped)
    gains_ros = player_rest_of_season_stats(other_player)

    all_stats_before = {t["name"]: dict(t["stats"]) for t in projected_standings}
    all_stats_after = dict(all_stats_before)
    all_stats_after[user_team_name] = apply_swap_delta(
        all_stats_before[user_team_name], loses_ros, gains_ros,
    )

    roto_before = score_roto(all_stats_before)
    roto_after = score_roto(all_stats_after)

    ev_roto_before = score_roto(all_stats_before, team_sds=team_sds)
    ev_roto_after = score_roto(all_stats_after, team_sds=team_sds)

    from fantasy_baseball.lineup.delta_roto import score_swap

    delta_roto = score_swap(ev_roto_before, ev_roto_after, user_team_name)

    return {
        "before": {
            "stats": all_stats_before,
            "roto": roto_before,
            "ev_roto": ev_roto_before,
        },
        "after": {
            "stats": all_stats_after,
            "roto": roto_after,
            "ev_roto": ev_roto_after,
        },
        "delta_roto": delta_roto.to_dict(),
        "categories": ALL_CATEGORIES,
        "user_team": user_team_name,
    }


def _compute_category_ranks(standings: StandingsSnapshot) -> dict[str, dict[str, int]]:
    """Compute per-category rank for each team (1 = best).

    For inverse categories (ERA, WHIP), lower value = rank 1.
    Uses epsilon comparison for float tie detection in rate stats.
    """
    ranks = {}
    for cat in ALL_CATEGORIES:
        reverse = cat not in INVERSE_CATS
        sorted_entries = sorted(standings.entries, key=lambda e: e.stats[cat], reverse=reverse)
        cat_ranks = {}
        prev_val = None
        prev_rank = 0
        for i, entry in enumerate(sorted_entries):
            val = entry.stats[cat]
            if prev_val is None or abs(val - prev_val) >= 1e-9:
                prev_rank = i + 1
                prev_val = val
            cat_ranks[entry.team_name] = prev_rank
        ranks[cat] = cat_ranks
    return ranks


def _compute_pending_moves_diff(
    today_roster: list[dict],
    future_roster: list[dict],
    team_name: str,
    team_key: str,
) -> list[dict]:
    """Compute pending-moves banner data from a roster diff.

    Compares the user's current roster against Yahoo's future-dated
    roster (via ``team.roster(day=next_tuesday)``) and returns the
    add/drop difference in the same shape the lineup UI banner
    expects.

    The diff uses normalized names so accent / casing variants don't
    produce spurious entries (e.g., "Julio Rodríguez" vs "Julio
    Rodriguez").

    Returns an empty list when the rosters match. When there are
    changes, returns a single move dict bundling all adds and all
    drops — matches the banner's existing multi-add/drop rendering.
    """
    from fantasy_baseball.utils.name_utils import normalize_name

    today_by_norm = {
        normalize_name(p["name"]): p for p in today_roster
    }
    future_by_norm = {
        normalize_name(p["name"]): p for p in future_roster
    }

    added_norms = set(future_by_norm) - set(today_by_norm)
    dropped_norms = set(today_by_norm) - set(future_by_norm)

    if not added_norms and not dropped_norms:
        return []

    adds = [
        {
            "name": future_by_norm[n]["name"],
            "positions": future_by_norm[n].get("positions", []),
        }
        for n in sorted(added_norms)
    ]
    drops = [
        {
            "name": today_by_norm[n]["name"],
            "positions": today_by_norm[n].get("positions", []),
        }
        for n in sorted(dropped_norms)
    ]

    return [{
        "team": team_name,
        "team_key": team_key,
        "adds": adds,
        "drops": drops,
    }]
