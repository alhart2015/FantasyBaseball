"""Cache management and data assembly for the season dashboard."""

import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.standings import CategoryStats, StandingsEntry, StandingsSnapshot
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES, HITTER_PROJ_KEYS, IL_STATUSES, INVERSE_STATS as INVERSE_CATS,
    PITCHER_PROJ_KEYS,
)
from fantasy_baseball.utils.positions import PITCHER_POSITIONS
from fantasy_baseball.utils.time_utils import local_now, local_today, next_tuesday

_refresh_lock = threading.Lock()
_refresh_status = {"running": False, "progress": "", "error": None}

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


def get_refresh_status() -> dict:
    with _refresh_lock:
        return dict(_refresh_status)


def _set_refresh_progress(msg: str) -> None:
    with _refresh_lock:
        _refresh_status["progress"] = msg

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

# Defaults for early-season teams missing stats
_STAT_DEFAULTS = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
                  "W": 0, "K": 0, "SV": 0, "ERA": 99.0, "WHIP": 99.0}


def _fill_stat_defaults(standings: list[dict]) -> None:
    """Ensure every team has all 10 stat keys (early season some are missing)."""
    for t in standings:
        filled = dict(_STAT_DEFAULTS)
        filled.update(t["stats"])
        t["stats"] = filled


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


CACHE_FILES = {
    "standings": "standings.json",
    "roster": "roster.json",
    "projections": "projections.json",
    "lineup_optimal": "lineup_optimal.json",
    "probable_starters": "probable_starters.json",
    "monte_carlo": "monte_carlo.json",
    "meta": "meta.json",
    "rankings": "rankings.json",
    "roster_audit": "roster_audit.json",
    "spoe": "spoe.json",
    "opp_rosters": "opp_rosters.json",
    "leverage": "leverage.json",
    "pending_moves": "pending_moves.json",
    "transaction_analyzer": "transaction_analyzer.json",
    "transactions": "transactions.json",
    "ros_projections": "ros_projections.json",
    "positions": "positions.json",
}


def read_cache(key: str, cache_dir: Path = CACHE_DIR) -> dict | list | None:
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
        raw = redis.get(f"cache:{key}")
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


def write_cache(key: str, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
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
                redis.set(f"cache:{key}", json.dumps(data))
            except Exception as e:
                print(f"[redis] write_cache({key}) failed: {e}")


def read_meta(cache_dir: Path = CACHE_DIR) -> dict:
    """Read cache metadata (last refresh time, week, etc.). Returns empty dict if missing."""
    return read_cache("meta", cache_dir) or {}


def _write_spoe_snapshot(spoe_result: dict) -> None:
    """Write a daily SPoE snapshot to Upstash under `spoe_snapshot:YYYY-MM-DD`.

    Separate from the main write_cache path because this key is not
    under the `cache:` prefix — it's a historical time series for the
    luck page to optionally render trend charts. No TTL; accumulates.
    """
    snapshot_date = spoe_result.get("snapshot_date")
    if not snapshot_date:
        return
    redis = _get_redis()
    if redis is None:
        return
    try:
        import json
        redis.set(
            f"spoe_snapshot:{snapshot_date}",
            json.dumps(spoe_result),
        )
    except Exception as exc:
        log.warning(f"Failed to write spoe_snapshot:{snapshot_date}: {exc}")


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
    standings: StandingsSnapshot, user_team_name: str
) -> dict:
    """Transform standings snapshot into display-ready structure with roto points and color codes.

    Args:
        standings: StandingsSnapshot with typed StandingsEntry objects.
        user_team_name: The authenticated user's team name for highlighting.

    Returns:
        {"teams": [...]} where each team has roto_points, is_user flag, color_classes, and rank.
    """
    if not standings.entries:
        return {"teams": []}

    # CategoryStats defaults (0.0 for counting, 99.0 for ERA/WHIP)
    # handle early-season missing data — no _fill_stat_defaults needed.
    all_stats = {e.team_name: e.stats for e in standings.entries}
    roto = score_roto(all_stats)

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
    standings: list[dict],
    hitters_proj: "pd.DataFrame",
    pitchers_proj: "pd.DataFrame",
    rest_of_season_hitters: "pd.DataFrame",
    rest_of_season_pitchers: "pd.DataFrame",
    user_leverage: dict[str, float],
    season_year: int,
) -> dict:
    """Build a fully enriched opponent lineup (projections, pace, dual wSGP).

    Args:
        roster: Raw roster from fetch_roster().
        opponent_name: Opponent team name (for leverage calculation).
        standings: Raw standings cache.
        hitters_proj: Blended hitter projections (with _name_norm column).
        pitchers_proj: Blended pitcher projections (with _name_norm column).
        rest_of_season_hitters: ROS hitter projections (may be empty DataFrame).
        rest_of_season_pitchers: ROS pitcher projections (may be empty DataFrame).
        user_leverage: User's leverage weights.
        season_year: Season year for game log lookup.

    Returns:
        Dict with "hitters" and "pitchers" lists, each entry containing
        projection stats, pace data, and dual wSGP (wsgp_them, wsgp_you).
    """
    from fantasy_baseball.analysis.pace import compute_overall_pace, compute_player_pace
    from fantasy_baseball.data.projections import match_roster_to_projections
    from fantasy_baseball.lineup.leverage import calculate_leverage
    from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
    from fantasy_baseball.utils.name_utils import normalize_name

    # Match roster to projections
    matched = match_roster_to_projections(roster, hitters_proj, pitchers_proj)

    # ROS projection lookup
    has_rest_of_season = not rest_of_season_hitters.empty or not rest_of_season_pitchers.empty
    if has_rest_of_season:
        rest_of_season_matched = match_roster_to_projections(roster, rest_of_season_hitters, rest_of_season_pitchers)
        rest_of_season_lookup = {normalize_name(p.name): p for p in rest_of_season_matched}
    else:
        rest_of_season_lookup = {}

    # Opponent leverage
    standings_snap = _standings_to_snapshot(standings)
    opp_leverage = calculate_leverage(standings_snap, opponent_name)

    # Load game log totals for pace
    hitter_logs, pitcher_logs = _load_game_log_totals(season_year)

    # Build enriched entries
    matched_names = set()
    enriched = []
    for player in matched:
        wsgp_them = calculate_weighted_sgp(player.rest_of_season, opp_leverage) if player.rest_of_season else 0.0
        wsgp_you = calculate_weighted_sgp(player.rest_of_season, user_leverage) if player.rest_of_season else 0.0
        norm = normalize_name(player.name)
        matched_names.add(norm)

        entry = player.to_flat_dict()
        entry["wsgp_them"] = wsgp_them
        entry["wsgp_you"] = wsgp_you

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
            entry["wsgp_them"] = 0.0
            entry["wsgp_you"] = 0.0
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
                                -h.get("wsgp_them", 0)))
    pitchers.sort(key=lambda p: (p.get("selected_position", "") in ("BN", "IL", "DL"),
                                 -p.get("wsgp_them", 0)))

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
    standings = read_cache("standings") or []
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

    for p in roster:
        player = Player.from_dict(p)
        pos = player.selected_position or "BN"
        is_pitcher = pos in PITCHER_POSITIONS or (
            pos == "BN" and set(player.positions).issubset(PITCHER_POSITIONS | {"BN"})
        )

        entry = {
            "name": player.name,
            "positions": player.positions,
            "selected_position": pos,
            "player_id": player.yahoo_id or "",
            "status": player.status,
            "wsgp": player.wsgp,
            "classification": player.classification,
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

        if is_pitcher:
            pitchers.append(entry)
        else:
            hitters.append(entry)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h["selected_position"], 99), -h["wsgp"]))
    pitchers.sort(key=lambda p: (p["is_bench"], -p["wsgp"]))

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
    optimal = read_cache("lineup_optimal")
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

    roto_before = score_roto(all_stats_before, team_sds=team_sds)
    roto_after = score_roto(all_stats_after, team_sds=team_sds)

    from fantasy_baseball.lineup.delta_roto import score_swap

    delta_roto = score_swap(roto_before, roto_after, user_team_name)

    return {
        "before": {
            "stats": all_stats_before,
            "roto": roto_before,
        },
        "after": {
            "stats": all_stats_after,
            "roto": roto_after,
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




def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    """Connect to Yahoo, fetch all data, run computations, and write cache files.

    Sets refresh status throughout so the UI can poll progress.
    """
    with _refresh_lock:
        _refresh_status["running"] = True
        _refresh_status["progress"] = "Starting..."
        _refresh_status["error"] = None

    from fantasy_baseball.web.job_logger import JobLogger
    logger = JobLogger("refresh")

    def _progress(msg):
        _set_refresh_progress(msg)
        logger.log(msg)
        log.info(msg)

    try:
        # Lazy imports — only loaded when refresh actually runs
        from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
        from fantasy_baseball.config import load_config
        from fantasy_baseball.data.mlb_schedule import get_week_schedule
        from fantasy_baseball.data.mlb_game_logs import fetch_game_log_totals
        from fantasy_baseball.lineup.leverage import calculate_leverage
        from fantasy_baseball.lineup.matchups import calculate_matchup_factors, get_team_batting_stats
        from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
        from fantasy_baseball.lineup.waivers import fetch_and_match_free_agents
        from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings, fetch_scoring_period
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.analysis.pace import compute_player_pace
        from fantasy_baseball.lineup.roster_audit import audit_roster

        project_root = Path(__file__).resolve().parents[3]

        # --- Step 1: Auth + league ---
        _progress("Authenticating with Yahoo...")
        sc = get_yahoo_session()
        config = load_config(project_root / "config" / "league.yaml")
        league = get_league(sc, config.league_id, config.game_code)

        # --- Step 2: Find user's team key ---
        _progress("Finding team...")
        teams = league.teams()
        user_team_key = None
        for key, team_info in teams.items():
            if team_info.get("name") == config.team_name:
                user_team_key = key
                break
        if user_team_key is None:
            # Fall back to first team if not found by name
            user_team_key = next(iter(teams))

        # --- Step 3: Fetch standings + roster ---
        _progress("Fetching standings...")
        standings = fetch_standings(league)
        _fill_stat_defaults(standings)
        write_cache("standings", standings, cache_dir)
        _progress(f"Fetched standings for {len(standings)} teams")

        # Compute the effective date for the next lineup lock. We fetch
        # all rosters at this date (via Yahoo's team.roster(day=...)) so
        # the audit/optimizer/waivers see the post-lock future state
        # without having to simulate pending transactions locally.
        # fetch_scoring_period returns Yahoo's Mon–Sun scoring week
        # (end_date is Sunday). The user's league locks lineups on
        # Tuesday morning, so the effective date is the next Tuesday
        # strictly after end_date — end_date + 1 would land on Monday,
        # one day too early.
        _progress("Computing effective date...")
        start_date, end_date = fetch_scoring_period(league)
        effective_date = next_tuesday(date.fromisoformat(end_date))
        _progress(f"Effective date (next lock): {effective_date}")

        standings_snap = _standings_to_snapshot(standings, effective_date)

        _progress("Fetching today's roster (for pending-moves diff)...")
        today_roster_raw = fetch_roster(league, user_team_key)

        _progress(f"Fetching future-dated roster for {effective_date}...")
        roster_raw = fetch_roster(league, user_team_key, day=effective_date)
        _progress(f"Fetched future roster: {len(roster_raw)} players")

        pending_moves = _compute_pending_moves_diff(
            today_roster_raw, roster_raw,
            team_name=config.team_name, team_key=user_team_key,
        )
        write_cache("pending_moves", pending_moves, cache_dir)
        if pending_moves:
            total_changes = sum(
                len(m["adds"]) + len(m["drops"]) for m in pending_moves
            )
            _progress(f"Pending moves: {total_changes} change(s) detected")

        # --- Step 4: Read preseason projections from Redis ---
        _progress("Loading projections...")
        import pandas as pd
        from fantasy_baseball.data.redis_store import (
            get_blended_projections as redis_get_blended,
            get_default_client as _redis_default_client,
        )
        _redis_client = _redis_default_client()
        if _redis_client is None:
            raise RuntimeError(
                "Redis client not configured: UPSTASH_REDIS_REST_URL / "
                "UPSTASH_REDIS_REST_TOKEN are not set in the environment. "
                "For local dev, put them in a .env file at the project root "
                "(get_default_client auto-loads it). On Render, set them in "
                "the service's environment variables."
            )
        _hitters_rows = redis_get_blended(_redis_client, "hitters") or []
        _pitchers_rows = redis_get_blended(_redis_client, "pitchers") or []
        if not _hitters_rows or not _pitchers_rows:
            raise RuntimeError(
                "Preseason projections not found in Redis "
                "(blended_projections:hitters / blended_projections:pitchers). "
                "Run `python scripts/build_db.py` once to populate them from "
                "the CSVs under data/projections/{season}/."
            )
        hitters_proj = pd.DataFrame(_hitters_rows)
        pitchers_proj = pd.DataFrame(_pitchers_rows)

        # Load ROS projections — blend latest dated CSV into Redis
        # (cache:ros_projections). No-op if no CSV dir exists locally
        # (Render has no CSVs on disk; the daily admin-triggered
        # _run_rest_of_season_fetch keeps Redis populated).
        _progress("Loading ROS projections...")
        projections_dir = project_root / "data" / "projections"
        from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros
        from fantasy_baseball.data.redis_store import (
            get_default_client, get_latest_roster_names,
        )
        try:
            rest_of_season_roster_names = get_latest_roster_names(get_default_client())
            blend_and_cache_ros(
                projections_dir,
                config.projection_systems, config.projection_weights,
                rest_of_season_roster_names, config.season_year,
                progress_cb=_progress,
            )
        except FileNotFoundError:
            # No local CSVs — fine on Render; the admin job keeps Redis populated.
            _progress("No local ROS CSV dir; relying on Redis cache")

        hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)
        _progress(f"Loaded {len(hitters_proj)} hitter + {len(pitchers_proj)} pitcher projections")

        # ROS projections live in Redis (cache:ros_projections). The
        # blend above just refreshed that key if local CSVs were
        # present; otherwise we fall back to whatever the daily admin
        # job wrote. Disk CSVs are no longer a fallback path.
        import pandas as pd
        rest_of_season_hitters = pd.DataFrame()
        rest_of_season_pitchers = pd.DataFrame()
        ros_cached = read_cache("ros_projections", cache_dir)
        if ros_cached:
            rest_of_season_hitters = pd.DataFrame(ros_cached.get("hitters", []))
            rest_of_season_pitchers = pd.DataFrame(ros_cached.get("pitchers", []))
        has_rest_of_season = not rest_of_season_hitters.empty or not rest_of_season_pitchers.empty
        if has_rest_of_season:
            _progress(f"Loaded ROS projections from Redis "
                      f"({len(rest_of_season_hitters)} hitters + {len(rest_of_season_pitchers)} pitchers)")

        preseason_hitters = hitters_proj
        preseason_pitchers = pitchers_proj
        if has_rest_of_season:
            rest_of_season_hitters["_name_norm"] = rest_of_season_hitters["name"].apply(normalize_name)
            rest_of_season_pitchers["_name_norm"] = rest_of_season_pitchers["name"].apply(normalize_name)
            _progress(f"Loaded {len(rest_of_season_hitters)} ROS hitters + {len(rest_of_season_pitchers)} ROS pitchers")
            # Use ROS projections as primary — they're the most current estimates
            hitters_proj = rest_of_season_hitters
            pitchers_proj = rest_of_season_pitchers
        else:
            _progress("WARNING: No ROS projections available — falling back to preseason")

        # --- Step 4b: Fetch opponent rosters (raw) ---
        _progress("Fetching opponent rosters...")
        from fantasy_baseball.data.projections import (
            hydrate_roster_entries,
            match_roster_to_projections,
        )
        from fantasy_baseball.models.league import League

        # Collect raw rosters keyed by team name — used only to feed the
        # Redis write below. League.from_redis will then be our source
        # of truth for roster data for the rest of the refresh.
        raw_rosters_by_team: dict[str, list[dict]] = {
            config.team_name: roster_raw,
        }

        def _fetch_opp(key_and_info):
            key, team_info = key_and_info
            tname = team_info.get("name", "")
            try:
                opp_raw = fetch_roster(league, key, day=effective_date)
                return (tname, opp_raw)
            except Exception:
                return None

        opp_items = [
            (key, info) for key, info in teams.items()
            if info.get("name", "") != config.team_name and key != user_team_key
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(_fetch_opp, opp_items):
                if result is None:
                    continue
                tname, opp_raw = result
                raw_rosters_by_team[tname] = opp_raw
        _progress(
            f"Fetched {len(raw_rosters_by_team)} rosters (user + opponents)"
        )

        # --- Step 4c: Write rosters + standings to Redis, then load League ---
        _progress("Writing roster snapshots to Redis...")
        from fantasy_baseball.data.redis_store import (
            get_default_client,
            write_roster_snapshot,
            write_standings_snapshot,
        )

        snapshot_date = effective_date.isoformat()
        for tname, team_raw in raw_rosters_by_team.items():
            # team_raw rows come from parse_roster: keys are "name",
            # "positions" (list), "selected_position", "player_id",
            # "status". Convert to the serialized shape the old
            # SQLite writer produced so downstream readers see an
            # identical blob.
            entries = [
                {
                    "slot": row["selected_position"],
                    "player_name": row["name"],
                    "positions": ", ".join(row.get("positions", [])),
                    "status": row.get("status") or "",
                    "yahoo_id": row.get("player_id") or "",
                }
                for row in team_raw
            ]
            write_roster_snapshot(
                get_default_client(), snapshot_date, tname, entries,
            )

        # Stat keys on the source dicts are UPPERCASE (R/HR/.../WHIP); the
        # old append_standings_snapshot lowercased them before writing, so
        # we preserve that shape here.
        snapshot_payload = {
            "teams": [
                {
                    "team": entry["name"],
                    "team_key": entry.get("team_key") or "",
                    "rank": entry.get("rank") or 0,
                    **{k.lower(): v for k, v in entry.get("stats", {}).items()},
                }
                for entry in standings
            ],
        }
        write_standings_snapshot(
            get_default_client(), snapshot_date, snapshot_payload,
        )

        _progress("Loading League from Redis...")
        league_model = League.from_redis(config.season_year)

        # --- Step 4d: Hydrate user roster + opponent rosters from League ---
        _progress("Hydrating user and opponent rosters...")
        user_team_model = league_model.team_by_name(config.team_name)
        user_roster_model = user_team_model.latest_roster()
        matched = hydrate_roster_entries(
            user_roster_model, hitters_proj, pitchers_proj,
        )

        opp_rosters: dict[str, list[Player]] = {}
        for team in league_model.teams:
            if team.name == config.team_name:
                continue
            if not team.rosters:
                continue
            latest = team.latest_roster()
            hydrated = hydrate_roster_entries(
                latest, hitters_proj, pitchers_proj,
            )
            if hydrated:
                opp_rosters[team.name] = hydrated
        _progress(f"Hydrated {len(opp_rosters)} opponent rosters")

        # Cache opponent rosters for on-demand trade search
        opp_rosters_flat = {
            tname: [p.to_dict() for p in roster]
            for tname, roster in opp_rosters.items()
        }
        write_cache("opp_rosters", opp_rosters_flat, cache_dir)

        # --- Step 4e: Build projected standings ---
        _progress("Projecting end-of-season standings...")
        from fantasy_baseball.scoring import project_team_stats

        all_team_rosters = {config.team_name: matched}
        all_team_rosters.update(opp_rosters)

        projected_standings = []
        for tname, roster in all_team_rosters.items():
            proj_stats = project_team_stats(roster, displacement=True)
            projected_standings.append({
                "name": tname,
                "team_key": "",
                "rank": 0,
                # project_team_stats returns a CategoryStats (dataclass);
                # serialize to a plain dict for the JSON cache write.
                "stats": proj_stats.to_dict(),
            })

        import math
        _season_start = date.fromisoformat(config.season_start)
        _season_end = date.fromisoformat(config.season_end)
        _total_days = (_season_end - _season_start).days
        _remaining_days = max(0, (_season_end - local_today()).days)
        fraction_remaining = (_remaining_days / _total_days) if _total_days > 0 else 0.0
        _sd_scale = math.sqrt(fraction_remaining)

        from fantasy_baseball.scoring import project_team_sds
        team_sds: dict[str, dict[str, float]] = {}
        for _tname, _troster in all_team_rosters.items():
            _raw_sds = project_team_sds(_troster, displacement=True)
            team_sds[_tname] = {c: sd * _sd_scale for c, sd in _raw_sds.items()}

        write_cache(
            "projections",
            {
                "projected_standings": projected_standings,
                "team_sds": team_sds,
                "fraction_remaining": fraction_remaining,
            },
            cache_dir,
        )
        _progress(f"Projected standings for {len(projected_standings)} teams")

        projected_standings_snap = _standings_to_snapshot(projected_standings, effective_date)

        # --- Step 5: Leverage weights ---
        _progress("Calculating leverage weights...")
        leverage = calculate_leverage(
            standings_snap, config.team_name,
            projected_standings=projected_standings_snap,
        )

        # --- Step 6: Match roster players to projections, compute wSGP ---
        _progress("Matching roster to projections...")
        from fantasy_baseball.models.player import Player

        # Match preseason projections for tooltip comparison
        preseason_matched = match_roster_to_projections(
            roster_raw, preseason_hitters, preseason_pitchers,
        )
        preseason_lookup = {normalize_name(p.name): p for p in preseason_matched}

        # Build Player objects from matched entries
        matched_names = set()
        roster_players: list[Player] = []
        for player in matched:
            norm = normalize_name(player.name)
            matched_names.add(norm)

            # Attach preseason stat bag
            pre_entry = preseason_lookup.get(norm)
            if pre_entry and pre_entry.rest_of_season:
                player.preseason = pre_entry.rest_of_season

            roster_players.append(player)

        # Include unmatched players
        for raw_player in roster_raw:
            if normalize_name(raw_player["name"]) not in matched_names:
                player = Player.from_dict({
                    **raw_player,
                    "player_type": PlayerType.PITCHER if set(raw_player.get("positions", [])) & PITCHER_POSITIONS else PlayerType.HITTER,
                })
                roster_players.append(player)

        _progress(f"Matched {len(roster_players)} players to projections")

        # --- Step 6b: Fetch MLB game logs ---
        _progress("Fetching MLB game logs...")
        fetch_game_log_totals(config.season_year, progress_cb=_progress)

        # --- Step 6c: Compute season-to-date pace vs projections ---
        _progress("Computing player pace...")
        hitter_logs, pitcher_logs = _load_game_log_totals(config.season_year)

        # Attach pace data to each roster player (pace compares actuals vs preseason)
        from fantasy_baseball.sgp.denominators import get_sgp_denominators
        sgp_denoms = get_sgp_denominators(config.sgp_overrides)
        for player in roster_players:
            norm = normalize_name(player.name)
            if player.player_type == PlayerType.HITTER:
                actuals = hitter_logs.get(norm, {})
                rest_of_season_keys = ["r", "hr", "rbi", "sb", "avg"]
            else:
                actuals = pitcher_logs.get(norm, {})
                rest_of_season_keys = ["w", "k", "sv", "era", "whip"]
            proj_keys = HITTER_PROJ_KEYS if player.player_type == PlayerType.HITTER else PITCHER_PROJ_KEYS
            pre_player = preseason_lookup.get(norm)
            if pre_player and pre_player.rest_of_season:
                projected = {k: getattr(pre_player.rest_of_season, k, 0) for k in proj_keys}
            else:
                projected = {k: 0 for k in proj_keys}
            rest_of_season_dict = {k: getattr(player.rest_of_season, k, 0) for k in rest_of_season_keys} if player.rest_of_season else None
            player.pace = compute_player_pace(
                actuals, projected, player.player_type,
                rest_of_season_stats=rest_of_season_dict, sgp_denoms=sgp_denoms,
            )

        # --- Step 6e: Compute wSGP on raw ROS stats ---
        # NOTE: recency blending was removed here because FanGraphs ROS
        # projections already incorporate early-season performance, and a
        # second layer of reliability weighting on top created inconsistencies
        # with projected_standings (see docs/superpowers/plans/2026-04-10-remove-recency-blending.md).
        for player in roster_players:
            if player.rest_of_season is not None:
                player.compute_wsgp(leverage)

        # --- Step 6d: Compute SGP rankings ---
        _progress("Computing SGP rankings...")
        from fantasy_baseball.sgp.rankings import (
            compute_sgp_rankings, compute_combined_sgp_rankings,
            compute_rankings_from_game_logs,
            rank_key, rank_key_from_positions, lookup_rank,
        )

        rest_of_season_ranks = compute_sgp_rankings(hitters_proj, pitchers_proj)
        preseason_ranks = compute_sgp_rankings(preseason_hitters, preseason_pitchers)
        current_ranks = compute_rankings_from_game_logs(hitter_logs, pitcher_logs)

        # Build combined lookup: {name::player_type: {ros, preseason, current}}
        all_keys = set(rest_of_season_ranks) | set(preseason_ranks) | set(current_ranks)
        rankings_lookup = {}
        for key in all_keys:
            rankings_lookup[key] = {
                "rest_of_season": rest_of_season_ranks.get(key),
                "preseason": preseason_ranks.get(key),
                "current": current_ranks.get(key),
            }

        write_cache("rankings", rankings_lookup, cache_dir)
        _progress(f"Ranked {len(rest_of_season_ranks)} ROS, {len(preseason_ranks)} preseason, {len(current_ranks)} current")

        # Attach ranks to roster players
        from fantasy_baseball.models.player import RankInfo
        for player in roster_players:
            rank_data = lookup_rank(rankings_lookup, player.fg_id, player.name, player.player_type)
            player.rank = RankInfo.from_dict(rank_data) if rank_data else RankInfo()

        # Classify roster players by league-wide value vs team fit
        from fantasy_baseball.lineup.player_classification import classify_roster
        rest_of_season_rank_lookup = {}
        for key, rank_data in rankings_lookup.items():
            ros = rank_data.get("rest_of_season")
            if ros is not None:
                rest_of_season_rank_lookup[key] = ros
        classifications = classify_roster(roster_players, rest_of_season_rank_lookup)
        for player in roster_players:
            player.classification = classifications.get(player.name, "")

        roster_flat = [p.to_flat_dict() for p in roster_players]
        write_cache("roster", roster_flat, cache_dir)

        # --- Step 7: Run lineup optimizer ---
        _progress("Optimizing lineup...")
        active_players = [p for p in roster_players if p.status not in IL_STATUSES]
        hitter_players = []
        pitcher_players = []
        for player in active_players:
            if set(player.positions) & PITCHER_POSITIONS:
                pitcher_players.append(player)
            else:
                hitter_players.append(player)

        optimal_hitters = optimize_hitter_lineup(
            hitter_players, leverage, config.roster_slots
        )
        optimal_pitchers_starters, optimal_pitchers_bench = optimize_pitcher_lineup(
            pitcher_players, leverage
        )

        # --- Step 8: Compare optimal to current, find moves ---
        _progress("Computing lineup moves...")
        moves = []
        for slot, player_name in optimal_hitters.items():
            for player in roster_players:
                if player.name == player_name:
                    current_slot = player.selected_position or "BN"
                    base_slot = slot.split("_")[0]
                    # Position is StrEnum; comparison is direct after
                    # enum normalization at the Player constructor
                    # boundary. See feat/player-position-enum.
                    bench_slots = {"BN", "IL", "DL"}
                    if current_slot != base_slot and (
                        current_slot in bench_slots or base_slot in bench_slots
                    ):
                        moves.append({
                            "action": "START",
                            "player": player_name,
                            "slot": base_slot,
                            "reason": f"wSGP: {player.wsgp:.1f}",
                        })
                    break

        optimal_data = {
            "hitter_lineup": optimal_hitters,
            "pitcher_starters": [p["name"] for p in optimal_pitchers_starters],
            "pitcher_bench": [p["name"] for p in optimal_pitchers_bench],
            "moves": moves,
        }
        write_cache("lineup_optimal", optimal_data, cache_dir)

        # --- Step 9: Probable starters ---
        _progress("Fetching schedule and matchup data...")
        # start_date and end_date were fetched earlier to compute effective_date
        schedule_cache_path = project_root / "data" / "weekly_schedule.json"
        schedule = get_week_schedule(start_date, end_date, schedule_cache_path)

        batting_stats_cache_path = project_root / "data" / "team_batting_stats.json"
        team_stats = get_team_batting_stats(batting_stats_cache_path)
        matchup_factors = calculate_matchup_factors(team_stats)

        pitcher_roster_for_schedule = [
            p for p in roster_players
            if set(p.positions) & PITCHER_POSITIONS
        ]
        from fantasy_baseball.lineup.matchups import get_probable_starters
        probable_starters = get_probable_starters(
            pitcher_roster_for_schedule, schedule or {},
            matchup_factors=matchup_factors, team_stats=team_stats,
        )
        write_cache("probable_starters", probable_starters, cache_dir)

        # --- Step 10: Roster audit ---
        _progress("Running roster audit...")
        fa_players, _ = fetch_and_match_free_agents(
            league, hitters_proj, pitchers_proj
        )

        # Cache positions for all known players (roster + opponents + FAs)
        from fantasy_baseball.utils.name_utils import normalize_name as _norm
        positions_map: dict[str, list[str]] = {}
        for p in roster_players:
            positions_map[_norm(p.name)] = list(p.positions)
        for _opp_roster in opp_rosters.values():
            for p in _opp_roster:
                positions_map[_norm(p.name)] = list(p.positions)
        for fa in fa_players:
            if fa.positions:
                positions_map[_norm(fa.name)] = list(fa.positions)
        write_cache("positions", positions_map, cache_dir)
        from fantasy_baseball.data.redis_store import set_positions, get_default_client
        set_positions(get_default_client(), positions_map)
        _progress(f"Cached positions for {len(positions_map)} players")

        audit_results = audit_roster(
            roster_players, fa_players, leverage, config.roster_slots,
            projected_standings=projected_standings,
            team_name=config.team_name,
            team_sds=team_sds,
        )
        write_cache("roster_audit", [e.to_dict() for e in audit_results], cache_dir)
        upgrades = sum(1 for e in audit_results if e.gap > 0)
        _progress(f"Roster audit: {upgrades} upgrade(s) found")

        # --- Step 11: Compute per-team leverage ---
        _progress("Computing leverage...")
        leverage_by_team: dict[str, dict] = {}
        for entry in standings_snap.entries:
            leverage_by_team[entry.team_name] = calculate_leverage(
                standings_snap, entry.team_name,
                projected_standings=projected_standings_snap,
            )
        write_cache("leverage", leverage_by_team, cache_dir)

        # --- Step 12: Monte Carlo simulation ---
        from fantasy_baseball.simulation import run_monte_carlo

        h_slots = sum(v for k, v in config.roster_slots.items()
                      if k not in ("P", "BN", "IL", "DL"))
        p_slots = config.roster_slots.get("P", 9)

        mc_rosters = all_team_rosters

        base_mc = run_monte_carlo(
            mc_rosters, h_slots, p_slots, config.team_name,
            n_iterations=1000, use_management=False,
            progress_cb=lambda i: _progress(f"Monte Carlo: iteration {i}/1000..."),
        )
        _progress("Pre-season Monte Carlo complete")
        mgmt_mc = run_monte_carlo(
            mc_rosters, h_slots, p_slots, config.team_name,
            n_iterations=1000, use_management=True,
            progress_cb=lambda i: _progress(f"MC + Roster Mgmt: iteration {i}/1000..."),
        )
        _progress("Pre-season + Mgmt Monte Carlo complete")

        # --- Step 13b: ROS Monte Carlo simulation ---
        rest_of_season_mc = None
        rest_of_season_mgmt_mc = None
        if has_rest_of_season:
            from fantasy_baseball.simulation import run_ros_monte_carlo

            season_start = date.fromisoformat(config.season_start)
            season_end = date.fromisoformat(config.season_end)
            total_days = (season_end - season_start).days
            remaining_days = max(0, (season_end - local_today()).days)
            fraction_remaining = remaining_days / total_days if total_days > 0 else 0

            # Build ROS rosters for all teams. hitters_proj/pitchers_proj
            # already ARE rest_of_season_hitters/rest_of_season_pitchers when has_rest_of_season is True
            # (see the assignment above), so opp_rosters is already
            # matched against ROS projections — just reuse it.
            rest_of_season_mc_rosters = {}
            if matched:
                rest_of_season_mc_rosters[config.team_name] = all_team_rosters.get(config.team_name, [])
            for tname, opp_players in opp_rosters.items():
                rest_of_season_mc_rosters[tname] = opp_players

            # Build actual standings dict
            actual_standings_dict = {
                s["name"]: s["stats"] for s in standings
            }

            if rest_of_season_mc_rosters:
                rest_of_season_mc = run_ros_monte_carlo(
                    team_rosters=rest_of_season_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=fraction_remaining,
                    h_slots=h_slots, p_slots=p_slots,
                    user_team_name=config.team_name,
                    n_iterations=1000, use_management=False,
                    progress_cb=lambda i: _progress(
                        f"Current MC: iteration {i}/1000..."
                    ),
                )
                _progress("Current Monte Carlo complete")
                rest_of_season_mgmt_mc = run_ros_monte_carlo(
                    team_rosters=rest_of_season_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=fraction_remaining,
                    h_slots=h_slots, p_slots=p_slots,
                    user_team_name=config.team_name,
                    n_iterations=1000, use_management=True,
                    progress_cb=lambda i: _progress(
                        f"Current MC + Mgmt: iteration {i}/1000..."
                    ),
                )
                _progress("Current + Mgmt Monte Carlo complete")

        write_cache("monte_carlo", {
            "base": base_mc,
            "with_management": mgmt_mc,
            "rest_of_season": rest_of_season_mc,
            "rest_of_season_with_management": rest_of_season_mgmt_mc,
        }, cache_dir)

        # --- Step 14: Compute season-to-date SPoE (luck analysis) ---
        # Reuses the league_model loaded in Step 4c. No separate DB
        # connection needed — SPoE walks Team.ownership_periods() on
        # the in-memory League object.
        _progress("Computing SPoE...")
        from fantasy_baseball.analysis.spoe import (
            build_preseason_lookup,
            compute_current_spoe,
        )

        preseason_lookup = build_preseason_lookup(
            preseason_hitters, preseason_pitchers,
        )
        spoe_result = compute_current_spoe(
            league_model,
            standings,
            preseason_lookup,
            config.season_start,
            config.season_end,
        )

        write_cache("spoe", spoe_result, cache_dir)
        _write_spoe_snapshot(spoe_result)
        _progress(f"SPoE computed for snapshot {spoe_result.get('snapshot_date')}")

        # --- Step 15: Transaction analyzer ---
        _progress("Analyzing transactions...")
        from fantasy_baseball.lineup.yahoo_roster import fetch_all_transactions
        from fantasy_baseball.analysis.transactions import (
            pair_standalone_moves,
            score_transaction,
            build_cache_output,
        )

        raw_txns = fetch_all_transactions(league)
        if raw_txns:
            # Load previously scored transactions from Redis/disk cache
            stored_txns = read_cache("transactions", cache_dir) or []
            existing_ids = {t["transaction_id"] for t in stored_txns}
            new_txns = [t for t in raw_txns
                        if t["transaction_id"] not in existing_ids]

            if new_txns:
                _progress(f"Scoring {len(new_txns)} new transaction(s)...")
                from fantasy_baseball.data.redis_store import (
                    get_default_client as _txn_redis_client,
                )
                _txn_client = _txn_redis_client()
                for txn in new_txns:
                    scores = score_transaction(
                        league_model, _txn_client, txn, config.season_year,
                    )
                    stored_txns.append({
                        "year": config.season_year,
                        **txn,
                        **scores,
                        "paired_with": None,
                    })

                # Re-pair all unpaired standalone moves
                unpaired = [t for t in stored_txns if not t.get("paired_with")]
                pairs = pair_standalone_moves(unpaired)
                by_id = {t["transaction_id"]: t for t in stored_txns}
                for drop_id, add_id in pairs:
                    by_id[drop_id]["paired_with"] = add_id
                    by_id[add_id]["paired_with"] = drop_id

            # Persist scored transactions to Redis and build display cache
            stored_txns.sort(key=lambda t: t.get("timestamp") or "")
            write_cache("transactions", stored_txns, cache_dir)
            cache_data = build_cache_output(stored_txns)
            write_cache("transaction_analyzer", cache_data, cache_dir)
            _progress(f"Analyzed {len(stored_txns)} total transaction(s)")

        # --- Step 16: Write meta ---
        _progress("Finalizing...")
        meta = {
            "last_refresh": local_now().strftime("%Y-%m-%d %H:%M"),
            "start_date": start_date,
            "end_date": end_date,
            "team_name": config.team_name,
        }
        write_cache("meta", meta, cache_dir)

        logger.finish("ok")
        _progress("Done")
        clear_opponent_cache()

    except Exception as exc:
        with _refresh_lock:
            _refresh_status["error"] = str(exc)
        logger.finish("error", str(exc))
        raise
    finally:
        with _refresh_lock:
            _refresh_status["running"] = False


