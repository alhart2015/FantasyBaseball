"""Cache management and data assembly for the season dashboard."""

import json
import logging
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.kv_store import KVStore, get_kv, is_remote
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.standings import ProjectedStandings, Standings, StandingsEntry
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES,
    HITTER_PROJ_KEYS,
    PITCHER_PROJ_KEYS,
    Category,
    OpportunityStat,
)
from fantasy_baseball.utils.constants import (
    INVERSE_STATS as INVERSE_CATS,
)
from fantasy_baseball.utils.positions import PITCHER_POSITIONS

log = logging.getLogger(__name__)


def _get_redis() -> KVStore | None:
    """Return the remote KV client on Render, ``None`` off-Render.

    Thin compatibility helper over ``kv_store.get_kv()``. The cache:*
    semantics in this module and in ``refresh_pipeline`` / ``job_logger``
    historically treated "Redis" as "remote or nothing" — off-Render,
    cache:* keys have dedicated JSON files on disk, so they don't need
    a local KV fallback. This helper preserves that semantic while
    keeping the RENDER gate in a single place (``kv_store.is_remote``).
    """
    return get_kv() if is_remote() else None

if TYPE_CHECKING:
    import pandas as pd

    from fantasy_baseball.models.player import Player

_opponent_cache: dict = {}
OPPONENT_CACHE_TTL_SECONDS = 900  # 15 minutes


def clear_opponent_cache() -> None:
    """Clear the opponent lineup in-memory cache (called on full refresh)."""
    _opponent_cache.clear()


CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


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
    """Read a cached JSON payload.

    On Render: Upstash is the source of truth; local disk serves as
    last-known-good fallback when Redis is unreachable. Off-Render
    (local dashboard, tests): disk only. The ``is_remote()`` gate in
    ``kv_store`` makes the remote path unreachable without
    ``RENDER=true``, so there is no code path from local to prod.
    """
    path = cache_dir / CACHE_FILES[key]

    redis = _get_redis()
    if redis is not None:
        try:
            raw = redis.get(redis_key(key))
        except Exception as e:
            print(f"[redis] read_cache({key}) failed: {e}")
            raw = None
        if raw is not None:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[redis] read_cache({key}) corrupt data, treating as miss")
                data = None
            if data is not None:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except OSError as e:
                    print(f"[redis] local write-back for {key} failed: {e}")
                return cast("dict | list", data)

    try:
        return cast("dict | list", json.loads(path.read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_cache(key: CacheKey, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
    """Atomically write a cached JSON payload. Writes to Redis only on Render."""
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

    redis = _get_redis()
    if redis is not None:
        try:
            redis.set(redis_key(key), json.dumps(data))
        except Exception as e:
            print(f"[redis] write_cache({key}) failed: {e}")


def read_meta(cache_dir: Path = CACHE_DIR) -> dict:
    """Read cache metadata (last refresh time, week, etc.). Returns empty dict if missing."""
    payload = read_cache(CacheKey.META, cache_dir)
    return payload if isinstance(payload, dict) else {}


def _load_game_log_totals(season_year: int) -> tuple[dict, dict]:
    """Load aggregated game log totals from Redis, keyed by normalized name.

    Returns (hitter_logs, pitcher_logs) where each is {normalized_name: {stat: value}}.
    The season_year parameter is accepted for signature compatibility but unused —
    Redis keys are not year-partitioned (current season only).
    """
    from fantasy_baseball.data.redis_store import get_game_log_totals
    from fantasy_baseball.utils.name_utils import normalize_name

    client = get_kv()
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
    standings: Standings,
    user_team_name: str,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
) -> dict:
    """Transform typed Standings into a display-ready structure with roto points and color codes.

    Args:
        standings: typed :class:`Standings` object.
        user_team_name: The authenticated user's team name for highlighting.
        team_sds: Per-team per-category standard deviations (``{team:
            {Category: sd}}``) for ERoto scoring. When provided,
            ``score_roto`` uses Gaussian pairwise win probabilities
            instead of deterministic rank-based scoring.

    Returns:
        ``{"teams": [...]}`` where each team dict contains:

        - ``stats``: a :class:`CategoryStats` object (Jinja indexes with
          ``Category`` enum).
        - ``roto_points``: ``{Category: float}`` per-category roto points.
        - ``roto_total``: the total roto points (Yahoo's when available).
        - ``score_roto_total``: always the raw ``score_roto`` total (diagnostic).
        - ``color_intensity``: ``{Category: float}`` with values in [-1, 1];
          tied categories are absent.
        - ``total_intensity``: float in [-1, 1] for the total column (absent
          when all teams tie on total).
        - ``rank``, ``is_user``, ``team_key``, ``sds`` (``{Category: sd}``).

    When every entry has ``yahoo_points_for`` set (live Yahoo standings
    path), the displayed total and rank come from Yahoo to match the
    official standings page exactly — display-precision ties in our
    rounded rate stats can't be broken locally. Per-category points
    still come from ``score_roto``, so category cells may not sum to
    the headline total; the gap is ±0.5 per tie and nets to zero.
    """
    if not standings.entries:
        return {"teams": []}

    # CategoryStats defaults (0.0 for counting, 99.0 for ERA/WHIP)
    # handle early-season missing data. Standings is structurally a
    # TeamStatsTable (team_name/stats on each entry); mypy can't see the
    # protocol variance through list[StandingsEntry] vs Sequence[TeamStatsRow].
    roto = score_roto(cast("Any", standings), team_sds=team_sds)

    has_yahoo_totals = all(e.yahoo_points_for is not None for e in standings.entries)
    yahoo_rank_by_name = {e.team_name: e.rank for e in standings.entries}

    teams: list[dict] = []
    team_totals: dict[str, float] = {}
    for entry in standings.entries:
        name = entry.team_name
        roto_cat_pts = roto[name]  # CategoryPoints
        score_roto_total = float(roto_cat_pts.total)

        if has_yahoo_totals:
            # yahoo_points_for is guaranteed non-None by has_yahoo_totals check.
            team_total = float(entry.yahoo_points_for)  # type: ignore[arg-type]
        else:
            team_total = score_roto_total

        team_totals[name] = team_total
        teams.append(
            {
                "name": name,
                "team_key": entry.team_key,
                "stats": entry.stats,
                "roto_points": {cat: roto_cat_pts[cat] for cat in ALL_CATEGORIES},
                "roto_total": team_total,
                "score_roto_total": score_roto_total,
                "is_user": name == user_team_name,
                "sds": team_sds.get(name, {}) if team_sds else {},
            }
        )

    intensity, total_intensity = _compute_color_intensity(standings, team_totals)
    for team in teams:
        team["color_intensity"] = intensity[team["name"]]
        if team["name"] in total_intensity:
            team["total_intensity"] = total_intensity[team["name"]]

    if has_yahoo_totals:
        teams.sort(key=lambda t: (-t["roto_total"], yahoo_rank_by_name[t["name"]]))
        for t in teams:
            t["rank"] = yahoo_rank_by_name[t["name"]]
    else:
        teams.sort(key=lambda t: t["roto_total"], reverse=True)
        for i, t in enumerate(teams):
            t["rank"] = i + 1

    return {"teams": teams}


def get_teams_list(standings: Standings, user_team_name: str) -> dict:
    """Build a team list for the opponent selector dropdown.

    Args:
        standings: typed :class:`Standings` (empty ``entries`` produces an
            empty result).
        user_team_name: The user's team name for flagging.

    Returns:
        {"teams": [...], "user_team_key": str | None}
    """
    if not standings.entries:
        return {"teams": [], "user_team_key": None}

    user_team_key: str | None = None
    teams: list[dict] = []
    for entry in standings.entries:
        is_user = entry.team_name == user_team_name
        if is_user:
            user_team_key = entry.team_key
        teams.append(
            {
                "name": entry.team_name,
                "team_key": entry.team_key,
                "rank": entry.rank,
                "is_user": is_user,
            }
        )

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
        roster,
        hitters_proj,
        pitchers_proj,
        context=f"opp-lineup:{opponent_name}",
    )

    # ROS projection lookup
    has_rest_of_season = not rest_of_season_hitters.empty or not rest_of_season_pitchers.empty
    if has_rest_of_season:
        rest_of_season_matched = match_roster_to_projections(
            roster,
            rest_of_season_hitters,
            rest_of_season_pitchers,
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
                entry["rest_of_season"] = {
                    k: getattr(rest_of_season_entry.rest_of_season, k, 0)
                    for k in ["r", "hr", "rbi", "sb", "avg"]
                }
            else:
                entry["rest_of_season"] = {
                    k: getattr(rest_of_season_entry.rest_of_season, k, 0)
                    for k in ["w", "k", "sv", "era", "whip"]
                }

        # Pace data
        ptype = player.player_type
        if ptype == PlayerType.HITTER:
            actuals = hitter_logs.get(norm, {})
        else:
            actuals = pitcher_logs.get(norm, {})
        proj_keys = HITTER_PROJ_KEYS if ptype == PlayerType.HITTER else PITCHER_PROJ_KEYS
        projected = {
            k: getattr(player.rest_of_season, k, 0) if player.rest_of_season else 0
            for k in proj_keys
        }
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
            pos == "BN" and set(p.get("positions", [])).issubset(PITCHER_POSITIONS | {"BN"})
        )
        if is_pitcher:
            pitchers.append(p)
        else:
            hitters.append(p)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(
        key=lambda h: (slot_rank.get(h.get("selected_position", ""), 99), -h.get("sgp", 0))
    )
    pitchers.sort(
        key=lambda p: (p.get("selected_position", "") in ("BN", "IL", "DL"), -p.get("sgp", 0))
    )

    return {
        "hitters": hitters,
        "pitchers": pitchers,
        "hitter_totals": _compute_team_totals_pace(hitters, "hitter", opponent_name),
        "pitcher_totals": _compute_team_totals_pace(pitchers, "pitcher", opponent_name),
    }


def format_monte_carlo_for_display(mc_data: dict, user_team_name: str) -> dict:
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
        teams.append(
            {
                "name": name,
                "median_pts": res["median_pts"],
                "p10": res["p10"],
                "p90": res["p90"],
                "first_pct": res["first_pct"],
                "top3_pct": res["top3_pct"],
                "is_user": name == user_team_name,
            }
        )
    teams.sort(key=lambda t: t["median_pts"], reverse=True)

    risk = []
    for cat, data in mc_data.get("category_risk", {}).items():
        if data["top3_pct"] >= 50:
            risk_class = "cat-top"
        elif data["bot3_pct"] >= 30:
            risk_class = "cat-bottom"
        else:
            risk_class = ""
        risk.append(
            {
                "cat": cat,
                "median_pts": data["median_pts"],
                "p10": data["p10"],
                "p90": data["p90"],
                "top3_pct": data["top3_pct"],
                "bot3_pct": data["bot3_pct"],
                "risk_class": risk_class,
            }
        )

    return {"teams": teams, "category_risk": risk}


HITTER_SLOTS_ORDER = [
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "IF",
    "OF",
    "OF",
    "OF",
    "OF",
    "UTIL",
    "UTIL",
    "BN",
    "IL",
]


def _compute_team_totals_pace(
    players: list[dict],
    player_type: str,
    team_name: str | None = None,
) -> dict:
    """Build a team totals row with pace highlighting.

    Actuals come from Yahoo standings (the source of truth for team totals —
    correctly accounts for players added/dropped mid-season). Expected values
    are PA/IP-weighted averages of individual player projections.

    Typed access: the standings cache deserializes into :class:`Standings`
    and we index by :class:`Category` / :class:`OpportunityStat` enum
    all the way to the template boundary, where we emit UPPERCASE
    stat-code keys (``"R"``, ``"HR"``, ``"PA"``, ...) because the Jinja
    template iterates fixed string lists.
    """
    from fantasy_baseball.analysis.pace import STAT_VARIANCE, _z_to_color

    active = [p for p in players if not p.get("is_bench", False)]

    # Look up this team's standings entry. Missing cache (unit tests,
    # pre-refresh) is fine — consumers get zero actuals.
    if team_name is None:
        meta = read_meta() or {}
        team_name = meta.get("team_name", "")
    team_entry: StandingsEntry | None = None
    raw = read_cache(CacheKey.STANDINGS)
    if isinstance(raw, dict):
        standings = Standings.from_json(raw)
        for entry in standings.entries:
            if entry.team_name == team_name:
                team_entry = entry
                break

    if player_type == "hitter":
        counting_cats: list[Category] = [Category.R, Category.HR, Category.RBI, Category.SB]
        rate_cats: dict[Category, tuple[str, bool]] = {Category.AVG: ("h", False)}
        opp_stat = OpportunityStat.PA
    else:
        counting_cats = [Category.W, Category.K, Category.SV]
        rate_cats = {Category.ERA: ("er", True), Category.WHIP: ("h_allowed", True)}
        opp_stat = OpportunityStat.IP

    totals: dict = {}

    # Opportunity stat (PA / IP) — pulled from StandingsEntry.extras.
    opp_actual = team_entry.extras.get(opp_stat, 0.0) if team_entry else 0.0
    totals[opp_stat.value] = {"actual": opp_actual, "color_class": "stat-neutral"}

    # Counting stats — actuals from standings, expected from player pace sums
    for cat in counting_cats:
        actual = team_entry.stats[cat] if team_entry else 0.0
        expected = sum(p.get("pace", {}).get(cat.value, {}).get("expected", 0) or 0 for p in active)
        if expected > 0:
            ratio = actual / expected
            variance = STAT_VARIANCE.get(cat.value.lower(), 0.0)
            z = (ratio - 1.0) / variance if variance > 0 else 0.0
        else:
            z = 0.0
        totals[cat.value] = {
            "actual": actual,
            "expected": round(expected, 1),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
        }

    # Rate stats — actuals from standings, expected as IP/PA-weighted proj avg
    for rate_cat, (component, is_inverse) in rate_cats.items():
        actual_val = team_entry.stats[rate_cat] if team_entry else 0.0
        opp_key = opp_stat.value
        proj_vals = [
            (
                p.get("pace", {}).get(rate_cat.value, {}).get("expected", 0),
                p.get("pace", {}).get(opp_key, {}).get("actual", 0),
            )
            for p in active
            if p.get("pace", {}).get(rate_cat.value, {}).get("expected")
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

        fmt_precision = 3 if rate_cat == Category.AVG else 2
        totals[rate_cat.value] = {
            "actual": round(actual_val, fmt_precision),
            "expected": round(expected_val, fmt_precision),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
        }

    return totals


def format_lineup_for_display(roster: list[dict], optimal: dict | None) -> dict:
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
    if isinstance(optimal, dict):
        return {"moves": optimal.get("moves", []), "is_optimal": len(optimal.get("moves", [])) == 0}
    return {"moves": [], "is_optimal": True}


def compute_comparison_standings(
    roster_player_name: str,
    other_player: "Player",
    user_roster: "list[Player]",
    projected_standings: ProjectedStandings,
    user_team_name: str,
    *,
    roster_player_projection: "Player | None" = None,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
) -> dict:
    """Compute before/after roto standings for a player swap.

    Uses the typed ``projected_standings`` as the single source of
    truth for team stats (built once during the refresh pipeline).
    The swap delta is applied via :func:`apply_swap_delta` rather than
    recomputing from the roster — this guarantees the "before" totals
    match the standings page exactly.

    When ``roster_player_projection`` is provided, its ROS stats are
    used for the dropped player's contribution instead of the roster
    cache entry.  This keeps the delta consistent with the browse page
    (which reads from ``ros_projections``).

    When ``team_sds`` is provided, ``score_roto`` uses EV-based
    pairwise Gaussian scoring so the comparison matches the roster
    audit for the same swap.

    Returns dict with before/after stats and roto, or {"error": ...}.
    The ``stats`` entries inside before/after are uppercase-string-keyed
    dicts (the shape :func:`apply_swap_delta` operates on and that the
    Flask/JSON boundary expects).
    """
    from fantasy_baseball.scoring import score_roto_dict
    from fantasy_baseball.trades.evaluate import (
        apply_swap_delta,
        find_player_by_name,
        player_rest_of_season_stats,
    )

    dropped = find_player_by_name(roster_player_name, user_roster)
    if dropped is None:
        return {"error": f"Player '{roster_player_name}' not found on roster"}

    loses_ros = player_rest_of_season_stats(roster_player_projection or dropped)
    gains_ros = player_rest_of_season_stats(other_player)

    all_stats_before = {e.team_name: e.stats.to_dict() for e in projected_standings.entries}
    all_stats_after = dict(all_stats_before)
    all_stats_after[user_team_name] = apply_swap_delta(
        all_stats_before[user_team_name],
        loses_ros,
        gains_ros,
    )

    roto_before = score_roto_dict(all_stats_before)
    roto_after = score_roto_dict(all_stats_after)

    ev_roto_before = score_roto_dict(all_stats_before, team_sds=team_sds)
    ev_roto_after = score_roto_dict(all_stats_after, team_sds=team_sds)

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


def _compute_color_intensity(
    standings: Standings,
    team_totals: dict[str, float],
) -> tuple[dict[str, dict[Category, float]], dict[str, float]]:
    """Per-team, per-category signed intensity in [-1, 1].

    For each category, intensity = 2 * ((value - min) / (max - min)) - 1,
    with ERA / WHIP (``INVERSE_CATS``) flipped so the lowest value is +1.0.
    Categories where every team is tied (``max == min``) are omitted —
    callers render those cells neutral.

    Returns a tuple:
        - ``per_cat``: ``{team_name: {Category: float}}`` — category intensities.
        - ``total``: ``{team_name: float}`` — intensity for the total column;
          teams are absent when every total is tied.
    """
    per_cat: dict[str, dict[Category, float]] = {e.team_name: {} for e in standings.entries}

    for cat in ALL_CATEGORIES:
        vals = {e.team_name: float(e.stats[cat]) for e in standings.entries}
        lo, hi = min(vals.values()), max(vals.values())
        if hi - lo < 1e-12:
            continue  # tied category — omit the key for every team
        span = hi - lo
        for name, v in vals.items():
            t = (v - lo) / span
            if cat in INVERSE_CATS:
                t = 1.0 - t
            per_cat[name][cat] = 2.0 * t - 1.0

    total: dict[str, float] = {}
    if team_totals:
        lo_t, hi_t = min(team_totals.values()), max(team_totals.values())
        if hi_t - lo_t >= 1e-12:
            span_t = hi_t - lo_t
            for name, v in team_totals.items():
                t = (v - lo_t) / span_t
                total[name] = 2.0 * t - 1.0

    return per_cat, total


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

    today_by_norm = {normalize_name(p["name"]): p for p in today_roster}
    future_by_norm = {normalize_name(p["name"]): p for p in future_roster}

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

    return [
        {
            "team": team_name,
            "team_key": team_key,
            "adds": adds,
            "drops": drops,
        }
    ]
