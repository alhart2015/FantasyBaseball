"""Cache management and data assembly for the season dashboard."""

import json
import logging
import os
import subprocess
from collections.abc import Mapping
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fantasy_baseball.category_odds import category_finish_odds
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.kv_store import KVStore, get_kv, is_remote
from fantasy_baseball.lineup.delta_roto import compute_one_for_one_band, score_swap
from fantasy_baseball.models.player import HitterStats, PitcherStats, PlayerType
from fantasy_baseball.models.positions import BENCH_SLOTS
from fantasy_baseball.models.standings import ProjectedStandings, Standings, StandingsEntry
from fantasy_baseball.scoring import score_roto, score_roto_dict
from fantasy_baseball.trades.evaluate import build_swap_standings, find_player_by_name
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES,
    COUNTING_STATS,
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


if TYPE_CHECKING:
    import pandas as pd

    from fantasy_baseball.models.player import Player

_opponent_cache: dict = {}
OPPONENT_CACHE_TTL_SECONDS = 900  # 15 minutes


def clear_opponent_cache() -> None:
    """Clear the opponent lineup in-memory cache (called on full refresh)."""
    _opponent_cache.clear()


# --- Cache provenance envelope ---------------------------------------------
# Every cache:* payload is stored as ``{"_meta": {...}, "_data": <payload>}``:
# write_cache wraps, read_cache unwraps. The envelope stamps the running code's
# git SHA and the UTC write time so version/time skew between keys (e.g. an old
# cache:projections vs a newer cache:standings_breakdown) is detectable by
# inspecting the stored blob instead of being invisible. Reads of bare
# (pre-envelope) payloads pass through unchanged for backward compatibility.
_ENVELOPE_META = "_meta"
_ENVELOPE_DATA = "_data"

_code_sha_cache: str | None = None
_code_sha_git_attempted: bool = False

_current_job: ContextVar[str | None] = ContextVar("fantasy_cache_job", default=None)


def set_cache_job(job: str | None) -> Token[str | None]:
    """Set the job label stamped into subsequent cache provenance envelopes.

    Entry points (the dashboard refresh, the ROS fetch) call this so every
    cache:* blob they write records its writer. Returns a token; pass it to
    :func:`reset_cache_job` in a ``finally`` to restore the prior label so a
    job set on a reused/synchronous worker thread cannot leak into the next
    job's writes.
    """
    return _current_job.set(job)


def reset_cache_job(token: Token[str | None]) -> None:
    """Restore the job label captured by a prior :func:`set_cache_job`."""
    _current_job.reset(token)


def _utc_now_iso() -> str:
    """UTC write timestamp for the cache provenance envelope."""
    return datetime.now(UTC).isoformat()


def _code_sha() -> str:
    """Short git SHA of the running code; memoized once a real SHA resolves.

    Prefers ``RENDER_GIT_COMMIT`` (set it in ``render.yaml`` so prod blobs
    carry a real SHA). Off Render it falls back to ``git rev-parse`` run in
    the repo root. On Render the git fallback is skipped -- the deployed slug
    may not be a git checkout, so forking git there is pure waste.

    Returns ``"unknown"`` when neither source resolves. The resolved SHA is
    memoized; a failure is NOT memoized as ``"unknown"`` (so a later call can
    still pick up RENDER_GIT_COMMIT), but git is forked at most ONCE per
    process (``_code_sha_git_attempted``) so a persistently-failing git off
    Render can't spawn a subprocess on every cache write.
    """
    global _code_sha_cache, _code_sha_git_attempted
    if _code_sha_cache is not None:
        return _code_sha_cache
    sha = os.environ.get("RENDER_GIT_COMMIT", "")
    if not sha and not is_remote() and not _code_sha_git_attempted:
        _code_sha_git_attempted = True
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
                cwd=Path(__file__).resolve().parents[3],
            ).stdout.strip()
        except Exception:
            sha = ""
    if sha:
        _code_sha_cache = sha
    return sha or "unknown"


def serialize_cache_payload(data: dict | list, extra_meta: dict | None = None) -> str:
    """Serialize a payload into the canonical enveloped cache string.

    Wraps ``data`` as ``{_meta: {_written_at, _sha, _job}, _data: data}`` and
    JSON-dumps it. ``write_cache`` writes this for cache:* keys; use
    :func:`write_cache_to` to write the same shape to an explicit KV client
    that bypasses ``write_cache`` (e.g. mirroring STREAK_SCORES to remote
    Upstash from a local refresh), so it reads back through ``read_cache``.

    ``extra_meta`` merges additional provenance fields into ``_meta`` (e.g.
    the ROS snapshot date that produced a projections blob) without touching
    ``_data``, so consumers are unaffected and the context is inspectable.
    """
    meta = {
        "_written_at": _utc_now_iso(),
        "_sha": _code_sha(),
        "_job": _current_job.get(),
    }
    if extra_meta:
        meta.update(extra_meta)
    envelope = {
        _ENVELOPE_META: meta,
        _ENVELOPE_DATA: data,
    }
    return json.dumps(envelope)


def _is_envelope(obj: object) -> bool:
    """True when ``obj`` is a provenance envelope produced by write_cache."""
    return (
        isinstance(obj, dict)
        and _ENVELOPE_META in obj
        and _ENVELOPE_DATA in obj
        and isinstance(obj[_ENVELOPE_META], dict)
    )


def read_cache(key: CacheKey) -> dict | list | None:
    """Read a cached payload from the KV store.

    Routes through ``kv_store.get_kv()``: Upstash on Render, SQLite
    locally. The ``RENDER`` gate in ``kv_store`` ensures off-Render
    callers cannot reach Upstash even with creds present. Transparently
    unwraps the provenance envelope; bare legacy payloads pass through.
    """
    kv = get_kv()
    try:
        raw = kv.get(redis_key(key))
    except Exception as e:
        log.warning(f"read_cache({key}) KV read failed: {e}")
        return None
    if raw is None:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"read_cache({key}) corrupt KV data, treating as miss")
        return None
    if _is_envelope(obj):
        return cast("dict | list", obj[_ENVELOPE_DATA])
    return cast("dict | list", obj)


def read_cache_dict(key: CacheKey) -> dict[str, Any] | None:
    """Read a cached payload, narrowed to dict.

    Returns ``None`` if the cache is missing, corrupt, or holds a list
    (unexpected shape for this key). Prefer this over ``read_cache``
    when the caller knows the key stores a dict — it lets mypy see the
    shape without ``cast()``.
    """
    payload = read_cache(key)
    return payload if isinstance(payload, dict) else None


def read_cache_list(key: CacheKey) -> list[Any] | None:
    """Read a cached payload, narrowed to list.

    Returns ``None`` if the cache is missing, corrupt, or holds a dict
    (unexpected shape for this key). See :func:`read_cache_dict`.
    """
    payload = read_cache(key)
    return payload if isinstance(payload, list) else None


def write_cache_to(
    client: KVStore, key: CacheKey, data: dict | list, extra_meta: dict | None = None
) -> None:
    """Write an enveloped cache:* value to an explicit KV client.

    Shared primitive behind :func:`write_cache` (which targets ``get_kv()``)
    and any path mirroring a cache:* key to a second client (e.g. the local
    refresh pushing STREAK_SCORES to remote Upstash), so both produce the
    identical envelope shape. ``extra_meta`` is stamped into the envelope
    ``_meta`` (see :func:`serialize_cache_payload`). Does not swallow errors
    -- the caller decides.
    """
    client.set(redis_key(key), serialize_cache_payload(data, extra_meta))


def write_cache(
    key: CacheKey,
    data: dict | list,
    extra_meta: dict | None = None,
    *,
    required: bool = True,
) -> None:
    """Write a cached payload to the KV store.

    Routes through ``kv_store.get_kv()``: Upstash on Render, SQLite
    locally. ``extra_meta`` is stamped into the envelope ``_meta``.

    ``required`` (default True): a write error propagates, so a swallowed
    write can't let a refresh report success after silently writing nothing
    (a partial cache that reads as complete and is never retried). The job's
    error handler then fails the run and QStash redelivers it.

    ``required=False``: for genuinely non-load-bearing keys (auxiliary
    dashboard panels -- e.g. leverage, SPoE, transactions, finish-odds). A
    write error is logged and swallowed so a transient blip on a cosmetic key
    doesn't abort -- and force a full re-run of -- the whole refresh after the
    load-bearing standings/roster/lineup already wrote successfully.
    """
    try:
        write_cache_to(get_kv(), key, data, extra_meta)
    except Exception:
        if required:
            raise
        log.warning("write_cache(%s) failed (non-required key); skipping", key, exc_info=True)


def read_meta() -> dict:
    """Read cache metadata (last refresh time, week, etc.). Returns empty dict if missing."""
    payload = read_cache(CacheKey.META)
    return payload if isinstance(payload, dict) else {}


def _load_game_log_totals() -> tuple[dict, dict]:
    """Load aggregated game log totals from Redis, keyed by normalized name.

    Returns (hitter_logs, pitcher_logs) where each is {normalized_name: {stat: value}}.
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


def _category_odds(rows: list[dict], cat: Category) -> dict | None:
    """User-team finish odds for one category's rows, or None if no user row.

    Percentages are rounded to whole numbers (display spec). ``rows`` may be
    in any order; ``category_finish_odds`` works over the team set.
    """
    user_index = next((i for i, r in enumerate(rows) if r["is_user"]), None)
    if user_index is None:
        return None
    odds = category_finish_odds(
        [r["value"] for r in rows],
        [r["sd"] for r in rows],
        user_index,
        higher_is_better=cat not in INVERSE_CATS,
    )
    return {
        "first_pct": round(odds.first_pct),
        "top3_pct": round(odds.top3_pct),
        "wins": odds.clear_wins,
        "opponents": odds.opponents,
    }


def _category_bars_one_flavor(data: dict | None) -> dict[str, dict]:
    """Reshape one standings display dict into per-category ranked rows + odds.

    Each category maps to ``{"rows": [...], "odds": {...} | None}``. ``rows``
    are ``{team, value, sd, is_user}`` sorted best-on-top: counting/AVG
    descending, ERA/WHIP ascending (lower is better). ``sd`` defaults to 0.0
    when a category is absent from a team's ``sds``. ``odds`` carries the user
    team's whole-number 1st/top-3 percentages, clear-win count, and opponent
    count (None when there is no user team in the data).
    """
    if not data or not data.get("teams"):
        return {}
    out: dict[str, dict] = {}
    for cat in ALL_CATEGORIES:
        rows = [
            {
                "team": team["name"],
                "value": team["stats"][cat],
                "sd": (team.get("sds") or {}).get(cat, 0.0),
                "is_user": team["is_user"],
            }
            for team in data["teams"]
        ]
        # reverse=True for "higher is better"; ERA/WHIP (INVERSE_CATS) sort
        # ascending so the lowest (best) team lands on top. Python's sort is
        # stable, so ties keep the input order.
        rows.sort(key=lambda r: r["value"], reverse=cat not in INVERSE_CATS)
        out[cat.value] = {"rows": rows, "odds": _category_odds(rows, cat)}
    return out


def format_category_bars_for_display(
    preseason_data: dict | None,
    current_projected_data: dict | None,
) -> dict[str, dict[str, dict]]:
    """Build the Category Bars chart payload from the two standings display dicts.

    Returns ``{"preseason": {CAT: {"rows": [...], "odds": {...}}}, "current": {...}}``
    where each row is ``{team, value, sd, is_user}`` sorted best-on-top and
    ``odds`` is the user team's per-category finish odds (see
    ``_category_bars_one_flavor``). A missing flavor (``None``, pre-refresh)
    yields an empty ``{}`` for that flavor.
    """
    return {
        "preseason": _category_bars_one_flavor(preseason_data),
        "current": _category_bars_one_flavor(current_projected_data),
    }


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
) -> dict:
    """Build a fully enriched opponent lineup (projections, pace, SGP).

    Args:
        roster: Raw roster from fetch_roster().
        opponent_name: Opponent team name (used for logging/context).
        hitters_proj: Blended hitter projections (with _name_norm column).
        pitchers_proj: Blended pitcher projections (with _name_norm column).
        rest_of_season_hitters: ROS hitter projections (may be empty DataFrame).
        rest_of_season_pitchers: ROS pitcher projections (may be empty DataFrame).

    Returns:
        Dict with "hitters" and "pitchers" lists, each entry containing
        projection stats, pace data, and per-player ROS-based SGP (matching the
        user lineup; falls back to 0.0 when no ROS projection matches).
    """
    from fantasy_baseball.analysis.pace import compute_overall_pace, compute_player_pace
    from fantasy_baseball.data.projections import match_roster_to_projections
    from fantasy_baseball.models.player import RankInfo
    from fantasy_baseball.sgp.rankings import lookup_rank
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
    hitter_logs, pitcher_logs = _load_game_log_totals()

    # Rankings (populated during refresh; absent on cold cache).
    rankings = read_cache_dict(CacheKey.RANKINGS) or {}

    # Build enriched entries
    matched_names = set()
    enriched = []
    for player in matched:
        norm = normalize_name(player.name)
        matched_names.add(norm)

        rank_data = lookup_rank(rankings, player.fg_id, player.name, player.player_type)
        if rank_data:
            player.rank = RankInfo.from_dict(rank_data)

        entry = player.to_flat_dict()
        entry.setdefault("sgp", 0.0)  # ROS-less fallback; overwritten below when ROS exists
        entry["delta_roto"] = None  # opponent rows don't have a swap delta

        # ROS projection: overwrite the SGP, the nested ros dict, AND the flat stat
        # keys so both the `sgp` column and the `h[rest_of_season_key]` tooltips
        # reflect ROS-source projections, not the blended preseason from
        # to_flat_dict. SGP MUST be ROS-based to match the user lineup
        # (format_lineup_for_display): the preseason blend carries a full-season
        # line, ~2x a starter's remaining-season value, so sourcing SGP from it
        # made the opponent column an apples-to-oranges comparison.
        rest_of_season_entry = rest_of_season_lookup.get(norm)
        if rest_of_season_entry and rest_of_season_entry.rest_of_season:
            ros_stats = rest_of_season_entry.rest_of_season
            entry["sgp"] = ros_stats.compute_sgp()
            ros_keys = (
                ["r", "hr", "rbi", "sb", "avg"]
                if player.player_type == PlayerType.HITTER
                else ["w", "k", "sv", "era", "whip", "ip"]
            )
            ros_dict = {k: getattr(ros_stats, k, 0) for k in ros_keys}
            entry["rest_of_season"] = ros_dict
            entry.update(ros_dict)
            entry["display_stats"] = _display_map(ros_stats, player.player_type, "ros")
        entry.setdefault("display_stats", {})

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
            entry["delta_roto"] = None
            entry["pace"] = {}
            entry["overall_pace"] = compute_overall_pace(entry["pace"])
            entry["display_stats"] = {}
            enriched.append(entry)

    # Split into hitters and pitchers
    hitters = []
    pitchers = []
    for p in enriched:
        pos = p.get("selected_position", "BN")
        p["is_bench"] = pos in BENCH_SLOTS
        p["is_il"] = "IL" in (p.get("status") or "") or pos == "IL"
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
    pitchers.sort(key=lambda p: (p.get("selected_position", "") in BENCH_SLOTS, -p.get("sgp", 0)))

    return {
        "hitters": hitters,
        "pitchers": pitchers,
        "hitter_totals": _compute_team_totals_pace(hitters, PlayerType.HITTER, opponent_name),
        "pitcher_totals": _compute_team_totals_pace(pitchers, PlayerType.PITCHER, opponent_name),
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
    player_type: PlayerType,
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

    if player_type == PlayerType.HITTER:
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


_legacy_moves_warning_logged = False


def _empty_moves() -> dict:
    """Fresh empty-shape moves dict. Avoids shared-list aliasing across callers."""
    return {"swaps": [], "unpaired_starts": [], "unpaired_benches": []}


def _normalize_moves(raw: object) -> dict:
    """Coerce cached ``moves`` field into the structured shape.

    Returns the empty shape for any of these cases:
    - ``None`` (no optimizer output yet)
    - a list (legacy cache from before the swap-pairs change)
    - a dict missing the expected keys

    Otherwise returns the raw dict with each list defaulted to ``[]``.

    Logs a warning ONCE per process when the legacy list shape is detected,
    so an operator can correlate "user reports the lineup looks optimal but
    isn't" with "we shipped the swap-pairs change and the cache hasn't been
    refreshed yet." The next refresh repopulates the cache in the new shape
    and the warning stops firing.
    """
    if isinstance(raw, list):
        global _legacy_moves_warning_logged
        if not _legacy_moves_warning_logged:
            log.warning(
                "Legacy list-shaped lineup moves cache detected; "
                "rendering as optimal until next refresh repopulates it."
            )
            _legacy_moves_warning_logged = True
        return _empty_moves()
    if not isinstance(raw, dict):
        return _empty_moves()
    return {
        "swaps": list(raw.get("swaps") or []),
        "unpaired_starts": list(raw.get("unpaired_starts") or []),
        "unpaired_benches": list(raw.get("unpaired_benches") or []),
    }


def _derive_ytd_stats(full, ros, player_type):
    """YTD actuals = full_season - rest_of_season, per counting component,
    clamped at >= 0. Rate stats (avg/era/whip) are recomputed from the
    subtracted components by the stats from_dict constructors. Returns a
    HitterStats/PitcherStats, or None when either input is missing.

    When ``full`` (full_season_projection) or ``ros`` (rest_of_season) is
    absent -- i.e. the player is unmatched in the full-season projection
    pool -- returns None, so the YTD view shows empty cells ('-') for that
    player via _display_map returning {}.
    """
    if full is None or ros is None:
        return None
    if player_type == PlayerType.HITTER:
        cols = ["pa", "ab", "h", "r", "hr", "rbi", "sb"]
        d = {k: max(0.0, getattr(full, k) - getattr(ros, k)) for k in cols}
        return HitterStats.from_dict(d)
    cols = ["ip", "w", "k", "sv", "er", "bb", "h_allowed"]
    d = {k: max(0.0, getattr(full, k) - getattr(ros, k)) for k in cols}
    return PitcherStats.from_dict(d)


def _display_map(stats, player_type, basis):
    """Per-category display values for the chosen basis, keyed by the same
    uppercase category names the tbody templates loop over. For the YTD basis,
    a player with zero volume (PA/IP) yields all-None values so the template
    renders '--' (matching today's no-games appearance)."""
    if stats is None:
        return {}
    if player_type == PlayerType.HITTER:
        m = {
            "PA": stats.pa,
            "R": stats.r,
            "HR": stats.hr,
            "RBI": stats.rbi,
            "SB": stats.sb,
            "AVG": stats.avg,
        }
        volume = stats.pa
    else:
        m = {
            "IP": stats.ip,
            "W": stats.w,
            "K": stats.k,
            "SV": stats.sv,
            "ERA": stats.era,
            "WHIP": stats.whip,
        }
        volume = stats.ip
    if basis == "ytd" and volume == 0:
        return {k: None for k in m}
    return m


def format_lineup_for_display(roster: list[dict], optimal: dict | None, basis: str = "ros") -> dict:
    """Format roster + optimizer output for the lineup template."""
    from fantasy_baseball.analysis.pace import compute_overall_pace
    from fantasy_baseball.models.player import Player

    # Validate basis once before the loop; unknown values fall back to "ros".
    if basis not in ("ros", "ytd", "total"):
        basis = "ros"

    hitters = []
    pitchers = []

    # Name -> roto_delta and band lookups built from optimizer output. Starters get a
    # delta; bench/IL players are absent (rendered as "--").
    roto_delta_by_name: dict[str, float] = {}
    band_by_name: dict[str, dict] = {}
    if optimal:
        for a in optimal.get("hitter_lineup", []) or []:
            if "name" in a and "roto_delta" in a:
                roto_delta_by_name[a["name"]] = a["roto_delta"]
            if "name" in a and a.get("band") is not None:
                band_by_name[a["name"]] = a["band"]
        for s in optimal.get("pitcher_starters", []) or []:
            if "name" in s and "roto_delta" in s:
                roto_delta_by_name[s["name"]] = s["roto_delta"]
            if "name" in s and s.get("band") is not None:
                band_by_name[s["name"]] = s["band"]

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
            "band": band_by_name.get(player.name),
            "games": p.get("games_this_week", 0),
            "is_bench": pos in BENCH_SLOTS,
            "is_il": "IL" in player.status or pos == "IL",
            "pace": player.pace or {},
            "overall_pace": compute_overall_pace(player.pace),
            "rank": player.rank.to_dict(),
            "preseason": player.preseason.to_dict() if player.preseason else None,
        }
        # Flatten ROS stats for template tooltip (h[rest_of_season_key] access pattern)
        if player.rest_of_season is not None:
            entry.update(player.rest_of_season.to_dict())

        # --- Per-basis selection (display-only): ROS / YTD / Total ---
        ros_stats = player.rest_of_season
        full_stats = player.full_season_projection or player.rest_of_season
        ytd_stats = _derive_ytd_stats(
            player.full_season_projection, player.rest_of_season, player.player_type
        )

        sgp_total = full_stats.compute_sgp() if full_stats is not None else None
        sgp_ytd = ytd_stats.compute_sgp() if ytd_stats is not None else 0.0

        basis_choice = {
            "ros": (ros_stats, ros_sgp, player.rank.rest_of_season),
            "ytd": (ytd_stats, sgp_ytd, player.rank.current),
            "total": (full_stats, sgp_total, player.rank.total),
        }
        sel_stats, sel_sgp, sel_rank = basis_choice[basis]

        entry["sgp"] = sel_sgp
        entry["rank_display"] = sel_rank
        entry["display_stats"] = _display_map(sel_stats, player.player_type, basis)

        if is_pitcher:
            pitchers.append(entry)
        else:
            hitters.append(entry)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h["selected_position"], 99), -(h["sgp"] or 0)))
    pitchers.sort(key=lambda p: (p["is_bench"], -(p["sgp"] or 0)))

    raw_moves = optimal.get("moves") if optimal else None
    moves = _normalize_moves(raw_moves)
    move_count = (
        len(moves["swaps"]) + len(moves["unpaired_starts"]) + len(moves["unpaired_benches"])
    )

    return {
        "hitters": hitters,
        "pitchers": pitchers,
        "hitter_totals": _compute_team_totals_pace(hitters, PlayerType.HITTER),
        "pitcher_totals": _compute_team_totals_pace(pitchers, PlayerType.PITCHER),
        "is_optimal": move_count == 0,
        "moves": moves,
    }


def run_optimize() -> dict:
    """Re-run lineup optimizer from cached data. Returns moves payload."""
    optimal = read_cache(CacheKey.LINEUP_OPTIMAL)
    moves = _normalize_moves(optimal.get("moves") if isinstance(optimal, dict) else None)
    move_count = (
        len(moves["swaps"]) + len(moves["unpaired_starts"]) + len(moves["unpaired_benches"])
    )
    return {"moves": moves, "is_optimal": move_count == 0}


def compute_comparison_standings(
    roster_player_name: str,
    other_player: "Player",
    user_roster: "list[Player]",
    projected_standings: ProjectedStandings,
    user_team_name: str,
    *,
    fraction_remaining: float = 1.0,
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
    dropped = find_player_by_name(roster_player_name, user_roster)
    if dropped is None:
        return {"error": f"Player '{roster_player_name}' not found on roster"}

    all_stats_before, all_stats_after = build_swap_standings(
        roster_player_projection or dropped,
        other_player,
        projected_standings,
        user_team_name,
    )

    roto_before = score_roto_dict(all_stats_before)
    roto_after = score_roto_dict(all_stats_after)

    ev_roto_before = score_roto_dict(all_stats_before, team_sds=team_sds)
    ev_roto_after = score_roto_dict(all_stats_after, team_sds=team_sds)

    delta_roto = score_swap(ev_roto_before, ev_roto_after, user_team_name)

    field_stats = projected_standings.field_stats(user_team_name)
    band = compute_one_for_one_band(
        roster_player_name,
        other_player,
        user_roster,
        field_stats,
        user_team_name,
        fraction_remaining,
        projected_standings=projected_standings,
        team_sds=team_sds,
    )

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
        "band": band.to_dict(),
        "categories": [c.value for c in ALL_CATEGORIES],
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


_COUNTING_STAT_VALUES: frozenset[str] = frozenset(c.value for c in COUNTING_STATS)


def _apply_counting_delta_to_leader(teams: dict[str, dict], categories: list[str]) -> None:
    """Mutate ``teams`` in place: replace each counting-stat total with the per-date
    gap to the leader. Counting stats are :data:`COUNTING_STATS` (R, HR, RBI, SB,
    W, K, SV). After transformation the per-date leader sits at ``0`` and every
    other team sits at ``value - leader_value`` (<= 0). Ratio stats (AVG, ERA,
    WHIP) are left as raw totals — direction differs (lower is better for
    ERA/WHIP), so a uniform "max is the leader" treatment would misrepresent
    them. ``None`` entries (team missing on that date) stay ``None``.
    """
    counting_cats = [cat for cat in categories if cat in _COUNTING_STAT_VALUES]
    if not counting_cats or not teams:
        return
    team_names = list(teams.keys())
    num_dates = len(teams[team_names[0]]["stats"][counting_cats[0]])
    for cat in counting_cats:
        for d in range(num_dates):
            present = [
                teams[name]["stats"][cat][d]
                for name in team_names
                if teams[name]["stats"][cat][d] is not None
            ]
            if not present:
                continue
            leader = max(present)
            for name in team_names:
                v = teams[name]["stats"][cat][d]
                if v is None:
                    continue
                teams[name]["stats"][cat][d] = v - leader


def build_trends_series(client, *, user_team: str) -> dict:
    """Read both history hashes and return the /api/trends/series payload.

    Shape:
        {
          "user_team": str,
          "categories":     list[str],  # ["R", "HR", ..., "WHIP"]
          "counting_stats": list[str],  # categories the per-date delta transform applies to
          "actual":    {"dates": [...], "teams": {name: {"roto_points": [...], "stats": {cat: [...]}}}},
          "projected": {"dates": [...], "teams": {name: {"roto_points": [...], "stats": {cat: [...]}}}},
        }

    Per-snapshot per-category totals come from ``score_roto``. For the
    actual series we prefer Yahoo's ``yahoo_points_for`` total when
    every entry on that snapshot has it, matching the /standings page.
    Teams that appear in some snapshots but not others get ``None`` on
    the missing dates so Chart.js renders a gap.

    Counting-stat tabs (R, HR, RBI, SB, W, K, SV) are emitted as the per-date
    distance from the leader rather than raw totals — the leader sits at 0 and
    other teams sit at ``value - leader_value`` (<= 0). This makes "how far
    behind first" the read on every counting-stat chart. Ratio stats (AVG, ERA,
    WHIP) remain raw totals because their best-direction differs.
    """
    from fantasy_baseball.data.redis_store import (
        get_projected_standings_history,
        get_standings_history,
    )

    categories = [c.value for c in ALL_CATEGORIES]

    actual_history = get_standings_history(client)
    projected_history = get_projected_standings_history(client)

    def _emit_actual() -> dict:
        if not actual_history:
            return {"dates": [], "teams": {}}
        dates = sorted(actual_history.keys())
        all_team_names: set[str] = set()
        for d in dates:
            for entry in actual_history[d].entries:
                all_team_names.add(entry.team_name)

        teams: dict[str, dict] = {
            name: {
                "roto_points": [],
                "stats": {cat: [] for cat in categories},
            }
            for name in all_team_names
        }
        for d in dates:
            standings = actual_history[d]
            roto = score_roto(cast("Any", standings))
            present = {e.team_name: e for e in standings.entries}
            yahoo_authoritative = bool(present) and all(
                e.yahoo_points_for is not None for e in present.values()
            )
            for name in all_team_names:
                row = present.get(name)
                if row is None:
                    teams[name]["roto_points"].append(None)
                    for cat in categories:
                        teams[name]["stats"][cat].append(None)
                    continue
                if yahoo_authoritative:
                    # yahoo_authoritative guarantees yahoo_points_for is non-None.
                    teams[name]["roto_points"].append(float(row.yahoo_points_for))  # type: ignore[arg-type]
                else:
                    teams[name]["roto_points"].append(float(roto[name].total))
                stats_dict = row.stats.to_dict()
                for cat in categories:
                    teams[name]["stats"][cat].append(stats_dict[cat])
        _apply_counting_delta_to_leader(teams, categories)
        return {"dates": dates, "teams": teams}

    def _emit_projected() -> dict:
        if not projected_history:
            return {"dates": [], "teams": {}}
        dates = sorted(projected_history.keys())
        all_team_names: set[str] = set()
        for d in dates:
            for entry in projected_history[d].entries:
                all_team_names.add(entry.team_name)

        teams: dict[str, dict] = {
            name: {
                "roto_points": [],
                "stats": {cat: [] for cat in categories},
            }
            for name in all_team_names
        }
        for d in dates:
            projected = projected_history[d]
            roto = score_roto(cast("Any", projected))
            present = {e.team_name: e for e in projected.entries}
            for name in all_team_names:
                row = present.get(name)
                if row is None:
                    teams[name]["roto_points"].append(None)
                    for cat in categories:
                        teams[name]["stats"][cat].append(None)
                    continue
                teams[name]["roto_points"].append(float(roto[name].total))
                stats_dict = row.stats.to_dict()
                for cat in categories:
                    teams[name]["stats"][cat].append(stats_dict[cat])
        _apply_counting_delta_to_leader(teams, categories)
        return {"dates": dates, "teams": teams}

    return {
        "user_team": user_team,
        "categories": categories,
        "counting_stats": sorted(_COUNTING_STAT_VALUES),
        "actual": _emit_actual(),
        "projected": _emit_projected(),
    }
