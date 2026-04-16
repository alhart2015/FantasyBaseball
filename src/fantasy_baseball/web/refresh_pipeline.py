"""Season dashboard refresh pipeline.

Orchestrates one full data refresh: auth with Yahoo, fetch rosters and
standings, blend projections, run the lineup optimizer, run Monte Carlo
simulations, compute SPoE, and write all cache artifacts. Entry point is
``run_full_refresh``. Progress is tracked in module-level state that the
web UI polls via ``get_refresh_status``.

Shared helpers (``_standings_to_snapshot``, ``_load_game_log_totals``,
``_compute_pending_moves_diff``, cache I/O) live in ``season_data`` and
are imported below. The dependency is one-way: this module imports from
``season_data``, never the reverse.
"""

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

from fantasy_baseball.utils.constants import (
    IL_STATUSES,
)
from fantasy_baseball.utils.positions import PITCHER_POSITIONS
from fantasy_baseball.utils.time_utils import (
    compute_effective_date,
    compute_fraction_remaining,
    local_now,
    local_today,
)

from fantasy_baseball.web.season_data import (
    CACHE_DIR,
    _compute_pending_moves_diff,
    _get_redis,
    _load_game_log_totals,
    _standings_to_snapshot,
    clear_opponent_cache,
    read_cache,
    write_cache,
)

_refresh_lock = threading.Lock()
_refresh_status = {"running": False, "progress": "", "error": None}

# Defaults for early-season teams missing stats
_STAT_DEFAULTS = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
                  "W": 0, "K": 0, "SV": 0, "ERA": 99.0, "WHIP": 99.0}


def get_refresh_status() -> dict:
    with _refresh_lock:
        return dict(_refresh_status)


def _set_refresh_progress(msg: str) -> None:
    with _refresh_lock:
        _refresh_status["progress"] = msg


def _fill_stat_defaults(standings: list[dict]) -> None:
    """Ensure every team has all 10 stat keys (early season some are missing)."""
    for t in standings:
        filled = dict(_STAT_DEFAULTS)
        filled.update(t["stats"])
        t["stats"] = filled


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
        redis.set(
            f"spoe_snapshot:{snapshot_date}",
            json.dumps(spoe_result),
        )
    except Exception as exc:
        log.warning(f"Failed to write spoe_snapshot:{snapshot_date}: {exc}")


class RefreshRun:
    """Encapsulates one execution of the season dashboard refresh.

    Each step from the original ``run_full_refresh`` is now a private
    method. The class holds shared state as instance attributes so
    methods don't need 10-arg signatures. Methods are NOT individually
    unit-tested; the integration test in
    ``tests/test_web/test_refresh_pipeline.py`` covers them collectively.

    Module-level state (``_refresh_lock``, ``_refresh_status``) is shared
    across threads and stays at module scope.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        from fantasy_baseball.web.job_logger import JobLogger
        self.cache_dir = cache_dir
        self.logger = JobLogger("refresh")

        # Shared state — populated as steps run. All initialized to None
        # so attribute access errors surface as clear AttributeErrors
        # rather than silent fall-through to the wrong type.
        self.config = None
        self.league = None              # Yahoo session-bound league
        self.league_model = None        # League dataclass loaded from Redis
        self.user_team_key = None
        self.standings = None
        self.standings_snap = None
        self.projected_standings = None
        self.projected_standings_snap = None
        self.team_sds = None
        self.fraction_remaining = None
        self.sd_scale = None
        self.effective_date = None
        self.start_date = None
        self.end_date = None
        self.roster_raw = None
        self.raw_rosters_by_team = None
        self.opp_rosters = None
        self.matched = None
        self.roster_players = None
        self.preseason_lookup = None
        self.preseason_hitters = None
        self.preseason_pitchers = None
        self.hitters_proj = None
        self.pitchers_proj = None
        self.has_rest_of_season = False
        self.hitter_logs = None
        self.pitcher_logs = None
        self.leverage = None
        self.rankings_lookup = None
        self.optimal_hitters = None
        self.optimal_pitchers_starters = None
        self.optimal_pitchers_bench = None
        self.fa_players = None

    def _progress(self, msg: str) -> None:
        _set_refresh_progress(msg)
        self.logger.log(msg)
        log.info(msg)

    def run(self) -> None:
        """Run the full refresh pipeline.

        Same try/except/finally protocol as the legacy ``run_full_refresh``:
        sets ``_refresh_status`` throughout, captures errors into
        ``_refresh_status['error']`` while still raising, and clears
        ``running`` in the ``finally`` block.
        """
        with _refresh_lock:
            _refresh_status["running"] = True
            _refresh_status["progress"] = "Starting..."
            _refresh_status["error"] = None

        try:
            self._authenticate()
            self._find_user_team()
            self._fetch_standings_and_roster()
            self._load_projections()
            self._fetch_opponent_rosters()
            self._write_snapshots_and_load_league()
            self._hydrate_rosters()
            self._build_projected_standings()
            self._compute_leverage()
            self._match_roster_to_projections()
            self._fetch_game_logs()
            self._compute_pace()
            self._compute_wsgp()
            self._compute_rankings()
            self._optimize_lineup()
            self._compute_moves()
            self._fetch_probable_starters()
            self._audit_roster()
            self._compute_per_team_leverage()
            self._run_monte_carlo()
            self._run_ros_monte_carlo()
            self._compute_spoe()
            self._analyze_transactions()
            self._write_meta()

            self.logger.finish("ok")
            self._progress("Done")
            from fantasy_baseball.web.season_data import clear_opponent_cache
            clear_opponent_cache()
        except Exception as exc:
            with _refresh_lock:
                _refresh_status["error"] = str(exc)
            self.logger.finish("error", str(exc))
            raise
        finally:
            with _refresh_lock:
                _refresh_status["running"] = False

    # Step methods — populated in subsequent tasks
    def _authenticate(self): raise NotImplementedError
    def _find_user_team(self): raise NotImplementedError
    def _fetch_standings_and_roster(self): raise NotImplementedError
    def _load_projections(self): raise NotImplementedError
    def _fetch_opponent_rosters(self): raise NotImplementedError
    def _write_snapshots_and_load_league(self): raise NotImplementedError
    def _hydrate_rosters(self): raise NotImplementedError
    def _build_projected_standings(self): raise NotImplementedError
    def _compute_leverage(self): raise NotImplementedError
    def _match_roster_to_projections(self): raise NotImplementedError
    def _fetch_game_logs(self): raise NotImplementedError
    def _compute_pace(self): raise NotImplementedError
    def _compute_wsgp(self): raise NotImplementedError
    def _compute_rankings(self): raise NotImplementedError
    def _optimize_lineup(self): raise NotImplementedError
    def _compute_moves(self): raise NotImplementedError
    def _fetch_probable_starters(self): raise NotImplementedError
    def _audit_roster(self): raise NotImplementedError
    def _compute_per_team_leverage(self): raise NotImplementedError
    def _run_monte_carlo(self): raise NotImplementedError
    def _run_ros_monte_carlo(self): raise NotImplementedError
    def _compute_spoe(self): raise NotImplementedError
    def _analyze_transactions(self): raise NotImplementedError
    def _write_meta(self): raise NotImplementedError


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
        from fantasy_baseball.lineup.roster_audit import audit_roster
        from fantasy_baseball.web.refresh_steps import (
            build_positions_map,
            compute_lineup_moves,
            merge_matched_and_raw_roster,
        )

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
        effective_date = compute_effective_date(end_date)
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
        from fantasy_baseball.scoring import build_projected_standings, build_team_sds

        all_team_rosters = {config.team_name: matched}
        all_team_rosters.update(opp_rosters)

        projected_standings = build_projected_standings(all_team_rosters)

        fraction_remaining = compute_fraction_remaining(
            date.fromisoformat(config.season_start),
            date.fromisoformat(config.season_end),
            local_today(),
        )
        _sd_scale = math.sqrt(fraction_remaining)

        team_sds = build_team_sds(all_team_rosters, _sd_scale)

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

        # Build Player objects from matched entries (+ any unmatched raw)
        roster_players = merge_matched_and_raw_roster(
            matched, roster_raw, preseason_lookup,
        )

        _progress(f"Matched {len(roster_players)} players to projections")

        # --- Step 6b: Fetch MLB game logs ---
        _progress("Fetching MLB game logs...")
        fetch_game_log_totals(config.season_year, progress_cb=_progress)

        # --- Step 6c: Compute season-to-date pace vs projections ---
        _progress("Computing player pace...")
        hitter_logs, pitcher_logs = _load_game_log_totals(config.season_year)

        # Attach pace data to each roster player (pace compares actuals vs preseason)
        from fantasy_baseball.sgp.denominators import get_sgp_denominators
        from fantasy_baseball.analysis.pace import attach_pace_to_roster
        sgp_denoms = get_sgp_denominators(config.sgp_overrides)
        attach_pace_to_roster(
            roster_players, hitter_logs, pitcher_logs,
            preseason_lookup, sgp_denoms,
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
        from fantasy_baseball.sgp.rankings import build_rankings_lookup
        rankings_lookup = build_rankings_lookup(
            rest_of_season_ranks, preseason_ranks, current_ranks,
        )

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
        moves = compute_lineup_moves(optimal_hitters, roster_players)

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
        positions_map = build_positions_map(roster_players, opp_rosters, fa_players)
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

            # fraction_remaining was computed in Step 4e and is reused here

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
