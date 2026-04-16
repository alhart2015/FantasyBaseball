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
        self.base_mc = None
        self.mgmt_mc = None
        self.rest_of_season_mc = None
        self.rest_of_season_mgmt_mc = None

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
            self._fetch_game_logs()
            self._load_projections()
            self._fetch_opponent_rosters()
            self._write_snapshots_and_load_league()
            self._hydrate_rosters()
            self._build_projected_standings()
            self._compute_leverage()
            self._match_roster_to_projections()
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

    # --- Step 1: Auth + league ---
    def _authenticate(self):
        from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
        from fantasy_baseball.config import load_config

        self._progress("Authenticating with Yahoo...")
        sc = get_yahoo_session()
        project_root = Path(__file__).resolve().parents[3]
        self.config = load_config(project_root / "config" / "league.yaml")
        self.league = get_league(sc, self.config.league_id, self.config.game_code)

    # --- Step 2: Find user's team key ---
    def _find_user_team(self):
        self._progress("Finding team...")
        teams = self.league.teams()
        for key, team_info in teams.items():
            if team_info.get("name") == self.config.team_name:
                self.user_team_key = key
                break
        if self.user_team_key is None:
            # Fall back to first team if not found by name
            self.user_team_key = next(iter(teams))

    # --- Step 3: Fetch standings + roster ---
    def _fetch_standings_and_roster(self):
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings, fetch_scoring_period

        self._progress("Fetching standings...")
        self.standings = fetch_standings(self.league)
        _fill_stat_defaults(self.standings)
        write_cache("standings", self.standings, self.cache_dir)
        self._progress(f"Fetched standings for {len(self.standings)} teams")

        # Compute the effective date for the next lineup lock. We fetch
        # all rosters at this date (via Yahoo's team.roster(day=...)) so
        # the audit/optimizer/waivers see the post-lock future state
        # without having to simulate pending transactions locally.
        # fetch_scoring_period returns Yahoo's Mon–Sun scoring week
        # (end_date is Sunday). The user's league locks lineups on
        # Tuesday morning, so the effective date is the next Tuesday
        # strictly after end_date — end_date + 1 would land on Monday,
        # one day too early.
        self._progress("Computing effective date...")
        self.start_date, self.end_date = fetch_scoring_period(self.league)
        self.effective_date = compute_effective_date(self.end_date)
        self._progress(f"Effective date (next lock): {self.effective_date}")

        self.standings_snap = _standings_to_snapshot(self.standings, self.effective_date)

        self._progress("Fetching today's roster (for pending-moves diff)...")
        today_roster_raw = fetch_roster(self.league, self.user_team_key)

        self._progress(f"Fetching future-dated roster for {self.effective_date}...")
        self.roster_raw = fetch_roster(self.league, self.user_team_key, day=self.effective_date)
        self._progress(f"Fetched future roster: {len(self.roster_raw)} players")

        pending_moves = _compute_pending_moves_diff(
            today_roster_raw, self.roster_raw,
            team_name=self.config.team_name, team_key=self.user_team_key,
        )
        write_cache("pending_moves", pending_moves, self.cache_dir)
        if pending_moves:
            total_changes = sum(
                len(m["adds"]) + len(m["drops"]) for m in pending_moves
            )
            self._progress(f"Pending moves: {total_changes} change(s) detected")

    # --- Step 4: Read preseason projections from Redis ---
    def _load_projections(self):
        from fantasy_baseball.utils.name_utils import normalize_name

        self._progress("Loading projections...")
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
        self.hitters_proj = pd.DataFrame(_hitters_rows)
        self.pitchers_proj = pd.DataFrame(_pitchers_rows)

        # Load ROS projections — blend latest dated CSV into Redis
        # (cache:ros_projections). No-op if no CSV dir exists locally
        # (Render has no CSVs on disk; the daily admin-triggered
        # _run_rest_of_season_fetch keeps Redis populated).
        self._progress("Loading ROS projections...")
        project_root = Path(__file__).resolve().parents[3]
        projections_dir = project_root / "data" / "projections"
        from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros
        from fantasy_baseball.data.redis_store import (
            get_default_client, get_latest_roster_names,
        )
        try:
            rest_of_season_roster_names = get_latest_roster_names(get_default_client())
            # _fetch_game_logs must run before this step so game_log_totals in
            # Redis reflect this refresh; otherwise the ROS blend normalizes
            # against last-refresh actuals and cache:ros_projections is stale.
            blend_and_cache_ros(
                projections_dir,
                self.config.projection_systems, self.config.projection_weights,
                rest_of_season_roster_names, self.config.season_year,
                progress_cb=self._progress,
            )
        except FileNotFoundError:
            # No local CSVs — fine on Render; the admin job keeps Redis populated.
            self._progress("No local ROS CSV dir; relying on Redis cache")

        self.hitters_proj["_name_norm"] = self.hitters_proj["name"].apply(normalize_name)
        self.pitchers_proj["_name_norm"] = self.pitchers_proj["name"].apply(normalize_name)
        self._progress(f"Loaded {len(self.hitters_proj)} hitter + {len(self.pitchers_proj)} pitcher projections")

        # ROS projections live in Redis (cache:ros_projections). The
        # blend above just refreshed that key if local CSVs were
        # present; otherwise we fall back to whatever the daily admin
        # job wrote. Disk CSVs are no longer a fallback path.
        import pandas as pd
        rest_of_season_hitters = pd.DataFrame()
        rest_of_season_pitchers = pd.DataFrame()
        ros_cached = read_cache("ros_projections", self.cache_dir)
        if ros_cached:
            rest_of_season_hitters = pd.DataFrame(ros_cached.get("hitters", []))
            rest_of_season_pitchers = pd.DataFrame(ros_cached.get("pitchers", []))
        self.has_rest_of_season = not rest_of_season_hitters.empty or not rest_of_season_pitchers.empty
        if self.has_rest_of_season:
            self._progress(f"Loaded ROS projections from Redis "
                      f"({len(rest_of_season_hitters)} hitters + {len(rest_of_season_pitchers)} pitchers)")

        self.preseason_hitters = self.hitters_proj
        self.preseason_pitchers = self.pitchers_proj
        if self.has_rest_of_season:
            rest_of_season_hitters["_name_norm"] = rest_of_season_hitters["name"].apply(normalize_name)
            rest_of_season_pitchers["_name_norm"] = rest_of_season_pitchers["name"].apply(normalize_name)
            self._progress(f"Loaded {len(rest_of_season_hitters)} ROS hitters + {len(rest_of_season_pitchers)} ROS pitchers")
            # Use ROS projections as primary — they're the most current estimates
            self.hitters_proj = rest_of_season_hitters
            self.pitchers_proj = rest_of_season_pitchers
        else:
            self._progress("WARNING: No ROS projections available — falling back to preseason")

    # --- Step 4b: Fetch opponent rosters (raw) ---
    def _fetch_opponent_rosters(self):
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster

        self._progress("Fetching opponent rosters...")

        # Collect raw rosters keyed by team name — used only to feed the
        # Redis write below. League.from_redis will then be our source
        # of truth for roster data for the rest of the refresh.
        self.raw_rosters_by_team = {
            self.config.team_name: self.roster_raw,
        }

        def _fetch_opp(key_and_info):
            key, team_info = key_and_info
            tname = team_info.get("name", "")
            try:
                opp_raw = fetch_roster(self.league, key, day=self.effective_date)
                return (tname, opp_raw)
            except Exception:
                return None

        teams = self.league.teams()
        opp_items = [
            (key, info) for key, info in teams.items()
            if info.get("name", "") != self.config.team_name and key != self.user_team_key
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(_fetch_opp, opp_items):
                if result is None:
                    continue
                tname, opp_raw = result
                self.raw_rosters_by_team[tname] = opp_raw
        self._progress(
            f"Fetched {len(self.raw_rosters_by_team)} rosters (user + opponents)"
        )

    # --- Step 4c: Write rosters + standings to Redis, then load League ---
    def _write_snapshots_and_load_league(self):
        from fantasy_baseball.models.league import League

        self._progress("Writing roster snapshots to Redis...")
        from fantasy_baseball.data.redis_store import (
            get_default_client,
            write_roster_snapshot,
            write_standings_snapshot,
        )

        snapshot_date = self.effective_date.isoformat()
        for tname, team_raw in self.raw_rosters_by_team.items():
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
                for entry in self.standings
            ],
        }
        write_standings_snapshot(
            get_default_client(), snapshot_date, snapshot_payload,
        )

        self._progress("Loading League from Redis...")
        self.league_model = League.from_redis(self.config.season_year)

    # --- Step 4d: Hydrate user roster + opponent rosters from League ---
    def _hydrate_rosters(self):
        from fantasy_baseball.data.projections import hydrate_roster_entries
        from fantasy_baseball.models.player import Player

        self._progress("Hydrating user and opponent rosters...")
        user_team_model = self.league_model.team_by_name(self.config.team_name)
        user_roster_model = user_team_model.latest_roster()
        self.matched = hydrate_roster_entries(
            user_roster_model, self.hitters_proj, self.pitchers_proj,
            context="user",
        )

        self.opp_rosters: dict[str, list[Player]] = {}
        for team in self.league_model.teams:
            if team.name == self.config.team_name:
                continue
            if not team.rosters:
                continue
            latest = team.latest_roster()
            hydrated = hydrate_roster_entries(
                latest, self.hitters_proj, self.pitchers_proj,
                context=f"opp:{team.name}",
            )
            if hydrated:
                self.opp_rosters[team.name] = hydrated
        self._progress(f"Hydrated {len(self.opp_rosters)} opponent rosters")

        # Cache opponent rosters for on-demand trade search
        opp_rosters_flat = {
            tname: [p.to_dict() for p in roster]
            for tname, roster in self.opp_rosters.items()
        }
        write_cache("opp_rosters", opp_rosters_flat, self.cache_dir)

    # --- Step 4e: Build projected standings ---
    def _build_projected_standings(self):
        self._progress("Projecting end-of-season standings...")
        from fantasy_baseball.scoring import build_projected_standings, build_team_sds

        all_team_rosters = {self.config.team_name: self.matched}
        all_team_rosters.update(self.opp_rosters)

        self.projected_standings = build_projected_standings(all_team_rosters)

        self.fraction_remaining = compute_fraction_remaining(
            date.fromisoformat(self.config.season_start),
            date.fromisoformat(self.config.season_end),
            local_today(),
        )
        self.sd_scale = math.sqrt(self.fraction_remaining)

        self.team_sds = build_team_sds(all_team_rosters, self.sd_scale)

        write_cache(
            "projections",
            {
                "projected_standings": self.projected_standings,
                "team_sds": self.team_sds,
                "fraction_remaining": self.fraction_remaining,
            },
            self.cache_dir,
        )
        self._progress(f"Projected standings for {len(self.projected_standings)} teams")

        self.projected_standings_snap = _standings_to_snapshot(self.projected_standings, self.effective_date)

    # --- Step 5: Leverage weights ---
    def _compute_leverage(self):
        from fantasy_baseball.lineup.leverage import calculate_leverage

        self._progress("Calculating leverage weights...")
        self.leverage = calculate_leverage(
            self.standings_snap, self.config.team_name,
            projected_standings=self.projected_standings_snap,
        )

    # --- Step 6: Match roster players to projections, compute wSGP ---
    def _match_roster_to_projections(self):
        from fantasy_baseball.data.projections import match_roster_to_projections
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.web.refresh_steps import merge_matched_and_raw_roster

        self._progress("Matching roster to projections...")

        # Match preseason projections for tooltip comparison
        preseason_matched = match_roster_to_projections(
            self.roster_raw, self.preseason_hitters, self.preseason_pitchers,
            context="preseason",
        )
        self.preseason_lookup = {normalize_name(p.name): p for p in preseason_matched}

        # Build Player objects from matched entries (+ any unmatched raw)
        self.roster_players = merge_matched_and_raw_roster(
            self.matched, self.roster_raw, self.preseason_lookup,
        )

        self._progress(f"Matched {len(self.roster_players)} players to projections")

    # --- Step 3b: Fetch MLB game logs (must precede ROS blend) ---
    def _fetch_game_logs(self):
        from fantasy_baseball.data.mlb_game_logs import fetch_game_log_totals

        self._progress("Fetching MLB game logs...")
        fetch_game_log_totals(self.config.season_year, progress_cb=self._progress)

    # --- Step 6c: Compute season-to-date pace vs projections ---
    def _compute_pace(self):
        self._progress("Computing player pace...")
        self.hitter_logs, self.pitcher_logs = _load_game_log_totals(self.config.season_year)

        # Attach pace data to each roster player (pace compares actuals vs preseason)
        from fantasy_baseball.sgp.denominators import get_sgp_denominators
        from fantasy_baseball.analysis.pace import attach_pace_to_roster
        sgp_denoms = get_sgp_denominators(self.config.sgp_overrides)
        attach_pace_to_roster(
            self.roster_players, self.hitter_logs, self.pitcher_logs,
            self.preseason_lookup, sgp_denoms,
        )

    # --- Step 6e: Compute wSGP on raw ROS stats ---
    def _compute_wsgp(self):
        # NOTE: recency blending was removed here because FanGraphs ROS
        # projections already incorporate early-season performance, and a
        # second layer of reliability weighting on top created inconsistencies
        # with projected_standings (see docs/superpowers/plans/2026-04-10-remove-recency-blending.md).
        for player in self.roster_players:
            if player.rest_of_season is not None:
                player.compute_wsgp(self.leverage)

    # --- Step 6d: Compute SGP rankings ---
    def _compute_rankings(self):
        self._progress("Computing SGP rankings...")
        from fantasy_baseball.sgp.rankings import (
            compute_sgp_rankings, compute_combined_sgp_rankings,
            compute_rankings_from_game_logs,
            rank_key, rank_key_from_positions, lookup_rank,
        )

        rest_of_season_ranks = compute_sgp_rankings(self.hitters_proj, self.pitchers_proj)
        preseason_ranks = compute_sgp_rankings(self.preseason_hitters, self.preseason_pitchers)
        current_ranks = compute_rankings_from_game_logs(self.hitter_logs, self.pitcher_logs)

        # Build combined lookup: {name::player_type: {ros, preseason, current}}
        from fantasy_baseball.sgp.rankings import build_rankings_lookup
        self.rankings_lookup = build_rankings_lookup(
            rest_of_season_ranks, preseason_ranks, current_ranks,
        )

        write_cache("rankings", self.rankings_lookup, self.cache_dir)
        self._progress(f"Ranked {len(rest_of_season_ranks)} ROS, {len(preseason_ranks)} preseason, {len(current_ranks)} current")

        # Attach ranks to roster players
        from fantasy_baseball.models.player import RankInfo
        for player in self.roster_players:
            rank_data = lookup_rank(self.rankings_lookup, player.fg_id, player.name, player.player_type)
            player.rank = RankInfo.from_dict(rank_data) if rank_data else RankInfo()

        # Classify roster players by league-wide value vs team fit
        from fantasy_baseball.lineup.player_classification import classify_roster
        rest_of_season_rank_lookup = {}
        for key, rank_data in self.rankings_lookup.items():
            ros = rank_data.get("rest_of_season")
            if ros is not None:
                rest_of_season_rank_lookup[key] = ros
        classifications = classify_roster(self.roster_players, rest_of_season_rank_lookup)
        for player in self.roster_players:
            player.classification = classifications.get(player.name, "")

        roster_flat = [p.to_flat_dict() for p in self.roster_players]
        write_cache("roster", roster_flat, self.cache_dir)

    # --- Step 7: Run lineup optimizer ---
    def _optimize_lineup(self):
        from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup

        self._progress("Optimizing lineup...")
        active_players = [p for p in self.roster_players if p.status not in IL_STATUSES]
        hitter_players = []
        pitcher_players = []
        for player in active_players:
            if set(player.positions) & PITCHER_POSITIONS:
                pitcher_players.append(player)
            else:
                hitter_players.append(player)

        self.optimal_hitters = optimize_hitter_lineup(
            hitter_players, self.leverage, self.config.roster_slots
        )
        self.optimal_pitchers_starters, self.optimal_pitchers_bench = optimize_pitcher_lineup(
            pitcher_players, self.leverage
        )

    # --- Step 8: Compare optimal to current, find moves ---
    def _compute_moves(self):
        from fantasy_baseball.web.refresh_steps import compute_lineup_moves

        self._progress("Computing lineup moves...")
        moves = compute_lineup_moves(self.optimal_hitters, self.roster_players)

        optimal_data = {
            "hitter_lineup": self.optimal_hitters,
            "pitcher_starters": [p["name"] for p in self.optimal_pitchers_starters],
            "pitcher_bench": [p["name"] for p in self.optimal_pitchers_bench],
            "moves": moves,
        }
        write_cache("lineup_optimal", optimal_data, self.cache_dir)

    # --- Step 9: Probable starters ---
    def _fetch_probable_starters(self):
        from fantasy_baseball.data.mlb_schedule import get_week_schedule
        from fantasy_baseball.lineup.matchups import calculate_matchup_factors, get_team_batting_stats

        self._progress("Fetching schedule and matchup data...")
        # start_date and end_date were fetched earlier to compute effective_date
        project_root = Path(__file__).resolve().parents[3]
        schedule_cache_path = project_root / "data" / "weekly_schedule.json"
        schedule = get_week_schedule(self.start_date, self.end_date, schedule_cache_path)

        batting_stats_cache_path = project_root / "data" / "team_batting_stats.json"
        team_stats = get_team_batting_stats(batting_stats_cache_path)
        matchup_factors = calculate_matchup_factors(team_stats)

        pitcher_roster_for_schedule = [
            p for p in self.roster_players
            if set(p.positions) & PITCHER_POSITIONS
        ]
        from fantasy_baseball.lineup.matchups import get_probable_starters
        probable_starters = get_probable_starters(
            pitcher_roster_for_schedule, schedule or {},
            matchup_factors=matchup_factors, team_stats=team_stats,
        )
        write_cache("probable_starters", probable_starters, self.cache_dir)

    # --- Step 10: Roster audit ---
    def _audit_roster(self):
        from fantasy_baseball.lineup.roster_audit import audit_roster
        from fantasy_baseball.lineup.waivers import fetch_and_match_free_agents
        from fantasy_baseball.web.refresh_steps import build_positions_map

        self._progress("Running roster audit...")
        self.fa_players, _ = fetch_and_match_free_agents(
            self.league, self.hitters_proj, self.pitchers_proj
        )

        # Cache positions for all known players (roster + opponents + FAs)
        positions_map = build_positions_map(self.roster_players, self.opp_rosters, self.fa_players)
        write_cache("positions", positions_map, self.cache_dir)
        from fantasy_baseball.data.redis_store import set_positions, get_default_client
        set_positions(get_default_client(), positions_map)
        self._progress(f"Cached positions for {len(positions_map)} players")

        audit_results = audit_roster(
            self.roster_players, self.fa_players, self.leverage, self.config.roster_slots,
            projected_standings=self.projected_standings,
            team_name=self.config.team_name,
            team_sds=self.team_sds,
        )
        write_cache("roster_audit", [e.to_dict() for e in audit_results], self.cache_dir)
        upgrades = sum(1 for e in audit_results if e.gap > 0)
        self._progress(f"Roster audit: {upgrades} upgrade(s) found")

    # --- Step 11: Compute per-team leverage ---
    def _compute_per_team_leverage(self):
        from fantasy_baseball.lineup.leverage import calculate_leverage

        self._progress("Computing leverage...")
        leverage_by_team: dict[str, dict] = {}
        for entry in self.standings_snap.entries:
            leverage_by_team[entry.team_name] = calculate_leverage(
                self.standings_snap, entry.team_name,
                projected_standings=self.projected_standings_snap,
            )
        write_cache("leverage", leverage_by_team, self.cache_dir)

    # --- Step 12: Monte Carlo simulation ---
    def _run_monte_carlo(self):
        from fantasy_baseball.simulation import run_monte_carlo

        h_slots = sum(v for k, v in self.config.roster_slots.items()
                      if k not in ("P", "BN", "IL", "DL"))
        p_slots = self.config.roster_slots.get("P", 9)

        all_team_rosters = {self.config.team_name: self.matched}
        all_team_rosters.update(self.opp_rosters)
        mc_rosters = all_team_rosters

        self.base_mc = run_monte_carlo(
            mc_rosters, h_slots, p_slots, self.config.team_name,
            n_iterations=1000, use_management=False,
            progress_cb=lambda i: self._progress(f"Monte Carlo: iteration {i}/1000..."),
        )
        self._progress("Pre-season Monte Carlo complete")
        self.mgmt_mc = run_monte_carlo(
            mc_rosters, h_slots, p_slots, self.config.team_name,
            n_iterations=1000, use_management=True,
            progress_cb=lambda i: self._progress(f"MC + Roster Mgmt: iteration {i}/1000..."),
        )
        self._progress("Pre-season + Mgmt Monte Carlo complete")

    # --- Step 13b: ROS Monte Carlo simulation ---
    def _run_ros_monte_carlo(self):
        self.rest_of_season_mc = None
        self.rest_of_season_mgmt_mc = None
        if self.has_rest_of_season:
            from fantasy_baseball.simulation import run_ros_monte_carlo

            # fraction_remaining was computed in Step 4e and is reused here

            h_slots = sum(v for k, v in self.config.roster_slots.items()
                          if k not in ("P", "BN", "IL", "DL"))
            p_slots = self.config.roster_slots.get("P", 9)

            all_team_rosters = {self.config.team_name: self.matched}
            all_team_rosters.update(self.opp_rosters)

            # Build ROS rosters for all teams. hitters_proj/pitchers_proj
            # already ARE rest_of_season_hitters/rest_of_season_pitchers when has_rest_of_season is True
            # (see the assignment above), so opp_rosters is already
            # matched against ROS projections — just reuse it.
            rest_of_season_mc_rosters = {}
            if self.matched:
                rest_of_season_mc_rosters[self.config.team_name] = all_team_rosters.get(self.config.team_name, [])
            for tname, opp_players in self.opp_rosters.items():
                rest_of_season_mc_rosters[tname] = opp_players

            # Build actual standings dict
            actual_standings_dict = {
                s["name"]: s["stats"] for s in self.standings
            }

            if rest_of_season_mc_rosters:
                self.rest_of_season_mc = run_ros_monte_carlo(
                    team_rosters=rest_of_season_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=self.fraction_remaining,
                    h_slots=h_slots, p_slots=p_slots,
                    user_team_name=self.config.team_name,
                    n_iterations=1000, use_management=False,
                    progress_cb=lambda i: self._progress(
                        f"Current MC: iteration {i}/1000..."
                    ),
                )
                self._progress("Current Monte Carlo complete")
                self.rest_of_season_mgmt_mc = run_ros_monte_carlo(
                    team_rosters=rest_of_season_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=self.fraction_remaining,
                    h_slots=h_slots, p_slots=p_slots,
                    user_team_name=self.config.team_name,
                    n_iterations=1000, use_management=True,
                    progress_cb=lambda i: self._progress(
                        f"Current MC + Mgmt: iteration {i}/1000..."
                    ),
                )
                self._progress("Current + Mgmt Monte Carlo complete")

        write_cache("monte_carlo", {
            "base": self.base_mc,
            "with_management": self.mgmt_mc,
            "rest_of_season": self.rest_of_season_mc,
            "rest_of_season_with_management": self.rest_of_season_mgmt_mc,
        }, self.cache_dir)

    # --- Step 14: Compute season-to-date SPoE (luck analysis) ---
    def _compute_spoe(self):
        # Reuses the league_model loaded in Step 4c. No separate DB
        # connection needed — SPoE walks Team.ownership_periods() on
        # the in-memory League object.
        self._progress("Computing SPoE...")
        from fantasy_baseball.analysis.spoe import (
            build_preseason_lookup,
            compute_current_spoe,
        )

        preseason_lookup = build_preseason_lookup(
            self.preseason_hitters, self.preseason_pitchers,
        )
        spoe_result = compute_current_spoe(
            self.league_model,
            self.standings,
            preseason_lookup,
            self.config.season_start,
            self.config.season_end,
        )

        write_cache("spoe", spoe_result, self.cache_dir)
        _write_spoe_snapshot(spoe_result)
        self._progress(f"SPoE computed for snapshot {spoe_result.get('snapshot_date')}")

    # --- Step 15: Transaction analyzer ---
    def _analyze_transactions(self):
        self._progress("Analyzing transactions...")
        from fantasy_baseball.lineup.yahoo_roster import fetch_all_transactions
        from fantasy_baseball.analysis.transactions import (
            pair_standalone_moves,
            score_transaction,
            build_cache_output,
        )

        raw_txns = fetch_all_transactions(self.league)
        if raw_txns:
            # Load previously scored transactions from Redis/disk cache
            stored_txns = read_cache("transactions", self.cache_dir) or []
            existing_ids = {t["transaction_id"] for t in stored_txns}
            new_txns = [t for t in raw_txns
                        if t["transaction_id"] not in existing_ids]

            if new_txns:
                self._progress(f"Scoring {len(new_txns)} new transaction(s)...")
                from fantasy_baseball.data.redis_store import (
                    get_default_client as _txn_redis_client,
                )
                _txn_client = _txn_redis_client()
                for txn in new_txns:
                    scores = score_transaction(
                        self.league_model, _txn_client, txn, self.config.season_year,
                    )
                    stored_txns.append({
                        "year": self.config.season_year,
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
            write_cache("transactions", stored_txns, self.cache_dir)
            cache_data = build_cache_output(stored_txns)
            write_cache("transaction_analyzer", cache_data, self.cache_dir)
            self._progress(f"Analyzed {len(stored_txns)} total transaction(s)")

    # --- Step 16: Write meta ---
    def _write_meta(self):
        self._progress("Finalizing...")
        meta = {
            "last_refresh": local_now().strftime("%Y-%m-%d %H:%M"),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "team_name": self.config.team_name,
        }
        write_cache("meta", meta, self.cache_dir)


def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    """Connect to Yahoo, fetch all data, run computations, and write cache files.

    Thin wrapper around RefreshRun for backward compatibility with
    existing callers (scripts/run_lineup.py, season_routes.py).
    """
    RefreshRun(cache_dir).run()
