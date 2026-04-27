"""Season dashboard refresh pipeline.

Orchestrates one full data refresh: auth with Yahoo, fetch rosters and
standings, blend projections, run the lineup optimizer, run Monte Carlo
simulations, compute SPoE, and write all cache artifacts. Entry point is
``run_full_refresh``. Progress is tracked in module-level state that the
web UI polls via ``get_refresh_status``.

Shared helpers (``_load_game_log_totals``, ``_compute_pending_moves_diff``,
cache I/O) live in ``season_data`` and are imported below. The dependency
is one-way: this module imports from ``season_data``, never the reverse.
"""

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fantasy_baseball.utils.constants import (
    IL_STATUSES,
    Category,
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
    CacheKey,
    _compute_pending_moves_diff,
    _get_redis,
    _load_game_log_totals,
    read_cache,
    write_cache,
)

if TYPE_CHECKING:
    import pandas as pd

    from fantasy_baseball.config import LeagueConfig
    from fantasy_baseball.lineup.optimizer import HitterAssignment, PitcherStarter
    from fantasy_baseball.models.league import League
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.models.standings import ProjectedStandings, Standings

log = logging.getLogger(__name__)

_refresh_lock = threading.Lock()
_refresh_status = {"running": False, "progress": "", "error": None}


def get_refresh_status() -> dict:
    with _refresh_lock:
        return dict(_refresh_status)


def _set_refresh_progress(msg: str) -> None:
    with _refresh_lock:
        _refresh_status["progress"] = msg


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


def build_standings_breakdown_payload(
    team_rosters: dict[str, list["Player"]],
    effective_date: date,
) -> dict:
    """Build the STANDINGS_BREAKDOWN cache payload for ``team_rosters``.

    One :class:`RosterBreakdown` per team, serialized to a JSON-safe
    dict and keyed by team name. ``effective_date`` is included to
    match the :class:`ProjectedStandings` payload shape.
    """
    from fantasy_baseball.scoring import compute_roster_breakdown

    teams_payload: dict[str, dict] = {}
    for team_name, roster in team_rosters.items():
        teams_payload[team_name] = compute_roster_breakdown(team_name, roster).to_dict()
    return {
        "effective_date": effective_date.isoformat(),
        "teams": teams_payload,
    }


def _load_projection_pair(payload: Any) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Convert a cached ``{"hitters": [...], "pitchers": [...]}`` payload to
    a ``(hitters_df, pitchers_df)`` pair with ``_name_norm`` attached.

    Returns ``(empty, empty)`` when the payload is missing or wrong-shaped.
    Skips ``_name_norm`` on empty frames so a partial blob doesn't crash on
    a missing ``name`` column.
    """
    import pandas as pd

    from fantasy_baseball.utils.name_utils import normalize_name

    if not isinstance(payload, dict):
        return pd.DataFrame(), pd.DataFrame()
    hitters = pd.DataFrame(payload.get("hitters", []) or [])
    pitchers = pd.DataFrame(payload.get("pitchers", []) or [])
    if not hitters.empty:
        hitters["_name_norm"] = hitters["name"].apply(normalize_name)
    if not pitchers.empty:
        pitchers["_name_norm"] = pitchers["name"].apply(normalize_name)
    return hitters, pitchers


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
        # rather than silent fall-through to the wrong type. Methods
        # that read these attributes assert-not-None at entry; the
        # ordering is enforced by ``run()``.
        self.config: LeagueConfig | None = None
        self.league: Any = None  # Yahoo session-bound league (untyped lib)
        self.league_model: League | None = None
        self.user_team_key: str | None = None
        self.standings: Standings | None = None
        self.projected_standings: ProjectedStandings | None = None
        self.preseason_projected_standings: ProjectedStandings | None = None
        self.team_sds: dict[str, dict[Category, float]] | None = None
        self.preseason_team_sds: dict[str, dict[Category, float]] | None = None
        self.fraction_remaining: float | None = None
        self.sd_scale: float | None = None
        self.effective_date: date | None = None
        self.start_date: str | None = None
        self.end_date: str | None = None
        self.roster_raw: list[dict[str, Any]] | None = None
        self.raw_rosters_by_team: dict[str, list[dict[str, Any]]] | None = None
        self.opp_rosters: dict[str, list[Player]] | None = None
        self.matched: list[Player] | None = None
        self.roster_players: list[Player] | None = None
        self.preseason_lookup: dict[str, Player] | None = None
        self.preseason_hitters: pd.DataFrame | None = None
        self.preseason_pitchers: pd.DataFrame | None = None
        self.hitters_proj: pd.DataFrame | None = None
        self.pitchers_proj: pd.DataFrame | None = None
        self.full_hitters_proj: pd.DataFrame | None = None
        self.full_pitchers_proj: pd.DataFrame | None = None
        self.has_rest_of_season: bool = False
        self.hitter_logs: dict[str, dict[str, Any]] | None = None
        self.pitcher_logs: dict[str, dict[str, Any]] | None = None
        self.leverage: dict[str, float] | None = None
        self.rankings_lookup: dict[str, dict[str, Any]] | None = None
        self.optimal_hitters: list[HitterAssignment] | None = None
        self.optimal_pitchers_starters: list[PitcherStarter] | None = None
        self.optimal_pitchers_bench: list[Player] | None = None
        self.fa_players: list[Player] | None = None
        self.rest_of_season_mc: dict[str, Any] | None = None
        self.rest_of_season_mgmt_mc: dict[str, Any] | None = None

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
            self._compute_rankings()
            self._optimize_lineup()
            self._compute_moves()
            self._fetch_probable_starters()
            self._audit_roster()
            self._compute_per_team_leverage()
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
        assert self.config is not None
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
        from fantasy_baseball.lineup.yahoo_roster import (
            fetch_roster,
            fetch_scoring_period,
            fetch_standings,
        )

        assert self.config is not None
        assert self.user_team_key is not None

        # Compute the effective date for the next lineup lock BEFORE
        # fetching standings so we can tag the Standings snapshot with
        # it. We fetch all rosters at this date (via Yahoo's
        # team.roster(day=...)) so the audit/optimizer/waivers see the
        # post-lock future state without having to simulate pending
        # transactions locally. fetch_scoring_period returns Yahoo's
        # Mon-Sun scoring week (end_date is Sunday). The user's league
        # locks lineups on Tuesday morning, so the effective date is the
        # next Tuesday strictly after end_date — end_date + 1 would land
        # on Monday, one day too early.
        self._progress("Computing effective date...")
        self.start_date, self.end_date = fetch_scoring_period(self.league)
        self.effective_date = compute_effective_date(self.end_date)
        self._progress(f"Effective date (next lock): {self.effective_date}")

        self._progress("Fetching standings...")
        self.standings = fetch_standings(self.league, effective_date=self.effective_date)
        write_cache(CacheKey.STANDINGS, self.standings.to_json(), self.cache_dir)
        self._progress(f"Fetched standings for {len(self.standings.entries)} teams")

        self._progress("Fetching today's roster (for pending-moves diff)...")
        today_roster_raw = fetch_roster(self.league, self.user_team_key)

        self._progress(f"Fetching future-dated roster for {self.effective_date}...")
        self.roster_raw = fetch_roster(self.league, self.user_team_key, day=self.effective_date)
        self._progress(f"Fetched future roster: {len(self.roster_raw)} players")

        pending_moves = _compute_pending_moves_diff(
            today_roster_raw,
            self.roster_raw,
            team_name=self.config.team_name,
            team_key=self.user_team_key,
        )
        write_cache(CacheKey.PENDING_MOVES, pending_moves, self.cache_dir)
        if pending_moves:
            total_changes = sum(len(m["adds"]) + len(m["drops"]) for m in pending_moves)
            self._progress(f"Pending moves: {total_changes} change(s) detected")

    # --- Step 4: Read preseason projections from Redis ---
    def _load_projections(self):
        from fantasy_baseball.utils.name_utils import normalize_name

        self._progress("Loading projections...")
        import pandas as pd

        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import (
            get_blended_projections as redis_get_blended,
        )

        _redis_client = get_kv()
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

        self.hitters_proj["_name_norm"] = self.hitters_proj["name"].apply(normalize_name)
        self.pitchers_proj["_name_norm"] = self.pitchers_proj["name"].apply(normalize_name)
        self._progress(
            f"Loaded {len(self.hitters_proj)} hitter + {len(self.pitchers_proj)} pitcher projections"
        )

        # ROS projections live in Redis (cache:ros_projections). The
        # daily admin-triggered _run_rest_of_season_fetch is the sole
        # authoritative writer — it downloads fresh CSVs from FanGraphs,
        # blends them, and writes the result to Redis. The refresh only
        # READS that key; it must not blend from disk CSVs, because the
        # only dated snapshot committed to git is stale (e.g. 2026-03-30)
        # and on Render the admin fetch and refresh frequently run on
        # different instances, so today's CSVs aren't on this instance's
        # disk. Blending here would overwrite fresh Redis with stale
        # March projections and regress the player-comparison tool back
        # to preseason values (see commit history: 2a11c1e established
        # Redis as authoritative, 9592b63 accidentally re-introduced the
        # overwrite).
        self.preseason_hitters = self.hitters_proj
        self.preseason_pitchers = self.pitchers_proj

        self._progress("Loading ROS projections from Redis...")
        ros_hitters, ros_pitchers = _load_projection_pair(
            read_cache(CacheKey.ROS_PROJECTIONS, self.cache_dir)
        )
        self.has_rest_of_season = not ros_hitters.empty or not ros_pitchers.empty
        if self.has_rest_of_season:
            self.hitters_proj = ros_hitters
            self.pitchers_proj = ros_pitchers
            self._progress(
                f"Loaded {len(ros_hitters)} ROS hitters + {len(ros_pitchers)} ROS pitchers"
            )
        else:
            self._progress("WARNING: No ROS projections available — falling back to preseason")

        # Full-season (ROS+YTD) projections populate Player.full_season_projection
        # for display + ProjectedStandings, while Player.rest_of_season stays
        # ROS-only for forward-looking decision paths. Same Redis+disk fallback
        # as the ROS load.
        self._progress("Loading full-season projections...")
        full_hitters, full_pitchers = _load_projection_pair(
            read_cache(CacheKey.FULL_SEASON_PROJECTIONS, self.cache_dir)
        )
        if not full_hitters.empty or not full_pitchers.empty:
            self.full_hitters_proj = full_hitters
            self.full_pitchers_proj = full_pitchers
            self._progress(
                f"Loaded {len(full_hitters)} full-season hitters + "
                f"{len(full_pitchers)} full-season pitchers"
            )
        else:
            self._progress(
                "WARNING: cache:full_season_projections missing — "
                "Player.full_season_projection will be unset"
            )

    # --- Step 4b: Fetch opponent rosters (raw) ---
    def _fetch_opponent_rosters(self):
        from fantasy_baseball.lineup.yahoo_roster import fetch_roster

        assert self.config is not None
        assert self.roster_raw is not None
        assert self.effective_date is not None

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
            (key, info)
            for key, info in teams.items()
            if info.get("name", "") != self.config.team_name and key != self.user_team_key
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(_fetch_opp, opp_items):
                if result is None:
                    continue
                tname, opp_raw = result
                self.raw_rosters_by_team[tname] = opp_raw
        self._progress(f"Fetched {len(self.raw_rosters_by_team)} rosters (user + opponents)")

    # --- Step 4c: Write rosters + standings to Redis, then load League ---
    def _write_snapshots_and_load_league(self):
        from fantasy_baseball.models.league import League

        assert self.config is not None
        assert self.effective_date is not None
        assert self.raw_rosters_by_team is not None
        assert self.standings is not None

        self._progress("Writing roster snapshots to Redis...")
        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import (
            write_roster_snapshot,
            write_standings_snapshot,
        )

        client = get_kv()
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
                client,
                snapshot_date,
                tname,
                entries,
            )

        # Canonical Standings snapshot — write_standings_snapshot keys
        # off standings.effective_date and serializes via .to_json().
        write_standings_snapshot(client, self.standings)

        self._progress("Loading League from Redis...")
        self.league_model = League.from_redis(self.config.season_year)

    # --- Step 4d: Hydrate user roster + opponent rosters from League ---
    def _hydrate_rosters(self):
        from fantasy_baseball.data.projections import hydrate_roster_entries

        assert self.config is not None
        assert self.league_model is not None
        assert self.hitters_proj is not None
        assert self.pitchers_proj is not None

        self._progress("Hydrating user and opponent rosters...")
        user_team_model = self.league_model.team_by_name(self.config.team_name)
        user_roster_model = user_team_model.latest_roster()
        self.matched = hydrate_roster_entries(
            user_roster_model,
            self.hitters_proj,
            self.pitchers_proj,
            full_hitters_proj=self.full_hitters_proj,
            full_pitchers_proj=self.full_pitchers_proj,
            context="user",
        )

        self.opp_rosters = {}
        for team in self.league_model.teams:
            if team.name == self.config.team_name:
                continue
            if not team.rosters:
                continue
            latest = team.latest_roster()
            hydrated = hydrate_roster_entries(
                latest,
                self.hitters_proj,
                self.pitchers_proj,
                full_hitters_proj=self.full_hitters_proj,
                full_pitchers_proj=self.full_pitchers_proj,
                context=f"opp:{team.name}",
            )
            if hydrated:
                self.opp_rosters[team.name] = hydrated
        self._progress(f"Hydrated {len(self.opp_rosters)} opponent rosters")

        # Cache opponent rosters for on-demand trade search
        opp_rosters_flat = {
            tname: [p.to_dict() for p in roster] for tname, roster in self.opp_rosters.items()
        }
        write_cache(CacheKey.OPP_ROSTERS, opp_rosters_flat, self.cache_dir)

    # --- Step 4e: Build projected standings ---
    def _build_projected_standings(self):
        self._progress("Projecting end-of-season standings...")
        from fantasy_baseball.data.projections import hydrate_roster_entries
        from fantasy_baseball.models.standings import ProjectedStandings
        from fantasy_baseball.scoring import build_team_sds, team_sds_to_json

        assert self.config is not None
        assert self.matched is not None
        assert self.opp_rosters is not None
        assert self.effective_date is not None
        assert self.league_model is not None
        assert self.preseason_hitters is not None
        assert self.preseason_pitchers is not None

        all_team_rosters = {self.config.team_name: self.matched}
        all_team_rosters.update(self.opp_rosters)

        self.projected_standings = ProjectedStandings.from_rosters(
            all_team_rosters, effective_date=self.effective_date
        )

        self.fraction_remaining = compute_fraction_remaining(
            date.fromisoformat(self.config.season_start),
            date.fromisoformat(self.config.season_end),
            local_today(),
        )
        self.sd_scale = math.sqrt(self.fraction_remaining)

        self.team_sds = build_team_sds(all_team_rosters, self.sd_scale)

        # Build preseason projected standings for the ERoto "Preseason" view.
        # When ROS projections are active, all_team_rosters are matched against
        # ROS — we need a separate pass against preseason projections.
        if self.has_rest_of_season:
            preseason_rosters: dict[str, list] = {}
            for team in self.league_model.teams:
                if not team.rosters:
                    continue
                latest = team.latest_roster()
                hydrated = hydrate_roster_entries(
                    latest,
                    self.preseason_hitters,
                    self.preseason_pitchers,
                    context=f"preseason:{team.name}",
                )
                if hydrated:
                    preseason_rosters[team.name] = hydrated
            self.preseason_projected_standings = ProjectedStandings.from_rosters(
                preseason_rosters, effective_date=self.effective_date
            )
            self.preseason_team_sds = build_team_sds(preseason_rosters, 1.0)
        else:
            self.preseason_projected_standings = self.projected_standings
            self.preseason_team_sds = build_team_sds(all_team_rosters, 1.0)

        write_cache(
            CacheKey.PROJECTIONS,
            {
                "projected_standings": self.projected_standings.to_json(),
                "team_sds": team_sds_to_json(self.team_sds),
                "fraction_remaining": self.fraction_remaining,
                "preseason_standings": self.preseason_projected_standings.to_json(),
                "preseason_team_sds": team_sds_to_json(self.preseason_team_sds),
            },
            self.cache_dir,
        )
        write_cache(
            CacheKey.STANDINGS_BREAKDOWN,
            build_standings_breakdown_payload(all_team_rosters, self.effective_date),
            self.cache_dir,
        )
        self._progress(f"Projected standings for {len(self.projected_standings.entries)} teams")

    # --- Step 5: Leverage weights ---
    def _compute_leverage(self):
        from fantasy_baseball.lineup.leverage import calculate_leverage

        assert self.config is not None
        assert self.standings is not None

        self._progress("Calculating leverage weights...")
        self.leverage = calculate_leverage(
            self.standings,
            self.config.team_name,
            projected_standings=self.projected_standings,
        )

    # --- Step 6: Match roster players to projections ---
    def _match_roster_to_projections(self):
        from fantasy_baseball.data.projections import match_roster_to_projections
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.web.refresh_steps import merge_matched_and_raw_roster

        assert self.roster_raw is not None
        assert self.preseason_hitters is not None
        assert self.preseason_pitchers is not None
        assert self.matched is not None

        self._progress("Matching roster to projections...")

        # Match preseason projections for tooltip comparison
        preseason_matched = match_roster_to_projections(
            self.roster_raw,
            self.preseason_hitters,
            self.preseason_pitchers,
            context="preseason",
        )
        self.preseason_lookup = {normalize_name(p.name): p for p in preseason_matched}

        # Build Player objects from matched entries (+ any unmatched raw)
        self.roster_players = merge_matched_and_raw_roster(
            self.matched,
            self.roster_raw,
            self.preseason_lookup,
        )

        self._progress(f"Matched {len(self.roster_players)} players to projections")

    # --- Step 3b: Fetch MLB game logs (must precede ROS blend) ---
    def _fetch_game_logs(self):
        from fantasy_baseball.data.mlb_game_logs import fetch_game_log_totals

        assert self.config is not None

        self._progress("Fetching MLB game logs...")
        fetch_game_log_totals(self.config.season_year, progress_cb=self._progress)

    # --- Step 6c: Compute season-to-date pace vs projections ---
    def _compute_pace(self):
        assert self.config is not None
        assert self.roster_players is not None
        assert self.preseason_lookup is not None

        self._progress("Computing player pace...")
        self.hitter_logs, self.pitcher_logs = _load_game_log_totals(self.config.season_year)

        # Attach pace data to each roster player (pace compares actuals vs preseason)
        from fantasy_baseball.analysis.pace import attach_pace_to_roster
        from fantasy_baseball.sgp.denominators import get_sgp_denominators

        sgp_denoms = get_sgp_denominators(self.config.sgp_overrides)
        attach_pace_to_roster(
            self.roster_players,
            self.hitter_logs,
            self.pitcher_logs,
            self.preseason_lookup,
            sgp_denoms,
        )

    # --- Step 6d: Compute SGP rankings ---
    def _compute_rankings(self):
        self._progress("Computing SGP rankings...")
        from fantasy_baseball.sgp.rankings import (
            compute_rankings_from_game_logs,
            compute_sgp_rankings,
            lookup_rank,
        )

        assert self.hitters_proj is not None
        assert self.pitchers_proj is not None
        assert self.preseason_hitters is not None
        assert self.preseason_pitchers is not None
        assert self.hitter_logs is not None
        assert self.pitcher_logs is not None
        assert self.roster_players is not None

        rest_of_season_ranks = compute_sgp_rankings(self.hitters_proj, self.pitchers_proj)
        preseason_ranks = compute_sgp_rankings(self.preseason_hitters, self.preseason_pitchers)
        current_ranks = compute_rankings_from_game_logs(self.hitter_logs, self.pitcher_logs)

        # Build combined lookup: {name::player_type: {ros, preseason, current}}
        from fantasy_baseball.sgp.rankings import build_rankings_lookup

        self.rankings_lookup = build_rankings_lookup(
            rest_of_season_ranks,
            preseason_ranks,
            current_ranks,
        )

        write_cache(CacheKey.RANKINGS, self.rankings_lookup, self.cache_dir)
        self._progress(
            f"Ranked {len(rest_of_season_ranks)} ROS, {len(preseason_ranks)} preseason, {len(current_ranks)} current"
        )

        # Attach ranks to roster players
        from fantasy_baseball.models.player import RankInfo

        for player in self.roster_players:
            rank_data = lookup_rank(
                self.rankings_lookup, player.fg_id, player.name, player.player_type
            )
            player.rank = RankInfo.from_dict(rank_data) if rank_data else RankInfo()

        roster_flat = [p.to_flat_dict() for p in self.roster_players]
        write_cache(CacheKey.ROSTER, roster_flat, self.cache_dir)

    # --- Step 7: Run lineup optimizer ---
    def _optimize_lineup(self):
        from fantasy_baseball.lineup.optimizer import (
            optimize_hitter_lineup,
            optimize_pitcher_lineup,
        )

        assert self.config is not None
        assert self.roster_players is not None
        assert self.projected_standings is not None

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
            hitters=hitter_players,
            full_roster=self.roster_players,
            projected_standings=self.projected_standings,
            team_name=self.config.team_name,
            roster_slots=self.config.roster_slots,
            team_sds=self.team_sds,
        )
        self.optimal_pitchers_starters, self.optimal_pitchers_bench = optimize_pitcher_lineup(
            pitchers=pitcher_players,
            full_roster=self.roster_players,
            projected_standings=self.projected_standings,
            team_name=self.config.team_name,
            slots=self.config.roster_slots.get("P", 9),
            team_sds=self.team_sds,
        )

    # --- Step 8: Compare optimal to current, find moves ---
    def _compute_moves(self):
        from fantasy_baseball.web.refresh_steps import compute_lineup_moves

        assert self.optimal_hitters is not None
        assert self.optimal_pitchers_starters is not None
        assert self.optimal_pitchers_bench is not None
        assert self.roster_players is not None

        self._progress("Computing lineup moves...")
        legacy_shape = {a.slot.value: a.name for a in self.optimal_hitters}
        moves = compute_lineup_moves(legacy_shape, self.roster_players)

        optimal_data = {
            "hitter_lineup": [a.to_dict() for a in self.optimal_hitters],
            "pitcher_starters": [s.to_dict() for s in self.optimal_pitchers_starters],
            "pitcher_bench": [p.name for p in self.optimal_pitchers_bench],
            "moves": moves,
        }
        write_cache(CacheKey.LINEUP_OPTIMAL, optimal_data, self.cache_dir)

    # --- Step 9: Probable starters ---
    def _fetch_probable_starters(self):
        from fantasy_baseball.data.mlb_schedule import get_week_schedule
        from fantasy_baseball.lineup.matchups import (
            calculate_matchup_factors,
            get_team_batting_stats,
        )

        assert self.start_date is not None
        assert self.end_date is not None
        assert self.roster_players is not None

        self._progress("Fetching schedule and matchup data...")
        # start_date and end_date were fetched earlier to compute effective_date
        project_root = Path(__file__).resolve().parents[3]
        schedule_cache_path = project_root / "data" / "weekly_schedule.json"
        schedule = get_week_schedule(self.start_date, self.end_date, schedule_cache_path)

        batting_stats_cache_path = project_root / "data" / "team_batting_stats.json"
        team_stats = get_team_batting_stats(batting_stats_cache_path)
        matchup_factors = calculate_matchup_factors(team_stats)

        pitcher_roster_for_schedule = [
            p for p in self.roster_players if set(p.positions) & PITCHER_POSITIONS
        ]
        from fantasy_baseball.lineup.matchups import get_probable_starters

        probable_starters = get_probable_starters(
            pitcher_roster_for_schedule,
            schedule or {},
            matchup_factors=matchup_factors,
            team_stats=team_stats,
        )
        write_cache(CacheKey.PROBABLE_STARTERS, probable_starters, self.cache_dir)

    # --- Step 10: Roster audit ---
    def _audit_roster(self):
        from fantasy_baseball.lineup.roster_audit import audit_roster
        from fantasy_baseball.lineup.waivers import fetch_and_match_free_agents
        from fantasy_baseball.web.refresh_steps import build_positions_map

        assert self.config is not None
        assert self.hitters_proj is not None
        assert self.pitchers_proj is not None
        assert self.roster_players is not None
        assert self.opp_rosters is not None
        assert self.projected_standings is not None
        assert self.optimal_hitters is not None
        assert self.optimal_pitchers_starters is not None

        self._progress("Running roster audit...")
        self.fa_players, _ = fetch_and_match_free_agents(
            self.league, self.hitters_proj, self.pitchers_proj
        )

        # Cache positions for all known players (roster + opponents + FAs)
        positions_map = build_positions_map(self.roster_players, self.opp_rosters, self.fa_players)
        write_cache(CacheKey.POSITIONS, positions_map, self.cache_dir)
        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import set_positions

        set_positions(get_kv(), positions_map)
        self._progress(f"Cached positions for {len(positions_map)} players")

        audit_results = audit_roster(
            self.roster_players,
            self.fa_players,
            self.config.roster_slots,
            projected_standings=self.projected_standings,
            team_name=self.config.team_name,
            team_sds=self.team_sds,
            optimal_hitters=self.optimal_hitters,
            optimal_pitchers=self.optimal_pitchers_starters,
        )
        write_cache(CacheKey.ROSTER_AUDIT, [e.to_dict() for e in audit_results], self.cache_dir)
        upgrades = sum(1 for e in audit_results if e.gap > 0)
        self._progress(f"Roster audit: {upgrades} upgrade(s) found")

    # --- Step 11: Compute per-team leverage ---
    def _compute_per_team_leverage(self):
        from fantasy_baseball.lineup.leverage import calculate_leverage

        assert self.standings is not None

        self._progress("Computing leverage...")
        leverage_by_team: dict[str, dict] = {}
        for entry in self.standings.entries:
            leverage_by_team[entry.team_name] = calculate_leverage(
                self.standings,
                entry.team_name,
                projected_standings=self.projected_standings,
            )
        write_cache(CacheKey.LEVERAGE, leverage_by_team, self.cache_dir)

    # --- Step 13b: ROS Monte Carlo simulation ---
    def _run_ros_monte_carlo(self):
        assert self.config is not None

        self.rest_of_season_mc = None
        self.rest_of_season_mgmt_mc = None
        if self.has_rest_of_season:
            from fantasy_baseball.simulation import run_ros_monte_carlo

            assert self.matched is not None
            assert self.opp_rosters is not None
            assert self.standings is not None
            assert self.fraction_remaining is not None

            # fraction_remaining was computed in Step 4e and is reused here

            h_slots = sum(
                v for k, v in self.config.roster_slots.items() if k not in ("P", "BN", "IL", "DL")
            )
            p_slots = self.config.roster_slots.get("P", 9)

            all_team_rosters = {self.config.team_name: self.matched}
            all_team_rosters.update(self.opp_rosters)

            # Build ROS rosters for all teams. hitters_proj/pitchers_proj
            # already ARE rest_of_season_hitters/rest_of_season_pitchers when has_rest_of_season is True
            # (see the assignment above), so opp_rosters is already
            # matched against ROS projections — just reuse it.
            rest_of_season_mc_rosters = {}
            if self.matched:
                rest_of_season_mc_rosters[self.config.team_name] = all_team_rosters.get(
                    self.config.team_name, []
                )
            for tname, opp_players in self.opp_rosters.items():
                rest_of_season_mc_rosters[tname] = opp_players

            # Build actual standings dict (uppercase-string-keyed per category)
            actual_standings_dict = {e.team_name: e.stats.to_dict() for e in self.standings.entries}

            if rest_of_season_mc_rosters:
                self.rest_of_season_mc = run_ros_monte_carlo(
                    team_rosters=rest_of_season_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=self.fraction_remaining,
                    h_slots=h_slots,
                    p_slots=p_slots,
                    user_team_name=self.config.team_name,
                    n_iterations=1000,
                    use_management=False,
                    progress_cb=lambda i: self._progress(f"Current MC: iteration {i}/1000..."),
                )
                self._progress("Current Monte Carlo complete")
                self.rest_of_season_mgmt_mc = run_ros_monte_carlo(
                    team_rosters=rest_of_season_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=self.fraction_remaining,
                    h_slots=h_slots,
                    p_slots=p_slots,
                    user_team_name=self.config.team_name,
                    n_iterations=1000,
                    use_management=True,
                    progress_cb=lambda i: self._progress(
                        f"Current MC + Mgmt: iteration {i}/1000..."
                    ),
                )
                self._progress("Current + Mgmt Monte Carlo complete")

        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import get_preseason_baseline

        _redis_client = get_kv()
        baseline = get_preseason_baseline(_redis_client, self.config.season_year) or {}
        if not baseline:
            self._progress("Preseason baseline missing — run scripts/freeze_preseason_baseline.py")

        write_cache(
            CacheKey.MONTE_CARLO,
            {
                "base": baseline.get("base"),
                "with_management": baseline.get("with_management"),
                "baseline_meta": baseline.get("meta"),
                "rest_of_season": self.rest_of_season_mc,
                "rest_of_season_with_management": self.rest_of_season_mgmt_mc,
            },
            self.cache_dir,
        )

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

        assert self.config is not None
        assert self.league_model is not None
        assert self.standings is not None
        assert self.preseason_hitters is not None
        assert self.preseason_pitchers is not None

        preseason_lookup = build_preseason_lookup(
            self.preseason_hitters,
            self.preseason_pitchers,
        )
        spoe_result = compute_current_spoe(
            self.league_model,
            self.standings,
            preseason_lookup,
            self.config.season_start,
            self.config.season_end,
        )

        write_cache(CacheKey.SPOE, spoe_result, self.cache_dir)
        _write_spoe_snapshot(spoe_result)
        self._progress(f"SPoE computed for snapshot {spoe_result.get('snapshot_date')}")

    # --- Step 15: Transaction analyzer ---
    def _analyze_transactions(self):
        self._progress("Analyzing transactions...")
        from fantasy_baseball.analysis.transactions import (
            _load_projections_for_date_redis,
            build_cache_output,
            pair_standalone_moves,
            score_transaction,
        )
        from fantasy_baseball.lineup.yahoo_roster import fetch_all_transactions

        assert self.config is not None
        assert self.league_model is not None
        assert self.projected_standings is not None

        raw_txns = fetch_all_transactions(self.league)
        if not raw_txns:
            return

        stored_txns_raw = read_cache(CacheKey.TRANSACTIONS, self.cache_dir) or []
        stored_txns: list[dict[str, Any]] = (
            stored_txns_raw if isinstance(stored_txns_raw, list) else []
        )
        existing_ids = {t["transaction_id"] for t in stored_txns}
        new_txns = [t for t in raw_txns if t["transaction_id"] not in existing_ids]

        if new_txns:
            self._progress(f"Scoring {len(new_txns)} new transaction(s)...")
            # Append unscored placeholders so pairing can match new-vs-stored.
            # ΔRoto is non-linear so we must pair BEFORE scoring — a paired
            # drop+add is one swap, not two independent scores that sum.
            for txn in new_txns:
                stored_txns.append(
                    {
                        "year": self.config.season_year,
                        **txn,
                        "paired_with": None,
                    }
                )

            unpaired = [t for t in stored_txns if not t.get("paired_with")]
            pairs = pair_standalone_moves(unpaired)
            by_id = {t["transaction_id"]: t for t in stored_txns}
            for drop_id, add_id in pairs:
                by_id[drop_id]["paired_with"] = add_id
                by_id[add_id]["paired_with"] = drop_id

            from fantasy_baseball.data.kv_store import get_kv

            _txn_client = get_kv()
            hitters_proj, pitchers_proj = _load_projections_for_date_redis(
                _txn_client,
            )
            season_start = date.fromisoformat(self.config.season_start)
            season_end = date.fromisoformat(self.config.season_end)

            for txn in new_txns:
                entry = by_id[txn["transaction_id"]]
                partner_id = entry.get("paired_with")
                partner = by_id.get(partner_id) if partner_id else None
                scores = score_transaction(
                    self.league_model,
                    entry,
                    self.projected_standings,
                    hitters_proj,
                    pitchers_proj,
                    season_start,
                    season_end,
                    partner=partner,
                    team_sds=self.team_sds,
                )
                entry.update(scores)

        stored_txns.sort(key=lambda t: t.get("timestamp") or "")
        write_cache(CacheKey.TRANSACTIONS, stored_txns, self.cache_dir)
        cache_data = build_cache_output(stored_txns)
        write_cache(CacheKey.TRANSACTION_ANALYZER, cache_data, self.cache_dir)
        self._progress(f"Analyzed {len(stored_txns)} total transaction(s)")

    # --- Step 16: Write meta ---
    def _write_meta(self):
        assert self.config is not None

        self._progress("Finalizing...")
        meta = {
            "last_refresh": local_now().strftime("%Y-%m-%d %H:%M"),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "team_name": self.config.team_name,
        }
        write_cache(CacheKey.META, meta, self.cache_dir)


def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    """Connect to Yahoo, fetch all data, run computations, and write cache files.

    Thin wrapper around RefreshRun for backward compatibility with
    existing callers (scripts/run_lineup.py, season_routes.py).
    """
    RefreshRun(cache_dir).run()
