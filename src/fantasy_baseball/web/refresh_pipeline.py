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
import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fantasy_baseball.models.positions import BENCH_SLOTS

# Streaks imports are deliberately deferred to inside ``_compute_streaks``:
# ``streaks.dashboard``, ``streaks.data.schema``, and ``streaks.pipeline``
# all transitively import ``duckdb`` at module load. duckdb is a [dev]
# extra and is not installed on Render — pulling it in at module import
# would 500 every route that lazy-imports refresh_pipeline (including the
# QStash-driven /api/refresh path that's load-bearing for the daily
# refresh). ``_compute_streaks`` is itself a no-op on Render (see
# ``is_remote()`` gate there), so the imports never actually need to run
# in production. See PR #72 for the parallel /lineup fix.
from fantasy_baseball.utils.constants import AB_PER_PA, Category, OpportunityStat
from fantasy_baseball.utils.positions import PITCHER_POSITIONS
from fantasy_baseball.utils.time_utils import (
    compute_effective_date,
    compute_fraction_remaining,
    local_now,
    local_today,
)
from fantasy_baseball.web.season_data import (
    CacheKey,
    _compute_pending_moves_diff,
    _load_game_log_totals,
    read_cache,
    read_cache_with_meta,
    reset_cache_job,
    set_cache_job,
    write_cache,
    write_cache_to,
)

if TYPE_CHECKING:
    import pandas as pd

    from fantasy_baseball.config import LeagueConfig
    from fantasy_baseball.lineup.optimizer import HitterAssignment, PitcherStarter
    from fantasy_baseball.mc_roster import EffectiveRoster
    from fantasy_baseball.models.league import League
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.models.standings import (
        CategoryStats,
        ProjectedStandings,
        Standings,
        TeamYtdComponents,
    )
    from fantasy_baseball.models.team import Team

log = logging.getLogger(__name__)

_refresh_lock = threading.Lock()
_refresh_status = {"running": False, "progress": "", "error": None}


def get_refresh_status() -> dict:
    with _refresh_lock:
        return dict(_refresh_status)


def try_acquire_refresh_slot() -> bool:
    """Atomically claim the refresh slot.

    Returns True if the caller acquired it (and may now spawn the worker),
    False if a refresh is already running. Without this, the route's
    check-then-spawn pattern races: two POSTs that hit between
    ``get_refresh_status`` and the worker setting ``running=True`` could
    both spawn refresh threads.
    """
    with _refresh_lock:
        if _refresh_status["running"]:
            return False
        _refresh_status["running"] = True
        _refresh_status["progress"] = "Starting..."
        _refresh_status["error"] = None
        return True


def release_refresh_slot() -> None:
    """Release the slot claimed by :func:`try_acquire_refresh_slot`.

    The full refresh and the ROS-projection fetch share this single slot
    (both sync MLB game logs and write the same cache keys, so they must be
    mutually exclusive in-process). Each worker must call this in a
    ``finally`` so a crash can't wedge the slot and block every later job.
    Idempotent -- safe to call when the slot is already free.
    """
    with _refresh_lock:
        _refresh_status["running"] = False


# Cross-instance refresh lock. The in-process slot above only mutexes within
# one Python process; on Render the daily QStash cron and the admin/QStash ROS
# fetch can land on different instances, and QStash's at-least-once delivery can
# redeliver a job. This durable KV lock makes the two heavy jobs mutually
# exclusive across instances too, so their shared game-log rollup
# read-modify-write cannot interleave and silently drop players from the totals.
_REFRESH_LOCK_TTL_SECONDS = 1800  # 30 min: exceeds any real job; self-heals after a hard crash


@contextmanager
def durable_refresh_lock() -> Iterator[bool]:
    """Hold the cross-instance refresh lock for the duration of the block.

    Yields True if acquired (proceed) or False if another instance already
    holds it (the caller should skip -- a duplicate/overlapping run), and
    releases the lock on exit. Both heavy job bodies wrap their work in this,
    so the acquire/skip/release contract lives in one place; any new heavy job
    should do the same. Held in the job BODY (not the route) so direct callers
    (scripts/refresh_remote.py) are protected too, not just QStash triggers.
    """
    import uuid

    from fantasy_baseball.data.kv_store import get_kv
    from fantasy_baseball.data.redis_store import acquire_refresh_lock, release_refresh_lock

    kv = get_kv()
    token = uuid.uuid4().hex
    acquired = acquire_refresh_lock(kv, token, _REFRESH_LOCK_TTL_SECONDS)
    try:
        yield acquired
    finally:
        if acquired:
            # Never raise out of the finally -- releasing is best-effort; the
            # TTL reclaims the lock if a release ever fails. Release against the
            # same store the acquire used.
            try:
                release_refresh_lock(kv, token)
            except Exception as exc:
                log.warning(f"Failed to release durable refresh lock: {exc}")


def refresh_lock_held() -> bool:
    """Best-effort read: is the durable refresh lock currently held?

    The job routes call this to return HTTP 503 (so QStash redelivers later)
    instead of silently dropping an overlapping run when another instance is
    already working. Returns False on any KV error -- the body's
    :func:`durable_refresh_lock` is the authoritative guard; this is only a
    fast-path so a clearly-busy trigger gets retried rather than skipped.
    """
    try:
        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import REFRESH_LOCK_KEY

        return get_kv().get(REFRESH_LOCK_KEY) is not None
    except Exception:
        return False


def _set_refresh_progress(msg: str) -> None:
    with _refresh_lock:
        _refresh_status["progress"] = msg


def _push_streak_scores_to_remote(payload: dict) -> None:
    """Mirror the freshly-computed STREAK_SCORES payload to remote Upstash.

    Local refreshes are the sole authoritative source of streak data
    (duckdb / streaks.duckdb only exist on dev boxes — Render's daily
    refresh short-circuits in ``_compute_streaks``). Without this push,
    the remote dashboard only sees streak updates after a manual
    local->remote sync. Pushing the single cache entry here removes
    that manual step.

    Skipped silently when Upstash creds are absent (CI / fresh checkout)
    and on failure, since streak data is non-load-bearing for the rest
    of the dashboard.
    """
    if not (
        os.environ.get("UPSTASH_REDIS_REST_URL") and os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    ):
        return
    try:
        from fantasy_baseball.data.cache_keys import CacheKey as _CacheKey
        from fantasy_baseball.data.kv_store import build_explicit_upstash_kv

        remote = build_explicit_upstash_kv()
        # Mirror with the same envelope/provenance as the local write_cache so
        # remote and local hold the identical shape (Render reads via read_cache).
        write_cache_to(remote, _CacheKey.STREAK_SCORES, payload)
    except Exception as exc:
        log.warning(f"Failed to mirror streak_scores to remote Upstash: {exc}")


def _write_spoe_snapshot(spoe_result: dict) -> None:
    """Write a daily SPoE snapshot under `spoe_snapshot:YYYY-MM-DD`.

    Separate from the main write_cache path because this key is not
    under the `cache:` prefix — it's a historical time series for the
    luck page to optionally render trend charts. No TTL; accumulates.
    Routes through ``get_kv()`` so it persists in both environments
    (Upstash on Render, SQLite locally).
    """
    snapshot_date = spoe_result.get("snapshot_date")
    if not snapshot_date:
        return
    from fantasy_baseball.data.kv_store import get_kv

    try:
        get_kv().set(
            f"spoe_snapshot:{snapshot_date}",
            json.dumps(spoe_result),
        )
    except Exception as exc:
        log.warning(f"Failed to write spoe_snapshot:{snapshot_date}: {exc}")


def _team_ytd_block(comps: "TeamYtdComponents") -> dict[str, float]:
    """Flatten a TeamYtdComponents to the JSON shape consumed by the
    breakdown modal.

    Keys are lowercase to mirror the per-player ``contribution_stats``
    schema (``HITTING_COUNTING`` + ``PITCHING_COUNTING`` -- r, hr, rbi,
    sb, h, ab, w, k, sv, ip, er, bb, h_allowed) so the modal can read
    team_ytd via the same ``colSpec.field`` path as per-player rows.

    ``bb_plus_h_allowed`` is exposed as the combined-sum key since YTD
    only exposes ``WHIP * IP`` (the sum); we can't split it into bb /
    h_allowed without per-game-log decomposition.
    """
    return {
        "r": comps.r,
        "hr": comps.hr,
        "rbi": comps.rbi,
        "sb": comps.sb,
        "h": comps.h,
        "ab": comps.ab,
        "w": comps.w,
        "k": comps.k,
        "sv": comps.sv,
        "ip": comps.ip,
        "er": comps.er,
        "bb_plus_h_allowed": comps.bb_plus_h_allowed,
    }


def build_standings_breakdown_payload(
    team_rosters: dict[str, list["Player"]],
    effective_date: date,
    *,
    fraction_remaining: float = 1.0,
    actual_standings: "Standings | None" = None,
) -> dict:
    """Build the STANDINGS_BREAKDOWN cache payload for ``team_rosters``.

    One :class:`RosterBreakdown` per team, serialized to a JSON-safe
    dict and keyed by team name. ``effective_date`` is included to
    match the :class:`ProjectedStandings` payload shape.

    Per-player rows are ROS-only (post-team-YTD refactor) so the modal
    can render the arithmetic exposed by the standings widget:
    ``team_YTD + sum(player ROS contribution rows) == projected_standings``
    for every team and every category. The team's YTD totals are emitted
    as a separate ``team_ytd`` block (derived from
    ``StandingsEntry.ytd_components()``) so the legacy per-player
    YTD-floor path is no longer needed here. When ``actual_standings``
    is ``None`` (pre-season or omitted), the block is all zeros so
    consumers don't need to branch on its presence.

    Uses the same two-pass DeltaRoto-optimal displacement as
    :meth:`ProjectedStandings.from_rosters`, on ROS components, so
    per-player ``contribution_stats[cat]`` aggregates here match the
    standings widget exactly:

    1. Pass 1 -- SGP-based displacement on ROS -> baseline ``{team: stats}``.
    2. Pass 2 -- each team picks its displacement targets via DeltaRoto,
       evaluating against frozen pass-1 baseline of other teams.

    Without the two-pass, the breakdown would show different scale
    factors than the standings (e.g., Mason Miller at sf=0.25 in the
    breakdown but at sf~=1.0 in the projected standings), which would
    desync the modal drilldown from the headline numbers.
    """
    from fantasy_baseball.models.standings import TeamYtdComponents, build_eos_baseline
    from fantasy_baseball.scoring import (
        LeagueContext,
        build_team_sds,
        compute_roster_breakdown,
    )

    ytd_by_team: dict[str, TeamYtdComponents] = {}
    if actual_standings is not None:
        for entry in actual_standings.entries:
            ytd_by_team[entry.team_name] = entry.ytd_components()

    # Pass 1: per-team END-OF-SEASON baseline consumed by the DeltaRoto picker.
    # Shared with ProjectedStandings.from_rosters so both pickers score against
    # the same YTD-inclusive baseline -- a ROS-only baseline here made the
    # per-player breakdown choose different displacement targets than the
    # standings widget and stop summing to the headline.
    baseline_stats = build_eos_baseline(team_rosters, ytd_by_team)
    # Match ProjectedStandings.from_rosters: damp the picker's SDs by
    # sqrt(fraction_remaining) so the breakdown's displacement decisions
    # agree with the standings widget and the canonical team_sds.
    team_sds = build_team_sds(team_rosters, sd_scale=fraction_remaining**0.5)

    teams_payload: dict[str, dict] = {}
    for team_name, roster in team_rosters.items():
        ctx = LeagueContext(
            baseline_other_team_stats={t: s for t, s in baseline_stats.items() if t != team_name},
            team_sds=team_sds,
            team_name=team_name,
            fraction_remaining=fraction_remaining,
        )
        # team_ytd is a first-class field on RosterBreakdown so it survives the
        # season_routes round-trip through from_dict/to_dict (used to backfill
        # contribution_stats on stale KV blobs). Passing it through here keeps
        # the team-YTD block out of the per-player rows -- per-player YTD
        # attribution is intentionally not done; only stats accrued while the
        # player was on the team count, and that bookkeeping lives at the team
        # level via TeamYtdComponents.
        team_ytd_components = ytd_by_team.get(team_name)
        if team_ytd_components is None:
            if actual_standings is not None:
                log.warning(
                    "Team YTD lookup miss for %r in build_standings_breakdown_payload; "
                    "falling back to zero YTD (team_rosters key not found in "
                    "actual_standings.entries -- apostrophe / whitespace / Unicode "
                    "normalization drift?)",
                    team_name,
                )
            team_ytd_components = TeamYtdComponents()
        breakdown = compute_roster_breakdown(
            team_name,
            roster,
            league_context=ctx,
            team_ytd=_team_ytd_block(team_ytd_components),
        )
        teams_payload[team_name] = breakdown.to_dict()
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

    def __init__(self) -> None:
        from fantasy_baseball.web.job_logger import JobLogger

        self.logger = JobLogger("refresh")

        self.config: LeagueConfig | None = None
        self.league: Any = None  # Yahoo session-bound league (untyped lib)
        self.teams_dict: dict[str, Team] | None = None
        self.league_model: League | None = None
        self.user_team_key: str | None = None
        self.standings: Standings | None = None
        # ytd_standings is self.standings augmented with team-YTD AB on
        # extras (computed in _build_projected_standings). Stored on
        # self so _audit_roster and _optimize_lineup can pass it to the
        # stash board and optimizer user-rows -- the same scale the
        # projected standings widget consumes.
        self.ytd_standings: Standings | None = None
        self.projected_standings: ProjectedStandings | None = None
        # Pass-1 end-of-season baseline ({team: CategoryStats}) shared with
        # ProjectedStandings.from_rosters and (Phase 4) the games-based MC so
        # the MC builds the same LeagueContext ERoto used -- one shared object,
        # no divergent recompute. Populated in _build_projected_standings.
        self.eos_baseline: dict[str, CategoryStats] | None = None
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
        # Raw game_log_totals rollups (string mlbam_id keys) captured in
        # _fetch_game_logs and reused for full-season derivation in
        # _load_projections without a second KV read.
        self.raw_hitter_totals: dict[str, Any] | None = None
        self.raw_pitcher_totals: dict[str, Any] | None = None
        self.leverage: dict[str, float] | None = None
        self.rankings_lookup: dict[str, dict[str, Any]] | None = None
        self.optimal_hitters: list[HitterAssignment] | None = None
        self.optimal_pitchers_starters: list[PitcherStarter] | None = None
        self.optimal_pitchers_bench: list[Player] | None = None
        self.fa_players: list[Player] | None = None
        self.rest_of_season_mc: dict[str, Any] | None = None

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
        # Stamp every cache:* blob this refresh writes with its writer; reset
        # in finally so a synchronous/reused worker thread doesn't leak the
        # label into the next job's writes.
        job_token = set_cache_job("refresh")
        with _refresh_lock:
            _refresh_status["running"] = True
            _refresh_status["progress"] = "Starting..."
            _refresh_status["error"] = None

        try:
            # The durable lock makes the two heavy jobs mutually exclusive
            # across Render instances / QStash redelivery (the in-process slot
            # only reaches within one process); two refreshes racing the
            # game-log rollup RMW silently drop players from the totals. Skip
            # this run when another instance already holds it.
            with durable_refresh_lock() as got_lock:
                if not got_lock:
                    self._progress("Another instance holds the refresh lock; skipping this run.")
                    self.logger.finish("skipped", "another instance holds the refresh lock")
                    return
                self._run_pipeline_steps()
        except Exception as exc:
            with _refresh_lock:
                _refresh_status["error"] = str(exc)
            self.logger.finish("error", str(exc))
            raise
        finally:
            reset_cache_job(job_token)
            release_refresh_slot()

    def _run_pipeline_steps(self) -> None:
        """The ordered refresh steps, run while holding the durable lock."""
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
        self._compute_draft_value()
        self._compute_streaks()
        self._write_meta()

        self.logger.finish("ok")
        self._progress("Done")
        from fantasy_baseball.web.season_data import clear_opponent_cache

        clear_opponent_cache()

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
        from fantasy_baseball.lineup.yahoo_roster import fetch_teams, find_user_team_key

        assert self.config is not None
        self._progress("Finding team...")
        self.teams_dict = fetch_teams(self.league)
        self.user_team_key = find_user_team_key(self.teams_dict, self.config.team_name)

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
        write_cache(CacheKey.STANDINGS, self.standings.to_json())
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
        write_cache(CacheKey.PENDING_MOVES, pending_moves)
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
        # Single read: pull the ROS data AND its provenance meta (snapshot date)
        # in one round-trip, rather than re-fetching + re-parsing the same blob
        # for the staleness warning below.
        ros_payload, ros_meta = read_cache_with_meta(CacheKey.ROS_PROJECTIONS)
        ros_hitters, ros_pitchers = _load_projection_pair(ros_payload)
        self.has_rest_of_season = not ros_hitters.empty or not ros_pitchers.empty
        if self.has_rest_of_season:
            self.hitters_proj = ros_hitters
            self.pitchers_proj = ros_pitchers
            self._progress(
                f"Loaded {len(ros_hitters)} ROS hitters + {len(ros_pitchers)} ROS pitchers"
            )
            self._warn_if_ros_blob_stale(ros_meta.get("_ros_snapshot_date"))
        else:
            self._progress("WARNING: No ROS projections available — falling back to preseason")

        # Full-season (ROS+YTD) projections populate Player.full_season_projection
        # for display + ProjectedStandings, while Player.rest_of_season stays
        # ROS-only for forward-looking decision paths.
        #
        # Re-derive full-season HERE from the ROS blob + the CURRENT game-log
        # totals (the same ones _build_projected_standings uses for the team-YTD
        # overlay) rather than reading cache:full_season_projections. That cached
        # blob is written by the SEPARATE _run_rest_of_season_fetch job and is
        # frozen at whenever that job last ran; consuming it would blend a stale
        # YTD vintage into the per-player full-season lines while the team-YTD
        # overlay is current, so the projected standings would mix vintages.
        # Deriving locally keeps the whole standings computation self-consistent
        # regardless of when the ROS fetch last ran. (game logs were synced in
        # _fetch_game_logs, the prior step, so these totals are fresh.)
        if self.has_rest_of_season:
            from fantasy_baseball.data.ros_pipeline import derive_full_season

            self._progress("Deriving full-season projections (ROS + current YTD)...")
            # Reuse the rollups _fetch_game_logs already read (no extra KV read);
            # the shared helper int-keys them with the same policy the ROS job
            # uses, so the two derivations can't drift.
            self.full_hitters_proj, self.full_pitchers_proj = derive_full_season(
                ros_hitters,
                ros_pitchers,
                self.raw_hitter_totals or {},
                self.raw_pitcher_totals or {},
            )
            self._progress(
                f"Derived {len(self.full_hitters_proj)} full-season hitters + "
                f"{len(self.full_pitchers_proj)} full-season pitchers"
            )
        else:
            # No ROS blob to add YTD onto; fall back to the cached full-season
            # blob (best-effort) so display still has something.
            self._progress("Loading full-season projections (no ROS; from cache)...")
            full_hitters, full_pitchers = _load_projection_pair(
                read_cache(CacheKey.FULL_SEASON_PROJECTIONS)
            )
            if not full_hitters.empty or not full_pitchers.empty:
                self.full_hitters_proj = full_hitters
                self.full_pitchers_proj = full_pitchers
            else:
                self._progress(
                    "WARNING: no ROS projections and cache:full_season_projections "
                    "missing -- Player.full_season_projection will be unset"
                )

    def _warn_if_ros_blob_stale(self, snapshot_date: str | None) -> None:
        """Surface the vintage of the ROS blob we just loaded.

        ``snapshot_date`` is the blob's stamped ``_ros_snapshot_date`` (its dir
        name), already in hand from the single read in ``_load_projections`` -- no
        second KV round-trip. full-season is re-derived here as
        ``ROS + CURRENT YTD``; the write-side guard keeps the last-good ROS blob
        during a FanGraphs outage, but as that frozen blob ages behind today,
        pairing its remaining-stats with today's YTD double-counts the games in
        between. Warn so the drift is visible -- defense in depth on the read side,
        mirroring the write-side refusal (same staleness arithmetic + threshold).
        """
        from fantasy_baseball.data.ros_pipeline import (
            ROS_SNAPSHOT_STALE_DAYS,
            parse_snapshot_date,
            ros_snapshot_days_stale,
        )

        snap = parse_snapshot_date(str(snapshot_date)) if snapshot_date else None
        if snap is None:
            return
        days = ros_snapshot_days_stale(snap)
        if days > ROS_SNAPSHOT_STALE_DAYS:
            self._progress(
                f"WARNING: ROS blob snapshot {snapshot_date} is {days} days old "
                f"(> {ROS_SNAPSHOT_STALE_DAYS}); full-season = YTD + ROS may double-count "
                f"~{days} days of games. Re-pull fresh ROS CSVs (FanGraphs fetch may be down)."
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

        def _fetch_opp(team: "Team"):
            try:
                opp_raw = fetch_roster(self.league, team.team_key, day=self.effective_date)
                return (team.name, opp_raw)
            except Exception as exc:
                log.warning(f"Opponent roster fetch failed for {team.name or team.team_key}: {exc}")
                return None

        assert self.teams_dict is not None
        opp_teams = [
            team
            for team in self.teams_dict.values()
            if team.name != self.config.team_name and team.team_key != self.user_team_key
        ]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for result in pool.map(_fetch_opp, opp_teams):
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

        def _hydrate(roster_model, context):
            # One shared frame bundle (ROS + full-season + preseason) so user and
            # opponents are hydrated identically -- in particular both get the
            # preseason line the displacement slot-share needs.
            return hydrate_roster_entries(
                roster_model,
                self.hitters_proj,
                self.pitchers_proj,
                full_hitters_proj=self.full_hitters_proj,
                full_pitchers_proj=self.full_pitchers_proj,
                preseason_hitters_proj=self.preseason_hitters,
                preseason_pitchers_proj=self.preseason_pitchers,
                context=context,
            )

        user_team_model = self.league_model.team_by_name(self.config.team_name)
        self.matched = _hydrate(user_team_model.latest_roster(), "user")

        self.opp_rosters = {}
        for team in self.league_model.teams:
            if team.name == self.config.team_name:
                continue
            if not team.rosters:
                continue
            hydrated = _hydrate(team.latest_roster(), f"opp:{team.name}")
            if hydrated:
                self.opp_rosters[team.name] = hydrated
        self._progress(f"Hydrated {len(self.opp_rosters)} opponent rosters")

        # Cache opponent rosters for on-demand trade search
        opp_rosters_flat = {
            tname: [p.to_dict() for p in roster] for tname, roster in self.opp_rosters.items()
        }
        write_cache(CacheKey.OPP_ROSTERS, opp_rosters_flat)

    # --- Step 4e: Build projected standings ---
    def _build_projected_standings(self):
        self._progress("Projecting end-of-season standings...")
        from fantasy_baseball.analysis.team_ytd_attribution import compute_team_ytd_ab
        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.projections import hydrate_roster_entries
        from fantasy_baseball.data.redis_store import build_hitter_ytd_game_logs
        from fantasy_baseball.models.standings import (
            ProjectedStandings,
            Standings,
            StandingsEntry,
        )
        from fantasy_baseball.scoring import build_team_sds, team_sds_to_json

        assert self.config is not None
        assert self.matched is not None
        assert self.opp_rosters is not None
        assert self.effective_date is not None
        assert self.league_model is not None
        assert self.preseason_hitters is not None
        assert self.preseason_pitchers is not None
        assert self.standings is not None

        all_team_rosters = {self.config.team_name: self.matched}
        all_team_rosters.update(self.opp_rosters)

        # Compute fraction_remaining first so the standings build can damp its
        # displacement-picker SDs by sqrt(fraction_remaining) -- matching the
        # canonical team_sds below and every other ERoto consumer.
        self.fraction_remaining = compute_fraction_remaining(
            date.fromisoformat(self.config.season_start),
            date.fromisoformat(self.config.season_end),
            local_today(),
        )
        self.sd_scale = math.sqrt(self.fraction_remaining)

        # Yahoo's team-standings response does not expose AB for this league,
        # so derive team-YTD AB from Team.ownership_periods() intersected with
        # per-game hitter logs and stuff it onto extras[OpportunityStat.AB].
        # ytd_components() then reads AB via Tier 1 of its sourcing precedence
        # and recombines AVG correctly downstream.
        #
        # The game logs are assembled from Upstash (the incrementally-synced
        # game_logs:{season}:* records) rather than read from
        # data/roster_game_logs.json -- that file is built by nothing in the
        # deployed pipeline and is absent on Render, which made the file
        # fallback yield AB=0 and silently collapse team-YTD AVG to ROS-only.
        game_logs = build_hitter_ytd_game_logs(get_kv(), self.config.season_year)
        ab_by_team = compute_team_ytd_ab(
            self.league_model,
            season_start=date.fromisoformat(self.config.season_start),
            season_end=date.fromisoformat(self.config.season_end),
            game_logs=game_logs,
        )
        ytd_standings = Standings(
            effective_date=self.standings.effective_date,
            entries=[
                StandingsEntry(
                    team_name=e.team_name,
                    team_key=e.team_key,
                    rank=e.rank,
                    stats=e.stats,
                    yahoo_points_for=e.yahoo_points_for,
                    extras={
                        **e.extras,
                        OpportunityStat.AB: ab_by_team.get(e.team_name, 0.0),
                    },
                )
                for e in self.standings.entries
            ],
        )

        from fantasy_baseball.models.standings import build_eos_baseline

        ytd_by_team = {e.team_name: e.ytd_components() for e in ytd_standings.entries}
        self.eos_baseline = build_eos_baseline(all_team_rosters, ytd_by_team)

        self.projected_standings = ProjectedStandings.from_rosters(
            all_team_rosters,
            effective_date=self.effective_date,
            actual_standings=ytd_standings,
            fraction_remaining=self.fraction_remaining,
            baseline_stats=self.eos_baseline,
        )

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
        )

        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import write_projected_standings_snapshot

        write_projected_standings_snapshot(get_kv(), self.projected_standings)

        write_cache(
            CacheKey.STANDINGS_BREAKDOWN,
            build_standings_breakdown_payload(
                all_team_rosters,
                self.effective_date,
                fraction_remaining=self.fraction_remaining,
                actual_standings=ytd_standings,
            ),
        )
        # Persist for _audit_roster (stash) and _optimize_lineup -- they
        # need the YTD-augmented standings so the user-row sees the same
        # AB attribution that the projected standings widget consumes.
        self.ytd_standings = ytd_standings
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
        )

        self._progress(f"Matched {len(self.roster_players)} players to projections")

    # --- Step 3b: Fetch MLB game logs (must precede ROS blend) ---
    def _fetch_game_logs(self):
        from fantasy_baseball.data.mlb_game_logs import fetch_game_log_totals

        assert self.config is not None

        self._progress("Fetching MLB game logs...")
        # Capture the rollups fetch_game_log_totals already reads back, so
        # _load_projections can derive full-season without a second KV read of
        # the same blobs.
        self.raw_hitter_totals, self.raw_pitcher_totals, _ = fetch_game_log_totals(
            self.config.season_year, progress_cb=self._progress
        )

    # --- Step 6c: Compute season-to-date pace vs projections ---
    def _compute_pace(self):
        assert self.config is not None
        assert self.roster_players is not None
        assert self.preseason_lookup is not None

        self._progress("Computing player pace...")
        self.hitter_logs, self.pitcher_logs = _load_game_log_totals()

        # Attach pace data to each roster player (pace compares actuals vs preseason)
        from fantasy_baseball.analysis.pace import attach_pace_to_roster
        from fantasy_baseball.sgp.denominators import get_sgp_denominators

        sgp_denoms = get_sgp_denominators()
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
            build_rankings_lookup,
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

        # Full-season (YTD + ROS) leaguewide ranking. The full-season pools are
        # already derived earlier in the pipeline; if they are absent (e.g. no
        # ROS projections), fall back to an empty ranking so rank.total is None.
        if self.full_hitters_proj is not None and self.full_pitchers_proj is not None:
            total_ranks = compute_sgp_rankings(self.full_hitters_proj, self.full_pitchers_proj)
        else:
            total_ranks = {}

        self.rankings_lookup = build_rankings_lookup(
            rest_of_season_ranks,
            preseason_ranks,
            current_ranks,
            total_ranks,
        )

        write_cache(CacheKey.RANKINGS, self.rankings_lookup)
        self._progress(
            f"Ranked {len(rest_of_season_ranks)} ROS, {len(preseason_ranks)} preseason, "
            f"{len(current_ranks)} current, {len(total_ranks)} total"
        )

        # Attach ranks to roster players
        from fantasy_baseball.models.player import RankInfo

        for player in self.roster_players:
            rank_data = lookup_rank(
                self.rankings_lookup, player.fg_id, player.name, player.player_type
            )
            player.rank = RankInfo.from_dict(rank_data) if rank_data else RankInfo()

        roster_flat = [p.to_flat_dict() for p in self.roster_players]
        write_cache(CacheKey.ROSTER, roster_flat)

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
        active_players = [p for p in self.roster_players if not p.is_on_il()]
        hitter_players = []
        pitcher_players = []
        for player in active_players:
            if set(player.positions) & PITCHER_POSITIONS:
                pitcher_players.append(player)
            else:
                hitter_players.append(player)

        # actual_standings threads team_YTD into the user-row baseline that
        # team_roto_total builds (see optimizer._resolve_user_ytd_components
        # and team_roto_total docstrings). Without this, the user row is
        # ROS-only while opponents (from ProjectedStandings.from_rosters) are
        # team_YTD + ROS, putting the user in a low-mu region of the
        # score_roto S-curve and silently saturating counting-cat deltas --
        # the same bug PR #110 fixed for the stash board.
        #
        # Use the YTD-augmented standings (AB on extras) -- self.standings
        # alone has no AB stat for this league, so AVG attribution would
        # silently collapse to zero AB. Fall back to self.standings as a
        # defensive net.
        opt_actual_standings = self.ytd_standings or self.standings
        self.optimal_hitters = optimize_hitter_lineup(
            hitters=hitter_players,
            full_roster=self.roster_players,
            projected_standings=self.projected_standings,
            team_name=self.config.team_name,
            roster_slots=self.config.roster_slots,
            team_sds=self.team_sds,
            fraction_remaining=self.fraction_remaining,
            actual_standings=opt_actual_standings,
        )
        self.optimal_pitchers_starters, self.optimal_pitchers_bench = optimize_pitcher_lineup(
            pitchers=pitcher_players,
            full_roster=self.roster_players,
            projected_standings=self.projected_standings,
            team_name=self.config.team_name,
            slots=self.config.roster_slots.get("P", 9),
            team_sds=self.team_sds,
            fraction_remaining=self.fraction_remaining,
            actual_standings=opt_actual_standings,
        )

    # --- Step 8: Compare optimal to current, find moves ---
    def _compute_moves(self):
        from fantasy_baseball.web.refresh_steps import compute_lineup_moves

        assert self.optimal_hitters is not None
        assert self.optimal_pitchers_starters is not None
        assert self.optimal_pitchers_bench is not None
        assert self.roster_players is not None

        self._progress("Computing lineup moves...")
        moves = compute_lineup_moves(
            optimal_hitters=self.optimal_hitters,
            optimal_pitchers=self.optimal_pitchers_starters,
            pitcher_bench=self.optimal_pitchers_bench,
            roster_players=self.roster_players,
        )

        optimal_data = {
            "hitter_lineup": [a.to_dict() for a in self.optimal_hitters],
            "pitcher_starters": [s.to_dict() for s in self.optimal_pitchers_starters],
            "pitcher_bench": [p.name for p in self.optimal_pitchers_bench],
            "moves": moves,
        }
        write_cache(CacheKey.LINEUP_OPTIMAL, optimal_data)

    # --- Step 9: Probable starters ---
    def _fetch_probable_starters(self):
        from fantasy_baseball.data.mlb_schedule import get_week_schedule
        from fantasy_baseball.lineup.matchups import (
            get_probable_starters,
            get_team_batting_stats,
        )
        from fantasy_baseball.lineup.upcoming_starts import filter_starting_pitchers

        assert self.start_date is not None
        assert self.end_date is not None
        assert self.roster_players is not None
        assert self.pitchers_proj is not None

        self._progress("Fetching schedule and matchup data...")
        project_root = Path(__file__).resolve().parents[3]
        schedule_cache_path = project_root / "data" / "weekly_schedule.json"
        # 14-day lookback: the upcoming-starts module needs each pitcher's
        # most recent start as the rotation anchor for projecting forward.
        schedule = get_week_schedule(
            self.start_date,
            self.end_date,
            schedule_cache_path,
            lookback_days=14,
        )

        batting_stats_cache_path = project_root / "data" / "team_batting_stats.json"
        team_stats = get_team_batting_stats(batting_stats_cache_path)

        sp_roster = filter_starting_pitchers(self.roster_players, self.pitchers_proj)

        probable_starters = get_probable_starters(
            sp_roster,
            schedule or {},
            team_stats=team_stats,
        )
        write_cache(CacheKey.PROBABLE_STARTERS, probable_starters, required=False)

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
        assert self.fraction_remaining is not None

        self._progress("Running roster audit...")
        self.fa_players, _ = fetch_and_match_free_agents(
            self.league,
            self.hitters_proj,
            self.pitchers_proj,
            preseason_hitters_proj=self.preseason_hitters,
            preseason_pitchers_proj=self.preseason_pitchers,
        )

        # Cache positions for all known players (roster + opponents + FAs)
        positions_map = build_positions_map(self.roster_players, self.opp_rosters, self.fa_players)
        write_cache(CacheKey.POSITIONS, positions_map)
        self._progress(f"Cached positions for {len(positions_map)} players")

        audit_results = audit_roster(
            self.roster_players,
            self.fa_players,
            self.config.roster_slots,
            projected_standings=self.projected_standings,
            team_name=self.config.team_name,
            fraction_remaining=self.fraction_remaining,
            team_sds=self.team_sds,
            optimal_hitters=self.optimal_hitters,
            optimal_pitchers=self.optimal_pitchers_starters,
        )
        write_cache(CacheKey.ROSTER_AUDIT, [e.to_dict() for e in audit_results])
        upgrades = sum(1 for e in audit_results if e.gap > 0)
        self._progress(f"Roster audit: {upgrades} upgrade(s) found")

        from fantasy_baseball.lineup.stash_value import StashResult, score_stash_candidates

        # The stash board is a non-critical add-on. A failure here must NOT
        # abort the rest of the refresh (Monte Carlo, standings, meta), so we
        # degrade to an empty cached board and continue -- mirroring the
        # streak-computation step.
        # Use the YTD-augmented standings (AB on extras) when available so
        # the stash baseline's user row matches the projected standings
        # scale. Fall back to self.standings defensively for code paths
        # that may not have populated ytd_standings yet.
        stash_actual_standings = self.ytd_standings or self.standings
        try:
            stash_result = score_stash_candidates(
                self.roster_players,
                self.fa_players,
                self.projected_standings,
                self.config.roster_slots,
                self.config.team_name,
                team_sds=self.team_sds,
                fraction_remaining=self.fraction_remaining,
                actual_standings=stash_actual_standings,
            )
            write_cache(CacheKey.STASH, stash_result.to_dict())
            self._progress(f"Stash board: {len(stash_result.candidates)} injured candidate(s)")
        except Exception:
            log.exception("Stash board computation failed; caching empty board")
            il_capacity = self.config.roster_slots.get("IL", 0)
            write_cache(
                CacheKey.STASH,
                StashResult(
                    open_il_slots=0,
                    cutline_rank=il_capacity,
                    candidates=[],
                    warning="Stash board unavailable this refresh.",
                ).to_dict(),
            )
            self._progress("Stash board: computation failed (empty board cached)")

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
        write_cache(CacheKey.LEVERAGE, leverage_by_team, required=False)

    # --- ROS Monte Carlo simulation ---
    def _run_ros_monte_carlo(self):
        assert self.config is not None

        if self.has_rest_of_season:
            from fantasy_baseball.simulation import run_ros_monte_carlo

            assert self.matched is not None
            assert self.opp_rosters is not None
            assert self.standings is not None
            assert self.fraction_remaining is not None

            non_hitter_slots = BENCH_SLOTS | {"P"}
            h_slots = sum(
                v for k, v in self.config.roster_slots.items() if k not in non_hitter_slots
            )
            p_slots = self.config.roster_slots.get("P", 9)

            # opp_rosters were already matched against ROS projections in
            # _hydrate_rosters (hitters_proj/pitchers_proj swap to ROS at
            # _load_projections), so they're MC-ready as-is.
            rest_of_season_mc_rosters: dict[str, list] = dict(self.opp_rosters)
            if self.matched:
                rest_of_season_mc_rosters[self.config.team_name] = self.matched

            # Build actual standings dict (uppercase-string-keyed per category),
            # threading each team's real accumulated AB/IP so the ROS blend
            # weights YTD by actual playing time rather than a league-typical
            # constant. Yahoo ships PA/IP on standings extras; AB is PA scaled
            # by the stable league AB-per-PA ratio.
            actual_standings_dict: dict[str, dict[str, float]] = {}
            for e in self.standings.entries:
                row = e.stats.to_dict()
                ip = e.extras.get(OpportunityStat.IP)
                pa = e.extras.get(OpportunityStat.PA)
                if ip is not None:
                    row["IP"] = float(ip)
                if pa is not None:
                    row["AB"] = float(pa) * AB_PER_PA
                actual_standings_dict[e.team_name] = row

            if os.environ.get("FB_SELECTION_ATTRIBUTION"):  # gated Phase 0 diagnostic
                from fantasy_baseball.mc_selection import (
                    format_attribution_table,
                    format_sd_calibration_table,
                    run_selection_attribution,
                )

                assert self.fraction_remaining is not None
                attr, sd_calib = run_selection_attribution(
                    rest_of_season_mc_rosters,
                    actual_standings_dict,
                    self.fraction_remaining,
                    h_slots,
                    p_slots,
                    n_iter=1000,
                    seed=42,
                    eos_baseline=self.eos_baseline,
                    team_sds=self.team_sds,
                )
                header = f"seed=42 n_iter=1000 fraction_remaining={self.fraction_remaining:.4f}\n"
                with open("phase0_attribution.txt", "w", encoding="ascii", errors="replace") as fh:
                    fh.write(header)
                    fh.write(format_attribution_table(attr))
                    if sd_calib is not None:
                        fh.write("\n\n=== SD calibration (counting cats only) ===\n")
                        fh.write(format_sd_calibration_table(sd_calib))
                self._progress("Selection-attribution diagnostic written")

            # Phase 4b: build each team's EffectiveRoster (fixed active set + IL
            # displacement + bench fill pool) from the LIVE Player rosters, using
            # the SAME LeagueContext (baseline, team_sds, fraction_remaining) the
            # standings build used -- so the MC's IL handling agrees with ERoto by
            # construction. Routes hitters through the ROS-direct body engine. Only
            # when the standings context is available; otherwise None -> top-k.
            effective_rosters: dict[str, EffectiveRoster] | None = None
            if self.eos_baseline is not None and self.team_sds is not None:
                from fantasy_baseball.mc_roster import build_effective_rosters

                effective_rosters = build_effective_rosters(
                    rest_of_season_mc_rosters,
                    self.eos_baseline,
                    self.team_sds,
                    self.fraction_remaining,
                )

            if rest_of_season_mc_rosters:
                self.rest_of_season_mc = run_ros_monte_carlo(
                    team_rosters=rest_of_season_mc_rosters,
                    actual_standings=actual_standings_dict,
                    fraction_remaining=self.fraction_remaining,
                    h_slots=h_slots,
                    p_slots=p_slots,
                    user_team_name=self.config.team_name,
                    n_iterations=1000,
                    progress_cb=lambda i: self._progress(f"Current MC: iteration {i}/1000..."),
                    effective_rosters=effective_rosters,
                )
                self._progress("Current Monte Carlo complete")

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
                "baseline_meta": baseline.get("meta"),
                "rest_of_season": self.rest_of_season_mc,
            },
            required=False,
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

        write_cache(CacheKey.SPOE, spoe_result, required=False)
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

        stored_txns_raw = read_cache(CacheKey.TRANSACTIONS) or []
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
        write_cache(CacheKey.TRANSACTIONS, stored_txns, required=False)
        cache_data = build_cache_output(stored_txns)
        write_cache(CacheKey.TRANSACTION_ANALYZER, cache_data, required=False)
        self._progress(f"Analyzed {len(stored_txns)} total transaction(s)")

    # --- Step 15a: Draft-value grade for the /transactions Draft Grade tab ---
    def _compute_draft_value(self) -> None:
        """Grade the draft (keepers + drafted picks) and cache it for the tab.

        Non-load-bearing: run_draft_value() can raise (reconstruction gate /
        missing keepers), so catch broadly, log, and continue -- a cosmetic
        panel must never abort a refresh whose load-bearing steps succeeded.
        required=False leaves any prior cache untouched on failure. Runs on
        Render too (all inputs are git-tracked or in the KV store; no duckdb).
        """
        self._progress("Computing draft value grade...")
        try:
            from fantasy_baseball.analysis.draft_value import (
                build_draft_value_cache,
                run_draft_value,
            )

            players, teams = run_draft_value()
            payload = build_draft_value_cache(players, teams)
            write_cache(CacheKey.DRAFT_VALUE, payload, required=False)
            self._progress(f"Draft value cached: {len(teams)} teams")
        except Exception:
            log.exception("Draft-value computation failed; cache unchanged")
            self._progress("Draft value computation failed (continuing)")

    # --- Step 15b: Compute streak scores for /streaks + lineup chips ---
    def _compute_streaks(self) -> None:
        """Run the full streak pipeline + serialize + write cache.

        Skipped on Render: duckdb is a [dev] extra (not installed on
        Render) and the streaks.duckdb file is gitignored (not deployed
        either). The cache is populated by running
        ``scripts/refresh_remote.py`` from a developer machine, which
        writes ``STREAK_SCORES`` directly to Upstash. The existing
        Upstash value survives the Render-side refresh untouched.

        Failures are logged but not re-raised — streak data is
        non-load-bearing for the rest of the dashboard. The DuckDB
        connection is closed in a ``finally`` so a failure inside
        ``compute_streak_report`` still releases the file lock.
        """
        from fantasy_baseball.data.kv_store import is_remote

        if is_remote():
            self._progress("Skipping streak compute on Render (run refresh_remote.py locally)")
            return

        # Local-only imports: see the module-level comment about the deferred
        # streaks imports for the full rationale.
        from fantasy_baseball.streaks.dashboard import serialize_report
        from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection
        from fantasy_baseball.streaks.pipeline import compute_streak_report

        self._progress("Computing streak scores...")
        try:
            assert self.config is not None
            assert self.league is not None
            project_root = Path(__file__).resolve().parents[3]
            conn = get_connection(DEFAULT_DB_PATH)
            try:
                report = compute_streak_report(
                    conn,
                    league=self.league,
                    team_name=self.config.team_name,
                    league_id=self.config.league_id,
                    projections_root=project_root / "data" / "projections",
                    scoring_season=self.config.season_year,
                    top_n_fas=50,
                )
            finally:
                conn.close()
            payload = serialize_report(report)
            write_cache(CacheKey.STREAK_SCORES, payload)
            _push_streak_scores_to_remote(payload)
            self._progress(
                f"Streak scores cached: {len(report.roster_rows)} roster, {len(report.fa_rows)} FAs"
            )
        except Exception:
            log.exception("Streak computation failed; cache unchanged")
            self._progress("Streak computation failed (continuing)")

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
        write_cache(CacheKey.META, meta)


def run_full_refresh() -> None:
    """Connect to Yahoo, fetch all data, run computations, and write cache.

    Thin wrapper around RefreshRun for backward compatibility with
    existing callers (scripts/run_lineup.py, season_routes.py).
    """
    RefreshRun().run()
