"""Revert guardrail for the manual-mode seam in ``RefreshRun``.

The seam is two additive, default-off keyword arguments on
``RefreshRun.__init__`` -- ``free_agent_source`` and ``job_label`` -- plus the
three branch conditions they feed. This file exists to prove the one claim that
makes the seam safe to ship inside the production Yahoo pipeline: **at their
defaults the new arguments change nothing.**

Every test in :class:`TestDefaultsAreInert` is a revert test. If one of them
fails, the seam has leaked into the live refresh path, and the fix is to revert
the seam rather than to loosen the test. The manual-mode tests then prove the
other half -- that with a source injected the audit genuinely runs against the
supplied pool -- and :class:`TestManualModeNeverReachesProd` pins the property
that keeps a hand-transcribed run from writing to production Upstash.
"""

import json
from datetime import date, datetime
from typing import ClassVar
from unittest.mock import patch

import pandas as pd
import pytest

from fantasy_baseball.config import LeagueConfig
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.lineup.waivers import FreeAgentRequest
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.web import refresh_pipeline
from fantasy_baseball.web.refresh_pipeline import RefreshRun
from tests._cache_helpers import unwrap_cache_value
from tests.test_web._refresh_fixture import patched_refresh_environment

# The exact strings today's pipeline uses. Spelled out here rather than
# imported so a change to either one has to be made twice, deliberately.
TODAYS_JOB_LABEL = "refresh"
TODAYS_SKIP_AUDIT_MESSAGE = "Skipping roster audit (stale-data mode: free agents need Yahoo)"

YAHOO_ENTRY_POINTS = [
    "fantasy_baseball.auth.yahoo_auth.get_yahoo_session",
    "fantasy_baseball.auth.yahoo_auth.get_league",
    "fantasy_baseball.lineup.yahoo_roster.fetch_teams",
    "fantasy_baseball.lineup.yahoo_roster.fetch_roster",
    "fantasy_baseball.lineup.yahoo_roster.fetch_standings",
    "fantasy_baseball.lineup.yahoo_roster.fetch_scoring_period",
    "fantasy_baseball.lineup.yahoo_roster.fetch_all_transactions",
    "fantasy_baseball.lineup.waivers.fetch_and_match_free_agents",
]


# --------------------------------------------------------------------- #
#                              Fixtures                                 #
# --------------------------------------------------------------------- #


@pytest.fixture
def kv_isolation(tmp_path, monkeypatch):
    """Per-test isolated SQLite KV (mirrors the fixture in test_refresh_pipeline)."""
    from fantasy_baseball.data import kv_store

    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


@pytest.fixture
def configured_test_env(monkeypatch):
    """Environment ``load_config`` expects (mirrors test_refresh_pipeline)."""
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")


def _config(**overrides) -> LeagueConfig:
    base = {
        "league_id": 123,
        "num_teams": 12,
        "game_code": "mlb",
        "team_name": "Team 01",
        "draft_position": 1,
        "keepers": [],
        "roster_slots": {"C": 1, "OF": 3, "Util": 1, "P": 9, "BN": 3, "IL": 2},
        "projection_systems": ["atc"],
        "projection_weights": {"atc": 1.0},
        "teams": {},
        "strategy": "no_punt_opp",
        "scoring_mode": "var",
        "season_year": 2026,
        "season_start": "2026-03-27",
        "season_end": "2026-09-28",
    }
    base.update(overrides)
    return LeagueConfig(**base)


def _hitter(name: str) -> Player:
    return Player(name=name, player_type=PlayerType.HITTER, positions=[Position.OF])


def _pitcher(name: str) -> Player:
    return Player(name=name, player_type=PlayerType.PITCHER, positions=[Position.P])


def _frame(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"name": n, "_name_norm": n.lower()} for n in names])


class _StubStash:
    """Stand-in for ``StashResult``; the stash board is not under test here."""

    candidates: ClassVar[list] = []

    def to_dict(self) -> dict:
        return {"candidates": []}


def _audit_ready_run(**kwargs) -> RefreshRun:
    """A ``RefreshRun`` carrying exactly what ``_audit_roster`` reads.

    Mirrors ``_build_refresh_run_for_streak_test`` in ``test_refresh_pipeline``:
    build the object, set the handful of fields the step under test touches, and
    leave everything else at its constructor default.
    """
    run = RefreshRun(**kwargs)
    run.config = _config()
    run.hitters_proj = _frame(["Rostered Bat", "Free Bat"])
    run.pitchers_proj = _frame(["Rostered Arm", "Free Arm"])
    run.preseason_hitters = _frame(["Rostered Bat"])
    run.preseason_pitchers = _frame(["Rostered Arm"])
    run.roster_players = [_hitter("Rostered Bat")]
    run.opp_rosters = {"Team 02": [_pitcher("Rostered Arm")]}
    run.projected_standings = object()
    run.optimal_hitters = []
    run.optimal_pitchers_starters = []
    run.fraction_remaining = 0.25
    run.rankings_lookup = {"free bat": {"ros_rank": 1}}
    return run


def _stub_audit_dependencies(monkeypatch) -> dict:
    """Patch out everything downstream of the FA fetch and record the calls.

    The audit math is not under test in this file -- the seam is. Recording
    ``audit_roster``'s arguments is how we prove the injected pool is what
    actually reaches it.
    """
    calls: dict = {"audit_roster": [], "stash": [], "write_cache": [], "progress": []}

    def _audit_roster(roster_players, fa_players, roster_slots, **kw):
        calls["audit_roster"].append({"roster": roster_players, "fa": fa_players})
        return []

    def _score_stash(roster_players, fa_players, *a, **kw):
        calls["stash"].append({"fa": fa_players})
        return _StubStash()

    monkeypatch.setattr("fantasy_baseball.lineup.roster_audit.audit_roster", _audit_roster)
    monkeypatch.setattr("fantasy_baseball.lineup.stash_value.score_stash_candidates", _score_stash)
    monkeypatch.setattr(
        refresh_pipeline,
        "write_cache",
        lambda key, payload, **kw: calls["write_cache"].append(key),
    )
    monkeypatch.setattr(
        refresh_pipeline,
        "_set_refresh_progress",
        lambda msg: calls["progress"].append(msg),
    )
    return calls


def _boom(*args, **kwargs):
    raise AssertionError("manual mode must not call Yahoo")


def _read_meta(client) -> dict:
    return unwrap_cache_value(client.get(redis_key(CacheKey.META)))


def _as_str(keys) -> list[str]:
    return [k.decode() if isinstance(k, bytes) else k for k in keys]


# --------------------------------------------------------------------- #
#      The revert tests: with the defaults, nothing changed at all      #
# --------------------------------------------------------------------- #


class TestDefaultsAreInert:
    """With ``free_agent_source=None`` every branch takes today's path."""

    def test_default_construction_is_not_manual_mode(self):
        run = RefreshRun()
        assert run.free_agent_source is None
        assert run.manual_mode is False

    def test_default_construction_under_skip_yahoo_is_not_manual_mode(self):
        """Stale-data mode alone must never look like manual mode.

        ``manual_mode`` keys off the injected source, not off ``skip_yahoo``.
        If the two ever collapse into one flag, existing stale-data runs start
        trying to audit against a pool nobody supplied.
        """
        assert RefreshRun(skip_yahoo=True).manual_mode is False

    def test_default_job_label_is_todays_hardcoded_string(self):
        run = RefreshRun()
        assert run.job_label == TODAYS_JOB_LABEL
        assert run.logger.job_name == TODAYS_JOB_LABEL

    def test_default_job_log_key_is_unchanged(self, kv_isolation):
        """``job_log:refresh:<date>:<ts>`` -- the key the /jobs page reads."""
        from fantasy_baseball.data.kv_store import get_kv

        RefreshRun().logger.finish("ok")
        keys = _as_str(get_kv().keys("job_log:*"))
        assert keys, "the default run wrote no job log at all"
        assert all(k.startswith("job_log:refresh:") for k in keys), keys

    def test_default_run_stamps_the_refresh_job_label(self, kv_isolation, monkeypatch):
        """``set_cache_job`` still receives the literal ``"refresh"``.

        Every ``cache:*`` blob's ``_meta._job`` comes from this call, so drift
        here silently relabels the provenance of the whole production cache.
        """
        seen = []
        real_set = refresh_pipeline.set_cache_job

        def _recording_set_cache_job(job):
            seen.append(job)
            return real_set(job)

        monkeypatch.setattr(refresh_pipeline, "set_cache_job", _recording_set_cache_job)
        monkeypatch.setattr(RefreshRun, "_run_pipeline_steps", lambda self: None)

        RefreshRun().run()

        assert seen == [TODAYS_JOB_LABEL]

    def test_default_audit_still_hard_skips_under_skip_yahoo(self, monkeypatch):
        """THE revert test.

        Today a stale-data run empties ``fa_players`` and returns before it
        writes anything, because a board computed against an empty FA pool
        would read as "no upgrades available" -- a claim, not a gap. That must
        still be exactly what happens when no source is injected.
        """
        calls = _stub_audit_dependencies(monkeypatch)
        run = _audit_ready_run(skip_yahoo=True)

        with patch(
            "fantasy_baseball.lineup.waivers.fetch_and_match_free_agents",
            side_effect=_boom,
        ):
            run._audit_roster()

        assert run.fa_players == []
        assert calls["audit_roster"] == [], "the audit ran in stale-data mode"
        assert calls["stash"] == [], "the stash board ran in stale-data mode"
        assert calls["write_cache"] == [], "stale-data mode overwrote a Yahoo-only cache"
        assert TODAYS_SKIP_AUDIT_MESSAGE in calls["progress"]

    def test_default_live_audit_still_uses_the_yahoo_fetch(self, monkeypatch):
        """The live path is unchanged: no source, no skip -> Yahoo supplies the pool."""
        calls = _stub_audit_dependencies(monkeypatch)
        run = _audit_ready_run(skip_yahoo=False)
        run.league = object()
        yahoo_pool = [_hitter("Free Bat")]
        seen: dict = {}

        def _fetch(league, hitters_proj, pitchers_proj, **kw):
            seen["league"] = league
            seen["preseason_hitters"] = kw.get("preseason_hitters_proj")
            return (yahoo_pool, None)

        with patch(
            "fantasy_baseball.lineup.waivers.fetch_and_match_free_agents",
            side_effect=_fetch,
        ):
            run._audit_roster()

        assert run.fa_players == yahoo_pool
        assert seen["league"] is run.league
        assert seen["preseason_hitters"] is run.preseason_hitters
        assert calls["audit_roster"][0]["fa"] == yahoo_pool

    def test_default_stale_run_still_carries_last_refresh_forward(self, monkeypatch):
        """The staleness badge must keep firing during a Yahoo outage."""
        monkeypatch.setattr(refresh_pipeline, "local_now", lambda: datetime(2026, 4, 24, 9, 0))
        monkeypatch.setattr(
            refresh_pipeline,
            "read_cache_dict",
            lambda key: {"last_refresh": "2026-04-19 07:30"},
        )
        assert RefreshRun(skip_yahoo=True)._last_refresh_stamp() == "2026-04-19 07:30"

    def test_default_live_run_still_stamps_now(self, monkeypatch):
        monkeypatch.setattr(refresh_pipeline, "local_now", lambda: datetime(2026, 4, 24, 9, 0))
        monkeypatch.setattr(refresh_pipeline, "read_cache_dict", lambda key: {})
        assert RefreshRun(skip_yahoo=False)._last_refresh_stamp() == "2026-04-24 09:00"


# --------------------------------------------------------------------- #
#            The manual path: the audit runs, Yahoo does not            #
# --------------------------------------------------------------------- #


class TestInjectedFreeAgentSource:
    def test_construction_flags_manual_mode(self):
        run = RefreshRun(skip_yahoo=True, free_agent_source=lambda req: [])
        assert run.manual_mode is True
        assert run.free_agent_source is not None

    def test_custom_job_label_reaches_the_logger(self):
        run = RefreshRun(free_agent_source=lambda req: [], job_label="manual")
        assert run.job_label == "manual"
        assert run.logger.job_name == "manual"

    def test_audit_runs_against_the_injected_pool(self, monkeypatch):
        """The functional change: ``skip_yahoo`` no longer short-circuits the
        audit, and the synthesized pool is what ``audit_roster`` receives."""
        calls = _stub_audit_dependencies(monkeypatch)
        pool = [_hitter("Free Bat"), _pitcher("Free Arm")]
        requests: list[FreeAgentRequest] = []

        def _source(req):
            requests.append(req)
            return pool

        run = _audit_ready_run(skip_yahoo=True, free_agent_source=_source)

        with patch(
            "fantasy_baseball.lineup.waivers.fetch_and_match_free_agents",
            side_effect=_boom,
        ):
            run._audit_roster()

        assert len(requests) == 1, "the injected source was not called exactly once"
        assert isinstance(requests[0], FreeAgentRequest)
        assert run.fa_players is pool
        assert calls["audit_roster"][0]["fa"] is pool, "audit_roster got a different pool"
        assert calls["stash"][0]["fa"] is pool, "the stash board got a different pool"
        assert TODAYS_SKIP_AUDIT_MESSAGE not in calls["progress"]

    def test_audit_writes_the_caches_stale_mode_leaves_alone(self, monkeypatch):
        calls = _stub_audit_dependencies(monkeypatch)
        run = _audit_ready_run(skip_yahoo=True, free_agent_source=lambda req: [_hitter("Free Bat")])

        with patch(
            "fantasy_baseball.lineup.waivers.fetch_and_match_free_agents",
            side_effect=_boom,
        ):
            run._audit_roster()

        assert calls["write_cache"] == [
            CacheKey.POSITIONS,
            CacheKey.ROSTER_AUDIT,
            CacheKey.STASH,
        ]

    def test_manual_mode_advances_last_refresh(self, monkeypatch):
        """Hand-transcribed league state is genuinely fresh.

        Carrying the last live Yahoo stamp forward would make the dashboard's
        ">24h old" badge lie in the other direction -- claiming the data is a
        month stale when it was transcribed this morning.
        """
        monkeypatch.setattr(refresh_pipeline, "local_now", lambda: datetime(2026, 4, 24, 9, 0))
        monkeypatch.setattr(
            refresh_pipeline,
            "read_cache_dict",
            lambda key: {"last_refresh": "2026-04-19 07:30"},
        )
        run = RefreshRun(skip_yahoo=True, free_agent_source=lambda req: [])
        assert run._last_refresh_stamp() == "2026-04-24 09:00"


class TestFreeAgentRequest:
    """``_free_agent_request`` is the only new data-shaping code in the seam."""

    def test_rostered_names_are_split_by_player_type(self):
        """A pooled set of bare names deletes the wrong players.

        Ohtani is kept as a batter in this league, so his arm is genuinely
        available; the catcher Will Smith being rostered must not remove the
        pitcher Will Smith from the pool.
        """
        run = _audit_ready_run()
        run.roster_players = [_hitter("Shohei Ohtani"), _hitter("Will Smith")]
        run.opp_rosters = {"Team 02": [_pitcher("Tarik Skubal")]}

        req = run._free_agent_request()

        assert req.rostered_hitters == frozenset({"shohei ohtani", "will smith"})
        assert req.rostered_pitchers == frozenset({"tarik skubal"})

    def test_opponent_rosters_are_included(self):
        run = _audit_ready_run()
        run.roster_players = [_hitter("Mine")]
        run.opp_rosters = {
            "Team 02": [_hitter("Theirs")],
            "Team 03": [_pitcher("Also Theirs")],
        }

        req = run._free_agent_request()

        assert req.rostered_hitters == frozenset({"mine", "theirs"})
        assert req.rostered_pitchers == frozenset({"also theirs"})

    def test_frames_and_rankings_are_passed_through_unchanged(self):
        run = _audit_ready_run()
        req = run._free_agent_request()

        assert req.hitters_proj is run.hitters_proj
        assert req.pitchers_proj is run.pitchers_proj
        assert req.preseason_hitters_proj is run.preseason_hitters
        assert req.preseason_pitchers_proj is run.preseason_pitchers
        assert req.rankings_lookup is run.rankings_lookup


# --------------------------------------------------------------------- #
#        Prod safety: the local -> remote write stays unreachable       #
# --------------------------------------------------------------------- #


class TestManualModeNeverReachesProd:
    """``_push_streak_scores_to_remote`` is the one local->PROD Upstash write in
    this module. A manual run works off a hand-transcribed, isolated KV store,
    so that write must stay unreachable -- otherwise a transcription typo lands
    in production."""

    @pytest.mark.parametrize("free_agent_source", [None, lambda req: []])
    def test_streaks_still_skipped_under_skip_yahoo(self, monkeypatch, free_agent_source):
        progress: list[str] = []
        monkeypatch.setattr(refresh_pipeline, "_set_refresh_progress", progress.append)
        monkeypatch.setattr(refresh_pipeline, "_push_streak_scores_to_remote", _boom)
        monkeypatch.setattr("fantasy_baseball.streaks.data.schema.get_connection", _boom)

        run = RefreshRun(skip_yahoo=True, free_agent_source=free_agent_source)
        run.config = _config()
        run.league = None

        run._compute_streaks()

        assert "Skipping streak compute (stale-data mode)" in progress

    def test_streak_push_has_exactly_one_call_site(self):
        """Structural check: if the push ever gains a second call site, the
        ``skip_yahoo`` gate on ``_compute_streaks`` stops being sufficient."""
        from pathlib import Path

        source = Path(refresh_pipeline.__file__).read_text(encoding="utf-8")
        call_sites = [
            line.strip()
            for line in source.splitlines()
            if "_push_streak_scores_to_remote(" in line and not line.startswith("def ")
        ]
        assert call_sites == ["_push_streak_scores_to_remote(payload)"], call_sites

    @pytest.mark.parametrize("free_agent_source", [None, lambda req: []])
    def test_transactions_still_skipped_under_skip_yahoo(self, monkeypatch, free_agent_source):
        progress: list[str] = []
        monkeypatch.setattr(refresh_pipeline, "_set_refresh_progress", progress.append)
        monkeypatch.setattr("fantasy_baseball.lineup.yahoo_roster.fetch_all_transactions", _boom)

        run = RefreshRun(skip_yahoo=True, free_agent_source=free_agent_source)
        run.config = _config()

        run._analyze_transactions()

        assert "Skipping transaction analysis (stale-data mode)" in progress


# --------------------------------------------------------------------- #
#                      End-to-end through run()                         #
# --------------------------------------------------------------------- #


class TestFullManualRefresh:
    """The whole pipeline, every Yahoo entry point armed to raise, one
    injected free-agent source."""

    def _manual_pool(self) -> list[Player]:
        from fantasy_baseball.models.player import HitterStats
        from tests.test_web._refresh_fixture import free_agents

        return [
            Player(
                name=fa["name"],
                player_type=PlayerType.HITTER,
                positions=[Position.parse(p) for p in fa["positions"]],
                selected_position=Position.parse("BN"),
                rest_of_season=HitterStats(
                    pa=580, ab=500, h=145, r=80, hr=20, rbi=75, sb=6, avg=0.280
                ),
            )
            for fa in free_agents()
        ]

    def test_manual_run_audits_without_yahoo_and_stamps_its_own_label(
        self,
        configured_test_env,
        fake_redis,
        monkeypatch,
    ):
        pool = self._manual_pool()
        requests: list[FreeAgentRequest] = []

        def _source(req):
            requests.append(req)
            return pool

        with patched_refresh_environment(fake_redis):
            # A live run first, to persist the league state the manual run
            # reads back (the same setup the existing stale-data tests use).
            refresh_pipeline.run_full_refresh()
            audit_before = fake_redis.get(redis_key(CacheKey.ROSTER_AUDIT))
            live_refresh = _read_meta(fake_redis)["last_refresh"]

            monkeypatch.setattr(refresh_pipeline, "local_today", lambda: date(2026, 4, 20))
            monkeypatch.setattr(refresh_pipeline, "local_now", lambda: datetime(2026, 4, 24, 9, 0))

            armed = [patch(target, side_effect=_boom) for target in YAHOO_ENTRY_POINTS]
            for p in armed:
                p.start()
            try:
                RefreshRun(
                    skip_yahoo=True,
                    free_agent_source=_source,
                    job_label="manual",
                ).run()
            finally:
                for p in armed:
                    p.stop()

        assert len(requests) == 1, "the manual run did not call the injected source"
        assert requests[0].rostered_hitters, "no rostered names reached the source"

        # The audit ran: the caches stale-data mode leaves frozen were rewritten.
        audit_after = fake_redis.get(redis_key(CacheKey.ROSTER_AUDIT))
        assert audit_after is not None
        assert audit_after != audit_before, "the manual run left the audit cache frozen"
        assert unwrap_cache_value(fake_redis.get(redis_key(CacheKey.POSITIONS)))
        assert unwrap_cache_value(fake_redis.get(redis_key(CacheKey.STASH))) is not None

        # Provenance: every blob this run wrote carries the manual label.
        assert json.loads(audit_after)["_meta"]["_job"] == "manual"

        # Manual data is fresh, so the staleness stamp advances.
        assert _read_meta(fake_redis)["last_refresh"] == "2026-04-24 09:00"
        assert _read_meta(fake_redis)["last_refresh"] != live_refresh

    def test_manual_run_writes_its_job_log_under_its_own_name(
        self,
        configured_test_env,
        fake_redis,
        monkeypatch,
    ):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
            monkeypatch.setattr(refresh_pipeline, "local_today", lambda: date(2026, 4, 20))

            armed = [patch(target, side_effect=_boom) for target in YAHOO_ENTRY_POINTS]
            for p in armed:
                p.start()
            try:
                RefreshRun(
                    skip_yahoo=True,
                    free_agent_source=lambda req: self._manual_pool(),
                    job_label="manual",
                ).run()
            finally:
                for p in armed:
                    p.stop()

            keys = _as_str(fake_redis.keys("job_log:*"))

        assert [k for k in keys if k.startswith("job_log:manual:")], f"no manual job log: {keys}"
        assert [k for k in keys if k.startswith("job_log:refresh:")], "live job log went missing"
