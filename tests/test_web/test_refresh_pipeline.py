"""Integration test for run_full_refresh.

Mocks Yahoo and uses fakeredis. Asserts shape of every cache artifact
plus cross-step invariants (in test_invariants below). Does NOT lock
down values — Monte Carlo has randomness and projections change weekly.
"""
import json
from pathlib import Path

import pytest

from fantasy_baseball.web import refresh_pipeline
from tests.test_web._refresh_fixture import patched_refresh_environment


def _read(cache_dir: Path, name: str):
    """Read a cache JSON file."""
    return json.loads((cache_dir / f"{name}.json").read_text())


@pytest.fixture
def configured_test_env(monkeypatch, fake_redis, tmp_path):
    """Set environment variables expected by load_config."""
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")
    return tmp_path


class TestRefreshShape:
    """Shape assertions: every expected cache file is written with the
    expected top-level keys and types."""

    def test_all_expected_cache_files_written(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)

        expected_files = [
            "standings", "pending_moves", "projections", "roster",
            "rankings", "lineup_optimal", "probable_starters", "positions",
            "roster_audit", "leverage", "monte_carlo", "spoe",
            "transaction_analyzer", "meta", "opp_rosters",
        ]
        for name in expected_files:
            path = cache_dir / f"{name}.json"
            assert path.exists(), f"Missing cache file: {name}.json"

    def test_standings_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "standings")
        assert isinstance(data, list)
        assert len(data) == 12
        for entry in data:
            assert {"name", "team_key", "rank", "stats"}.issubset(entry.keys())
            assert {"R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"}.issubset(
                entry["stats"].keys()
            )

    def test_projections_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "projections")
        assert {"projected_standings", "team_sds", "fraction_remaining"} <= data.keys()
        assert isinstance(data["projected_standings"], list)
        assert isinstance(data["team_sds"], dict)
        assert isinstance(data["fraction_remaining"], (int, float))

    def test_lineup_optimal_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "lineup_optimal")
        assert {"hitter_lineup", "pitcher_starters", "pitcher_bench", "moves"} <= data.keys()
        assert isinstance(data["moves"], list)

    def test_monte_carlo_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "monte_carlo")
        assert "base" in data
        assert "with_management" in data
        # ROS keys may be None when has_rest_of_season=False (next task)

    def test_meta_shape(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "meta")
        assert {"last_refresh", "start_date", "end_date", "team_name"} <= data.keys()
        assert data["team_name"] == "Team 01"


class TestRefreshInvariants:
    """Cross-step contracts — these catch wiring regressions."""

    @pytest.fixture(autouse=True)
    def _run_refresh(self, configured_test_env, fake_redis):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        self.cache_dir = cache_dir

    def test_every_team_in_standings_appears_in_projected_standings(self):
        standings = _read(self.cache_dir, "standings")
        projections = _read(self.cache_dir, "projections")
        standings_names = {t["name"] for t in standings}
        projected_names = {t["name"] for t in projections["projected_standings"]}
        assert standings_names == projected_names

    def test_every_roster_player_has_pace(self):
        roster = _read(self.cache_dir, "roster")
        for player in roster:
            assert "pace" in player, f"{player.get('name')} missing pace"
            assert player["pace"] is not None

    def test_lineup_moves_only_reference_roster_players(self):
        roster = _read(self.cache_dir, "roster")
        optimal = _read(self.cache_dir, "lineup_optimal")
        roster_names = {p["name"] for p in roster}
        for move in optimal["moves"]:
            assert move["player"] in roster_names, (
                f"Move references {move['player']!r} not on roster"
            )

    def test_positions_map_covers_roster_and_opponents_and_fas(self):
        positions = _read(self.cache_dir, "positions")
        roster = _read(self.cache_dir, "roster")
        opp_rosters = _read(self.cache_dir, "opp_rosters")
        from fantasy_baseball.utils.name_utils import normalize_name
        # Roster players
        for p in roster:
            assert normalize_name(p["name"]) in positions
        # Opponent players
        for opp_name, opp_roster in opp_rosters.items():
            for p in opp_roster:
                assert normalize_name(p["name"]) in positions

    def test_meta_last_refresh_is_set(self):
        meta = _read(self.cache_dir, "meta")
        assert meta["last_refresh"]  # truthy

    def test_meta_team_name_matches_config(self):
        meta = _read(self.cache_dir, "meta")
        assert meta["team_name"] == "Team 01"


class TestMonteCarloROSBranch:
    @pytest.mark.parametrize("has_ros", [True, False])
    def test_monte_carlo_keys_match_ros_availability(
        self, configured_test_env, fake_redis, has_ros,
    ):
        cache_dir = configured_test_env
        with patched_refresh_environment(
            fake_redis, has_rest_of_season=has_ros, cache_dir=cache_dir,
        ):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "monte_carlo")
        assert data["base"] is not None
        assert data["with_management"] is not None
        if has_ros:
            assert data["rest_of_season"] is not None
            assert data["rest_of_season_with_management"] is not None
        else:
            assert data["rest_of_season"] is None
            assert data["rest_of_season_with_management"] is None


class TestROSProjectionsRedisAuthoritative:
    """The refresh must NEVER overwrite cache:ros_projections.

    The daily admin-triggered ROS fetch is the sole authoritative
    writer. Blending from disk CSVs in the refresh path regressed
    Jakob Junis from ~19 saves (fresh FanGraphs data in Redis) back
    to preseason-era ~4 saves (committed git snapshot from March 30),
    which broke the player-comparison tool downstream.
    """

    def test_refresh_does_not_call_blend_and_cache_ros(
        self, configured_test_env, fake_redis, monkeypatch,
    ):
        """Regression guard: refresh must not invoke blend_and_cache_ros.

        Blending from disk CSVs on Render overwrites fresh Redis data
        with stale git-committed snapshots whenever admin fetch and
        refresh run on different instances.
        """
        from fantasy_baseball.data import ros_pipeline

        call_count = {"n": 0}

        def _blow_up(*args, **kwargs):
            call_count["n"] += 1
            raise AssertionError(
                "refresh must not call blend_and_cache_ros; see "
                "comment in refresh_pipeline._load_projections"
            )

        monkeypatch.setattr(ros_pipeline, "blend_and_cache_ros", _blow_up)
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        assert call_count["n"] == 0

    def test_refresh_preserves_existing_ros_projections(
        self, configured_test_env, fake_redis,
    ):
        """Redis ros_projections must be unchanged after refresh runs.

        Simulates: admin fetch wrote today's ROS blend (Junis sv=19).
        Running refresh must not clobber that with a stale disk blend.
        """
        from fantasy_baseball.data.cache_keys import CacheKey, redis_key
        cache_dir = configured_test_env

        # The fixture seeds its own ros_projections. Run the refresh
        # and assert whatever the fixture wrote is still there afterward
        # (byte-for-byte — no second write).
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            before = fake_redis.get(redis_key(CacheKey.ROS_PROJECTIONS))
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
            after = fake_redis.get(redis_key(CacheKey.ROS_PROJECTIONS))
        assert before == after, (
            "Refresh modified cache:ros_projections — it must be read-only "
            "for the refresh path (only the admin fetch writes this key)"
        )


class TestPreseasonBaseline:
    """The refresh reads preseason_baseline:{year} from Redis; if
    missing, the base/with_management cache fields are None but the
    refresh still completes."""

    def test_baseline_present_populates_cache(
        self, configured_test_env, fake_redis,
    ):
        cache_dir = configured_test_env
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "monte_carlo")
        assert data["base"] is not None
        assert data["with_management"] is not None
        assert "team_results" in data["base"]
        assert data["baseline_meta"]["roster_date"] == "2026-03-27"

    def test_baseline_missing_leaves_none(
        self, configured_test_env, fake_redis,
    ):
        cache_dir = configured_test_env
        # Intentionally strip the baseline that the fixture seeds.
        with patched_refresh_environment(fake_redis, cache_dir=cache_dir):
            fake_redis.delete("preseason_baseline:2026")
            refresh_pipeline.run_full_refresh(cache_dir=cache_dir)
        data = _read(cache_dir, "monte_carlo")
        assert data["base"] is None
        assert data["with_management"] is None
        assert data["baseline_meta"] is None
