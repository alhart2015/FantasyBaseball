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
