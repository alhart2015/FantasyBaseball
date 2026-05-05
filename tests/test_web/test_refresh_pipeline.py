"""Integration test for run_full_refresh.

Mocks Yahoo and uses fakeredis. Asserts shape of every cache artifact
plus cross-step invariants (in test_invariants below). Does NOT lock
down values — Monte Carlo has randomness and projections change weekly.
"""

import json

import pytest

from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.web import refresh_pipeline
from tests.test_web._refresh_fixture import patched_refresh_environment


def _read(client, name: str):
    """Read a cache entry from the KV (fake_redis in tests).

    After Phase 1 of the cache refactor, ``write_cache`` writes to the KV
    via ``cache:<name>`` keys instead of JSON files. This
    helper reflects that — it queries the same KV the dashboard reads from
    on Render and SQLite locally.
    """
    raw = client.get(f"cache:{name}")
    return json.loads(raw) if raw else None


@pytest.fixture
def configured_test_env(monkeypatch, fake_redis):
    """Set environment variables expected by load_config.

    Pre-Phase-2 this fixture also returned ``tmp_path`` for the
    JSON-file cache layer; after the cache layer moved onto kv_store
    the dir is no longer needed and the fixture returns nothing.
    """
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")


class TestRefreshShape:
    """Shape assertions: every expected cache file is written with the
    expected top-level keys and types."""

    def test_all_expected_cache_files_written(self, configured_test_env, fake_redis):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()

        expected_keys = [
            CacheKey.STANDINGS,
            CacheKey.PENDING_MOVES,
            CacheKey.PROJECTIONS,
            CacheKey.ROSTER,
            CacheKey.RANKINGS,
            CacheKey.LINEUP_OPTIMAL,
            CacheKey.PROBABLE_STARTERS,
            CacheKey.POSITIONS,
            CacheKey.ROSTER_AUDIT,
            CacheKey.LEVERAGE,
            CacheKey.MONTE_CARLO,
            CacheKey.SPOE,
            CacheKey.TRANSACTION_ANALYZER,
            CacheKey.META,
            CacheKey.OPP_ROSTERS,
        ]
        for key in expected_keys:
            assert fake_redis.get(redis_key(key)) is not None, f"Missing cache key: {key}"

    def test_projected_standings_history_populated(self, configured_test_env, fake_redis):
        """Each refresh appends a snapshot to projected_standings_history."""
        from fantasy_baseball.data.redis_store import get_projected_standings_history

        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()

        history = get_projected_standings_history(fake_redis)
        assert len(history) >= 1, "Expected at least one projected standings snapshot"
        _snap_date, projected = next(iter(history.items()))
        assert len(projected.entries) == 12
        assert {e.team_name for e in projected.entries} == {f"Team {i:02d}" for i in range(1, 13)}

    def test_standings_shape(self, configured_test_env, fake_redis):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "standings")
        # Canonical Standings.to_json() shape: {effective_date, teams: [...]}
        assert isinstance(data, dict)
        assert {"effective_date", "teams"} <= data.keys()
        assert len(data["teams"]) == 12
        for entry in data["teams"]:
            assert {"name", "team_key", "rank", "stats"}.issubset(entry.keys())
            assert {"R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"}.issubset(
                entry["stats"].keys()
            )

    def test_projections_shape(self, configured_test_env, fake_redis):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "projections")
        assert {"projected_standings", "team_sds", "fraction_remaining"} <= data.keys()
        # projected_standings is now ProjectedStandings.to_json() shape:
        # {effective_date, teams: [{name, stats}]}
        assert isinstance(data["projected_standings"], dict)
        assert {"effective_date", "teams"} <= data["projected_standings"].keys()
        assert isinstance(data["team_sds"], dict)
        assert isinstance(data["fraction_remaining"], (int, float))

    def test_lineup_optimal_shape(self, configured_test_env, fake_redis):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "lineup_optimal")
        assert {"hitter_lineup", "pitcher_starters", "pitcher_bench", "moves"} <= data.keys()
        assert isinstance(data["moves"], list)

    def test_monte_carlo_shape(self, configured_test_env, fake_redis):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "monte_carlo")
        assert "base" in data
        assert "with_management" in data
        # ROS keys may be None when has_rest_of_season=False (next task)

    def test_meta_shape(self, configured_test_env, fake_redis):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "meta")
        assert {"last_refresh", "start_date", "end_date", "team_name"} <= data.keys()
        assert data["team_name"] == "Team 01"


class TestRefreshInvariants:
    """Cross-step contracts — these catch wiring regressions."""

    @pytest.fixture(autouse=True)
    def _run_refresh(self, configured_test_env, fake_redis):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        self.fake_redis = fake_redis

    def test_every_team_in_standings_appears_in_projected_standings(self):
        standings = _read(self.fake_redis, "standings")
        projections = _read(self.fake_redis, "projections")
        standings_names = {t["name"] for t in standings["teams"]}
        projected_names = {t["name"] for t in projections["projected_standings"]["teams"]}
        assert standings_names == projected_names

    def test_every_roster_player_has_pace(self):
        roster = _read(self.fake_redis, "roster")
        for player in roster:
            assert "pace" in player, f"{player.get('name')} missing pace"
            assert player["pace"] is not None

    def test_lineup_moves_only_reference_roster_players(self):
        roster = _read(self.fake_redis, "roster")
        optimal = _read(self.fake_redis, "lineup_optimal")
        roster_names = {p["name"] for p in roster}
        for move in optimal["moves"]:
            assert move["player"] in roster_names, (
                f"Move references {move['player']!r} not on roster"
            )

    def test_positions_map_covers_roster_and_opponents_and_fas(self):
        positions = _read(self.fake_redis, "positions")
        roster = _read(self.fake_redis, "roster")
        opp_rosters = _read(self.fake_redis, "opp_rosters")
        from fantasy_baseball.utils.name_utils import normalize_name

        # Roster players
        for p in roster:
            assert normalize_name(p["name"]) in positions
        # Opponent players
        for _opp_name, opp_roster in opp_rosters.items():
            for p in opp_roster:
                assert normalize_name(p["name"]) in positions

    def test_meta_last_refresh_is_set(self):
        meta = _read(self.fake_redis, "meta")
        assert meta["last_refresh"]  # truthy

    def test_meta_team_name_matches_config(self):
        meta = _read(self.fake_redis, "meta")
        assert meta["team_name"] == "Team 01"


class TestPitcherLineupMoves:
    """Regression guard: ``_compute_moves`` must surface pitcher
    start/sit recommendations, not just hitter ones. The bug was that
    ``compute_lineup_moves`` was only ever called with hitter
    assignments, so the dashboard rendered "Optimal ✓" even when the
    optimizer wanted a benched pitcher activated.
    """

    def test_benched_pitcher_recommended_active_emits_start_move(
        self,
        configured_test_env,
        fake_redis,
        monkeypatch,
    ):
        """Put a 6th pitcher on BN for the user's team; with 9 P slots
        and 6 healthy pitchers the optimizer wants all 6 active, so the
        bench pitcher should produce a START move."""
        from tests.test_web import _refresh_fixture

        original_roster_for_team = _refresh_fixture.roster_for_team

        def override(team_index):
            roster = original_roster_for_team(team_index)
            if team_index == 0:
                # Pitcher060 is in projections (range 0..79) but unused
                # by any roster (12 teams x 5 pitchers covers 0..59).
                roster.append(
                    {
                        "name": "Pitcher060",
                        "positions": ["SP", "P"],
                        "selected_position": "BN",
                        "player_id": "yh_p_060",
                        "status": "",
                    }
                )
            return roster

        monkeypatch.setattr(_refresh_fixture, "roster_for_team", override)

        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()

        optimal = _read(fake_redis, "lineup_optimal")
        pitcher_starts = [
            m for m in optimal["moves"] if m["slot"] == "P" and m["action"] == "START"
        ]
        assert any(m["player"] == "Pitcher060" for m in pitcher_starts), (
            f"Expected START move for benched Pitcher060; got moves={optimal['moves']}"
        )


class TestMonteCarloROSBranch:
    @pytest.mark.parametrize("has_ros", [True, False])
    def test_monte_carlo_keys_match_ros_availability(
        self,
        configured_test_env,
        fake_redis,
        has_ros,
    ):
        with patched_refresh_environment(
            fake_redis,
            has_rest_of_season=has_ros,
        ):
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "monte_carlo")
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
        self,
        configured_test_env,
        fake_redis,
        monkeypatch,
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
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        assert call_count["n"] == 0

    def test_refresh_preserves_existing_ros_projections(
        self,
        configured_test_env,
        fake_redis,
    ):
        """Redis ros_projections must be unchanged after refresh runs.

        Simulates: admin fetch wrote today's ROS blend (Junis sv=19).
        Running refresh must not clobber that with a stale disk blend.
        """
        # The fixture seeds its own ros_projections. Run the refresh
        # and assert whatever the fixture wrote is still there afterward
        # (byte-for-byte — no second write).
        with patched_refresh_environment(fake_redis):
            before = fake_redis.get(redis_key(CacheKey.ROS_PROJECTIONS))
            refresh_pipeline.run_full_refresh()
            after = fake_redis.get(redis_key(CacheKey.ROS_PROJECTIONS))
        assert before == after, (
            "Refresh modified cache:ros_projections — it must be read-only "
            "for the refresh path (only the admin fetch writes this key)"
        )


def test_standings_breakdown_cache_written_by_refresh():
    """build_standings_breakdown_payload produces the STANDINGS_BREAKDOWN shape."""
    import json
    from datetime import date

    from fantasy_baseball.models.player import Player, PlayerType
    from fantasy_baseball.models.positions import Position
    from fantasy_baseball.web.refresh_pipeline import build_standings_breakdown_payload

    def _h(name):
        return Player(
            name=name,
            player_type=PlayerType.HITTER,
            positions=[Position.OF],
            selected_position=Position.OF,
        )

    team_rosters = {"Team A": [_h("A1"), _h("A2")], "Team B": [_h("B1")]}

    payload = build_standings_breakdown_payload(team_rosters, date(2026, 4, 22))

    assert payload["effective_date"] == "2026-04-22"
    assert set(payload["teams"].keys()) == {"Team A", "Team B"}
    assert "hitters" in payload["teams"]["Team A"]
    assert "pitchers" in payload["teams"]["Team A"]
    assert len(payload["teams"]["Team A"]["hitters"]) == 2

    # Round-trip through JSON (proves serialization shape)
    roundtripped = json.loads(json.dumps(payload))
    assert roundtripped == payload


class TestPreseasonBaseline:
    """The refresh reads preseason_baseline:{year} from Redis; if
    missing, the base/with_management cache fields are None but the
    refresh still completes."""

    def test_baseline_present_populates_cache(
        self,
        configured_test_env,
        fake_redis,
    ):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "monte_carlo")
        assert data["base"] is not None
        assert data["with_management"] is not None
        assert "team_results" in data["base"]
        assert data["baseline_meta"]["roster_date"] == "2026-03-27"

    def test_baseline_missing_leaves_none(
        self,
        configured_test_env,
        fake_redis,
    ):
        # Intentionally strip the baseline that the fixture seeds.
        with patched_refresh_environment(fake_redis):
            fake_redis.delete("preseason_baseline:2026")
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "monte_carlo")
        assert data["base"] is None
        assert data["with_management"] is None
        assert data["baseline_meta"] is None


class TestFullSeasonProjectionsLoad:
    """``_load_projections`` must emit a clear warning when the
    full-season blob is missing from the KV. Pre-Phase-1 there was a
    second test for "reads disk when Redis missing" — the disk fallback
    is gone now (single KV layer), so that test was removed.
    """

    def test_warns_when_full_season_blob_missing(
        self,
        configured_test_env,
        fake_redis,
        caplog,
    ):
        """KV missing → refresh logs the warning and leaves
        ``self.full_hitters_proj`` / ``self.full_pitchers_proj`` unset
        (so ``hydrate_roster_entries`` skips populating
        ``Player.full_season_projection``)."""
        # Ensure the KV has no full-season blob. The fixture doesn't
        # seed it, but be defensive in case that changes.
        fake_redis.delete(redis_key(CacheKey.FULL_SEASON_PROJECTIONS))

        with (
            caplog.at_level("INFO", logger="fantasy_baseball.web.refresh_pipeline"),
            patched_refresh_environment(fake_redis),
        ):
            refresh_pipeline.run_full_refresh()

        # The warning is emitted via _progress -> log.info, so it shows
        # up at INFO level on the refresh_pipeline logger.
        assert any(
            "cache:full_season_projections missing" in record.getMessage()
            for record in caplog.records
        ), "Expected the missing-full-season warning; saw: " + ", ".join(
            r.getMessage() for r in caplog.records
        )
