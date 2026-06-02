"""Integration test for run_full_refresh.

Mocks Yahoo and uses fakeredis. Asserts shape of every cache artifact
plus cross-step invariants (in test_invariants below). Does NOT lock
down values — Monte Carlo has randomness and projections change weekly.
"""

import json

import pytest

from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.web import refresh_pipeline
from tests._cache_helpers import unwrap_cache_value
from tests.test_web._refresh_fixture import patched_refresh_environment


def _read(client, name: str):
    """Read a cache entry from the KV (fake_redis in tests).

    After Phase 1 of the cache refactor, ``write_cache`` writes to the KV
    via ``cache:<name>`` keys instead of JSON files. This
    helper reflects that — it queries the same KV the dashboard reads from
    on Render and SQLite locally.
    """
    return unwrap_cache_value(client.get(f"cache:{name}"))


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
            CacheKey.STASH,
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
        assert isinstance(data["moves"], dict)
        assert {"swaps", "unpaired_starts", "unpaired_benches"} <= data["moves"].keys()
        assert isinstance(data["moves"]["swaps"], list)
        assert isinstance(data["moves"]["unpaired_starts"], list)
        assert isinstance(data["moves"]["unpaired_benches"], list)

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

    def test_build_projected_standings_passes_actual_standings(
        self, configured_test_env, fake_redis, monkeypatch
    ):
        """The refresh pipeline must thread ``self.standings`` (Yahoo team
        YTD) into ``ProjectedStandings.from_rosters`` so end-of-season
        projections use team_YTD + ROS arithmetic instead of per-player
        full-season totals. Without this, mid-season acquisitions get
        double-counted (their old team's YTD plus their new team's ROS
        both flow into the new team's projection).

        The preseason fallback call (inside ``if self.has_rest_of_season``)
        must NOT receive actual_standings -- preseason rosters paired with
        preseason projections should project ROS-only, which collapses to
        full-season at season start.
        """
        from fantasy_baseball.models.standings import ProjectedStandings, Standings

        captured: list[dict] = []
        orig = ProjectedStandings.from_rosters.__func__

        def spy(cls, team_rosters, effective_date, **kwargs):
            captured.append(
                {
                    "actual_standings": kwargs.get("actual_standings"),
                    "fraction_remaining": kwargs.get("fraction_remaining", 1.0),
                }
            )
            return orig(cls, team_rosters, effective_date, **kwargs)

        monkeypatch.setattr(ProjectedStandings, "from_rosters", classmethod(spy))

        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()

        # _build_projected_standings invokes from_rosters twice when
        # has_rest_of_season=True (fixture default): once for the main
        # projection (must receive Standings) and once for the preseason
        # baseline (must NOT receive Standings -- by design, see step
        # comment in refresh_pipeline._build_projected_standings).
        assert len(captured) >= 2, f"Expected at least 2 from_rosters calls; got {len(captured)}"
        main_call = captured[0]
        preseason_call = captured[1]

        assert isinstance(main_call["actual_standings"], Standings), (
            "Main from_rosters call must receive self.standings (Yahoo "
            "team YTD); got actual_standings="
            f"{main_call['actual_standings']!r}"
        )
        # AB extras attached via team_ytd_attribution.compute_team_ytd_ab so
        # ytd_components() can recombine AVG via Tier 1 of its AB sourcing.
        # The value may be 0 (test fixture may not exercise game logs) but
        # the key must be present on every entry.
        from fantasy_baseball.utils.constants import OpportunityStat

        assert all(OpportunityStat.AB in e.extras for e in main_call["actual_standings"].entries), (
            "expected AB in extras for every standings entry"
        )
        assert preseason_call["actual_standings"] is None, (
            "Preseason from_rosters call must NOT pass actual_standings "
            "-- preseason rosters + preseason projections produce ROS-only "
            "(which equals full-season pre-season); got "
            f"actual_standings={preseason_call['actual_standings']!r}"
        )

    def test_build_projected_standings_sources_game_logs_from_upstash(
        self, configured_test_env, fake_redis, monkeypatch
    ):
        """compute_team_ytd_ab must receive game_logs assembled from Upstash
        (build_hitter_ytd_game_logs), not fall back to the un-built
        data/roster_game_logs.json file. That file is absent on Render, so the
        fallback yields AB=0 and team-YTD AVG silently degrades to ROS-only.
        """
        from fantasy_baseball.analysis import team_ytd_attribution
        from fantasy_baseball.data import redis_store

        # The fixture seeds the hitter rollup (name-keyed: hitter000..) during
        # the run but no per-player logs. Seed a per-player log for one of those
        # rollup ids -- that key is not touched by the run, so the Upstash bridge
        # has real data to return.
        redis_store.set_player_game_log(
            fake_redis,
            2026,
            "hitter000",
            "hitting",
            {"name": "hitter000", "games": [{"date": "2026-04-15", "ab": 4}]},
        )

        captured: dict = {}
        orig = team_ytd_attribution.compute_team_ytd_ab

        def spy(league, *args, **kwargs):
            captured["game_logs"] = kwargs.get("game_logs")
            return orig(league, *args, **kwargs)

        monkeypatch.setattr(team_ytd_attribution, "compute_team_ytd_ab", spy)

        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()

        assert captured.get("game_logs") is not None, (
            "pipeline must pass game_logs= sourced from Upstash, not rely on "
            "the absent data/roster_game_logs.json file"
        )
        # Exactly the bridge's output, and non-empty (proves it read per-player
        # logs, not the file fallback).
        expected = redis_store.build_hitter_ytd_game_logs(fake_redis, 2026)
        assert captured["game_logs"] == expected
        assert captured["game_logs"].get("hitter000", {}).get("games") == [
            {"date": "2026-04-15", "ab": 4}
        ]


class TestFullSeasonVintage:
    """The refresh must re-derive full-season (ROS + YTD) from CURRENT game
    logs, not consume the ROS-fetch job's frozen cache:full_season_projections.
    The two jobs run on independent schedules, so the cached full-season blob
    carries the YTD vintage from whenever the ROS fetch last ran, while the
    refresh computes its team-YTD overlay from current game logs -- blending
    the two mixes vintages in the headline projected standings.
    """

    def test_load_projections_derives_full_season_from_current_ytd(self, monkeypatch, fake_redis):
        from fantasy_baseball.data import redis_store
        from fantasy_baseball.web import season_data

        monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)
        monkeypatch.setattr(season_data, "get_kv", lambda: fake_redis)

        # Preseason blended projections are required by _load_projections.
        redis_store.set_blended_projections(
            fake_redis, "hitters", [{"name": "Foo Bar", "mlbam_id": 5, "hr": 99}]
        )
        redis_store.set_blended_projections(
            fake_redis, "pitchers", [{"name": "Pitch Er", "mlbam_id": 6, "k": 99}]
        )

        # ROS-remaining blob: 10 HR / 10 K left.
        season_data.write_cache(
            CacheKey.ROS_PROJECTIONS,
            {
                "hitters": [{"name": "Foo Bar", "mlbam_id": 5, "hr": 10.0}],
                "pitchers": [{"name": "Pitch Er", "mlbam_id": 6, "k": 10.0}],
            },
        )
        # CURRENT YTD from game logs: 20 HR / 20 K so far.
        redis_store.set_game_log_totals(fake_redis, "hitters", {"5": {"name": "Foo Bar", "hr": 20}})
        redis_store.set_game_log_totals(
            fake_redis, "pitchers", {"6": {"name": "Pitch Er", "k": 20}}
        )

        # A STALE cached full-season the refresh must IGNORE (e.g. the ROS job
        # ran days ago against a smaller YTD): 999 is unreachable from 10 + 20.
        season_data.write_cache(
            CacheKey.FULL_SEASON_PROJECTIONS,
            {
                "hitters": [{"name": "Foo Bar", "mlbam_id": 5, "hr": 999.0}],
                "pitchers": [{"name": "Pitch Er", "mlbam_id": 6, "k": 999.0}],
            },
        )

        run = refresh_pipeline.RefreshRun()
        run._load_projections()

        h_row = run.full_hitters_proj[run.full_hitters_proj["name"] == "Foo Bar"].iloc[0]
        p_row = run.full_pitchers_proj[run.full_pitchers_proj["name"] == "Pitch Er"].iloc[0]
        # ROS (10) + current YTD (20) = 30, NOT the stale cache's 999.
        assert h_row["hr"] == 30.0
        assert p_row["k"] == 30.0


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
        moves = optimal["moves"]
        for swap in moves["swaps"]:
            assert swap["start"]["player"] in roster_names, (
                f"Swap start references {swap['start']['player']!r} not on roster"
            )
            assert swap["bench"]["player"] in roster_names, (
                f"Swap bench references {swap['bench']['player']!r} not on roster"
            )
        for move in moves["unpaired_starts"] + moves["unpaired_benches"]:
            assert move["player"] in roster_names, (
                f"Unpaired move references {move['player']!r} not on roster"
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
        moves = optimal["moves"]

        def name_of(move_side):
            return move_side["player"]

        # Pitcher060 should appear as a START (either inside a swap or unpaired).
        starting_names = {name_of(s["start"]) for s in moves["swaps"]}
        starting_names |= {name_of(m) for m in moves["unpaired_starts"]}
        assert "Pitcher060" in starting_names, (
            f"Expected Pitcher060 to be activated; got moves={moves}"
        )
        # And the START should be a P-target (no hitter slot for a pitcher).
        for swap in moves["swaps"]:
            if swap["start"]["player"] == "Pitcher060":
                assert swap["start"]["to"] == "P"
        for m in moves["unpaired_starts"]:
            if m["player"] == "Pitcher060":
                assert m["to"] == "P"


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


def test_breakdown_payload_includes_team_ytd_block_when_actual_standings_given():
    """When actual_standings is passed, each team's breakdown payload carries
    a team_ytd block with components from StandingsEntry.ytd_components()."""
    from datetime import date

    from fantasy_baseball.models.standings import (
        CategoryStats,
        Standings,
        StandingsEntry,
    )
    from fantasy_baseball.utils.constants import OpportunityStat
    from fantasy_baseball.web.refresh_pipeline import build_standings_breakdown_payload

    actual = Standings(
        effective_date=date(2026, 6, 2),
        entries=[
            StandingsEntry(
                team_name="Test",
                team_key="t",
                rank=1,
                stats=CategoryStats(
                    r=120,
                    hr=30,
                    rbi=110,
                    sb=15,
                    avg=0.275,
                    w=15,
                    k=300,
                    sv=8,
                    era=3.50,
                    whip=1.20,
                ),
                extras={
                    OpportunityStat.IP: 300.0,
                    OpportunityStat.AB: 800.0,
                },
            ),
        ],
    )

    payload = build_standings_breakdown_payload(
        team_rosters={"Test": []},  # empty roster -> ROS rows empty
        effective_date=date(2026, 6, 2),
        fraction_remaining=0.5,
        actual_standings=actual,
    )

    team_ytd = payload["teams"]["Test"]["team_ytd"]
    # Fix #6 (partial): team_ytd keys now mirror the per-player
    # contribution_stats schema (lowercase) so the modal can read
    # team_ytd via the same colSpec.field path. BB/H_allowed remain
    # combined since YTD WHIP*IP only gives the sum.
    assert team_ytd["r"] == 120
    assert team_ytd["hr"] == 30
    assert team_ytd["rbi"] == 110
    assert team_ytd["sb"] == 15
    assert team_ytd["ab"] == 800.0
    assert team_ytd["ip"] == 300.0
    assert team_ytd["w"] == 15
    assert team_ytd["k"] == 300
    assert team_ytd["sv"] == 8
    # H derived as AVG * AB.
    assert team_ytd["h"] == pytest.approx(0.275 * 800.0)
    # ER derived as ERA * IP / 9.
    assert team_ytd["er"] == pytest.approx(3.50 * 300.0 / 9.0)
    # BB + H_allowed derived as WHIP * IP.
    assert team_ytd["bb_plus_h_allowed"] == pytest.approx(1.20 * 300.0)


def test_breakdown_displacement_responds_to_opponent_ytd():
    """The breakdown's Pass-1 baseline must include YTD (mirroring
    ProjectedStandings.from_rosters), so the DeltaRoto displacement picker reacts
    to opponents' season totals. With a ROS-only Pass-1 baseline the picker is
    blind to YTD and chooses different displacement targets than the standings
    widget -- so the per-player breakdown stops summing to the headline total
    (the surfaces-disagree bug).

    Same rosters, two actual_standings differing ONLY in the opponent's YTD SV:
    the user team's chosen displacement target must change. (A ROS-only baseline
    ignores actual_standings entirely, so the two would be identical.)
    """
    from datetime import date

    from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
    from fantasy_baseball.models.positions import Position
    from fantasy_baseball.models.standings import CategoryStats, Standings, StandingsEntry
    from fantasy_baseball.utils.constants import OpportunityStat
    from fantasy_baseball.web.refresh_pipeline import build_standings_breakdown_payload

    def _p(name, slot, ros_ip, ros_k, ros_sv=0, pre_ip=None):
        ros = PitcherStats(
            ip=ros_ip,
            w=ros_ip * 0.05,
            k=ros_k,
            sv=ros_sv,
            er=ros_ip * 0.4,
            bb=ros_ip * 0.3,
            h_allowed=ros_ip * 0.9,
            era=3.6,
            whip=1.2,
        )
        full = PitcherStats(
            ip=ros_ip + 40,
            w=(ros_ip + 40) * 0.05,
            k=ros_k + 40,
            sv=ros_sv,
            er=(ros_ip + 40) * 0.4,
            bb=(ros_ip + 40) * 0.3,
            h_allowed=(ros_ip + 40) * 0.9,
            era=3.6,
            whip=1.2,
        )
        pre = PitcherStats(
            ip=pre_ip or ros_ip + 60, w=0, k=0, sv=0, er=0, bb=0, h_allowed=0, era=0, whip=0
        )
        return Player(
            name=name,
            player_type=PlayerType.PITCHER,
            rest_of_season=ros,
            full_season_projection=full,
            preseason=pre,
            selected_position=Position.parse(slot),
        )

    # A cheap-to-activate IL closer (high preseason -> small slot-share) whose SV
    # makes its activation pivotal when the opponent's season SV is high.
    me = [
        _p("Returner", "IL", ros_ip=20, ros_k=18, ros_sv=10, pre_ip=200),
        _p("SP1", "P", 150, 170),
        _p("SP2", "P", 145, 165),
        _p("RP1", "P", 30, 35, ros_sv=12, pre_ip=65),
    ]
    opp = [
        _p("OppSP1", "P", 160, 180),
        _p("OppSP2", "P", 150, 170),
        _p("OppRP", "P", 40, 45, ros_sv=5, pre_ip=70),
    ]
    rosters = {"Me": me, "Opp": opp}

    def _standings(opp_sv):
        def entry(n, sv):
            return StandingsEntry(
                team_name=n,
                team_key=n,
                rank=1,
                stats=CategoryStats(
                    r=400, hr=100, rbi=400, sb=50, avg=0.26, w=10, k=300, sv=sv, era=3.5, whip=1.2
                ),
                extras={OpportunityStat.IP: 300.0, OpportunityStat.AB: 800.0},
            )

        return Standings(
            effective_date=date(2026, 6, 2), entries=[entry("Me", 50), entry("Opp", opp_sv)]
        )

    def _factors(opp_sv):
        payload = build_standings_breakdown_payload(
            rosters, date(2026, 6, 2), fraction_remaining=0.6, actual_standings=_standings(opp_sv)
        )
        return {p["name"]: round(p["scale_factor"], 2) for p in payload["teams"]["Me"]["pitchers"]}

    weak_opp = _factors(opp_sv=5)
    strong_opp = _factors(opp_sv=75)
    # The displacement target changes with the opponent's season SV. A ROS-only
    # Pass-1 baseline would make these identical.
    assert weak_opp != strong_opp, (
        f"Displacement ignored opponent YTD (Pass-1 baseline missing YTD): {weak_opp}"
    )


def test_breakdown_payload_team_ytd_zero_when_no_actual_standings():
    """When actual_standings is None (pre-season or omitted), the team_ytd
    block is all zeros so consumers can still render the section without
    branching on its presence."""
    from datetime import date

    from fantasy_baseball.web.refresh_pipeline import build_standings_breakdown_payload

    payload = build_standings_breakdown_payload(
        team_rosters={"Test": []},
        effective_date=date(2026, 3, 27),
        actual_standings=None,
    )

    team_ytd = payload["teams"]["Test"]["team_ytd"]
    assert team_ytd["r"] == 0
    assert team_ytd["k"] == 0
    assert team_ytd["ab"] == 0
    assert team_ytd["h"] == 0
    assert team_ytd["ip"] == 0
    assert team_ytd["er"] == 0


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
    """``_load_projections`` DERIVES full-season (ROS + current YTD) from the
    ROS blob and the freshly-synced game logs, rather than reading the
    separate ROS-fetch job's frozen ``cache:full_season_projections``. So when
    ROS is present, deleting that cached blob has no effect on the standings
    and the old missing-blob warning does not fire. (Previously the refresh
    read the cached blob, which could blend a stale YTD vintage into the
    per-player full-season lines -- see TestFullSeasonVintage.)
    """

    def test_full_season_derived_so_missing_cached_blob_is_irrelevant(
        self,
        configured_test_env,
        fake_redis,
        caplog,
    ):
        # The fixture seeds cache:ros_projections, so has_rest_of_season is
        # True and full-season is derived. Delete the cached full-season blob
        # to prove the refresh does not depend on it.
        fake_redis.delete(redis_key(CacheKey.FULL_SEASON_PROJECTIONS))

        with (
            caplog.at_level("INFO", logger="fantasy_baseball.web.refresh_pipeline"),
            patched_refresh_environment(fake_redis),
        ):
            refresh_pipeline.run_full_refresh()

        messages = [r.getMessage() for r in caplog.records]
        # Full-season is derived from ROS + current YTD...
        assert any("Deriving full-season projections" in m for m in messages), (
            "Expected the derive step; saw: " + ", ".join(messages)
        )
        # ...so the obsolete missing-cached-blob warning must NOT fire.
        assert not any("cache:full_season_projections missing" in m for m in messages)


class TestProbableStartersWiring:
    """Task 10: ``_fetch_probable_starters`` must request a 14-day
    lookback from ``get_week_schedule`` and pre-filter the roster to
    SP-eligible + projected gs > 0 before invoking
    ``get_probable_starters``. The upcoming-starts module needs each
    pitcher's most recent start as the rotation anchor; closers/RPs
    should not propagate downstream as fake "starting" pitchers."""

    def test_fetch_probable_starters_passes_lookback_and_filters_sp(
        self,
        monkeypatch,
        configured_test_env,
        fake_redis,
    ):
        import pandas as pd

        from fantasy_baseball.data import mlb_schedule
        from fantasy_baseball.lineup import matchups as _matchups
        from fantasy_baseball.models.player import Player, PlayerType
        from fantasy_baseball.models.positions import Position
        from fantasy_baseball.web import refresh_pipeline as rp_mod

        captured: dict = {}

        def fake_get_week_schedule(start_date, end_date, cache_path, lookback_days=0):
            captured["lookback_days"] = lookback_days
            return {
                "probable_pitchers": [],
                "start_date": start_date,
                "end_date": end_date,
            }

        def fake_get_probable_starters(pitcher_roster, schedule, **kwargs):
            captured["roster_names"] = [p.name for p in pitcher_roster]
            return []

        # Patch get_week_schedule at the source module — _fetch_probable_starters
        # imports it lazily inside the method, so the binding is resolved at
        # call time.
        monkeypatch.setattr(mlb_schedule, "get_week_schedule", fake_get_week_schedule)
        monkeypatch.setattr(_matchups, "get_probable_starters", fake_get_probable_starters)
        monkeypatch.setattr(_matchups, "get_team_batting_stats", lambda _path: {})
        # Don't actually hit the KV — write_cache is fine to no-op.
        monkeypatch.setattr(rp_mod, "write_cache", lambda *a, **kw: None)

        run = rp_mod.RefreshRun()
        run.start_date = "2026-04-13"
        run.end_date = "2026-04-19"
        run.roster_players = [
            Player(
                name="Bryan Woo",
                player_type=PlayerType.PITCHER,
                positions=[Position.SP, Position.P],
                selected_position=Position.P,
                team="SEA",
            ),
            Player(
                name="Mason Miller",
                player_type=PlayerType.PITCHER,
                positions=[Position.RP, Position.P],
                selected_position=Position.P,
                team="OAK",
            ),
        ]
        run.pitchers_proj = pd.DataFrame(
            [
                {"name": "Bryan Woo", "gs": 22, "ip": 130.0},
                {"name": "Mason Miller", "gs": 0, "ip": 60.0},
            ]
        )

        run._fetch_probable_starters()

        assert captured["lookback_days"] == 14, (
            f"Expected lookback_days=14; got {captured.get('lookback_days')!r}"
        )
        assert "Bryan Woo" in captured["roster_names"], (
            f"SP with gs>0 should pass through; got {captured['roster_names']}"
        )
        assert "Mason Miller" not in captured["roster_names"], (
            f"RP with gs=0 should be filtered out; got {captured['roster_names']}"
        )


# --------------------------------------------------------------------- #
#         Task 8 — RefreshRun._compute_streaks (streak cache step)       #
# --------------------------------------------------------------------- #


@pytest.fixture
def kv_isolation(tmp_path, monkeypatch):
    """Per-test isolated SQLite KV.

    Mirrors the fixture in ``test_season_routes.py``. ``write_cache``
    and ``get_kv`` route through ``kv_store.get_kv()``; pointing the KV
    at an empty tmp_path file lets ``_compute_streaks`` write through
    the real cache layer without polluting other tests' state.
    """
    from fantasy_baseball.data import kv_store

    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def _build_refresh_run_for_streak_test():
    """Build a minimal ``RefreshRun`` for unit-testing ``_compute_streaks``.

    Mirrors how ``test_fetch_probable_starters_passes_lookback_and_filters_sp``
    builds a bare ``RefreshRun`` and sets just the attributes the step
    reads. ``_compute_streaks`` needs ``self.config`` (team_name,
    league_id, season_year), ``self.league`` (opaque sentinel — the
    streak pipeline is patched out so it's never dereferenced), and the
    inherited ``self.logger`` / ``self._progress``.
    """
    from fantasy_baseball.config import LeagueConfig
    from fantasy_baseball.web.refresh_pipeline import RefreshRun

    run = RefreshRun()
    run.config = LeagueConfig(
        league_id=123,
        num_teams=12,
        game_code="mlb",
        team_name="t",
        draft_position=1,
        keepers=[],
        roster_slots={},
        projection_systems=["atc"],
        projection_weights={"atc": 1.0},
        sgp_overrides={},
        teams={},
        strategy="no_punt_opp",
        scoring_mode="var",
        season_year=2026,
        season_start="2026-03-27",
        season_end="2026-09-28",
    )
    # Opaque sentinel — compute_streak_report is monkeypatched so the
    # value is never inspected.
    run.league = object()
    return run


def test_compute_streaks_writes_cache(monkeypatch, kv_isolation) -> None:
    """_compute_streaks wraps compute_streak_report + serializes + writes cache."""
    from datetime import date

    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
    from fantasy_baseball.streaks.reports.sunday import Report, ReportRow

    score = PlayerCategoryScore(
        player_id=1,
        category="hr",
        label="hot",
        probability=0.6,
        drivers=(Driver(feature="barrel_pct", z_score=1.0),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="X",
        positions=("OF",),
        player_id=1,
        composite=1,
        scores={"hr": score},
        max_probability=0.6,
    )
    fake_report = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="t",
        league_id=1,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(),
        skipped=(),
    )

    # ``_compute_streaks`` lazy-imports compute_streak_report and
    # get_connection inside the function (see refresh_pipeline.py module
    # docstring for why), so patches must target the source modules, not
    # ``refresh_pipeline``.
    monkeypatch.setattr(
        "fantasy_baseball.streaks.pipeline.compute_streak_report",
        lambda *a, **kw: fake_report,
    )
    monkeypatch.setattr(
        "fantasy_baseball.streaks.data.schema.get_connection",
        lambda *a, **kw: _FakeConn(),
    )

    run = _build_refresh_run_for_streak_test()
    run._compute_streaks()

    cached = kv_store.get_kv().get(redis_key(CacheKey.STREAK_SCORES))
    assert cached is not None
    payload = unwrap_cache_value(cached)
    assert payload["team_name"] == "t"
    assert len(payload["roster_rows"]) == 1


def test_compute_streaks_swallows_failures(monkeypatch, kv_isolation, caplog) -> None:
    """A failure in compute_streak_report logs but doesn't crash the pipeline."""
    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key

    def _boom(*a, **kw):
        raise RuntimeError("DuckDB unhappy")

    monkeypatch.setattr("fantasy_baseball.streaks.pipeline.compute_streak_report", _boom)
    monkeypatch.setattr(
        "fantasy_baseball.streaks.data.schema.get_connection",
        lambda *a, **kw: _FakeConn(),
    )

    run = _build_refresh_run_for_streak_test()
    with caplog.at_level("ERROR", logger="fantasy_baseball.web.refresh_pipeline"):
        run._compute_streaks()  # must not raise

    cached = kv_store.get_kv().get(redis_key(CacheKey.STREAK_SCORES))
    assert cached is None  # not overwritten on failure

    # ``log.exception`` attaches the original exception via ``exc_info``;
    # the formatted record has the traceback text, but ``record.message``
    # is just the log call's static string. Check both surfaces so we
    # confirm the underlying error was actually captured (not silently
    # swallowed with a generic message).
    def _record_carries_boom(r) -> bool:
        if "DuckDB unhappy" in r.getMessage():
            return True
        return bool(r.exc_info and "DuckDB unhappy" in str(r.exc_info[1]))

    assert any(_record_carries_boom(r) for r in caplog.records)


def test_compute_streaks_closes_connection_on_failure(monkeypatch, kv_isolation) -> None:
    """The DuckDB connection must be closed even when streak compute raises."""

    closed = {"n": 0}

    class _TrackingConn:
        def close(self) -> None:
            closed["n"] += 1

    monkeypatch.setattr(
        "fantasy_baseball.streaks.data.schema.get_connection",
        lambda *a, **kw: _TrackingConn(),
    )

    def _boom(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("fantasy_baseball.streaks.pipeline.compute_streak_report", _boom)

    run = _build_refresh_run_for_streak_test()
    run._compute_streaks()
    assert closed["n"] == 1, "connection must be closed on failure"


def test_compute_streaks_skipped_on_render(monkeypatch, fake_redis) -> None:
    """On Render, ``_compute_streaks`` must early-return without touching DuckDB.

    Render has no duckdb installed and no streaks.duckdb file -- the
    cache is populated from a developer machine via
    ``scripts/refresh_remote.py``. The Render-side daily refresh must
    NOT attempt streak compute (which would crash on the ``import
    duckdb`` inside ``streaks.pipeline``) and must NOT overwrite the
    cached STREAK_SCORES that the developer machine wrote.

    Hermetic: ``get_kv`` is faked to an in-memory store so the RENDER=true
    path never builds a real Upstash client. The previous version called
    ``kv_store.get_kv()`` directly under ``RENDER=true``, which built a
    live client from the .env creds and wrote the sentinel to *prod*
    Upstash -- a prod-polluting external dependency that made the test
    flaky in full-suite ordering. Unroutable creds are set as a second
    layer of defense.
    """
    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key

    monkeypatch.setenv("RENDER", "true")
    # Defense in depth: even if a real client were built, it must not be
    # able to reach prod Upstash (the repo .env holds real creds).
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "tok")
    # The cache the Render refresh reads/writes is faked in-memory; this
    # test is about the is_remote() gate, not the Upstash backend.
    monkeypatch.setattr(kv_store, "get_kv", lambda: fake_redis)

    # Seed an existing cache so we can prove the Render-side refresh
    # didn't clobber it.
    fake_redis.set(redis_key(CacheKey.STREAK_SCORES), '{"sentinel": "do-not-overwrite"}')

    # The gate must short-circuit before _compute_streaks opens DuckDB.
    # _compute_streaks swallows exceptions, so a raise alone wouldn't fail
    # the test -- track the call instead and assert it never happened.
    opened = {"n": 0}

    def _tracking_get_connection(*a, **kw):
        opened["n"] += 1
        raise AssertionError("get_connection must not be called on Render")

    monkeypatch.setattr(
        "fantasy_baseball.streaks.data.schema.get_connection", _tracking_get_connection
    )

    run = _build_refresh_run_for_streak_test()
    run._compute_streaks()  # must not raise

    assert opened["n"] == 0, "is_remote() gate must short-circuit before DuckDB access"
    cached = fake_redis.get(redis_key(CacheKey.STREAK_SCORES))
    assert cached == '{"sentinel": "do-not-overwrite"}', "Render refresh overwrote the cache"


def test_compute_streaks_mirrors_to_remote_upstash(monkeypatch, kv_isolation) -> None:
    """Local _compute_streaks pushes STREAK_SCORES to remote Upstash so
    Render reads fresh data without a manual sync step. The local SQLite
    write still happens; the remote push is in addition.
    """
    from datetime import date

    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
    from fantasy_baseball.streaks.reports.sunday import Report, ReportRow

    score = PlayerCategoryScore(
        player_id=1,
        category="hr",
        label="hot",
        probability=0.6,
        drivers=(Driver(feature="barrel_pct", z_score=1.0),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="X",
        positions=("OF",),
        player_id=1,
        composite=1,
        scores={"hr": score},
        max_probability=0.6,
    )
    fake_report = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="t",
        league_id=1,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(),
        skipped=(),
    )
    monkeypatch.setattr(
        "fantasy_baseball.streaks.pipeline.compute_streak_report",
        lambda *a, **kw: fake_report,
    )
    monkeypatch.setattr(
        "fantasy_baseball.streaks.data.schema.get_connection",
        lambda *a, **kw: _FakeConn(),
    )

    # Stand-in remote: capture set() calls without touching the network.
    class _FakeRemote:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def set(self, key: str, value: str) -> None:
            self.calls.append((key, value))

    fake_remote = _FakeRemote()

    # Creds must be present for the mirror path to engage.
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://example.invalid")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "tok")
    monkeypatch.setattr(
        "fantasy_baseball.data.kv_store.build_explicit_upstash_kv",
        lambda: fake_remote,
    )

    run = _build_refresh_run_for_streak_test()
    run._compute_streaks()

    assert len(fake_remote.calls) == 1, "remote.set must be invoked exactly once"
    key, value = fake_remote.calls[0]
    assert key == redis_key(CacheKey.STREAK_SCORES)
    stored = json.loads(value)
    # The mirrored remote value carries the same provenance envelope as the
    # local write_cache, so Render reads it through the envelope-aware
    # read_cache rather than seeing a bare (shape-mismatched) blob.
    assert "_meta" in stored
    payload = stored["_data"]
    assert payload["team_name"] == "t"
    assert len(payload["roster_rows"]) == 1


def test_compute_streaks_no_remote_mirror_without_creds(monkeypatch, kv_isolation) -> None:
    """Without Upstash env vars, the remote mirror is skipped quietly —
    don't crash a fresh local clone that has no .env yet.
    """
    from datetime import date

    from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
    from fantasy_baseball.streaks.reports.sunday import Report, ReportRow

    score = PlayerCategoryScore(
        player_id=1,
        category="hr",
        label="hot",
        probability=0.6,
        drivers=(Driver(feature="barrel_pct", z_score=1.0),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="X",
        positions=("OF",),
        player_id=1,
        composite=1,
        scores={"hr": score},
        max_probability=0.6,
    )
    fake_report = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="t",
        league_id=1,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(),
        skipped=(),
    )
    monkeypatch.setattr(
        "fantasy_baseball.streaks.pipeline.compute_streak_report",
        lambda *a, **kw: fake_report,
    )
    monkeypatch.setattr(
        "fantasy_baseball.streaks.data.schema.get_connection",
        lambda *a, **kw: _FakeConn(),
    )

    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)

    # If the mirror path ran, it would call this and fail the test.
    def _explode() -> None:
        raise AssertionError("build_explicit_upstash_kv must not be called without creds")

    monkeypatch.setattr(
        "fantasy_baseball.data.kv_store.build_explicit_upstash_kv",
        _explode,
    )

    run = _build_refresh_run_for_streak_test()
    run._compute_streaks()  # must not raise


def test_refresh_pipeline_imports_without_duckdb() -> None:
    """Regression: ``refresh_pipeline`` must load without duckdb installed.

    Render's daily QStash-driven refresh hits ``/api/refresh``, which
    lazy-imports ``refresh_pipeline``. If anything at the module-level
    of ``refresh_pipeline`` (or its imports) does ``import duckdb``,
    the route 500s and the daily refresh dies. This test runs the
    import in a subprocess with ``sys.modules['duckdb'] = None`` so
    any latent ``import duckdb`` fails the subprocess.
    """
    import subprocess
    import sys

    src = (
        "import sys\n"
        "sys.modules['duckdb'] = None\n"  # force ImportError on `import duckdb`
        "import fantasy_baseball.web.refresh_pipeline  # noqa: F401\n"
        "from fantasy_baseball.web.refresh_pipeline import run_full_refresh, RefreshRun\n"
        "assert callable(run_full_refresh)\n"
        "assert RefreshRun is not None\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"refresh_pipeline pulled in duckdb-tainted modules at load.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


class TestStashBoardDegradedMode:
    """Regression guard: a failure in score_stash_candidates must NOT abort
    the full refresh. The pipeline must degrade to an empty cached board and
    continue writing all downstream cache keys (e.g. CacheKey.META)."""

    def test_stash_failure_degrades_gracefully(
        self,
        configured_test_env,
        fake_redis,
        monkeypatch,
        caplog,
    ):
        """score_stash_candidates raises -> refresh completes, empty board cached."""
        import fantasy_baseball.lineup.stash_value as _stash_mod

        monkeypatch.setattr(
            _stash_mod,
            "score_stash_candidates",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        # Refresh must not raise even though stash computation fails.
        with (
            caplog.at_level("ERROR", logger="fantasy_baseball.web.refresh_pipeline"),
            patched_refresh_environment(fake_redis),
        ):
            refresh_pipeline.run_full_refresh()

        # A downstream cache key written AFTER _audit_roster must be present,
        # proving the refresh continued past the failed stash step.
        assert fake_redis.get(redis_key(CacheKey.META)) is not None, (
            "CacheKey.META missing -- refresh must continue past stash failure"
        )

        # The stash cache must contain the empty-board sentinel.
        raw = fake_redis.get(redis_key(CacheKey.STASH))
        assert raw is not None, "CacheKey.STASH must be written even on failure"

        stash_data = unwrap_cache_value(raw)
        assert stash_data["candidates"] == [], (
            f"Expected empty candidates list; got {stash_data['candidates']!r}"
        )
        assert stash_data["warning"] == "Stash board unavailable this refresh.", (
            f"Expected warning message; got {stash_data.get('warning')!r}"
        )

        # The failure must have been logged (not silently swallowed).
        def _record_carries_boom(r) -> bool:
            if "boom" in r.getMessage():
                return True
            return bool(r.exc_info and "boom" in str(r.exc_info[1]))

        assert any(_record_carries_boom(r) for r in caplog.records), (
            "Expected the stash failure to be logged via log.exception; "
            "got records: " + str([r.getMessage() for r in caplog.records])
        )


def test_build_standings_breakdown_payload_warns_on_team_name_mismatch(caplog):
    """Fix #7: when ``actual_standings`` carries entries whose team_name
    does not match any team_rosters key, ``build_standings_breakdown_payload``
    must log a warning naming the team rather than silently zeroing YTD.
    """
    from datetime import date

    from fantasy_baseball.models.standings import (
        CategoryStats,
        Standings,
        StandingsEntry,
    )
    from fantasy_baseball.web.refresh_pipeline import build_standings_breakdown_payload

    # actual carries "Other", team_rosters carries "Test" -> miss.
    actual = Standings(
        effective_date=date(2026, 6, 2),
        entries=[
            StandingsEntry(
                team_name="Other",
                team_key="o",
                rank=1,
                stats=CategoryStats(),
            ),
        ],
    )

    with caplog.at_level(
        "WARNING",
        logger="fantasy_baseball.web.refresh_pipeline",
    ):
        build_standings_breakdown_payload(
            team_rosters={"Test": []},
            effective_date=date(2026, 6, 2),
            actual_standings=actual,
        )

    msgs = [r.getMessage() for r in caplog.records]
    assert any("Test" in m for m in msgs), f"Expected a warning naming 'Test'; got: {msgs}"


class TestAuditAndOptimizerUseYtdStandings:
    """Fix #3: ``_audit_roster`` and ``_optimize_lineup`` must pass the
    augmented ``self.ytd_standings`` (with team-YTD AB stuffed onto
    extras) to their consumers, not the un-augmented ``self.standings``.

    Without the fix, the stash board and lineup optimizer user-rows
    silently see no AB attribution -- yielding ROS-only AVG decisions
    while the projected standings widget (and everyone else) sees the
    YTD-augmented baseline. The two views disagree.
    """

    def test_audit_roster_passes_augmented_ytd_standings_to_score_stash(
        self, configured_test_env, fake_redis, monkeypatch
    ):
        """Capture the actual_standings argument passed to
        score_stash_candidates. It must carry the augmented AB on extras
        (i.e., be ``self.ytd_standings``, not ``self.standings``).
        """
        from fantasy_baseball.utils.constants import OpportunityStat

        captured: dict = {}

        def _spy_score_stash(*args, **kwargs):
            captured["actual_standings"] = kwargs.get("actual_standings")
            from fantasy_baseball.lineup.stash_value import StashResult

            return StashResult(open_il_slots=0, cutline_rank=0, candidates=[])

        monkeypatch.setattr(
            "fantasy_baseball.web.refresh_pipeline.score_stash_candidates",
            _spy_score_stash,
            raising=False,
        )

        # The score_stash_candidates symbol is imported inside _audit_roster,
        # so we patch the source module instead.
        import fantasy_baseball.lineup.stash_value as _stash_mod

        monkeypatch.setattr(_stash_mod, "score_stash_candidates", _spy_score_stash)

        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()

        actual = captured.get("actual_standings")
        assert actual is not None, "score_stash_candidates not called"
        # Augmented ytd_standings carries OpportunityStat.AB on at least
        # one entry. Bare self.standings would not.
        ab_present = any(OpportunityStat.AB in e.extras for e in actual.entries)
        assert ab_present, (
            "score_stash_candidates received un-augmented standings "
            "(no team-YTD AB in extras); expected self.ytd_standings"
        )

    def test_optimize_lineup_passes_augmented_ytd_standings(
        self, configured_test_env, fake_redis, monkeypatch
    ):
        """``optimize_hitter_lineup`` / ``optimize_pitcher_lineup`` must
        receive the augmented ``self.ytd_standings`` (AB on extras),
        not ``self.standings``.
        """
        from fantasy_baseball.utils.constants import OpportunityStat

        captured: dict = {}

        def _spy_hitter(*args, **kwargs):
            captured.setdefault("hitter", kwargs.get("actual_standings"))
            return []

        def _spy_pitcher(*args, **kwargs):
            captured.setdefault("pitcher", kwargs.get("actual_standings"))
            return [], []

        import fantasy_baseball.web.refresh_pipeline as _rp

        monkeypatch.setattr(_rp, "optimize_hitter_lineup", _spy_hitter, raising=False)
        monkeypatch.setattr(_rp, "optimize_pitcher_lineup", _spy_pitcher, raising=False)
        # The symbols are imported inside _optimize_lineup; patch the source
        # module so the late import picks up the spy.
        import fantasy_baseball.lineup.optimizer as _opt_mod

        monkeypatch.setattr(_opt_mod, "optimize_hitter_lineup", _spy_hitter)
        monkeypatch.setattr(_opt_mod, "optimize_pitcher_lineup", _spy_pitcher)

        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()

        for which in ("hitter", "pitcher"):
            actual = captured.get(which)
            assert actual is not None, f"optimize_{which}_lineup not called"
            ab_present = any(OpportunityStat.AB in e.extras for e in actual.entries)
            assert ab_present, (
                f"optimize_{which}_lineup received un-augmented standings "
                f"(no team-YTD AB in extras); expected self.ytd_standings"
            )


class _FakeConn:
    """Tiny stand-in for a DuckDB connection — only ``close()`` is exercised."""

    def close(self) -> None:
        return None
