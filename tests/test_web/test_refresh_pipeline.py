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
    import json
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
    payload = json.loads(cached)
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


def test_compute_streaks_skipped_on_render(monkeypatch, kv_isolation) -> None:
    """On Render, ``_compute_streaks`` must early-return without touching DuckDB.

    Render has no duckdb installed and no streaks.duckdb file — the
    cache is populated from a developer machine via
    ``scripts/refresh_remote.py``. The Render-side daily refresh must
    NOT attempt streak compute (which would crash on the ``import
    duckdb`` inside ``streaks.pipeline``) and must NOT overwrite the
    cached STREAK_SCORES that the developer machine wrote.
    """
    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key

    monkeypatch.setenv("RENDER", "true")
    kv_store._reset_singleton()

    # Seed an existing cache so we can prove the Render-side refresh
    # didn't clobber it.
    kv_store.get_kv().set(redis_key(CacheKey.STREAK_SCORES), '{"sentinel": "do-not-overwrite"}')

    # If _compute_streaks tried to run, this patch would make it crash —
    # but the is_remote() gate must short-circuit before reaching it.
    def _explode(*a, **kw):
        raise AssertionError("compute_streak_report must not be called on Render")

    monkeypatch.setattr("fantasy_baseball.streaks.pipeline.compute_streak_report", _explode)

    run = _build_refresh_run_for_streak_test()
    run._compute_streaks()  # must not raise

    cached = kv_store.get_kv().get(redis_key(CacheKey.STREAK_SCORES))
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
    payload = json.loads(value)
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


class _FakeConn:
    """Tiny stand-in for a DuckDB connection — only ``close()`` is exercised."""

    def close(self) -> None:
        return None
