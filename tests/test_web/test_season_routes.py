import json
import re
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import redis_key
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.web.season_app import create_app
from fantasy_baseball.web.season_data import CacheKey


@pytest.fixture
def client():
    """Pre-authenticated test client. The whole site is behind login,
    so most tests need a session that has already passed the gate."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
        yield client


@pytest.fixture
def unauth_client():
    """Anonymous test client for verifying the login gate itself."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def kv_isolation(tmp_path, monkeypatch):
    """Per-test isolated SQLite KV.

    After Phase 2 of the cache refactor, ``read_cache``/``write_cache``
    route through ``kv_store.get_kv()`` instead of JSON files in a
    ``cache_dir``. Tests that exercise the dashboard's read-then-render
    flow seed the KV here and let the route handlers read the same KV.
    """
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def test_index_redirects_to_standings(client):
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/standings" in resp.headers["Location"]


def test_standings_page_renders(client):
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/standings")
    assert resp.status_code == 200
    assert b"Standings" in resp.data


def test_lineup_page_renders(client):
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/lineup")
    assert resp.status_code == 200
    assert b"Lineup" in resp.data


def test_trades_page_renders(client):
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/waivers-trades")
    assert resp.status_code == 200
    assert b"Trades" in resp.data


def test_players_page_renders(client):
    resp = client.get("/players")
    assert resp.status_code == 200
    assert b"pos-filter" in resp.data


def test_sidebar_nav_links_present(client):
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/standings")
    html = resp.data.decode()
    assert 'href="/standings"' in html
    assert 'href="/lineup"' in html
    assert 'href="/waivers-trades"' in html


def test_active_page_highlighted(client):
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/standings")
    html = resp.data.decode()
    assert "active" in html


def _mock_standings():
    """Canonical Standings.to_json() shape (post-refactor cache payload)."""
    teams = [
        (
            "Hart of the Order",
            {
                "R": 300,
                "HR": 90,
                "RBI": 290,
                "SB": 50,
                "AVG": 0.270,
                "W": 35,
                "K": 600,
                "SV": 25,
                "ERA": 3.50,
                "WHIP": 1.18,
            },
        ),
        (
            "SkeleThor",
            {
                "R": 310,
                "HR": 85,
                "RBI": 295,
                "SB": 40,
                "AVG": 0.265,
                "W": 38,
                "K": 580,
                "SV": 30,
                "ERA": 3.40,
                "WHIP": 1.15,
            },
        ),
    ]
    return {
        "effective_date": "2026-04-01",
        "teams": [
            {"name": n, "team_key": f"key_{i}", "rank": i + 1, "stats": s}
            for i, (n, s) in enumerate(teams)
        ],
    }


def test_standings_renders_table_with_data(client):
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict") as mock_cache,
        patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg,
    ):
        mock_cache.side_effect = lambda k: _mock_standings() if k == CacheKey.STANDINGS else {}
        mock_cfg.return_value.team_name = "Hart of the Order"
        resp = client.get("/standings")
        assert resp.status_code == 200
        assert b"Hart of the Order" in resp.data
        assert b"user-team" in resp.data


def test_refresh_status_not_running(client):
    resp = client.get("/api/refresh-status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["running"] is False


def test_unauthed_api_returns_json_401_not_redirect(unauth_client):
    """Unauthenticated /api/* GETs must return JSON 401, not redirect to /login.

    The mobile lineup page relies on `r.json()` to read errors; a 302 to the HTML
    login page makes JSON parsing fail with a confusing browser-specific error.
    """
    resp = unauth_client.get("/api/opponent/mlb.l.1.t.1/lineup")
    assert resp.status_code == 401
    assert resp.content_type.startswith("application/json")
    assert resp.get_json() == {"error": "Authentication required"}


def test_unauthed_html_page_still_redirects_to_login(unauth_client):
    """Non-API routes should still redirect to /login when unauthenticated."""
    resp = unauth_client.get("/logs")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_unauthed_standings_redirects_to_login(unauth_client):
    """Once the site is fully gated, even the default landing page
    must demand auth instead of leaking standings data."""
    resp = unauth_client.get("/standings")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_unauthed_teams_api_returns_json_401(unauth_client):
    """Previously-public JSON APIs must now 401, not leak data."""
    resp = unauth_client.get("/api/teams")
    assert resp.status_code == 401
    assert resp.get_json() == {"error": "Authentication required"}


def test_bearer_token_still_works_for_protected_route(unauth_client, monkeypatch):
    """QStash cron hits /api/refresh-status with a Bearer token; the
    global gate must accept that path without a session cookie."""
    monkeypatch.setenv("ADMIN_PASSWORD", "test-pw")
    resp = unauth_client.get("/api/refresh-status", headers={"Authorization": "Bearer test-pw"})
    assert resp.status_code == 200


def test_login_page_is_accessible_without_auth(unauth_client):
    resp = unauth_client.get("/login")
    assert resp.status_code == 200
    assert b"Login" in resp.data


def test_login_then_access_protected_route(unauth_client, monkeypatch):
    """Round-trip: POST /login, follow the redirect, hit /standings."""
    monkeypatch.setenv("ADMIN_PASSWORD", "test-pw")
    resp = unauth_client.post("/login", data={"password": "test-pw"})
    assert resp.status_code == 302
    # session cookie is now set on the client; protected GET should pass
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = unauth_client.get("/standings")
    assert resp.status_code == 200


def test_logs_page_renders(client, kv_isolation):
    # Isolated empty KV -> get_all_logs() returns [] -> page renders empty.
    resp = client.get("/logs")
    assert resp.status_code == 200
    assert b"Job Logs" in resp.data


# --- Refresh / ROS-fetch mutual exclusion -----------------------------------
# The full refresh and the ROS-projection fetch both sync MLB game logs
# (a read-modify-write of the shared rollup) and write the same cache keys.
# They must be mutually exclusive in-process. Both routes gate on the single
# refresh slot (try_acquire_refresh_slot); the ROS worker releases it when done.


@pytest.fixture
def free_refresh_slot():
    """Guarantee the shared refresh slot is free before and after the test,
    so a held slot never leaks into the rest of the suite."""
    from fantasy_baseball.web import refresh_pipeline

    refresh_pipeline.release_refresh_slot()
    yield
    refresh_pipeline.release_refresh_slot()


def test_fetch_ros_route_rejected_when_slot_held(client, monkeypatch, free_refresh_slot):
    """A ROS fetch must not start while another heavy job holds the slot."""
    from fantasy_baseball.web import refresh_pipeline, season_routes

    thread_ctor = MagicMock()
    monkeypatch.setattr(season_routes.threading, "Thread", thread_ctor)
    assert refresh_pipeline.try_acquire_refresh_slot() is True

    resp = client.post("/api/fetch-ros-projections")

    assert resp.get_json()["status"] == "already_running"
    thread_ctor.assert_not_called()  # no worker spawned


def test_fetch_ros_route_acquires_slot_when_free(
    client, monkeypatch, fake_redis, free_refresh_slot
):
    """When free, the ROS fetch starts AND claims the slot, so a concurrent
    refresh (or second fetch) is locked out."""
    from fantasy_baseball.web import refresh_pipeline, season_routes

    # The route reads the durable lock (refresh_lock_held) on the slot-free
    # path; isolate get_kv so it reads an empty test KV, not the local DB.
    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)
    fake_thread = MagicMock()
    monkeypatch.setattr(season_routes.threading, "Thread", MagicMock(return_value=fake_thread))

    resp = client.post("/api/fetch-ros-projections")

    assert resp.get_json()["status"] == "started"
    fake_thread.start.assert_called_once()
    assert refresh_pipeline.get_refresh_status()["running"] is True


def test_refresh_route_rejected_while_ros_fetch_holds_slot(client, monkeypatch, free_refresh_slot):
    """The slot is shared: a refresh cannot start while a ROS fetch holds it."""
    from fantasy_baseball.web import refresh_pipeline, season_routes

    monkeypatch.setattr(season_routes.threading, "Thread", MagicMock())
    # Simulate a ROS fetch in progress (it acquired the shared slot).
    assert refresh_pipeline.try_acquire_refresh_slot() is True

    resp = client.post("/api/refresh")

    assert resp.get_json()["status"] == "already_running"


def test_ros_fetch_worker_releases_slot(client, monkeypatch, tmp_path, free_refresh_slot):
    """The ROS worker must release the slot when it finishes, even on error,
    so a failed fetch doesn't wedge the slot and block all future jobs."""
    from fantasy_baseball.web import refresh_pipeline, season_routes

    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "kv.db"))
    kv_store._reset_singleton()

    def _boom(*_a, **_k):
        raise RuntimeError("game-log sync failed")

    monkeypatch.setattr("fantasy_baseball.data.mlb_game_logs.fetch_game_log_totals", _boom)
    assert refresh_pipeline.try_acquire_refresh_slot() is True

    # Runs synchronously here; the inner error is logged, the slot released.
    season_routes._run_rest_of_season_fetch()

    assert refresh_pipeline.get_refresh_status()["running"] is False
    kv_store._reset_singleton()


def _thread_that_fails_to_start(*_a, **_k):
    t = MagicMock()
    t.start.side_effect = RuntimeError("can't start new thread")
    return t


def test_fetch_ros_route_releases_slot_when_spawn_fails(
    client, monkeypatch, fake_redis, free_refresh_slot
):
    """If the worker thread fails to spawn after the slot is acquired, the
    slot must be released so the spawn failure can't wedge all future jobs."""
    from fantasy_baseball.web import refresh_pipeline, season_routes

    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)
    monkeypatch.setattr(season_routes.threading, "Thread", _thread_that_fails_to_start)

    with pytest.raises(RuntimeError):
        client.post("/api/fetch-ros-projections")

    assert refresh_pipeline.get_refresh_status()["running"] is False


def test_refresh_route_releases_slot_when_spawn_fails(
    client, monkeypatch, fake_redis, free_refresh_slot
):
    """Same spawn-failure guard for the full-refresh route."""
    from fantasy_baseball.web import refresh_pipeline, season_routes

    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)
    monkeypatch.setattr(season_routes.threading, "Thread", _thread_that_fails_to_start)

    with pytest.raises(RuntimeError):
        client.post("/api/refresh")

    assert refresh_pipeline.get_refresh_status()["running"] is False


def test_route_returns_503_when_lock_held_by_other_instance(
    client, monkeypatch, fake_redis, free_refresh_slot
):
    """When another instance holds the durable lock (the in-process slot is
    free here), the route returns 503 so QStash redelivers the overlapping run
    later instead of silently dropping it. Fixes the skip-with-no-retry gap.
    """
    from fantasy_baseball.data import redis_store
    from fantasy_baseball.web import season_routes

    # Another instance holds the durable lock.
    assert redis_store.acquire_refresh_lock(fake_redis, "other-instance", 1800) is True
    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)
    thread_ctor = MagicMock()
    monkeypatch.setattr(season_routes.threading, "Thread", thread_ctor)

    resp = client.post("/api/refresh")

    assert resp.status_code == 503
    assert resp.get_json()["status"] == "locked_by_other_instance"
    thread_ctor.assert_not_called()  # no worker spawned


def test_ros_fetch_skips_when_durable_lock_held_by_other_instance(
    monkeypatch, fake_redis, free_refresh_slot
):
    """A second instance must not run the ROS fetch while another holds the
    DURABLE lock. The in-process slot only mutexes within one process; across
    Render instances / QStash redelivery the durable KV lock is the guard. If
    it didn't hold, two jobs would race the game-log rollup read-modify-write
    and silently drop players from the totals.
    """
    from fantasy_baseball.data import redis_store
    from fantasy_baseball.web import season_routes

    # Simulate another instance already holding the cross-instance lock.
    assert redis_store.acquire_refresh_lock(fake_redis, "other-instance", 1800) is True
    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)

    # If the job did NOT skip, this would be called as its first data step.
    game_logs = MagicMock(side_effect=AssertionError("job should have skipped"))
    monkeypatch.setattr("fantasy_baseball.data.mlb_game_logs.fetch_game_log_totals", game_logs)

    season_routes._run_rest_of_season_fetch()

    game_logs.assert_not_called()  # skipped before touching the shared rollup


def test_ros_fetch_runs_and_releases_durable_lock_when_free(
    monkeypatch, fake_redis, free_refresh_slot
):
    """When the durable lock is free the job claims it, runs, and releases it
    so the next job can acquire -- the lock must not wedge after a clean run.
    """
    from fantasy_baseball.data import redis_store
    from fantasy_baseball.web import season_routes

    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)

    # Make the first data step a no-op-ish failure so we don't run the whole
    # pipeline; the durable-lock acquire/release still wraps it.
    monkeypatch.setattr(
        "fantasy_baseball.data.mlb_game_logs.fetch_game_log_totals",
        MagicMock(side_effect=RuntimeError("stop after lock acquired")),
    )

    season_routes._run_rest_of_season_fetch()

    # Lock was released in finally, so a later instance can acquire it.
    assert redis_store.acquire_refresh_lock(fake_redis, "next-instance", 1800) is True


def test_ros_fetch_skips_blend_when_no_system_fetched(monkeypatch, fake_redis, free_refresh_slot):
    """Fetch-success gate: when every system fails to fetch (e.g. FanGraphs
    Cloudflare-403), the job must NOT call blend_and_cache_ros -- which would
    pick the newest on-disk snapshot (a stale prior-run or committed dir) and
    overwrite the last-good cache:ros_projections -- and must release the slot."""
    from unittest.mock import MagicMock

    from fantasy_baseball.web import refresh_pipeline, season_routes

    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)
    monkeypatch.setattr(
        "fantasy_baseball.data.mlb_game_logs.fetch_game_log_totals",
        lambda *a, **k: None,
    )
    # Every system returns an error -> zero fresh CSVs produced this run.
    monkeypatch.setattr(
        "fantasy_baseball.data.fangraphs_fetch.fetch_rest_of_season_projections",
        lambda *a, **k: {"steamer": "error: no data returned for hitters"},
    )
    blend = MagicMock()
    monkeypatch.setattr("fantasy_baseball.data.ros_pipeline.blend_and_cache_ros", blend)

    season_routes._run_rest_of_season_fetch()

    blend.assert_not_called()  # gate skipped the blend; last-good ROS untouched
    assert refresh_pipeline.get_refresh_status()["running"] is False


def test_full_standings_page_with_cached_data(client, kv_isolation):
    """Integration test: standings page renders correctly with all cached data present."""
    from fantasy_baseball.web import season_data

    standings = {
        "effective_date": "2026-04-01",
        "teams": [
            {
                "name": "Hart of the Order",
                "team_key": "k1",
                "rank": 1,
                "stats": {
                    "R": 300,
                    "HR": 90,
                    "RBI": 290,
                    "SB": 50,
                    "AVG": 0.270,
                    "W": 35,
                    "K": 600,
                    "SV": 25,
                    "ERA": 3.50,
                    "WHIP": 1.18,
                },
            },
            {
                "name": "SkeleThor",
                "team_key": "k2",
                "rank": 2,
                "stats": {
                    "R": 310,
                    "HR": 85,
                    "RBI": 295,
                    "SB": 40,
                    "AVG": 0.265,
                    "W": 38,
                    "K": 580,
                    "SV": 30,
                    "ERA": 3.40,
                    "WHIP": 1.15,
                },
            },
        ],
    }
    season_data.write_cache(CacheKey.STANDINGS, standings)
    season_data.write_cache(CacheKey.META, {"last_refresh": "8:32 AM", "week": "3"})

    with patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cfg.return_value.team_name = "Hart of the Order"

        resp = client.get("/standings")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "Hart of the Order" in html
        assert "SkeleThor" in html
        assert "8:32 AM" in html


def test_full_lineup_page_with_cached_data(client, kv_isolation):
    """Integration test: lineup page renders with cached roster data."""
    from fantasy_baseball.web import season_data

    roster = [
        {
            "name": "Adley Rutschman",
            "positions": ["C"],
            "selected_position": "C",
            "player_id": "123",
            "status": "",
        },
        {
            "name": "Corbin Burnes",
            "positions": ["SP"],
            "selected_position": "P",
            "player_id": "456",
            "status": "",
        },
    ]
    optimal = {"hitters": {}, "pitchers": {}, "moves": []}
    season_data.write_cache(CacheKey.ROSTER, roster)
    season_data.write_cache(CacheKey.LINEUP_OPTIMAL, optimal)
    season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"})

    resp = client.get("/lineup")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "Adley Rutschman" in html
    assert "Corbin Burnes" in html
    assert "Optimal" in html  # should show optimal button since no moves


def test_band_cell_macro_maps_verdict_to_gap_color_and_renders_tooltip():
    """band_cell renders the shared gap-badge rectangle (verdict->color)
    plus a rich hover tooltip with deltaRoto / SD / P(helps)."""
    app = create_app()
    with app.app_context():
        tmpl = app.jinja_env.from_string(
            "{% from 'season/macros.html' import band_cell %}{{ band_cell(b) }}"
        )

        real = tmpl.render(b={"mean": 1.1, "sd": 0.4, "p_positive": 0.82, "verdict": "real"})
        assert "gap-badge gap-positive" in real  # rectangle, green verdict
        assert 'class="tooltip"' in real  # rich hover, not just title=
        assert "+1.1" in real  # mean on the badge
        assert "P(helps)" in real and "82%" in real
        assert "Std dev" in real

        coin = tmpl.render(b={"mean": 0.0, "sd": 0.3, "p_positive": 0.5, "verdict": "coin-flip"})
        assert "gap-badge gap-marginal" in coin

        down = tmpl.render(b={"mean": -1.2, "sd": 0.5, "p_positive": 0.1, "verdict": "downgrade"})
        assert "gap-badge gap-negative" in down

        # An unexpected verdict must degrade to a neutral badge, not 500 the page.
        unknown = tmpl.render(b={"mean": 0.0, "sd": 0.2, "p_positive": 0.5, "verdict": "???"})
        assert "gap-badge gap-marginal" in unknown


def test_lineup_delta_roto_renders_band_cell_with_tooltip(client, kv_isolation):
    """A lineup row with an optimizer band shows the colored rectangle + the
    hover tooltip, and the shared tooltip JS partial is wired into the page."""
    from fantasy_baseball.web import season_data

    roster = [
        {
            "name": "Adley Rutschman",
            "positions": ["C"],
            "selected_position": "C",
            "player_id": "1",
            "status": "",
        }
    ]
    optimal = {
        "hitter_lineup": [
            {
                "name": "Adley Rutschman",
                "roto_delta": 1.1,
                "band": {"mean": 1.1, "sd": 0.4, "p_positive": 0.82, "verdict": "real"},
            }
        ],
        "pitcher_starters": [],
    }
    season_data.write_cache(CacheKey.ROSTER, roster)
    season_data.write_cache(CacheKey.LINEUP_OPTIMAL, optimal)
    season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"})

    resp = client.get("/lineup")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "gap-badge gap-positive" in html  # same rectangle as roster audit
    assert "P(helps)" in html and "82%" in html  # rich hover content
    assert "Std dev" in html
    assert "function bindTooltips()" in html  # shared tooltip partial included


def test_roster_audit_delta_roto_renders_band_cell_with_tooltip(client, kv_isolation):
    """The roster-audit deltaRoto column keeps the rectangle and now gains the
    same rich hover tooltip + the shared tooltip JS partial."""
    from fantasy_baseball.web import season_data

    audit = [
        {
            "slot": "C",
            "player": "Adley Rutschman",
            "player_type": "hitter",
            "positions": ["C"],
            "player_sgp": 2.0,
            "best_fa": "Backup Catcher",
            "best_fa_positions": ["C"],
            "best_fa_type": "hitter",
            "best_fa_sgp": 2.6,
            "candidates": [
                {
                    "name": "Backup Catcher",
                    "positions": ["C"],
                    "player_type": "hitter",
                    "sgp": 2.6,
                    "gap": 0.6,
                    "delta_roto": {"total": 1.1},
                    "band": {"mean": 1.1, "sd": 0.4, "p_positive": 0.82, "verdict": "real"},
                }
            ],
        }
    ]
    season_data.write_cache(CacheKey.ROSTER_AUDIT, audit)
    season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"})

    resp = client.get("/roster-audit")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "gap-badge gap-positive" in html
    assert "P(helps)" in html and "82%" in html
    assert "function bindTooltips()" in html  # shared tooltip partial included


def test_full_trades_page_renders(client):
    """Integration test: trades page renders without waiver data."""
    resp = client.get("/waivers-trades")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "Trade Finder" in html


def test_compare_missing_params(client):
    """Missing required params should return 400."""
    resp = client.get("/api/players/compare")
    assert resp.status_code == 400

    resp2 = client.get("/api/players/compare?roster_player=X")
    assert resp2.status_code == 400


def test_lineup_accepts_basis_param(client):
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/lineup?basis=ytd")
    assert resp.status_code == 200


def test_lineup_tbodies_returns_html_for_basis(client, kv_isolation):
    _seed_minimum_lineup_caches(["Adley Rutschman", "Corbin Burnes"])

    resp = client.get("/lineup/tbodies?basis=total")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["basis"] == "total"
    assert "hitters_html" in data
    assert "pitchers_html" in data


def test_lineup_tbodies_unknown_basis_falls_back(client, kv_isolation):
    _seed_minimum_lineup_caches(["Adley Rutschman"])

    resp = client.get("/lineup/tbodies?basis=bogus")
    assert resp.status_code == 200
    assert resp.get_json()["basis"] == "ros"


def test_lineup_tbodies_404_without_roster(client, kv_isolation):
    # Isolated empty KV (no roster seeded) -> route returns 404.
    resp = client.get("/lineup/tbodies?basis=ros")
    assert resp.status_code == 404


def test_standings_page_includes_breakdown_json_when_cache_present(client, kv_isolation):
    """When STANDINGS_BREAKDOWN cache exists, its JSON is embedded in the page."""
    from fantasy_baseball.web import season_data

    payload = {
        "teams": {
            "Team A": {
                "team_name": "Team A",
                "hitters": [
                    {
                        "name": "H1",
                        "player_type": "hitter",
                        "status": "active",
                        "scale_factor": 1.0,
                        "raw_stats": {
                            "hr": 20,
                            "r": 60,
                            "rbi": 70,
                            "sb": 5,
                            "h": 120,
                            "ab": 450,
                        },
                    }
                ],
                "pitchers": [],
            }
        }
    }
    season_data.write_cache(CacheKey.STANDINGS_BREAKDOWN, payload)
    season_data.write_cache(CacheKey.STANDINGS, _mock_standings())

    with patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cfg.return_value.team_name = "Hart of the Order"

        resp = client.get("/standings")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert 'id="breakdown-data"' in body
        assert '"Team A"' in body


def test_standings_page_omits_breakdown_json_when_cache_missing(client):
    """When STANDINGS_BREAKDOWN cache is absent, no embedded JSON tag."""
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/standings")
    body = resp.get_data(as_text=True)
    assert 'id="breakdown-data"' not in body


def test_standings_passes_baseline_meta_to_template(client, kv_isolation):
    """When monte_carlo.baseline_meta is present in the cache, it is
    rendered into the page as the freeze-date caption."""
    from fantasy_baseball.web import season_data

    season_data.write_cache(
        CacheKey.MONTE_CARLO,
        {
            "base": {"team_results": {}, "category_risk": {}},
            "baseline_meta": {
                "frozen_at": "2026-04-17T00:00:00Z",
                "roster_date": "2026-03-27",
                "season_year": 2026,
            },
            "rest_of_season": None,
        },
    )
    season_data.write_cache(CacheKey.STANDINGS, _mock_standings())

    with patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cfg.return_value.team_name = "Team 01"

        resp = client.get("/standings")
        assert resp.status_code == 200
        assert b"2026-03-27" in resp.data


def test_standings_passes_distributions_to_template(client, kv_isolation):
    """The standings route reshapes the cached rest_of_season distributions
    block and passes it to the template as the `distributions` kwarg, marking
    the user's team server-side and dropping the raw user_team string."""
    from fantasy_baseball.web import season_data

    season_data.write_cache(
        CacheKey.MONTE_CARLO,
        {
            "base": None,
            "baseline_meta": None,
            "rest_of_season": {
                "team_results": {},
                "category_risk": {},
                "distributions": {
                    "user_team": "Team 01",
                    "overall": {
                        "x": [60.0, 70.0, 80.0],
                        "teams": {
                            "Team 01": {"y": [0.1, 0.2, 0.1], "median": 75.0},
                            "Team 02": {"y": [0.2, 0.1, 0.1], "median": 65.0},
                        },
                    },
                    "category_totals": {
                        "HR": {
                            "x": [200.0, 250.0, 300.0],
                            "teams": {
                                "Team 01": {"y": [0.1, 0.2, 0.1], "median": 270.0},
                                "Team 02": {"y": [0.2, 0.1, 0.1], "median": 240.0},
                            },
                        },
                    },
                    "category_points": {},
                },
            },
        },
    )
    season_data.write_cache(CacheKey.STANDINGS, _mock_standings())

    with patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cfg.return_value.team_name = "Team 01"
        with patch(
            "fantasy_baseball.web.season_routes.render_template", return_value=""
        ) as rendered:
            client.get("/standings")

    dist = rendered.call_args.kwargs["distributions"]
    assert "overall" in dist
    assert dist["overall"]["rows"]
    assert any(r["is_user"] for r in dist["overall"]["rows"])
    assert "user_team" not in dist


def test_standings_distributions_empty_without_mc(client, kv_isolation):
    """With no MONTE_CARLO cache seeded, the module-scope empty-state default
    reaches the template and the route does not crash."""
    from fantasy_baseball.web import season_data

    season_data.write_cache(CacheKey.STANDINGS, _mock_standings())

    with patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cfg.return_value.team_name = "Team 01"
        with patch(
            "fantasy_baseball.web.season_routes.render_template", return_value=""
        ) as rendered:
            client.get("/standings")

    dist = rendered.call_args.kwargs["distributions"]
    assert dist == {"overall": {"x": [], "rows": []}, "category_totals": {}, "category_points": {}}


def test_standings_embeds_distributions_node(client, kv_isolation):
    """The Distributions view embeds the reshaped distributions block as a
    JSON <script> node (#distributions-data) the canvas renderer reads. The
    embedded payload carries the server-marked is_user flag and drops the raw
    user_team string."""
    from fantasy_baseball.web import season_data

    season_data.write_cache(
        CacheKey.MONTE_CARLO,
        {
            "base": None,
            "baseline_meta": None,
            "rest_of_season": {
                "team_results": {},
                "category_risk": {},
                "distributions": {
                    "user_team": "Team 01",
                    "overall": {
                        "x": [60.0, 70.0, 80.0],
                        "teams": {
                            "Team 01": {"y": [0.1, 0.2, 0.1], "median": 75.0},
                            "Team 02": {"y": [0.2, 0.1, 0.1], "median": 65.0},
                        },
                    },
                    "category_totals": {},
                    "category_points": {},
                },
            },
        },
    )
    season_data.write_cache(CacheKey.STANDINGS, _mock_standings())

    with patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cfg.return_value.team_name = "Team 01"
        body = client.get("/standings").get_data(as_text=True)

    match = re.search(
        r'<script type="application/json" id="distributions-data">(.*?)</script>',
        body,
        re.DOTALL,
    )
    assert match is not None, "distributions-data script tag not found"
    dist = json.loads(match.group(1))
    assert dist["overall"]["rows"]
    assert any(r["is_user"] for r in dist["overall"]["rows"])
    assert "user_team" not in dist


def _seed_browse_caches():
    """Seed ros_projections + positions + roster + opp_rosters + audit
    into the active KV store. Returns the seeded names so tests can assert
    on specific players.
    """
    kv = get_kv()
    of_hitters = [
        {
            "name": "OF FA A",
            "player_type": "hitter",
            "team": "BOS",
            "r": 90,
            "hr": 30,
            "rbi": 100,
            "sb": 10,
            "h": 160,
            "ab": 550,
        },
        {
            "name": "OF FA B",
            "player_type": "hitter",
            "team": "NYY",
            "r": 80,
            "hr": 25,
            "rbi": 90,
            "sb": 8,
            "h": 150,
            "ab": 540,
        },
        {
            "name": "OF FA C",
            "player_type": "hitter",
            "team": "LAD",
            "r": 70,
            "hr": 20,
            "rbi": 80,
            "sb": 6,
            "h": 140,
            "ab": 530,
        },
        {
            "name": "OF Mine",
            "player_type": "hitter",
            "team": "ATL",
            "r": 95,
            "hr": 32,
            "rbi": 105,
            "sb": 12,
            "h": 165,
            "ab": 555,
        },
        {
            "name": "OF Opp",
            "player_type": "hitter",
            "team": "HOU",
            "r": 88,
            "hr": 28,
            "rbi": 95,
            "sb": 9,
            "h": 155,
            "ab": 545,
        },
    ]
    sp_pitchers = [
        {
            "name": "SP FA A",
            "player_type": "pitcher",
            "team": "BOS",
            "w": 12,
            "k": 180,
            "sv": 0,
            "ip": 180.0,
            "er": 60,
            "bb": 50,
            "h_allowed": 150,
        },
    ]
    kv.set(
        redis_key(CacheKey.ROS_PROJECTIONS),
        json.dumps({"hitters": of_hitters, "pitchers": sp_pitchers}),
    )
    kv.set(
        redis_key(CacheKey.POSITIONS),
        json.dumps(
            {
                "of fa a": ["OF"],
                "of fa b": ["OF"],
                "of fa c": ["OF"],
                "of mine": ["OF"],
                "of opp": ["OF"],
                "sp fa a": ["P"],
            }
        ),
    )
    kv.set(redis_key(CacheKey.ROSTER), json.dumps([{"name": "OF Mine", "player_type": "hitter"}]))
    kv.set(
        redis_key(CacheKey.OPP_ROSTERS),
        json.dumps(
            {
                "Rivals": [{"name": "OF Opp", "player_type": "hitter"}],
            }
        ),
    )
    kv.set(redis_key(CacheKey.ROSTER_AUDIT), json.dumps([]))
    return {
        "fa_a": "OF FA A",
        "fa_b": "OF FA B",
        "fa_c": "OF FA C",
        "mine": "OF Mine",
        "opp": "OF Opp",
    }


def test_browse_specific_position_returns_rostered_and_top_fas(client, kv_isolation):
    names = _seed_browse_caches()
    resp = client.get("/api/players/browse?pos=OF&fa_limit=2&fa_offset=0")
    assert resp.status_code == 200
    body = resp.get_json()
    returned = {p["name"] for p in body["players"]}
    assert returned == {names["mine"], names["opp"], names["fa_a"], names["fa_b"]}
    assert body["has_more_fa"] is True
    assert body["next_fa_offset"] == 2


def test_browse_load_more_paginates_fas_only(client, kv_isolation):
    names = _seed_browse_caches()
    resp = client.get("/api/players/browse?pos=OF&fa_limit=2&fa_offset=2")
    assert resp.status_code == 200
    body = resp.get_json()
    returned = {p["name"] for p in body["players"]}
    assert returned == {names["fa_c"]}
    assert body["has_more_fa"] is False
    assert body["next_fa_offset"] == 3


def test_browse_all_hit_caps_at_20_fas(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/browse?pos=ALL_HIT&fa_offset=0")
    assert resp.status_code == 200
    body = resp.get_json()
    fa_count = sum(1 for p in body["players"] if p["owner"] is None)
    rostered_count = sum(1 for p in body["players"] if p["owner"] is not None)
    assert fa_count == 3
    assert rostered_count == 2
    assert body["has_more_fa"] is False


def test_browse_invalid_pos_returns_400(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/browse?pos=Bogus")
    assert resp.status_code == 400


def test_browse_missing_pos_returns_400(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/browse")
    assert resp.status_code == 400


def test_browse_all_hit_paginates_with_fa_offset(client, kv_isolation):
    _seed_browse_caches()
    # Default fa_limit for ALL_HIT is 20, but the seed only has 3 FAs.
    # Asking for fa_offset=2 with fa_limit=2 should return one FA, no rostered.
    resp = client.get("/api/players/browse?pos=ALL_HIT&fa_limit=2&fa_offset=2")
    assert resp.status_code == 200
    body = resp.get_json()
    fa_count = sum(1 for p in body["players"] if p["owner"] is None)
    rostered_count = sum(1 for p in body["players"] if p["owner"] is not None)
    assert fa_count == 1
    assert rostered_count == 0
    assert body["has_more_fa"] is False


def test_browse_empty_cache_returns_empty_envelope(client, kv_isolation):
    # Do not seed — ROS_PROJECTIONS missing.
    resp = client.get("/api/players/browse?pos=OF")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"players": [], "has_more_fa": False, "next_fa_offset": 0}


def test_browse_sp_rp_split_on_sv_threshold(client, kv_isolation):
    # Three pitchers: sv=0 (SP), sv=4 (SP — strict <), sv=5 (RP — boundary).
    kv = get_kv()
    kv.set(
        redis_key(CacheKey.ROS_PROJECTIONS),
        json.dumps(
            {
                "hitters": [],
                "pitchers": [
                    {
                        "name": "SP Zero",
                        "player_type": "pitcher",
                        "team": "BOS",
                        "w": 12,
                        "k": 180,
                        "sv": 0,
                        "ip": 180.0,
                        "er": 60,
                        "bb": 50,
                        "h_allowed": 150,
                    },
                    {
                        "name": "SP Four",
                        "player_type": "pitcher",
                        "team": "NYY",
                        "w": 10,
                        "k": 160,
                        "sv": 4,
                        "ip": 170.0,
                        "er": 65,
                        "bb": 55,
                        "h_allowed": 155,
                    },
                    {
                        "name": "RP Five",
                        "player_type": "pitcher",
                        "team": "LAD",
                        "w": 3,
                        "k": 70,
                        "sv": 5,
                        "ip": 60.0,
                        "er": 22,
                        "bb": 20,
                        "h_allowed": 50,
                    },
                ],
            }
        ),
    )
    kv.set(
        redis_key(CacheKey.POSITIONS),
        json.dumps(
            {
                "sp zero": ["P"],
                "sp four": ["P"],
                "rp five": ["P"],
            }
        ),
    )
    kv.set(redis_key(CacheKey.ROSTER), json.dumps([]))
    kv.set(redis_key(CacheKey.OPP_ROSTERS), json.dumps({}))
    kv.set(redis_key(CacheKey.ROSTER_AUDIT), json.dumps([]))

    sp_resp = client.get("/api/players/browse?pos=SP")
    assert sp_resp.status_code == 200
    sp_names = {p["name"] for p in sp_resp.get_json()["players"]}
    assert sp_names == {"SP Zero", "SP Four"}

    rp_resp = client.get("/api/players/browse?pos=RP")
    assert rp_resp.status_code == 200
    rp_names = {p["name"] for p in rp_resp.get_json()["players"]}
    assert rp_names == {"RP Five"}


def test_browse_response_includes_delta_roto_for_fa_with_audit_hit(client, kv_isolation):
    """Pin delta_roto shape: FA whose name appears in roster_audit candidates
    surfaces the precomputed dict so the frontend Compute button can mutate
    in place.
    """
    kv = get_kv()
    kv.set(
        redis_key(CacheKey.ROS_PROJECTIONS),
        json.dumps(
            {
                "hitters": [
                    {
                        "name": "Roster OF",
                        "player_type": "hitter",
                        "team": "ATL",
                        "r": 50,
                        "hr": 10,
                        "rbi": 40,
                        "sb": 2,
                        "h": 110,
                        "ab": 450,
                    },
                    {
                        "name": "FA Stud",
                        "player_type": "hitter",
                        "team": "BOS",
                        "r": 90,
                        "hr": 30,
                        "rbi": 100,
                        "sb": 10,
                        "h": 160,
                        "ab": 550,
                    },
                ],
                "pitchers": [],
            }
        ),
    )
    kv.set(
        redis_key(CacheKey.POSITIONS),
        json.dumps(
            {
                "roster of": ["OF"],
                "fa stud": ["OF"],
            }
        ),
    )
    kv.set(
        redis_key(CacheKey.ROSTER),
        json.dumps([{"name": "Roster OF", "player_type": "hitter", "positions": ["OF"]}]),
    )
    kv.set(redis_key(CacheKey.OPP_ROSTERS), json.dumps({}))
    kv.set(
        redis_key(CacheKey.ROSTER_AUDIT),
        json.dumps(
            [
                {
                    "player": "Roster OF",
                    "candidates": [
                        {
                            "name": "FA Stud",
                            "delta_roto": {
                                "total": 1.5,
                                "categories": {"R": {"roto_delta": 0.5}, "HR": {"roto_delta": 1.0}},
                            },
                        },
                    ],
                },
            ]
        ),
    )

    resp = client.get("/api/players/browse?pos=OF")
    assert resp.status_code == 200
    body = resp.get_json()
    fa = next(p for p in body["players"] if p["name"] == "FA Stud")
    assert fa["delta_roto"] == {
        "total": 1.5,
        "categories": {"R": {"roto_delta": 0.5}, "HR": {"roto_delta": 1.0}},
    }
    # Rostered player gets no delta_roto.
    rostered = next(p for p in body["players"] if p["name"] == "Roster OF")
    assert rostered["delta_roto"] is None


def test_browse_hitter_response_includes_required_stat_fields(client, kv_isolation):
    """The frontend table renders per-type stat fields directly. Pin the
    legacy field names so a refactor of _build_player_record can't silently
    drop them.
    """
    _seed_browse_caches()
    resp = client.get("/api/players/browse?pos=OF")
    assert resp.status_code == 200
    fa = next(p for p in resp.get_json()["players"] if p["name"] == "OF FA A")
    for field in ("R", "HR", "RBI", "SB", "AVG", "h", "ab"):
        assert field in fa, f"missing hitter field: {field}"
    # And no pitcher fields leak through.
    for field in ("W", "K", "SV", "ERA", "WHIP", "ip", "er", "bb", "h_allowed"):
        assert field not in fa, f"unexpected pitcher field on hitter: {field}"


def test_browse_pitcher_response_includes_required_stat_fields(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/browse?pos=SP")
    assert resp.status_code == 200
    sp = next(p for p in resp.get_json()["players"] if p["name"] == "SP FA A")
    for field in ("W", "K", "SV", "ERA", "WHIP", "ip", "er", "bb", "h_allowed"):
        assert field in sp, f"missing pitcher field: {field}"
    for field in ("R", "HR", "RBI", "SB", "AVG", "h", "ab"):
        assert field not in sp, f"unexpected hitter field on pitcher: {field}"


def test_find_returns_substring_matches(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/find?q=fa")
    assert resp.status_code == 200
    body = resp.get_json()
    names = {p["name"] for p in body["players"]}
    # "fa" matches every FA-named player (OF FA A/B/C and SP FA A).
    assert "OF FA A" in names
    assert "OF FA B" in names
    assert "OF FA C" in names
    assert "SP FA A" in names

    # Case-insensitivity: uppercase query returns the same matches.
    resp_upper = client.get("/api/players/find?q=FA")
    assert resp_upper.status_code == 200
    assert {p["name"] for p in resp_upper.get_json()["players"]} == names


def test_find_rejects_short_query(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/find?q=a")
    assert resp.status_code == 400


def test_find_missing_q_returns_400(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/find")
    assert resp.status_code == 400


def test_find_caps_at_25_results(client, kv_isolation):
    kv = get_kv()
    hitters = [
        {
            "name": f"Smithers {i}",
            "player_type": "hitter",
            "team": "BOS",
            "r": 50,
            "hr": 10,
            "rbi": 40,
            "sb": 2,
            "h": 100,
            "ab": 400,
        }
        for i in range(30)
    ]
    kv.set(redis_key(CacheKey.ROS_PROJECTIONS), json.dumps({"hitters": hitters, "pitchers": []}))
    kv.set(redis_key(CacheKey.POSITIONS), json.dumps({f"smithers {i}": ["OF"] for i in range(30)}))
    kv.set(redis_key(CacheKey.ROSTER), json.dumps([]))
    kv.set(redis_key(CacheKey.OPP_ROSTERS), json.dumps({}))
    kv.set(redis_key(CacheKey.ROSTER_AUDIT), json.dumps([]))
    resp = client.get("/api/players/find?q=smith")
    assert resp.status_code == 200
    assert len(resp.get_json()["players"]) == 25


def test_lookup_returns_players_in_request_order(client, kv_isolation):
    names = _seed_browse_caches()
    resp = client.get(f"/api/players/lookup?keys={names['fa_b']}::hitter,{names['mine']}::hitter")
    assert resp.status_code == 200
    body = resp.get_json()
    returned = [p["name"] for p in body["players"]]
    assert returned == [names["fa_b"], names["mine"]]


def test_lookup_silently_drops_misses(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/lookup?keys=Nobody::hitter,OF FA A::hitter")
    assert resp.status_code == 200
    body = resp.get_json()
    assert [p["name"] for p in body["players"]] == ["OF FA A"]


def test_lookup_missing_keys_returns_400(client, kv_isolation):
    _seed_browse_caches()
    resp = client.get("/api/players/lookup")
    assert resp.status_code == 400


def test_lookup_handles_malformed_inputs(client, kv_isolation):
    _seed_browse_caches()
    # Blank keys param -> 400 (same as missing).
    assert client.get("/api/players/lookup?keys=").status_code == 400
    # All-malformed pairs -> 200 with empty list (no separator, bad type).
    body = client.get("/api/players/lookup?keys=NoSeparator,Soto::nope").get_json()
    assert body == {"players": []}


def test_lookup_normalizes_case_for_matching(client, kv_isolation):
    names = _seed_browse_caches()
    # Lowercase request matches the upper-cased seeded name via normalize_name.
    resp = client.get(f"/api/players/lookup?keys={names['fa_a'].lower()}::hitter")
    assert resp.status_code == 200
    assert [p["name"] for p in resp.get_json()["players"]] == [names["fa_a"]]


# --- /lineup streak chip injection ----------------------------------------------------


def _seed_streak_cache_for(
    name: str,
    *,
    composite: int,
    hot_cat: str,
    prob: float,
) -> None:
    """Seed CacheKey.STREAK_SCORES with one roster row for ``name``.

    Mirrors the helper in tests/test_web/test_streaks_route.py — kept
    separate so this test module stays self-contained.
    """
    from datetime import date

    from fantasy_baseball.streaks.dashboard import serialize_report
    from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
    from fantasy_baseball.streaks.reports.sunday import (
        DriverLine,
        Report,
        ReportRow,
    )
    from fantasy_baseball.web.season_data import write_cache

    score = PlayerCategoryScore(
        player_id=1,
        category=hot_cat,  # type: ignore[arg-type]
        label="hot",
        probability=prob,
        drivers=(Driver(feature="barrel_pct", z_score=1.0),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name=name,
        positions=("OF",),
        player_id=1,
        composite=composite,
        scores={hot_cat: score},  # type: ignore[dict-item]
        max_probability=prob,
    )
    dl = DriverLine(
        player_name=name,
        category=hot_cat,  # type: ignore[arg-type]
        label="hot",
        probability=prob,
        drivers=(Driver(feature="barrel_pct", z_score=1.0),),
    )
    rpt = Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="Hart of the Order",
        league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(dl,),
        skipped=(),
    )
    write_cache(CacheKey.STREAK_SCORES, serialize_report(rpt))


def _seed_minimum_lineup_caches(hitter_names: list[str]) -> None:
    """Seed the minimum cache entries the /lineup route needs to render hitter rows.

    Writes ROSTER (each name as a hitter at OF) plus an empty LINEUP_OPTIMAL
    so format_lineup_for_display produces hitter dicts that flow into the
    tbody partial.
    """
    from fantasy_baseball.web import season_data

    roster = [
        {
            "name": name,
            "positions": ["OF"],
            "selected_position": "OF",
            "player_id": str(i + 1),
            "status": "",
        }
        for i, name in enumerate(hitter_names)
    ]
    season_data.write_cache(CacheKey.ROSTER, roster)
    season_data.write_cache(CacheKey.LINEUP_OPTIMAL, {"hitters": {}, "pitchers": {}, "moves": []})
    season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"})


def test_lineup_injects_streak_chip_when_cache_present(client, kv_isolation) -> None:
    """When STREAK_SCORES is in cache, the lineup hitters table renders chips."""
    _seed_streak_cache_for("Roster Guy", composite=2, hot_cat="hr", prob=0.62)
    _seed_minimum_lineup_caches(hitter_names=["Roster Guy"])

    resp = client.get("/lineup")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "streak-chip" in body
    assert "HOT &middot; HR" in body or "HOT · HR" in body


def test_lineup_renders_dash_chip_when_no_streak_cache(client, kv_isolation) -> None:
    _seed_minimum_lineup_caches(hitter_names=["Roster Guy"])
    resp = client.get("/lineup")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "streak-chip streak-neutral" in body


# --- /api/il-return-plan ---------------------------------------------------------------


def _il_fake_cache(monkeypatch, values: dict) -> None:
    """Patch read_cache_dict/read_cache_list to read from an in-memory map.

    Mirrors the fake-cache helper in test_optimize_trade_lineup_route.py so the
    IL planner route's optimizer loop runs against a tiny roster (fast) instead
    of a full-league shape.
    """

    def fake_read_cache_dict(key, *_a, **_k):
        v = values.get(key.value)
        return v if isinstance(v, dict) else None

    def fake_read_cache_list(key, *_a, **_k):
        v = values.get(key.value)
        return v if isinstance(v, list) else None

    import fantasy_baseball.web.season_routes as routes

    monkeypatch.setattr(routes, "read_cache_dict", fake_read_cache_dict)
    monkeypatch.setattr(routes, "read_cache_list", fake_read_cache_list)


class _IlFakeCfg:
    """Tiny league config so optimize_*_lineup finishes in milliseconds.

    Capacity (non-IL slots) = OF3 + UTIL1 + BN1 = 5. A 5-body active/bench
    roster plus one IL-slot player overflows by 1 when the IL player is
    activated, forcing a drop and producing non-empty plans.
    """

    team_name = "Hart"
    roster_slots: ClassVar[dict[str, int]] = {
        "OF": 3,
        "UTIL": 1,
        "BN": 1,
        "IL": 1,
    }


def _patch_il_config(monkeypatch) -> None:
    import fantasy_baseball.web.season_routes as routes

    monkeypatch.setattr(routes, "_load_config", lambda: _IlFakeCfg())


def _il_roster_and_projections():
    """5 active/bench hitters + 1 IL-slot hitter, plus a 2-team projection."""
    from fantasy_baseball.models.player import HitterStats, Player

    def _hit(name, slot):
        return Player(
            name=name,
            player_type="hitter",
            positions=["OF"],
            selected_position=slot,
            rest_of_season=HitterStats(pa=600, ab=500, h=125, r=70, hr=20, rbi=60, sb=5, avg=0.250),
        ).to_dict()

    roster = [_hit(f"M{i}", "OF" if i < 3 else "UTIL" if i == 3 else "BN") for i in range(5)]
    roster.append(_hit("IL Guy", "IL"))

    standings_stats = {
        "R": 1000,
        "HR": 250,
        "RBI": 750,
        "SB": 80,
        "AVG": 0.260,
        "W": 70,
        "K": 1200,
        "SV": 50,
        "ERA": 3.80,
        "WHIP": 1.25,
    }
    projected_standings = {
        "effective_date": "2026-04-01",
        "teams": [
            {"name": "Hart", "stats": dict(standings_stats)},
            {"name": "Rival", "stats": dict(standings_stats)},
        ],
    }
    return roster, projected_standings


def test_il_return_plan_route_404_without_data(client, kv_isolation):
    # Empty (isolated) KV -> route reports missing roster data (not a 500).
    resp = client.get("/api/il-return-plan?activate=abc")
    assert resp.status_code == 404


def test_il_return_plan_route_returns_plan_shape(client, monkeypatch):
    roster, ps = _il_roster_and_projections()
    _il_fake_cache(
        monkeypatch,
        {
            "roster": roster,
            "projections": {
                "projected_standings": ps,
                "team_sds": None,
                "fraction_remaining": 1.0,
            },
        },
    )
    _patch_il_config(monkeypatch)

    resp = client.get("/api/il-return-plan")  # no activate -> activate all IL
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data.keys()) >= {"activating", "capacity", "overflow", "plans"}
    assert isinstance(data["plans"], list)
    assert isinstance(data["capacity"], int)
    # The IL-slot player is the one activated.
    assert data["activating"] == ["IL Guy"]


def test_stash_route_renders_ranked_board(client, kv_isolation):
    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.web import season_data

    payload = {
        "open_il_slots": 1,
        "cutline_rank": 2,
        "candidates": [
            {
                "name": "Blake Snell",
                "player_type": "pitcher",
                "status": "IL15",
                "owned": False,
                "stash_value": 4.2,
                "band": {"mean": 4.2, "sd": 1.1, "p_positive": 0.91, "verdict": "real"},
                "recommended_drop": None,
            }
        ],
        "warning": None,
    }
    season_data.write_cache(CacheKey.STASH, payload)
    season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"})

    resp = client.get("/stash")
    assert resp.status_code == 200
    html = resp.data.decode()
    assert "Blake Snell" in html
    assert "Grab &amp; Stash" in html or "Grab & Stash" in html


def test_stash_below_cutline_owned_flagged_droppable(client, kv_isolation):
    from fantasy_baseball.data.cache_keys import CacheKey
    from fantasy_baseball.web import season_data

    payload = {
        "open_il_slots": 0,
        "cutline_rank": 1,
        "candidates": [
            {
                "name": "Better FA",
                "player_type": "pitcher",
                "status": "IL15",
                "owned": False,
                "stash_value": 4.0,
                "band": {"mean": 4.0, "sd": 1.0, "p_positive": 0.9, "verdict": "real"},
                "recommended_drop": "Weak Owned Stash",
            },
            {
                "name": "Weak Owned Stash",
                "player_type": "pitcher",
                "status": "IL60",
                "owned": True,
                "stash_value": 1.0,
                "band": {"mean": 1.0, "sd": 0.8, "p_positive": 0.6, "verdict": "lean"},
                "recommended_drop": None,
            },
        ],
        "warning": None,
    }
    season_data.write_cache(CacheKey.STASH, payload)
    season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"})
    html = client.get("/stash").data.decode()
    assert "below-cutline" in html  # the weak owned stash is below the cutline
    assert "Weak Owned Stash" in html


def test_standings_route_does_not_fabricate_contribution_stats_for_stale_blob(
    client,
):
    """A stale KV blob lacking contribution_stats must NOT have it fabricated
    by the route. raw_stats is the full-season projection, so the old
    raw_stats * scale_factor fallback rendered full_season * factor -- the
    pre-#110 YTD double-count (team YTD is added separately at the team
    level). Per the repo rule "a wrong answer that looks plausible is worse
    than no answer," a stale blob renders honest zeros (contribution_stats
    absent/empty) rather than plausible-but-wrong numbers. This pins the
    removal of the back-compat fabrication in PlayerContribution.from_dict.
    """
    from fantasy_baseball.web.season_data import CacheKey

    stale_payload = {
        "effective_date": "2026-05-29",
        "teams": {
            "Hart of the Order": {
                "team_name": "Hart of the Order",
                "hitters": [
                    {
                        "name": "Test Hitter",
                        "player_type": "hitter",
                        "status": "active",
                        "scale_factor": 1.0,
                        "raw_stats": {
                            "r": 80.0,
                            "hr": 25.0,
                            "rbi": 70.0,
                            "sb": 5.0,
                            "h": 130.0,
                            "ab": 500.0,
                        },
                        # contribution_stats intentionally absent (stale blob).
                    }
                ],
                "pitchers": [
                    {
                        "name": "Test Displaced",
                        "player_type": "pitcher",
                        "status": "displaced",
                        "scale_factor": 0.5,
                        "raw_stats": {
                            "w": 10.0,
                            "k": 200.0,
                            "sv": 0.0,
                            "ip": 180.0,
                            "er": 60.0,
                            "bb": 50.0,
                            "h_allowed": 160.0,
                        },
                        # contribution_stats intentionally absent.
                    }
                ],
            }
        },
    }

    def fake_read_cache_dict(key):
        if key == CacheKey.STANDINGS_BREAKDOWN:
            return stale_payload
        if key == CacheKey.STANDINGS:
            # Minimal valid standings blob so the `if raw_standings:` branch
            # in the route is entered and raw_breakdown is actually read.
            return {
                "effective_date": "2026-05-29",
                "teams": [
                    {
                        "name": "Hart of the Order",
                        "team_key": "key_1",
                        "rank": 1,
                        "stats": {
                            "R": 300,
                            "HR": 90,
                            "RBI": 290,
                            "SB": 50,
                            "AVG": 0.270,
                            "W": 35,
                            "K": 600,
                            "SV": 25,
                            "ERA": 3.50,
                            "WHIP": 1.18,
                        },
                    }
                ],
            }
        return None

    # Patch the cache reader at the season_routes import site.
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict") as m,
        patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg,
    ):
        mock_cfg.return_value.team_name = "Hart of the Order"
        m.side_effect = fake_read_cache_dict
        response = client.get("/standings")

    assert response.status_code == 200
    body = response.get_data(as_text=True)

    # The route must NOT invent contribution_stats for a stale blob. Find the
    # standings_breakdown JSON block.
    match = re.search(
        r'<script[^>]*id="breakdown-data"[^>]*>(.*?)</script>',
        body,
        re.DOTALL,
    )
    assert match, "Expected breakdown-data script tag in standings.html output"
    breakdown_json = json.loads(match.group(1).strip())

    # contribution_stats must be empty -- NOT the fabricated full_season * factor.
    # The old bug produced K = 200 * 0.5 = 100.0 and HR = 25 * 1.0 = 25.0.
    pitcher = breakdown_json["teams"]["Hart of the Order"]["pitchers"][0]
    assert pitcher.get("contribution_stats", {}) == {}, (
        f"Route fabricated contribution_stats for a stale blob: {pitcher.get('contribution_stats')}"
    )
    # raw_stats still round-trips for the display column.
    assert abs(pitcher["raw_stats"]["k"] - 200.0) < 1e-6

    hitter = breakdown_json["teams"]["Hart of the Order"]["hitters"][0]
    assert hitter.get("contribution_stats", {}) == {}


def test_standings_route_preserves_team_ytd_block_through_round_trip(client):
    """Regression test: build_standings_breakdown_payload writes a team_ytd
    block per team into cache:standings_breakdown; the route reads it,
    round-trips each team payload through ``RosterBreakdown.from_dict``/
    ``to_dict``, and emits the JSON consumed by the template. The team_ytd
    block MUST survive that round-trip -- before commit 29fa623 it was
    silently stripped because ``RosterBreakdown`` had no ``team_ytd``
    field, leaving the modal unable to render the team-YTD header row
    and breaking the widget-vs-modal arithmetic invariant
    (team_ytd + sum(player rows) == widget headline).

    Exercises the actual route path so a future regression that drops
    team_ytd from ``RosterBreakdown.to_dict``, the season_routes
    round-trip, or the template serialization will fail this test.
    """
    from fantasy_baseball.web.season_data import CacheKey

    payload_with_team_ytd = {
        "effective_date": "2026-06-02",
        "teams": {
            "Hart of the Order": {
                "team_name": "Hart of the Order",
                "hitters": [
                    {
                        "name": "Hitter A",
                        "player_type": "hitter",
                        "status": "active",
                        "scale_factor": 1.0,
                        "raw_stats": {
                            "r": 70.0,
                            "hr": 20.0,
                            "rbi": 65.0,
                            "sb": 4.0,
                            "h": 110.0,
                            "ab": 420.0,
                        },
                        "contribution_stats": {
                            "r": 70.0,
                            "hr": 20.0,
                            "rbi": 65.0,
                            "sb": 4.0,
                            "h": 110.0,
                            "ab": 420.0,
                        },
                    }
                ],
                "pitchers": [],
                # The actual block the refresh pipeline writes. Keys match
                # _team_ytd_block in refresh_pipeline.py at this point in
                # the branch (uppercase). The route round-trip MUST preserve
                # the block as-is regardless of casing.
                "team_ytd": {
                    "R": 120.0,
                    "HR": 30.0,
                    "RBI": 110.0,
                    "SB": 15.0,
                    "W": 15.0,
                    "K": 300.0,
                    "SV": 8.0,
                    "H": 220.0,
                    "AB": 800.0,
                    "IP": 300.0,
                    "ER": 116.67,
                    "BB_plus_H_allowed": 360.0,
                },
            }
        },
    }

    def fake_read_cache_dict(key):
        if key == CacheKey.STANDINGS_BREAKDOWN:
            return payload_with_team_ytd
        if key == CacheKey.STANDINGS:
            return _mock_standings()
        return None

    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict") as m,
        patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg,
    ):
        mock_cfg.return_value.team_name = "Hart of the Order"
        m.side_effect = fake_read_cache_dict
        response = client.get("/standings")

    assert response.status_code == 200
    body = response.get_data(as_text=True)

    match = re.search(
        r'<script[^>]*id="breakdown-data"[^>]*>(.*?)</script>',
        body,
        re.DOTALL,
    )
    assert match, "Expected breakdown-data script tag in standings.html output"
    breakdown_json = json.loads(match.group(1).strip())

    team_payload = breakdown_json["teams"]["Hart of the Order"]
    assert "team_ytd" in team_payload, (
        "Route stripped team_ytd from the breakdown payload -- "
        "RosterBreakdown.from_dict/to_dict round-trip regressed."
    )
    team_ytd = team_payload["team_ytd"]
    assert team_ytd["R"] == 120.0
    assert team_ytd["HR"] == 30.0
    assert team_ytd["K"] == 300.0
    assert team_ytd["AB"] == 800.0
    assert team_ytd["BB_plus_H_allowed"] == 360.0


def test_standings_route_team_ytd_absent_when_legacy_payload(client):
    """Backwards-compat: legacy KV blobs written before the team_ytd
    field landed lack the block. The route must still render (default
    to an empty dict on read) instead of crashing the standings page.
    """
    from fantasy_baseball.web.season_data import CacheKey

    legacy_payload = {
        "effective_date": "2026-05-29",
        "teams": {
            "Hart of the Order": {
                "team_name": "Hart of the Order",
                "hitters": [],
                "pitchers": [],
                # No team_ytd key -- mimics a stale blob from before the
                # team-YTD refactor.
            }
        },
    }

    def fake_read_cache_dict(key):
        if key == CacheKey.STANDINGS_BREAKDOWN:
            return legacy_payload
        if key == CacheKey.STANDINGS:
            return _mock_standings()
        return None

    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict") as m,
        patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg,
    ):
        mock_cfg.return_value.team_name = "Hart of the Order"
        m.side_effect = fake_read_cache_dict
        response = client.get("/standings")

    assert response.status_code == 200
    body = response.get_data(as_text=True)

    match = re.search(
        r'<script[^>]*id="breakdown-data"[^>]*>(.*?)</script>',
        body,
        re.DOTALL,
    )
    assert match, "Expected breakdown-data script tag in standings.html output"
    breakdown_json = json.loads(match.group(1).strip())

    team_payload = breakdown_json["teams"]["Hart of the Order"]
    # from_dict defaults missing team_ytd to {} so the modal can still
    # render the row (zero values) instead of crashing on undefined.
    assert team_payload.get("team_ytd") == {}
