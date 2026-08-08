import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import redis_key
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.web.season_app import create_app
from fantasy_baseball.web.season_data import CacheKey
from fantasy_baseball.web.trajectory_view import VIEWS, filter_state

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    # Continuation probability is surfaced on the chip label, not just the tooltip.
    assert "62%" in body


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
    sgp_overrides: ClassVar[dict[str, float]] = {}


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
    # Route now returns the two-scenario envelope; the plan shape lives under as_projected.
    assert set(data.keys()) >= {"as_projected", "if_healthy", "adjusted", "tops_differ"}
    ap = data["as_projected"]
    assert set(ap.keys()) >= {"activating", "capacity", "overflow", "plans"}
    assert isinstance(ap["plans"], list)
    assert isinstance(ap["capacity"], int)
    # The IL-slot player is the one activated.
    assert ap["activating"] == ["IL Guy"]


def test_il_return_plan_returns_scenario_envelope(client):
    from unittest.mock import patch

    from fantasy_baseball.lineup.il_return_planner import (
        IlReturnPlanResult,
        IlReturnScenarios,
    )

    scenarios = IlReturnScenarios(
        as_projected=IlReturnPlanResult(activating=["Cruz"], capacity=23, overflow=1, plans=[]),
        if_healthy=IlReturnPlanResult(activating=["Cruz"], capacity=23, overflow=1, plans=[]),
        adjusted=[
            {
                "name": "Cruz",
                "player_type": "hitter",
                "vol_unit": "PA",
                "vol_projected": 175.0,
                "vol_healthy": 223.0,
            }
        ],
        tops_differ=True,
    )
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_list",
            return_value=[
                {
                    "name": "Cruz",
                    "player_type": "hitter",
                    "player_id": 11370,
                    "status": "IL10",
                    "selected_position": "IL",
                    "positions": ["OF"],
                }
            ],
        ),
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value={
                "projected_standings": {"teams": []},
                "team_sds": None,
                "fraction_remaining": 0.41,
            },
        ),
        patch("fantasy_baseball.web.season_routes._projected_from_cache", return_value=object()),
        patch("fantasy_baseball.web.season_routes._team_sds_from_cache", return_value=None),
        patch(
            "fantasy_baseball.lineup.il_return_planner.plan_il_returns_scenarios",
            return_value=scenarios,
        ),
    ):
        resp = client.get("/api/il-return-plan?activate=11370")
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"as_projected", "if_healthy", "adjusted", "tops_differ"}
    assert body["tops_differ"] is True
    assert body["adjusted"][0]["vol_unit"] == "PA"


def test_il_return_plan_if_healthy_null_when_no_adjustment(client):
    from unittest.mock import patch

    from fantasy_baseball.lineup.il_return_planner import (
        IlReturnPlanResult,
        IlReturnScenarios,
    )

    scenarios = IlReturnScenarios(
        as_projected=IlReturnPlanResult(activating=["Buxton"], capacity=23, overflow=0, plans=[]),
        if_healthy=None,
    )
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_list",
            return_value=[
                {
                    "name": "Buxton",
                    "player_type": "hitter",
                    "player_id": 9590,
                    "status": "IL10",
                    "selected_position": "BN",
                    "positions": ["OF"],
                }
            ],
        ),
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value={
                "projected_standings": {"teams": []},
                "team_sds": None,
                "fraction_remaining": 0.41,
            },
        ),
        patch("fantasy_baseball.web.season_routes._projected_from_cache", return_value=object()),
        patch("fantasy_baseball.web.season_routes._team_sds_from_cache", return_value=None),
        patch(
            "fantasy_baseball.lineup.il_return_planner.plan_il_returns_scenarios",
            return_value=scenarios,
        ),
    ):
        resp = client.get("/api/il-return-plan?activate=9590")
    assert resp.status_code == 200
    assert resp.get_json()["if_healthy"] is None


def test_roster_audit_page_renders_il_returns_scenario_js(client):
    from unittest.mock import patch

    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None),
        patch("fantasy_baseball.web.season_routes.read_cache_list", return_value=None),
    ):
        resp = client.get("/roster-audit")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    # The dual-scenario client logic is present (static JS is served
    # unconditionally, outside the {% if audit %} guard).
    assert "if_healthy" in html
    assert "renderPlans" in html
    # Both column labels and both headline branches are present, so a deletion
    # of the labels or either headline branch fails CI (the runtime rendering is
    # JS and is verified manually in Step 5).
    assert "As projected" in html
    assert "If healthy" in html
    assert "depends on" in html  # tops_differ == true branch
    assert "Robust:" in html  # tops_differ == false branch


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


def _txn_draft_cache(txn_teams, draft_teams):
    """side_effect for read_cache_dict keyed by CacheKey.value."""
    values = {
        CacheKey.TRANSACTION_ANALYZER.value: {"teams": txn_teams}
        if txn_teams is not None
        else None,
        CacheKey.DRAFT_VALUE.value: {"horizon": "proj", "teams": draft_teams}
        if draft_teams is not None
        else None,
    }

    def _fake(key, *_a, **_k):
        v = values.get(key.value)
        return v if isinstance(v, dict) else None

    return _fake


_DRAFT_TEAM = {
    "team": "Hart of the Order",
    "avg_value": 4.2,
    "sum_value": 58.1,
    "credited_count": 14,
    "players": [
        {
            "name": "Juan Soto",
            "display_name": "Juan Soto",
            "player_type": "hitter",
            "kind": "keeper",
            "slot": None,
            "preseason_var": 38.1,
            "est_var_proj": 44.3,
            "value_proj": 12.3,
            "value_ytd": 3.1,
            "skill": 6.1,
            "luck": 6.2,
        }
    ],
}


def test_transactions_draft_empty_placeholder(client):
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        side_effect=_txn_draft_cache([], None),
    ):
        resp = client.get("/transactions")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "tab-strip" in body  # tab strip present even with empty draft data
    assert "No draft data" in body


def test_transactions_empty_txn_but_populated_draft(client):
    # Post-draft / pre-first-transaction: txn empty, draft populated.
    # Guards the hoist-out-of-conditional restructure AND that both tabs render.
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        side_effect=_txn_draft_cache([], [_DRAFT_TEAM]),
    ):
        resp = client.get("/transactions")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "tab-strip" in body
    assert "Draft Grade" in body  # both tab labels render
    assert "switchTab" in body  # tab JS hoisted, present regardless of txn_data
    assert "toggleTxnDetail" in body  # expand JS hoisted too
    assert "Juan Soto" in body  # draft rows render


def test_trajectory_page_renders_with_a_cold_cache(client):
    """A missing board must SAY so. The page is fed by an offline push script, not by
    the refresh pipeline, so "no board yet" is a normal state and rendering an empty
    table would read as "nobody scored"."""
    with patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=None):
        resp = client.get("/trajectory")
    assert resp.status_code == 200
    assert b"No trajectory board cached yet" in resp.data
    assert b"push_trajectory_board.py" in resp.data


def test_trajectory_page_reports_a_payload_it_cannot_read(client):
    """Rather than 500ing, or worse, reading the point fields off by one.

    The trigger changed on 2026-08-06 (Hart's call): points are named fields now, so
    there is no `PAYLOAD_VERSION` to mismatch -- an unreadable payload is one whose
    points are the OLD positional arrays, which is exactly what prod held when this
    landed. The route-level guarantee is unchanged and is why this test exists: the
    page renders its error, it does not 500 and it does not render a board.
    """
    positional = {
        "base_season": 2026,
        "max_horizon": 3,
        "generated_at": "positional-blob",
        "players": [
            {
                "id": 1,
                "name": "Old Schema",
                "pool": "hitter",
                "age": 27,
                "slot": "OF",
                "floor": 4.0,
                "now": 10.0,
                "prior": 10.0,
                "support": 0.9,
                "extrapolated": 0,
                "sgp": [[1, 8.0, 3.0, 13.0, 50.0, 0]],
            }
        ],
    }
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        return_value=positional,
    ):
        resp = client.get("/trajectory")
    assert resp.status_code == 200
    assert b"push_trajectory_board" in resp.data
    assert b"Old Schema" not in resp.data, "a board was rendered off an unreadable payload"


def test_trajectory_page_renders_a_board(client):

    from fantasy_baseball.trajectory.board import BoardRow
    from fantasy_baseball.trajectory.sweep import sweep_pool, to_payload
    from tests._trajectory_panel import synthetic_panel

    panel = synthetic_panel()
    swept = sweep_pool(
        [BoardRow(1, "Testy McTestface", "hitter", 27, 20.0, 19.0, "OF", 4.0)],
        panel,
        "hitter",
        (1, 2),
    )
    payload = to_payload(swept, base_season=2026, max_horizon=2, generated_at="2026-08-04T09:00:00")

    with patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=payload):
        resp = client.get("/trajectory?end=2028")
    assert resp.status_code == 200
    assert b"Testy McTestface" in resp.data
    # The vintage is load-bearing: this board does not move with a dashboard refresh.
    assert b"2026-08-04T09:00:00" in resp.data
    # Per-year columns appear once the range spans more than one season.
    assert b"'27" in resp.data and b"'28" in resp.data

    # ONE mechanism for the control state. Every control is a URL from `board_url`; the
    # dropdowns used to sit in a <form> that needed a hidden input per filter to carry
    # the ones it did not own, so the state was encoded twice and a filter added to only
    # one of them silently reset whenever a dropdown changed.
    html = resp.data.decode()
    assert 'type="hidden"' not in html, "control state must live in the URL, not in inputs"
    assert "<form" not in html, "no form on this page -- the selects navigate"
    assert "scale=var" in html and "pool=hitter" in html, "controls carry the full state"

    # Prose must not assert one scale's semantics while the other is selected. The Now
    # column is floor-subtracted on VAR and raw on SGP, so where negatives are EXPECTED
    # differs: on VAR they show up in every column, while on SGP the board's own min-SGP
    # cut keeps Now positive and only the band's p10 goes under.
    with patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=payload):
        var_html = client.get("/trajectory?end=2028&scale=var").data.decode()
        sgp_html = client.get("/trajectory?end=2028&scale=sgp").data.decode()
    assert "So negatives are expected on this scale" in var_html
    assert "So negatives are expected on this scale" not in sgp_html, (
        "VAR-only claim leaked into the SGP view"
    )
    assert "Negatives are rarer on this scale but not absent" in sgp_html

    # And neither view may claim VAR is clamped. That claim was true until #331 and is
    # the thing the reader most needs corrected: a below-replacement player now reads a
    # negative rather than rendering identically to a replacement-level one.
    for scale, html_for_scale in (("var", var_html), ("sgp", sgp_html)):
        assert "clamped at zero" not in html_for_scale, (
            f"{scale}: VAR has not been clamped since #331"
        )
    assert "reads negative rather than 0" in var_html


def _trajectory_payload():
    """A two-player board for the route tests.

    The route reads the payload from the cache BEFORE it touches rosters, and
    renders `board=None` when it is absent -- off Render that read hits the
    local SQLite mirror, so without this patch there are no controls on the page
    and every dropdown assertion is answering the wrong question. Same
    construction as `test_trajectory_page_renders_a_board`.

    `min_sgp` is set because `push_trajectory_board.py` always stamps it and the
    "not scored" line prints it -- a fixture without it renders a page no real
    push could produce.
    """
    from fantasy_baseball.trajectory.board import BoardRow
    from fantasy_baseball.trajectory.sweep import sweep_pool, to_payload
    from tests._trajectory_panel import synthetic_panel

    swept = sweep_pool(
        [
            BoardRow(1, "Testy McTestface", "hitter", 27, 20.0, 19.0, "OF", 4.0),
            BoardRow(2, "Someone Else", "hitter", 27, 12.0, 11.0, "OF", 4.0),
        ],
        synthetic_panel(),
        "hitter",
        (1, 2),
    )
    return to_payload(
        swept,
        base_season=2026,
        max_horizon=2,
        min_sgp=2.0,
        generated_at="2026-08-07T09:00:00",
    )


def _trajectory_spots():
    """Names must match the payload's BoardRows or the join yields no teams.

    THE OPPONENT SORTS BEFORE MY OWN TEAM on purpose. It used to be "Zebras", and
    "Hart of the Order" < "Zebras", so the my-team-first assertion below passed
    under plain alphabetical order as well -- hardcoding `promoted = ()` in
    `index_rosters` left it green. "Aardvarks" makes it fail unless promotion
    actually happened.

    "Unpriced Prospect" matches no board row on purpose: he is what the "not
    scored" line exists to render. He is alone on "Nobodies" so that selecting
    that team also pins the line's PLACEMENT -- it sits outside the
    `{% if not board.rows %}` branch precisely so a team with nothing scored
    still gets the list that explains its empty page.
    """
    from fantasy_baseball.data.rosters import RosterSpot

    return [
        RosterSpot("Testy McTestface", "testy mctestface", "hitter", "Aardvarks", "1", ""),
        RosterSpot("Someone Else", "someone else", "hitter", "Hart of the Order", "2", ""),
        RosterSpot("Unpriced Prospect", "unpriced prospect", "hitter", "Nobodies", "3", ""),
    ]


def test_trajectory_page_offers_a_team_dropdown(client):
    """Rendered from live rosters, with my own team promoted to the top."""
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        resp = client.get("/trajectory")
    assert resp.status_code == 200
    assert b"All teams" in resp.data

    # Compare the OPTION positions, not the raw page. "Hart of the Order" is
    # also the site header, so `resp.data.index(b"Hart of the Order")` finds the
    # chrome and the ordering assertion passes however the dropdown is sorted.
    body = resp.data.decode()
    mine = body.index(">Hart of the Order</option>")
    theirs = body.index(">Aardvarks</option>")
    assert mine < theirs, "my own team must lead the dropdown"
    assert body.index(">All teams</option>") < mine, "the default sits above it"


def test_trajectory_page_lists_a_selected_teams_unpriced_players(client):
    """Spec requirement 7, inherited from #322: a rostered player the model could
    not price must be NAMED, not silently missing.

    "Nobodies" holds exactly one such player and nothing else, so this also pins
    where the line renders: a team with zero scored rows still gets it, which is
    the whole reason it sits outside the empty-board branch.
    """
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        resp = client.get("/trajectory?team=Nobodies")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Nothing scored at this timeframe" in body, "no rows, so placement is meaningful"
    # Back to "not scored" and one cause. It briefly read "not shown here" and
    # hedged between two, because a spot losing a name-key contest was filed here
    # too. Ownership is membership now, so a player whose key has a row shows under
    # his team instead of landing here, and the only way onto this list is that the
    # board had no row for him at all.
    assert "not scored:" in body
    assert "Unpriced Prospect" in body
    assert "could not price them" in body


def test_trajectory_page_hides_the_dropdown_when_no_rosters_arrived(client):
    """Empty `spots` -- an unreachable Upstash cannot be told from an empty
    league, so the control is not rendered at all rather than rendered empty.

    The payload IS patched, so a board renders and its other controls appear.
    Without that this test would pass whether or not the dropdown was ever
    built, since it asserts an absence.
    """
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch("fantasy_baseball.data.rosters.live_rosters", return_value=[]),
    ):
        resp = client.get("/trajectory")
    assert resp.status_code == 200
    assert b"Testy McTestface" in resp.data, "a board rendered, so absence is meaningful"
    assert b"All teams" not in resp.data


def test_trajectory_page_survives_a_team_param_with_no_rosters(client):
    """A stale bookmark must not 500."""
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch("fantasy_baseball.data.rosters.live_rosters", return_value=[]),
    ):
        resp = client.get("/trajectory?team=Nobody+FC")
    assert resp.status_code == 200
    assert b"Testy McTestface" in resp.data


@pytest.mark.parametrize("view", VIEWS)
def test_trajectory_controls_carry_every_filter_on_every_link(client, view):
    """`filter_state` is the single place that knows full filter state. A filter it
    names that a link does not emit gets silently reset when any other control is
    used -- the hidden-input failure the macro's own comment records.

    RUN ON EVERY VIEW, because the route builds `cur` ONCE PER VIEW. Covering only
    one is how `team` came to be a hardcoded "all" on the teams branch while its
    neighbours were pass-throughs: the filter survived the link shape check and died
    on the round trip.

    THE LITERAL BELOW IS THE GUARD, and duplicating the list is the point of it. This
    once read `expected = [f"{n}=" for n in filter_state(view, None, {})]`, deriving
    its expectation from the very thing it guards: a key DELETED from `filter_state`
    shrank both sides and the test stayed green. Measured -- deleting `"per"` left all
    173 tests in this file passing, while a reader on /trajectory?view=teams&per=25 who
    clicked a scale pill silently dropped back to 5 rows per team.

    The equality assertion keeps the literal honest in the other direction: a filter
    ADDED to `filter_state` and not to the links is what the derived version did catch,
    and the earlier literal had drifted two behind before it was removed.
    """
    names = [
        "view",
        "end",
        "pool",
        "scale",
        "per",
        "top",
        "team",
        "player",
        "n",
        "pid",
        "ppool",
    ]
    assert sorted(names) == sorted(filter_state(view, None, {})), (
        "the literal above and `filter_state` must name the same filters -- add to both"
    )
    expected = [f"{name}=" for name in names]
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        resp = client.get(
            f"/trajectory?view={view}&end=2028&pool=hitter&scale=sgp&top=25&per=3&team=Aardvarks"
        )
    assert resp.status_code == 200
    body = resp.data.decode()

    # Every control link must carry every one of them, or using one resets the others.
    links = re.findall(r'href="(/trajectory\?[^"]*)"', body)
    assert links, "the control bar rendered no links"
    for link in links:
        for param in expected:
            assert param in link, f"{view}: {param} missing from {link}"


def test_a_trajectory_view_round_trip_keeps_the_team_filter(client):
    """Measured, and the reason the link-shape check above is not enough: on
    /trajectory?team=Aardvarks the "By team" pill carried team=Aardvarks, but the
    "League" pill on the page it landed on came back team=all. `top` and `per` both
    survived the same trip, so the loss was silent and looked like a deliberate reset.

    Asserted in both directions on the SAME filter, so a fix to one branch that
    leaves the other hardcoded still fails here.
    """
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        board = client.get("/trajectory?view=board&team=Aardvarks").data.decode()
        teams = client.get("/trajectory?view=teams&team=Aardvarks").data.decode()

    def pill(body: str, label: str) -> str:
        match = re.search(rf'href="(/trajectory\?[^"]*)">{label}</a>', body)
        assert match, f"no {label} pill rendered"
        return match.group(1)

    assert "team=Aardvarks" in pill(board, "By team"), "board -> teams keeps the filter"
    assert "team=Aardvarks" in pill(teams, "League"), "teams -> board keeps it too"


def test_trajectory_teams_view_renders_a_block_for_every_team_including_empty_ones(client):
    """`_trajectory_spots()` rosters THREE teams -- Aardvarks and Hart of the Order
    each have one scored player, and Nobodies has only "Unpriced Prospect", who the
    board never scored. All three must still render a block: #323's first named
    failure mode is a team whose players were all filtered out vanishing instead of
    showing its unpriced list. `build_teams_board` seeds its block map from the
    ROSTERS (`index.teams`), not from the scored rows, precisely so Nobodies can't
    silently drop out -- this test pins that guarantee at the template layer.
    """
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        resp = client.get("/trajectory?view=teams")
    assert resp.status_code == 200
    body = resp.data.decode()

    # Every block emits `class="team-block` regardless of whether it scored
    # anything -- unlike the "from the best N" summary below, this markup is not
    # conditional on `block.scored`, so counting it pins the BLOCK count (3) rather
    # than the scored-block count (2). This is the assertion that actually backs
    # the test's name.
    assert body.count('class="team-block') == 3, "one block per rostered team"

    # NOT a bare `"Hart of the Order" in body` -- that string is also the site header,
    # so it is present whether or not a single block rendered. Assert on markup only a
    # block emits. (The same trap made a dropdown-ordering test vacuous on #336.)
    assert body.count("from the best 5") == 2, (
        "one summary per SCORED block; Nobodies has none, so this is 2, not 3"
    )
    assert "Aardvarks" in body
    assert "Testy McTestface" in body, "a block's rows render"

    # The empty team (Nobodies) must still appear, with its unpriced list -- the
    # actual content #323 requires -- and must NOT get a summary clause, since
    # "0 of 0 scored" summarizes nothing.
    assert "Nobodies" in body, "the zero-scored team's block still renders"
    assert "Unpriced Prospect" in body, "its unscored list renders"
    assert "0 of 0 scored" not in body, "no summary clause for a team with nothing scored"


def test_trajectory_teams_view_discloses_its_vintage(client):
    """`?view=teams` is a bookmarkable, shareable URL that a trade conversation starts
    from, and the board does NOT move with a dashboard refresh -- it is as fresh as the
    last offline push. The league board says so; this view rendered none of the three
    vintage fields `meta` carries, so it gave no signal the numbers may be weeks old.
    """
    payload = dict(
        _trajectory_payload(),
        generated_at="2026-08-07T09:00:00",
        season_elapsed=0.7,
        panel_vintage={"hitter": "h.csv", "pitcher": "p.csv"},
    )
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=payload),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        resp = client.get("/trajectory?view=teams")
    assert resp.status_code == 200
    body = resp.data.decode()

    assert "2026-08-07T09:00:00" in body, "the build timestamp"
    assert "h.csv / p.csv" in body, "the panels it was fitted on"
    assert "70% complete" in body, "how much of the base season the pacing rests on"
    assert "does not refresh with the dashboard" in body


def test_trajectory_teams_view_keeps_a_flagged_rows_marker(client):
    """The league board marks a row evaluated outside its own support (!) and a
    band-fallback row (!!). The teams view printed both as bare totals -- and those
    totals are summed into `block.total`, which ORDERS the page, so a block could sort
    above its neighbour on precisely the estimates the model is least sure about.

    The flags are set on the payload rather than shaped out of the panel: `extrapolated`
    is per player and `band_fell_back` per year, so setting them directly is the only way
    to reach BOTH branches of the template's elif from one render.
    """
    payload = _trajectory_payload()
    # A distinct vintage: `_ranked_rows` caches on `generated_at` plus the shape, and
    # these mutations change neither -- without this the other route tests' rows are
    # served here, unflagged.
    payload["generated_at"] = "2026-08-07T09:00:00-flagged"
    payload["players"][0]["extrapolated"] = 1
    for year in payload["players"][1]["sgp"]:
        year["band_fell_back"] = 1

    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict", return_value=payload),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        teams = client.get("/trajectory?view=teams").data.decode()
        board = client.get("/trajectory?view=board").data.decode()

    # ON THE MARKUP THE FLAG EMITS, never on a bare "(!)" in the page. Both templates
    # ALSO explain the flags in prose -- `<strong>(!)</strong>` in the teams view's
    # vintage line and in the league board's footer -- so a substring check passes with
    # every flag deleted. Measured: it did. (Same trap as the dropdown-ordering test on
    # #336, where "Hart of the Order" was also the site header.)
    for body, where in ((teams, "teams view"), (board, "league board")):
        assert ">(!)</span>" in body, f"{where}: the thin-support marker"
        assert ">(!!)</span>" in body, f"{where}: the band-fallback marker"
    # Rendered from the constant, not restated as prose -- #310 may move the threshold.
    assert "Under 10% of the fitting weight" in teams


def test_trajectory_defaults_to_the_league_board(client):
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch(
            "fantasy_baseball.data.rosters.live_rosters",
            return_value=_trajectory_spots(),
        ),
    ):
        plain = client.get("/trajectory")
        explicit = client.get("/trajectory?view=board")
        junk = client.get("/trajectory?view=nonsense")
    for resp in (plain, explicit, junk):
        assert resp.status_code == 200
        assert "All teams" in resp.data.decode(), "the league board's team dropdown"


def test_trajectory_teams_view_falls_back_when_no_rosters_arrived(client):
    """A stale bookmark must degrade to the league board, not render an empty page."""
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            return_value=_trajectory_payload(),
        ),
        patch("fantasy_baseball.data.rosters.live_rosters", return_value=[]),
    ):
        resp = client.get("/trajectory?view=teams")
    assert resp.status_code == 200
    assert "Testy McTestface" in resp.data.decode(), "the league board rendered instead"


def _trajectory_chart(payload):
    """The `cache:trajectory_chart_data` blob paired with `payload` (#344).

    A SECOND key, carrying the board's own `generated_at` -- the player view refuses
    extras stamped for a different board, and only that view reads them at all.
    """
    from fantasy_baseball.trajectory.sweep import to_chart_payload

    return to_chart_payload(
        {
            (p["id"], p["pool"]): {
                "history": [[25, 14.0], [26, 16.0]],
                "comps": [
                    {
                        "name": "Andre Ethier",
                        "season": 2007,
                        "rmse": 1.25,
                        "path": [12.7, 14.4, 12.1, 9.2, 12.2],
                    },
                    {
                        "name": "Bryan Reynolds",
                        "season": 2020,
                        "rmse": 1.31,
                        "path": [11.0, 12.0, 11.5, 10.0, 9.5],
                    },
                ],
            }
            for p in payload["players"]
        },
        generated_at=str(payload["generated_at"]),
    )


def _trajectory_reads(board, chart=None, seen=None, *, narrow=False):
    """A cache-reader side effect serving the board and chart keys SEPARATELY.

    `return_value` cannot express this any more: the two trajectory keys hold different
    blobs, and handing the board back for both would put a list where the chart lookup
    expects a mapping. `seen` collects the keys asked for, which is how the board and
    teams tests assert that they never reach for the chart data at all.

    `narrow` reproduces what `read_cache_dict` DOES to a stored list -- collapse it to
    None. Without it the fake is more forgiving than the real reader, and a route
    switched back to `read_cache_dict` for the chart key passes a test written to prove
    it must not be.
    """

    def read(key):
        if seen is not None:
            seen.append(key)
        value = chart if key is CacheKey.TRAJECTORY_CHART_DATA else board
        return None if narrow and not isinstance(value, dict) else value

    return read


@contextmanager
def _trajectory_cache(board, chart=None, seen=None):
    """Patch BOTH cache readers the trajectory route uses, off one dispatcher.

    The board comes through `read_cache_dict` and the chart through `read_cache` -- the
    latter because the narrowing reader collapses a stored list to None, which the page
    would then report as "no chart data" rather than as a blob it cannot read. Each is
    faked with its OWN narrowing behaviour, so which reader the route picked is
    observable here. Patching only one would let the other reach the real KV, where a
    locally pushed board (or none at all) decides the test.
    """
    with (
        patch(
            "fantasy_baseball.web.season_routes.read_cache_dict",
            side_effect=_trajectory_reads(board, chart, seen, narrow=True),
        ),
        patch(
            "fantasy_baseball.web.season_routes.read_cache",
            side_effect=_trajectory_reads(board, chart, seen),
        ),
    ):
        yield


def _trajectory_board_and_chart():
    """The route fixture and the chart blob that pairs with it."""
    payload = _trajectory_payload()
    return payload, _trajectory_chart(payload)


def test_trajectory_player_view_renders_a_chart_for_a_resolved_name(client):
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "trajectory-chart" in body, "the canvas the chart draws into"
    assert "closest realized paths" in body, "labelled as illustration, not evidence"
    # The TABLE markup, not the `#trajectory-chart-data` JSON island -- that island
    # also serializes `board.comps` verbatim, so a plain substring match on "Andre
    # Ethier"/"1.25" is satisfied by the JSON alone and stays green even if the
    # honesty table renders nothing.
    assert "<td>Andre Ethier</td>" in body, "comps are named in the table"
    assert "<td>1.25</td>" in body, "each comp shows its RMSE in the table"


def test_trajectory_player_view_states_the_five_year_comp_rule(client):
    """A reader who notices no comp is recent must find the rule, not infer a bug."""
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert "five realized seasons" in resp.data.decode()


def test_trajectory_player_view_with_no_name_renders_the_search_box(client):
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player")
    assert resp.status_code == 200
    assert 'name="player"' in resp.data.decode(), "the search input"


def test_trajectory_player_view_unknown_name_does_not_500(client):
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Nobody+At+All")
    assert resp.status_code == 200
    assert "No player named" in resp.data.decode()


def test_trajectory_player_view_degrades_on_a_legacy_positional_points_blob(client):
    """A pre-#332 blob storing points positionally must degrade, not 500.

    That shape is what is deployed in production today, and the guard against it lives
    in `_unpack` -- ONE guard, because since 757fb9fc there is one reader: the player
    view goes through `player_from_row` instead of unpacking `row["sgp"]` itself. This
    docstring used to describe that deleted second reader and told a future editor the
    duplication was deliberate. What it guards now is that the single guard's
    `ValueError` still reaches this page as a degraded banner rather than a 500 -- the
    route catches `ValueError`, so a change that let a `TypeError` escape instead
    would break exactly here."""
    payload = _trajectory_payload()
    payload["players"] = [
        {**p, "sgp": [[1, 14.0, 10.0, 18.0, 100.0, 0], [2, 15.0, 11.0, 19.0, 90.0, 0]]}
        for p in payload["players"]
    ]
    with _trajectory_cache(payload):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert resp.status_code == 200
    assert "re-run scripts/push_trajectory_board.py" in resp.data.decode()


def test_trajectory_player_view_discloses_the_var_netting_on_axis_and_table(client):
    """Spec requirement 6: on the VAR scale every series is netted against the
    searched player's OWN slot floor, and the axis label must say so -- not just
    repeat "VAR" as if that were self-explanatory. The numbers table's header is the
    no-JS fallback for the same disclosure (#324 F2).

    ONE STRING, BOTH SURFACES. The chart's y-axis title and this header used to be
    built independently -- a Jinja `{% set %}` and a JS template literal, each
    interpolating `floor` -- so they could disagree about a subtraction. Asserting they
    are equal is what makes `PlayerView.axis_label` the only place the rule lives.
    """
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface&scale=var")
    body = resp.data.decode()
    # Testy McTestface's slot floor is 4.0 -- see `_trajectory_payload`'s BoardRow.
    assert "<th>VAR (SGP - 4.00 slot floor)</th>" in body
    assert _chart_island(body)["axis_label"] == "VAR (SGP - 4.00 slot floor)"
    assert "netted against" in body, "the comp caption names the rule too"


def test_trajectory_player_view_sgp_scale_keeps_a_plain_label(client):
    """Nothing is netted on the SGP scale (`board.floor` is 0.0 there) -- the label
    must not claim a subtraction that did not happen."""
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface&scale=sgp")
    body = resp.data.decode()
    assert "<th>SGP</th>" in body
    assert _chart_island(body)["axis_label"] == "SGP", "the chart says the same thing"
    assert "slot floor" not in body


def test_trajectory_player_view_discloses_vintage_and_the_history_gap(client):
    """Sibling templates (trajectory.html, trajectory_teams.html) both print the
    build vintage / pace note; this one printed neither. Also explains why the solid
    line stops a year before the dashed one starts (#324 F3)."""
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    body = resp.data.decode()
    assert "Built 2026-08-07T09:00:00" in body, "the same vintage stamp the sibling views print"
    assert "still in progress" in body, "explains the gap between history and projection"


def test_trajectory_player_view_explains_a_missing_chart_key(client):
    """A board with no `cache:trajectory_chart_data` beside it -- the shape currently
    deployed in production, and the shape between a reader deploy and the first push
    that writes the new key. The page must say what's missing rather than rendering an
    empty comps table as if the model scored zero comps.

    NOT the mismatch note: nothing arrived, so nothing disagrees with the board, and the
    fix is "push it", not "the two blobs are out of step"."""
    with _trajectory_cache(_trajectory_payload()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "trajectory-chart" in body, "the projection still renders regardless"
    assert body.count("predates") == 2, "one note for the missing comps, one for history"
    assert "different build" not in body, "nothing arrived, so nothing can mismatch"
    comps_section = body[body.index("Closest realized paths") : body.index("The numbers")]
    assert "<td>" not in comps_section, "no fabricated comp rows"


def test_trajectory_player_view_refuses_chart_data_from_another_build(client):
    """The failure mode the split creates (#344). Two keys can be refreshed
    independently, so a board from noon can sit beside extras from Tuesday -- a stale
    career line under a fresh projection, silent, with both halves plausible.

    Asserts the chart is NOT DRAWN, not merely that a note appeared: everything the
    chart is handed goes through the JSON island, so an implementation that printed the
    warning and drew the stale line anyway fails here.
    """
    board, chart = _trajectory_board_and_chart()
    chart["generated_at"] = "2020-01-01T00:00:00-05:00"
    with _trajectory_cache(board, chart):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert resp.status_code == 200
    body = resp.data.decode()

    island = _chart_island(body)
    assert island["history"] == [], "a career line from another build must not be drawn"
    assert island["comps"] == [], "nor its comps"
    assert island["projection"], "the board's own fit is unaffected"
    assert "<td>Andre Ethier</td>" not in body, "and no comp table rows either"

    assert "different build" in body, "the mismatch has its own explanation"
    assert "predates" not in body, "which is NOT the predates-the-feature note"


@pytest.mark.parametrize("shape", ["board-under-the-chart-key", "top-level-list"])
def test_trajectory_player_view_survives_a_chart_blob_it_cannot_read(client, shape):
    """A foreign shape must degrade to the mismatch note, never to a 500.

    The board written to the chart key is the reachable case -- one push produces both,
    so the stamps AGREE and the vintage check waves it through to `players.get(...)`,
    which a list does not have. The route catches `(ValueError, KeyError)` only, so an
    unguarded `AttributeError` takes the whole page down while the projection it would
    have rendered is sitting right there in the board blob.

    The top-level list is why the chart key is read with `read_cache`: `read_cache_dict`
    narrows a stored list to None, which is indistinguishable from a key that was never
    written, and the page would then blame the board for predating the feature.
    """
    board, _ = _trajectory_board_and_chart()
    foreign = (
        {"generated_at": board["generated_at"], "players": board["players"]}
        if shape == "board-under-the-chart-key"
        else board["players"]
    )
    with _trajectory_cache(board, foreign):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")

    assert resp.status_code == 200, "an unreadable chart blob must not 500 the page"
    body = resp.data.decode()
    assert _chart_island(body)["projection"], "the board's own fit still renders"
    assert _chart_island(body)["history"] == []
    assert "different build" in body, "reported as out of step, not as a missing feature"
    assert "predates" not in body


def _two_way_trajectory_payload():
    """The route fixture with `Testy McTestface` carried by a hitter row AND a pitcher
    row -- the live board's Shohei Ohtani shape, where the two rows share an id and an
    age and differ only by slot and pool. Returns the board and its paired chart blob,
    which is keyed `(id, pool)` and so carries a separate entry for each of the two."""
    payload = _trajectory_payload()
    hitter = payload["players"][0]
    payload["players"] = [
        *payload["players"],
        {**hitter, "pool": "pitcher", "slot": "SP", "floor": 3.0},
    ]
    return payload, _trajectory_chart(payload)


def test_trajectory_player_candidates_are_links_that_resolve_the_ambiguity(client):
    """An ambiguous name was a PERMANENT dead end: the candidates rendered as plain
    `<li>` text and the search form's only input was `player`, so no query string could
    pick one. On the live board this fires for Ohtani, whose two rows share id 660271
    AND age 31 -- the list offered two lines identical in the field it named.

    Two narrowing keys, deliberately NOT the board's `pool` filter: `ppool` separates a
    two-way player, `pid` separates same-pool namesakes (the live board has two hitters
    named Max Muncy, sharing neither).
    """
    with _trajectory_cache(*_two_way_trajectory_payload()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert resp.status_code == 200
    body = resp.data.decode()

    candidates = body[body.index("More than one player") : body.index("</ul>")]
    assert "pitcher" in candidates and "hitter" in candidates, "the pool is the discriminator"
    links = re.findall(r'href="(/trajectory\?[^"]*)"', candidates)
    assert links, "a candidate a reader cannot click is a dead end"
    assert all("pid=" in link and "ppool=" in link for link in links)

    picked = next(link for link in links if "ppool=pitcher" in link)
    with _trajectory_cache(*_two_way_trajectory_payload()):
        resolved = client.get(picked.replace("&amp;", "&"))
    assert resolved.status_code == 200
    resolved_body = resolved.data.decode()
    assert "More than one player" not in resolved_body, "the pick resolved it"
    assert "trajectory-chart" in resolved_body
    assert ", SP." in resolved_body, "the PITCHER row, not the hitter one"


def test_trajectory_player_narrowing_survives_a_control_click(client):
    """`pid`/`ppool` are in `filter_state`, so every control link carries them the way
    `player` and `n` do. Without that, toggling the scale on a resolved two-way player
    drops him straight back to the candidate list."""
    with _trajectory_cache(*_two_way_trajectory_payload()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface&ppool=pitcher")
    body = resp.data.decode()
    links = re.findall(r'href="(/trajectory\?[^"]*)"', body)
    assert links
    for link in links:
        assert "ppool=pitcher" in link, f"narrowing dropped from {link}"


def test_trajectory_player_view_names_the_fix_for_a_row_missing_a_field(client):
    """A one-word banner is not an error message.

    `player_from_row` requires `id`, `pool`, `now`, `prior`, `support` and
    `extrapolated`, none of which this view uses. A stale blob missing one raised
    `KeyError('now')`, the route caught it, and `str(exc)` rendered a red banner reading
    literally `'now'` -- no indication the payload was the problem. `_unpack` already
    gives the positional-blob case an actionable sentence; this is the same one.
    """
    payload = _trajectory_payload()
    payload["players"] = [{k: v for k, v in p.items() if k != "now"} for p in payload["players"]]
    with _trajectory_cache(payload):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "re-run scripts/push_trajectory_board.py" in body, "say what to do"
    # Jinja escapes the quotes around the field name, hence the entities.
    assert "missing &#39;now&#39;" in body, "and still name the field that is missing"


def test_trajectory_player_view_offers_the_by_team_pill(client):
    """The player view was a one-way door: `_trajectory_controls.html` gates the "By
    team" pill on `{% if teams %}` and this template passed `[]`, so there was no link
    back to the teams view.

    It passes `[]` because the player branch deliberately SKIPS the `live_rosters()`
    read -- a real per-request Yahoo call `build_player_view` has no use for. That skip
    stays. The macro was conflating two questions: does the teams view exist (what the
    pill needs) and which teams populate the dropdown (what the list is for).
    """
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "view=teams" in body, "no link back to the teams view"
    # And the dropdown the team LIST feeds stays absent -- the skip is what makes the
    # player view cheap, and this is what proves the two questions were separated
    # rather than the roster read quietly re-added.
    assert "All teams" not in body


def _chart_island(body: str) -> dict:
    """The player page's `#trajectory-chart-data` JSON island, parsed.

    Everything the chart is given comes through here, so a test asserting what the
    chart draws asserts this rather than the JS -- there is no JS runtime in this suite.
    """
    match = re.search(
        r'<script type="application/json" id="trajectory-chart-data">(.*?)</script>',
        body,
        re.DOTALL,
    )
    assert match, "the chart's data island"
    return json.loads(match.group(1))


def _trajectory_chart_js_source() -> str:
    return (
        PROJECT_ROOT / "src" / "fantasy_baseball" / "web" / "static" / "trajectory_chart.js"
    ).read_text()


def test_trajectory_chart_js_disables_the_default_aspect_ratio():
    """No JS runtime lives in this suite, so this is a source-text assertion, not a
    faked browser check -- honest per #324 F8's instruction not to write one that
    cannot fail.

    Chart.js's OWN defaults (`responsive: true, maintainAspectRatio: true`, ratio 2)
    ignore `.chart-wrapper`'s fixed 360px box (season.css) and draw the canvas at
    roughly twice its height, painting over the honesty paragraph beneath it
    (#324 F1). `season_trends.js`'s `buildChart` sets this same pair for the same
    box; `trajectory_chart.js` must match it rather than inherit the default."""
    src = _trajectory_chart_js_source()
    # Anchored on the option lines themselves, not a bare substring: the comment
    # above quotes `responsive: true, maintainAspectRatio: true` to explain the
    # defaults, so a substring check is satisfied by the prose that documents the
    # BROKEN form and stays green with the real option deleted.
    assert re.search(r"^\s*maintainAspectRatio: false,$", src, re.M)
    assert re.search(r"^\s*responsive: true,$", src, re.M)


def test_trajectory_chart_js_discloses_the_var_netting_on_the_axis():
    """Spec requirement 6: the y-axis title must say what was subtracted on the VAR
    scale, not just repeat the scale name (#324 F2).

    The label is now built ONCE, server-side, as `PlayerView.axis_label`, and this
    file reads the finished string -- so what has to be pinned here is that the axis
    uses it, and the SERVER's rule is pinned where it now lives (see
    `test_trajectory_player_view_discloses_the_var_netting_on_axis_and_table`, which
    asserts the chart's island and the table header carry the SAME string). It
    previously asserted the JS template literal that built a second copy of the same
    rule, which no longer exists.

    Both halves matter: without the `data.axis_label` read the chart is rebuilding
    the label locally again, and without `title: { display: true` Chart.js draws no
    y-axis title at all and the disclosure is silently gone.
    """
    src = _trajectory_chart_js_source()
    assert re.search(r"y: \{ title: \{ display: true, text: data\.axis_label \} \}", src)
    # And nothing left re-deriving it: the island no longer carries either input.
    assert "data.floor" not in src
    assert "data.scale" not in src


def test_trajectory_chart_js_filters_the_internal_p10_series_from_tooltips_too():
    """`_p10` is the internal fill-target dataset, already hidden from the legend by
    label; Chart.js's default tooltip has no such filter, so a hover near the lower
    band edge would otherwise show a series literally called "_p10" (#324 F6).

    Matched on the FILTER, not on the one-line spelling `tooltip: { filter:` this used
    to assert. That literal broke the moment the tooltip grew a second key (#346's
    paced-point label) -- a formatting fact about a block, not the rule being pinned.
    """
    src = _trajectory_chart_js_source()
    assert re.search(r"tooltip:\s*\{\s*filter:", src)
    assert 'item.dataset.label !== "_p10"' in src


def test_the_player_page_ships_the_paced_point_to_the_chart(client):
    """The gap at the base season is the most useful point on the chart. It has to
    reach the JS island, not just the view model -- and it comes off the BOARD's `now`,
    so it is there even though the chart blob's history stops a year earlier."""
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    body = resp.data.decode()
    island = _chart_island(body)

    assert island["paced"], "the island carries the point"
    assert island["paced_label"] == "2026 pace", "and says what it is"
    assert island["paced"][0] > max(pt[0] for pt in island["history"]), (
        "the paced season is the career line's last point"
    )
    assert "pace</td>" in body, "the numbers table marks the row"


def test_trajectory_player_chart_data_is_truncated_to_the_projected_horizons(client):
    """The fixture's comp paths are stored 5 long; `_trajectory_payload` sweeps 2
    horizons. The route must serve what `build_player_view` already truncated, not
    the raw stored path -- a page showing 5 comp points against a 2-point projection
    would draw off the end of the chart's x-axis."""
    with _trajectory_cache(*_trajectory_board_and_chart()):
        resp = client.get("/trajectory?view=player&player=Testy+McTestface")
    chart_data = _chart_island(resp.data.decode())
    assert len(chart_data["projection"]) == 2, "the fixture sweeps 2 horizons"
    assert len(chart_data["comps"][0]["path"]) == 2, "not the fixture's stored 5"


def test_trajectory_player_view_ambiguous_name_renders_no_chart(client):
    """Two players sharing a normalized name must not silently pick one -- the
    disambiguation list renders instead of a chart for either man's career."""
    payload = _trajectory_payload()
    first = payload["players"][0]
    payload["players"] = [*payload["players"], {**first, "id": first["id"] + 10_000}]
    with _trajectory_cache(payload, _trajectory_chart(payload)):
        resp = client.get(f"/trajectory?view=player&player={first['name'].replace(' ', '+')}")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "trajectory-chart" not in body, "an ambiguous name must not render a chart"
    assert "More than one player is named" in body


def test_trajectory_end_and_pool_survive_a_round_trip_through_the_player_view(client):
    """`filter_state` gives `end_year`/`pool` to `PlayerView` the same treatment as
    `top`/`team`: a pass-through from the query string, not a hardcoded default. A
    League -> Player -> League round trip must land back on the same timeframe and
    pool, not silently reset to `end_years[0]`/"both" -- the literal-`"all"` bug this
    module's docstring already names, one field over.
    """
    with _trajectory_cache(_trajectory_payload()):
        resp = client.get("/trajectory?view=player&end=2028&pool=pitcher")
    assert resp.status_code == 200
    body = resp.data.decode()
    # The League pill's own link is the round trip: it must carry the values
    # forward rather than resetting them.
    league_href = re.search(r'href="([^"]*)">League</a>', body).group(1)
    assert "end=2028" in league_href
    assert "pool=pitcher" in league_href


def test_trajectory_search_form_carries_the_filters_it_passes_through(client):
    """The search form is the other round trip. It is a GET form, so every filter it
    omits is silently reset on submit -- the exact failure `_trajectory_controls.html`'s
    docstring names ("missing an input silently reset that filter"). The pass-through in
    `filter_state` is worth nothing if searching a second player drops the state on the
    way out.
    """
    with _trajectory_cache(_trajectory_payload()):
        resp = client.get("/trajectory?view=player&end=2028&pool=pitcher&top=25&team=Aardvarks")
    assert resp.status_code == 200
    form = re.search(
        r'<form method="get" class="trajectory-search">(.*?)</form>',
        resp.data.decode(),
        re.S,
    ).group(1)
    for name, value in (
        ("end", "2028"),
        ("pool", "pitcher"),
        ("top", "25"),
        ("team", "Aardvarks"),
    ):
        assert f'name="{name}" value="{value}"' in form, f"search drops {name}"


def test_trajectory_player_view_hides_the_inert_through_and_pool_controls(client):
    """The "Through" dropdown and the pool pills do nothing on the player view --
    `build_player_view` takes no `end` and searches one resolved name, not a pool.
    Offering them invites a reader to believe they filter something."""
    with _trajectory_cache(_trajectory_payload()):
        resp = client.get("/trajectory?view=player")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "<label>Through" not in body
    assert ">Hitters<" not in body and ">Pitchers<" not in body


def test_trajectory_player_view_renders_no_per_team_selector(client):
    """The Top/Team/Per-team block is a three-way branch now; a bare `else` would
    show the teams view's "Per team" selector on the player page, which has no
    per-team concept at all."""
    with _trajectory_cache(_trajectory_payload()):
        resp = client.get("/trajectory?view=player")
    assert resp.status_code == 200
    assert "Per team" not in resp.data.decode()


def test_trajectory_player_and_n_pass_through_on_the_league_board(client):
    """`filter_state`'s `player`/`n` fields are the player view's own pass-through
    story in reverse: on every OTHER view they must come from the query string, so a
    search in progress survives a trip to League and back."""
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        return_value=_trajectory_payload(),
    ):
        resp = client.get("/trajectory?player=Testy+McTestface&n=7")
    assert resp.status_code == 200
    body = resp.data.decode()
    player_href = re.search(r'href="([^"]*)">Player</a>', body).group(1)
    assert "player=Testy" in player_href
    assert "n=7" in player_href


def test_the_three_trajectory_views_coexist(client):
    """Each renders its own thing, and the other two are still reachable."""
    with (
        _trajectory_cache(*_trajectory_board_and_chart()),
        patch("fantasy_baseball.data.rosters.live_rosters", return_value=_trajectory_spots()),
    ):
        board = client.get("/trajectory")
        teams = client.get("/trajectory?view=teams")
        player = client.get("/trajectory?view=player&player=Testy+McTestface")
    assert all(r.status_code == 200 for r in (board, teams, player))
    assert "All teams" in board.data.decode()
    assert "team-block" in teams.data.decode()
    assert "trajectory-chart" in player.data.decode()


@pytest.mark.parametrize("url", ["/trajectory", "/trajectory?view=teams"])
def test_the_default_views_never_read_the_chart_data_key(client, url):
    """THE POINT OF THE SPLIT (#344). History and comps left the board because only the
    player view renders them; a board or teams request that still reached for them would
    have moved ~1.1 MB of Upstash egress and a JSON parse, not removed it.

    Asserts on WHICH KEYS ARE READ, not on the output: both views render identically
    whether or not the extra read happened, so output can never catch the regression.
    """
    board, chart = _trajectory_board_and_chart()
    seen: list = []
    with (
        _trajectory_cache(board, chart, seen),
        patch("fantasy_baseball.data.rosters.live_rosters", return_value=_trajectory_spots()),
    ):
        resp = client.get(url)
    assert resp.status_code == 200
    assert CacheKey.TRAJECTORY_BOARD in seen, "the board itself is still read"
    assert CacheKey.TRAJECTORY_CHART_DATA not in seen

    # And the player view DOES read it -- otherwise this test passes on a route that
    # never reads the key at all, and the chart would silently be gone.
    seen.clear()
    with _trajectory_cache(board, chart, seen):
        client.get("/trajectory?view=player&player=Testy+McTestface")
    assert CacheKey.TRAJECTORY_CHART_DATA in seen


# `test_the_stored_and_displayed_comp_ceilings_agree` was here: it asserted the view's
# clamp ceiling equalled the push script's stored count. Both are now the one
# `comp_paths.MAX_COMPS`, so the parity it policed is structural and there is nothing
# left for the two to drift apart on.
