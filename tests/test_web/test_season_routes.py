from unittest.mock import patch

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.web.season_app import create_app
from fantasy_baseball.web.season_data import CacheKey


@pytest.fixture
def client():
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


def test_unauthed_api_returns_json_401_not_redirect(client):
    """Unauthenticated /api/* GETs must return JSON 401, not redirect to /login.

    The mobile lineup page relies on `r.json()` to read errors; a 302 to the HTML
    login page makes JSON parsing fail with a confusing browser-specific error.
    """
    resp = client.get("/api/opponent/mlb.l.1.t.1/lineup")
    assert resp.status_code == 401
    assert resp.content_type.startswith("application/json")
    assert resp.get_json() == {"error": "Authentication required"}


def test_unauthed_html_page_still_redirects_to_login(client):
    """Non-API routes should still redirect to /login when unauthenticated."""
    resp = client.get("/logs")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_logs_page_renders(client):
    with client.session_transaction() as sess:
        sess["authenticated"] = True
    with patch("fantasy_baseball.web.job_logger._get_redis", return_value=None):
        resp = client.get("/logs")
    assert resp.status_code == 200
    assert b"Job Logs" in resp.data


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
            "with_management": {"team_results": {}, "category_risk": {}},
            "baseline_meta": {
                "frozen_at": "2026-04-17T00:00:00Z",
                "roster_date": "2026-03-27",
                "season_year": 2026,
            },
            "rest_of_season": None,
            "rest_of_season_with_management": None,
        },
    )
    season_data.write_cache(CacheKey.STANDINGS, _mock_standings())

    with patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cfg.return_value.team_name = "Team 01"

        resp = client.get("/standings")
        assert resp.status_code == 200
        assert b"2026-03-27" in resp.data


# --- /api/sync-from-remote ----------------------------------------------------


def _auth(client):
    with client.session_transaction() as sess:
        sess["authenticated"] = True


def test_sync_from_remote_requires_auth(client):
    """Unauth'd requests return 401 (matches other write endpoints)."""
    resp = client.post("/api/sync-from-remote")
    assert resp.status_code == 401


def test_sync_from_remote_rejects_on_render(client, monkeypatch):
    """On Render, the Upstash KV is authoritative — sync would be a no-op
    at best and destructive at worst. The endpoint refuses with 400."""
    _auth(client)
    monkeypatch.setattr("fantasy_baseball.web.season_routes.is_remote", lambda: True, raising=False)
    # is_remote is imported lazily inside the route, so patch the source.
    monkeypatch.setattr("fantasy_baseball.data.kv_store.is_remote", lambda: True)
    resp = client.post("/api/sync-from-remote")
    assert resp.status_code == 400
    assert "local-only" in resp.get_json()["error"]


def test_sync_from_remote_calls_sync_helper(client, monkeypatch):
    """Off-Render, the endpoint invokes ``sync_remote_to_local`` and
    returns its summary."""
    _auth(client)

    from fantasy_baseball.data import kv_sync

    calls = []

    class _FakeStats:
        def summary(self):
            return "10 string keys, 0 hash keys (0 fields)"

    def fake_sync():
        calls.append(1)
        return _FakeStats()

    monkeypatch.setattr(kv_sync, "sync_remote_to_local", fake_sync)
    resp = client.post("/api/sync-from-remote")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert "10 string keys" in body["summary"]
    assert calls == [1]


def test_sync_from_remote_surfaces_errors_as_500(client, monkeypatch):
    """If the underlying sync helper raises, the endpoint returns 500
    with the error message in the body — better than a stack trace."""
    _auth(client)

    from fantasy_baseball.data import kv_sync

    def boom():
        raise RuntimeError("Upstash creds missing")

    monkeypatch.setattr(kv_sync, "sync_remote_to_local", boom)
    resp = client.post("/api/sync-from-remote")
    assert resp.status_code == 500
    assert "Upstash creds missing" in resp.get_json()["error"]
