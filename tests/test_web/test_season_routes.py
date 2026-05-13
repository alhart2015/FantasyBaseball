import json
from unittest.mock import patch

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import redis_key
from fantasy_baseball.data.kv_store import get_kv
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
