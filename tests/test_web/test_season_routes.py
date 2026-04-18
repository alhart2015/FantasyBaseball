import pytest
from unittest.mock import patch

from fantasy_baseball.web.season_app import create_app
from fantasy_baseball.web.season_data import CacheKey


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_redirects_to_standings(client):
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/standings" in resp.headers["Location"]


def test_standings_page_renders(client):
    resp = client.get("/standings")
    assert resp.status_code == 200
    assert b"Standings" in resp.data


def test_lineup_page_renders(client):
    resp = client.get("/lineup")
    assert resp.status_code == 200
    assert b"Lineup" in resp.data


def test_trades_page_renders(client):
    resp = client.get("/waivers-trades")
    assert resp.status_code == 200
    assert b"Trades" in resp.data


def test_players_page_renders(client):
    resp = client.get("/players")
    assert resp.status_code == 200
    assert b"pos-filter" in resp.data


def test_sidebar_nav_links_present(client):
    resp = client.get("/standings")
    html = resp.data.decode()
    assert 'href="/standings"' in html
    assert 'href="/lineup"' in html
    assert 'href="/waivers-trades"' in html


def test_active_page_highlighted(client):
    resp = client.get("/standings")
    html = resp.data.decode()
    assert 'active' in html


def _mock_standings():
    teams = [
        ("Hart of the Order", {"R": 300, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
                               "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}),
        ("SkeleThor", {"R": 310, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
                       "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}),
    ]
    return [{"name": n, "team_key": f"key_{i}", "rank": i + 1, "stats": s}
            for i, (n, s) in enumerate(teams)]


def test_standings_renders_table_with_data(client):
    with patch("fantasy_baseball.web.season_routes.read_cache") as mock_cache, \
         patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cache.side_effect = lambda k: _mock_standings() if k == "standings" else {}
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


def test_logs_page_renders(client):
    with client.session_transaction() as sess:
        sess["authenticated"] = True
    with patch("fantasy_baseball.web.job_logger._get_redis", return_value=None):
        resp = client.get("/logs")
    assert resp.status_code == 200
    assert b"Job Logs" in resp.data


def test_full_standings_page_with_cached_data(client, tmp_path):
    """Integration test: standings page renders correctly with all cached data present."""
    from fantasy_baseball.web import season_data

    old_cache_dir = season_data.CACHE_DIR
    season_data.CACHE_DIR = tmp_path

    try:
        standings = [
            {"name": "Hart of the Order", "team_key": "k1", "rank": 1,
             "stats": {"R": 300, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
                       "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}},
            {"name": "SkeleThor", "team_key": "k2", "rank": 2,
             "stats": {"R": 310, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
                       "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}},
        ]
        season_data.write_cache(CacheKey.STANDINGS, standings, tmp_path)
        season_data.write_cache(CacheKey.META, {"last_refresh": "8:32 AM", "week": "3"}, tmp_path)

        with patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc, \
             patch("fantasy_baseball.web.season_routes.read_meta") as mock_rm, \
             patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
            mock_rc.side_effect = lambda k: season_data.read_cache(k, tmp_path)
            mock_rm.return_value = season_data.read_meta(tmp_path)
            mock_cfg.return_value.team_name = "Hart of the Order"

            resp = client.get("/standings")
            assert resp.status_code == 200
            html = resp.data.decode()
            assert "Hart of the Order" in html
            assert "SkeleThor" in html
            assert "8:32 AM" in html
    finally:
        season_data.CACHE_DIR = old_cache_dir


def test_full_lineup_page_with_cached_data(client, tmp_path):
    """Integration test: lineup page renders with cached roster data."""
    from fantasy_baseball.web import season_data

    old_cache_dir = season_data.CACHE_DIR
    season_data.CACHE_DIR = tmp_path

    try:
        roster = [
            {"name": "Adley Rutschman", "positions": ["C"], "selected_position": "C",
             "player_id": "123", "status": ""},
            {"name": "Corbin Burnes", "positions": ["SP"], "selected_position": "P",
             "player_id": "456", "status": ""},
        ]
        optimal = {"hitters": {}, "pitchers": {}, "moves": []}
        season_data.write_cache(CacheKey.ROSTER, roster, tmp_path)
        season_data.write_cache(CacheKey.LINEUP_OPTIMAL, optimal, tmp_path)
        season_data.write_cache(CacheKey.META, {"last_refresh": "9:00 AM"}, tmp_path)

        with patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc, \
             patch("fantasy_baseball.web.season_routes.read_meta") as mock_rm:
            mock_rc.side_effect = lambda k: season_data.read_cache(k, tmp_path)
            mock_rm.return_value = season_data.read_meta(tmp_path)

            resp = client.get("/lineup")
            assert resp.status_code == 200
            html = resp.data.decode()
            assert "Adley Rutschman" in html
            assert "Corbin Burnes" in html
            assert "Optimal" in html  # should show optimal button since no moves
    finally:
        season_data.CACHE_DIR = old_cache_dir


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


def test_standings_passes_baseline_meta_to_template(client, tmp_path):
    """When monte_carlo.baseline_meta is present in the cache, it is
    rendered into the page as the freeze-date caption."""
    from fantasy_baseball.web import season_data

    old_cache_dir = season_data.CACHE_DIR
    season_data.CACHE_DIR = tmp_path
    try:
        season_data.write_cache(CacheKey.MONTE_CARLO, {
            "base": {"team_results": {}, "category_risk": {}},
            "with_management": {"team_results": {}, "category_risk": {}},
            "baseline_meta": {
                "frozen_at": "2026-04-17T00:00:00Z",
                "roster_date": "2026-03-27",
                "season_year": 2026,
            },
            "rest_of_season": None,
            "rest_of_season_with_management": None,
        }, tmp_path)
        season_data.write_cache(CacheKey.STANDINGS, _mock_standings(), tmp_path)

        with patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc, \
             patch("fantasy_baseball.web.season_routes.read_meta") as mock_rm, \
             patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
            mock_rc.side_effect = lambda k: season_data.read_cache(k, tmp_path)
            mock_rm.return_value = season_data.read_meta(tmp_path)
            mock_cfg.return_value.team_name = "Team 01"

            resp = client.get("/standings")
            assert resp.status_code == 200
            assert b"2026-03-27" in resp.data
    finally:
        season_data.CACHE_DIR = old_cache_dir
