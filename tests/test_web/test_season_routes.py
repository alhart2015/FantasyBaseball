import pytest
from unittest.mock import patch

from fantasy_baseball.web.season_app import create_app


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


def test_waivers_trades_page_renders(client):
    resp = client.get("/waivers-trades")
    assert resp.status_code == 200
    assert b"Waivers" in resp.data


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


def test_trade_standings_returns_404_without_data(client):
    with patch("fantasy_baseball.web.season_routes.read_cache", return_value=None):
        resp = client.get("/api/trade/0/standings")
        assert resp.status_code == 404


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
        season_data.write_cache("standings", standings, tmp_path)
        season_data.write_cache("meta", {"last_refresh": "8:32 AM", "week": "3"}, tmp_path)

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
        season_data.write_cache("roster", roster, tmp_path)
        season_data.write_cache("lineup_optimal", optimal, tmp_path)
        season_data.write_cache("meta", {"last_refresh": "9:00 AM"}, tmp_path)

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


def test_full_waivers_page_with_cached_data(client, tmp_path):
    """Integration test: waivers page renders with cached trade/waiver data."""
    from fantasy_baseball.web import season_data

    old_cache_dir = season_data.CACHE_DIR
    season_data.CACHE_DIR = tmp_path

    try:
        waivers = [
            {"add": "Tyler O'Neill", "drop": "Masataka Yoshida", "sgp_gain": 1.8,
             "categories": {"HR": 9, "AVG": -0.008}, "add_positions": "OF",
             "projected_stats": ".262 / 28 HR / 75 RBI / 12 SB"},
        ]
        trades = [
            {"send": "Nick Pivetta", "send_positions": ["SP"],
             "receive": "Josh Hader", "receive_positions": ["RP"],
             "opponent": "SkeleThor",
             "hart_delta": 2, "opp_delta": 1,
             "hart_cat_deltas": {"SV": 18, "ERA": -0.20},
             "opp_cat_deltas": {"W": 4, "K": 35},
             "hart_wsgp_gain": 2.1,
             "send_rank": {"ros": 45}, "receive_rank": {"ros": 50},
             "pitch": "You're getting the #45 overall player for your #50 — straight swap."},
        ]
        season_data.write_cache("waivers", waivers, tmp_path)
        season_data.write_cache("trades", trades, tmp_path)
        season_data.write_cache("meta", {"last_refresh": "9:00 AM"}, tmp_path)

        with patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc, \
             patch("fantasy_baseball.web.season_routes.read_meta") as mock_rm:
            mock_rc.side_effect = lambda k: season_data.read_cache(k, tmp_path)
            mock_rm.return_value = season_data.read_meta(tmp_path)

            resp = client.get("/waivers-trades")
            assert resp.status_code == 200
            html = resp.data.decode()
            assert "Tyler O&#39;Neill" in html or "Tyler O'Neill" in html
            assert "Josh Hader" in html
            assert "SkeleThor" in html
            assert "+2.0 roto pts" in html  # hart_delta displayed as roto points
    finally:
        season_data.CACHE_DIR = old_cache_dir


def test_compare_missing_params(client):
    """Missing required params should return 400."""
    resp = client.get("/api/players/compare")
    assert resp.status_code == 400

    resp2 = client.get("/api/players/compare?roster_player=X")
    assert resp2.status_code == 400
