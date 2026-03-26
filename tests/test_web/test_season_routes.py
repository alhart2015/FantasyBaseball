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


def test_standings_renders_table_with_data(client):
    with patch("fantasy_baseball.web.season_routes.read_cache") as mock_cache, \
         patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cache.side_effect = lambda k: _mock_standings() if k == "standings" else {}
        mock_cfg.return_value.team_name = "Hart of the Order"
        resp = client.get("/standings")
        assert resp.status_code == 200
        assert b"Hart of the Order" in resp.data
        assert b"user-team" in resp.data
