import json

import pytest

from fantasy_baseball.web.app import create_app


@pytest.fixture
def state_file(tmp_path):
    path = tmp_path / "draft_state.json"
    state = {
        "current_pick": 15,
        "current_round": 2,
        "picking_team": 6,
        "is_user_pick": False,
        "picks_until_user_turn": 4,
        "user_roster": ["Juan Soto"],
        "drafted_players": ["Juan Soto", "Elly De La Cruz"],
        "recommendations": [
            {
                "name": "Gerrit Cole",
                "var": 8.2,
                "best_position": "P",
                "positions": ["SP"],
                "need_flag": True,
                "note": "fills P need",
            }
        ],
        "balance": {
            "totals": {
                "R": 110,
                "HR": 35,
                "RBI": 100,
                "SB": 10,
                "AVG": 0.290,
                "W": 0,
                "K": 0,
                "SV": 0,
                "ERA": 0.0,
                "WHIP": 0.0,
            },
            "warnings": ["SB is low (10, target ~100)"],
        },
        "available_players": [
            {
                "name": "Gerrit Cole",
                "positions": ["SP"],
                "var": 8.2,
                "player_type": "pitcher",
                "w": 16,
                "k": 250,
                "sv": 0,
                "era": 2.80,
                "whip": 1.05,
            },
        ],
        "filled_positions": {"OF": 1},
    }
    path.write_text(json.dumps(state))
    return path


@pytest.fixture
def client(state_file):
    app = create_app(state_path=state_file)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestDashboardRoute:
    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_contains_dashboard_title(self, client):
        resp = client.get("/")
        assert b"Draft Dashboard" in resp.data

    def test_index_contains_htmx_script(self, client):
        resp = client.get("/")
        assert b"htmx.org" in resp.data


class TestApiStateRoute:
    def test_api_state_returns_200(self, client):
        resp = client.get("/api/state")
        assert resp.status_code == 200

    def test_api_state_returns_json(self, client):
        resp = client.get("/api/state")
        assert resp.content_type == "application/json"

    def test_api_state_contains_pick_info(self, client):
        resp = client.get("/api/state")
        data = json.loads(resp.data)
        assert data["current_pick"] == 15
        assert data["current_round"] == 2

    def test_api_state_missing_file_returns_empty(self, tmp_path):
        app = create_app(state_path=tmp_path / "missing.json")
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.get("/api/state")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data == {}


class TestStaticAssets:
    def test_css_served(self, client):
        resp = client.get("/static/style.css")
        assert resp.status_code == 200
