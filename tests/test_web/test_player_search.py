import json

import pytest

from fantasy_baseball.data import redis_store
from fantasy_baseball.web.season_app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test"
    with app.test_client() as c:
        yield c


@pytest.fixture
def redis_with_players(fake_redis, monkeypatch):
    """Seed Redis with ROS projections, preseason projections, and game
    log totals for player search tests, and redirect ``get_default_client``
    to the fake client.

    Also overrides ``season_routes.read_cache`` so the test does not
    hit whatever local ``data/cache/*.json`` files happen to be in the
    repo — returns only the keys the route cares about.
    """
    monkeypatch.setattr(redis_store, "_default_client", fake_redis)
    monkeypatch.setattr(redis_store, "_default_client_initialized", True)

    ros_payload = {
        "hitters": [
            {
                "year": 2026, "snapshot_date": "2026-04-01",
                "fg_id": "15640", "name": "Aaron Judge", "team": "NYY",
                "player_type": "hitter",
                "pa": 600, "ab": 500, "h": 145, "r": 95, "hr": 38,
                "rbi": 92, "sb": 7, "avg": 0.290,
                "w": None, "k": None, "sv": None, "ip": None,
                "er": None, "bb": None, "h_allowed": None,
                "era": None, "whip": None, "adp": 5.0,
            }
        ],
        "pitchers": [
            {
                "year": 2026, "snapshot_date": "2026-04-01",
                "fg_id": "28027", "name": "Gerrit Cole", "team": "NYY",
                "player_type": "pitcher",
                "pa": None, "ab": None, "h": None, "r": None, "hr": None,
                "rbi": None, "sb": None, "avg": None,
                "w": 14, "k": 200, "sv": 0, "ip": 190, "er": 60, "bb": 40,
                "h_allowed": 140, "era": 2.84, "whip": 0.95, "adp": 20.0,
            }
        ],
    }
    fake_redis.set("cache:ros_projections", json.dumps(ros_payload))

    redis_store.set_blended_projections(fake_redis, "hitters", [
        {
            "year": 2026, "fg_id": "15640", "name": "Aaron Judge",
            "team": "NYY", "player_type": "hitter",
            "pa": 650, "ab": 550, "h": 160, "r": 110, "hr": 45,
            "rbi": 120, "sb": 5, "avg": 0.291,
            "w": None, "k": None, "sv": None, "ip": None,
            "er": None, "bb": None, "h_allowed": None,
            "era": None, "whip": None, "adp": 5.0,
        }
    ])
    redis_store.set_blended_projections(fake_redis, "pitchers", [])
    redis_store.set_game_log_totals(fake_redis, "hitters", {})
    redis_store.set_game_log_totals(fake_redis, "pitchers", {})

    # Override the cache layer so disk-resident data/cache/*.json files
    # in the repo do not pollute test results. ros_projections is read
    # from Redis directly by api_player_search via read_cache (which
    # also reads Redis, but falls back to disk first).
    from fantasy_baseball.web import season_routes

    def _fake_read_cache(key, *args, **kwargs):
        if key == "ros_projections":
            return ros_payload
        if key in ("rankings",):
            return {}
        if key in ("roster", "standings"):
            return []
        if key in ("projections",):
            return {}
        if key == "positions":
            return {}
        return None

    monkeypatch.setattr(season_routes, "read_cache", _fake_read_cache)

    yield fake_redis


def test_players_page_renders(client):
    resp = client.get("/players")
    assert resp.status_code == 200
    assert b"pos-filter" in resp.data


def test_search_returns_matching_players(client, redis_with_players):
    resp = client.get("/api/players/search?q=judge")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert len(data) == 1
    assert data[0]["name"] == "Aaron Judge"
    assert data[0]["player_type"] == "hitter"
    assert data[0]["rest_of_season"]["hr"] == 38
    assert data[0]["preseason"]["hr"] == 45


def test_search_requires_min_2_chars(client):
    resp = client.get("/api/players/search?q=j")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data == []


def test_search_no_results(client, redis_with_players):
    resp = client.get("/api/players/search?q=nonexistent")
    data = json.loads(resp.data)
    assert data == []
