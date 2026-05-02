"""Coverage for /trends and /api/trends/series."""

import json

import pytest

from fantasy_baseball.data import redis_store


@pytest.fixture
def app(monkeypatch, fake_redis):
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "http://fake")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")

    from fantasy_baseball.data import kv_store

    monkeypatch.setattr(kv_store, "get_kv", lambda: fake_redis)

    from fantasy_baseball.web import season_data, season_routes  # noqa: F401
    from fantasy_baseball.web.season_app import create_app

    application = create_app()
    application.config["TESTING"] = True
    application.config["SECRET_KEY"] = "test"
    return application


def test_trends_page_renders(app):
    client = app.test_client()
    resp = client.get("/trends")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "chart-actual" in body
    assert "chart-projected" in body


def test_api_trends_series_empty(app):
    client = app.test_client()
    resp = client.get("/api/trends/series")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["user_team"]
    # Matches the canonical ALL_CATEGORIES ordering used everywhere else
    # in the dashboard (HITTING_CATEGORIES + PITCHING_CATEGORIES, with SV
    # placed at the end of the pitching block — see utils/constants.py).
    assert payload["categories"] == [
        "R",
        "HR",
        "RBI",
        "SB",
        "AVG",
        "W",
        "K",
        "ERA",
        "WHIP",
        "SV",
    ]
    assert payload["actual"] == {"dates": [], "teams": {}}
    assert payload["projected"] == {"dates": [], "teams": {}}


def test_api_trends_series_with_data(app, fake_redis):
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-15",
        json.dumps(
            {
                "effective_date": "2026-04-15",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {
                            "R": 45,
                            "HR": 12,
                            "RBI": 40,
                            "SB": 8,
                            "AVG": 0.268,
                            "W": 3,
                            "K": 85,
                            "SV": 4,
                            "ERA": 3.21,
                            "WHIP": 1.14,
                        },
                        "yahoo_points_for": 78.5,
                        "extras": {},
                    }
                ],
            }
        ),
    )
    client = app.test_client()
    payload = client.get("/api/trends/series").get_json()
    assert payload["actual"]["dates"] == ["2026-04-15"]
    assert "Alpha" in payload["actual"]["teams"]
    assert payload["actual"]["teams"]["Alpha"]["roto_points"] == [78.5]
