"""Coverage for the Trends standings subpage and /api/trends/series."""

import json
from unittest.mock import patch

import pytest

from fantasy_baseball.data import redis_store
from fantasy_baseball.web.season_data import CacheKey


def _mock_standings():
    """Minimal two-team standings payload so /standings renders its data
    branch (the Trends subpage markup lives inside `{% if standings %}`)."""
    stats = {
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
    }
    return {
        "effective_date": "2026-04-01",
        "teams": [
            {"name": "Hart of the Order", "team_key": "k1", "rank": 1, "stats": stats},
            {"name": "SkeleThor", "team_key": "k2", "rank": 2, "stats": stats},
        ],
    }


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


def _authed_client(app):
    """Helper: every test below needs to be past the login gate."""
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["authenticated"] = True
    return client


def test_trends_subpage_renders_within_standings(app):
    # Trends moved from its own /trends route to a subpage of /standings
    # (a pill in the top toggle, the way Distributions is). The two Chart.js
    # canvases and the lazy-loader script now live in the standings response.
    client = _authed_client(app)
    with (
        patch("fantasy_baseball.web.season_routes.read_cache_dict") as mock_cache,
        patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg,
    ):
        mock_cache.side_effect = lambda k: _mock_standings() if k == CacheKey.STANDINGS else {}
        mock_cfg.return_value.team_name = "Hart of the Order"
        resp = client.get("/standings")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "chart-actual" in body
    assert "chart-projected" in body
    assert "season_trends.js" in body
    assert 'data-view="trends"' in body
    # The trends_chart macro emits per-category tabs from two loops --
    # `hitting_categories` then `pitching_categories`. Assert one tab from EACH:
    # Jinja's default Undefined iterates to empty, so dropping or misspelling
    # either context var silently blanks half the tabs with no error anywhere.
    assert 'data-tab="HR"' in body  # hitting_categories
    assert 'data-tab="WHIP"' in body  # pitching_categories
    # Second render site: the Distributions strip is its own inline nav, not the
    # trends_chart macro, so it reads the same two vars through separate markup
    # and a typo in one site is caught only by that site's pair.
    assert 'data-distmetric="SB"' in body  # hitting_categories
    assert 'data-distmetric="SV"' in body  # pitching_categories
    # The seam itself: without this span both strips still render every tab, so
    # the assertions above stay green while the split silently degrades to
    # wrapping wherever the width runs out. One per strip, Trends macro twice.
    # Matched on the bare class name so adding a second class doesn't zero it.
    assert body.count("tab-strip-break") == 3


def test_trends_route_removed(app):
    # The standalone page is gone; only the subpage + API remain.
    client = _authed_client(app)
    assert client.get("/trends").status_code == 404


def test_api_trends_series_empty(app):
    client = _authed_client(app)
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
    # season_trends.js reads counting_stats from this payload to decide which
    # tabs get the "distance from 1st" y-axis title. Locks the API/UI contract.
    assert sorted(payload["counting_stats"]) == ["HR", "K", "R", "RBI", "SB", "SV", "W"]
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
    client = _authed_client(app)
    payload = client.get("/api/trends/series").get_json()
    assert payload["actual"]["dates"] == ["2026-04-15"]
    assert "Alpha" in payload["actual"]["teams"]
    assert payload["actual"]["teams"]["Alpha"]["roto_points"] == [78.5]
