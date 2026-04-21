"""Tests for POST /api/evaluate-trade."""

from __future__ import annotations

from typing import ClassVar

import pytest

from fantasy_baseball.web.season_app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def _fake_cache(monkeypatch, values: dict):
    """Patch read_cache to return the values dict keyed by CacheKey.value."""

    def fake_read_cache(key, *_args, **_kwargs):
        return values.get(key.value)

    import fantasy_baseball.web.season_routes as routes

    monkeypatch.setattr(routes, "read_cache", fake_read_cache)


def _auth(client):
    with client.session_transaction() as sess:
        sess["authenticated"] = True


def player_key_json(p: dict) -> str:
    return f"{p['name']}::{p['player_type']}"


def test_evaluate_trade_returns_400_on_missing_opponent(client, monkeypatch):
    _auth(client)
    _fake_cache(
        monkeypatch,
        {
            "roster": [],
            "opp_rosters": {},
            "projections": {},
            "ros_projections": {"hitters": [], "pitchers": []},
        },
    )
    resp = client.post(
        "/api/evaluate-trade",
        json={"send": ["A::hitter"], "receive": ["B::hitter"]},
    )
    assert resp.status_code == 400
    assert "opponent" in resp.get_json()["error"].lower()


def test_evaluate_trade_returns_legal_result_shape(client, monkeypatch):
    _auth(client)
    from fantasy_baseball.models.player import HitterStats, Player

    def _hit(name, r=70):
        return Player(
            name=name,
            player_type="hitter",
            positions=["OF"],
            rest_of_season=HitterStats(pa=600, ab=500, h=125, r=r, hr=20, rbi=60, sb=5, avg=0.250),
        ).to_dict()

    me = [_hit(f"M{i}") for i in range(23)]
    for i, p in enumerate(me):
        p["selected_position"] = "BN" if i >= 21 else "OF"
    opp = [_hit(f"R{i}") for i in range(23)]
    for i, p in enumerate(opp):
        p["selected_position"] = "BN" if i >= 21 else "OF"

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
            {"name": "T3", "stats": dict(standings_stats)},
            {"name": "T4", "stats": dict(standings_stats)},
        ],
    }

    _fake_cache(
        monkeypatch,
        {
            "roster": me,
            "opp_rosters": {"Rival": opp},
            "projections": {
                "projected_standings": projected_standings,
                "team_sds": None,
            },
            "ros_projections": {"hitters": [], "pitchers": []},
        },
    )

    import fantasy_baseball.web.season_routes as routes

    class _FakeCfg:
        team_name = "Hart"
        roster_slots: ClassVar[dict[str, int]] = {
            "C": 1,
            "1B": 1,
            "2B": 1,
            "3B": 1,
            "SS": 1,
            "IF": 1,
            "OF": 4,
            "UTIL": 2,
            "P": 9,
            "BN": 2,
            "IL": 2,
        }

    monkeypatch.setattr(routes, "_load_config", lambda: _FakeCfg())

    payload = {
        "opponent": "Rival",
        "send": ["M0::hitter"],
        "receive": ["R0::hitter"],
        "my_drops": [],
        "opp_drops": [],
        "my_adds": [],
        "my_active_ids": [player_key_json(p) for p in me[1:21]] + ["R0::hitter"],
    }
    resp = client.post("/api/evaluate-trade", json=payload)
    data = resp.get_json()
    assert resp.status_code == 200, data
    assert "legal" in data
    assert "delta_total" in data
    assert set(data["categories"].keys()) >= {"R", "HR", "ERA"}
