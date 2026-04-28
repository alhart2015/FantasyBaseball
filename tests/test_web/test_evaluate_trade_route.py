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
    """Patch read_cache_dict/list to return the values dict keyed by CacheKey.value."""

    def fake_read_cache_dict(key, *_args, **_kwargs):
        val = values.get(key.value)
        return val if isinstance(val, dict) else None

    def fake_read_cache_list(key, *_args, **_kwargs):
        val = values.get(key.value)
        return val if isinstance(val, list) else None

    import fantasy_baseball.web.season_routes as routes

    monkeypatch.setattr(routes, "read_cache_dict", fake_read_cache_dict)
    monkeypatch.setattr(routes, "read_cache_list", fake_read_cache_list)


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


def test_evaluate_trade_response_includes_view_blocks(client, monkeypatch):
    """The response carries roto / ev_roto / stat_totals blocks for the 3-mode UI."""
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
        ],
    }

    _fake_cache(
        monkeypatch,
        {
            "roster": me,
            "opp_rosters": {"Rival": opp},
            "projections": {"projected_standings": projected_standings, "team_sds": None},
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

    new_active = [player_key_json(p) for p in me[:21]]
    new_active.remove(player_key_json(me[0]))
    new_active.append(player_key_json(opp[0]))

    resp = client.post(
        "/api/evaluate-trade",
        json={
            "opponent": "Rival",
            "send": [player_key_json(me[0])],
            "receive": [player_key_json(opp[0])],
            "my_drops": [],
            "opp_drops": [],
            "my_adds": [],
            "my_active_ids": new_active,
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["legal"] is True

    for view_name in ("roto", "ev_roto", "stat_totals"):
        assert view_name in body, f"missing {view_name} block"
        block = body[view_name]
        assert "delta_total" in block
        assert "categories" in block
        for cat in ("R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"):
            assert cat in block["categories"], f"{view_name} missing {cat}"
            cv = block["categories"][cat]
            assert {"before", "after", "delta"} <= cv.keys()

    # stat_totals.delta_total is always 0.0 (summing across mixed-unit cats is meaningless)
    assert body["stat_totals"]["delta_total"] == 0.0

    # When team_sds is None, score_roto_dict returns integer roto regardless of mode,
    # so roto and ev_roto deltas should match for every category.
    for cat in ("R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"):
        assert (
            body["roto"]["categories"][cat]["delta"] == body["ev_roto"]["categories"][cat]["delta"]
        )
