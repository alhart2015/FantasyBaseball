"""Tests for POST /api/optimize-trade-lineup."""

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
    def fake_read_cache_dict(key, *_a, **_k):
        v = values.get(key.value)
        return v if isinstance(v, dict) else None

    def fake_read_cache_list(key, *_a, **_k):
        v = values.get(key.value)
        return v if isinstance(v, list) else None

    import fantasy_baseball.web.season_routes as routes

    monkeypatch.setattr(routes, "read_cache_dict", fake_read_cache_dict)
    monkeypatch.setattr(routes, "read_cache_list", fake_read_cache_list)


def _auth(client):
    with client.session_transaction() as sess:
        sess["authenticated"] = True


def player_key_json(p: dict) -> str:
    return f"{p['name']}::{p['player_type']}"


class _FakeCfg:
    """Mirrors the league config shape consumed by the route.

    Trimmed to a tiny shape (4 active hitter slots + 1 bench + 1 IL,
    no pitchers) so ``optimize_hitter_lineup``'s combinatorial search
    finishes in milliseconds. Full-league shapes (12 active hitters,
    9 pitchers) make the Hungarian-on-every-subset loop take minutes
    in unit tests, which is unacceptable.
    """

    team_name = "Hart"
    roster_slots: ClassVar[dict[str, int]] = {
        "OF": 3,
        "UTIL": 1,
        "BN": 1,
        "IL": 1,
    }


def _patch_config(monkeypatch):
    import fantasy_baseball.web.season_routes as routes

    monkeypatch.setattr(routes, "_load_config", lambda: _FakeCfg())


def _setup_minimal_rosters():
    """5-hitter roster on each side (4 active + 1 bench, no pitchers).

    Sized to match ``_FakeCfg.roster_slots`` so legality passes for
    a 1-for-1 swap and the optimizer finishes immediately.
    """
    from fantasy_baseball.models.player import HitterStats, Player

    def _hit(name):
        return Player(
            name=name,
            player_type="hitter",
            positions=["OF"],
            rest_of_season=HitterStats(pa=600, ab=500, h=125, r=70, hr=20, rbi=60, sb=5, avg=0.250),
        ).to_dict()

    me = [_hit(f"M{i}") for i in range(5)]
    for i, p in enumerate(me):
        p["selected_position"] = "BN" if i >= 4 else "OF"
    opp = [_hit(f"R{i}") for i in range(5)]
    for i, p in enumerate(opp):
        p["selected_position"] = "BN" if i >= 4 else "OF"

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
    return me, opp, projected_standings


def test_optimize_route_returns_400_on_missing_opponent(client, monkeypatch):
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
    resp = client.post("/api/optimize-trade-lineup", json={"send": [], "receive": []})
    assert resp.status_code == 400
    assert "opponent" in resp.get_json()["error"].lower()


def test_optimize_route_returns_slots_for_legal_trade(client, monkeypatch):
    _auth(client)
    me, opp, ps = _setup_minimal_rosters()
    _fake_cache(
        monkeypatch,
        {
            "roster": me,
            "opp_rosters": {"Rival": opp},
            "projections": {"projected_standings": ps, "team_sds": None},
            "ros_projections": {"hitters": [], "pitchers": []},
        },
    )
    _patch_config(monkeypatch)

    resp = client.post(
        "/api/optimize-trade-lineup",
        json={
            "opponent": "Rival",
            "send": [player_key_json(me[0])],
            "receive": [player_key_json(opp[0])],
            "my_drops": [],
            "opp_drops": [],
            "my_adds": [],
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert "my_slots" in body and "opp_slots" in body

    # Every non-IL player on each side gets a slot assignment (BN counts).
    expected_my_keys = {player_key_json(p) for p in me} - {player_key_json(me[0])} | {
        player_key_json(opp[0])
    }
    assert set(body["my_slots"].keys()) == expected_my_keys

    expected_opp_keys = {player_key_json(p) for p in opp} - {player_key_json(opp[0])} | {
        player_key_json(me[0])
    }
    assert set(body["opp_slots"].keys()) == expected_opp_keys


def test_optimize_route_numbers_repeated_slots(client, monkeypatch):
    """Multi-count slots (OF=3) come back as OF1/OF2/OF3, not bare "OF".

    The trade-builder frontend's slotList() emits numbered IDs for any
    slot with count > 1; the panel renders by exact ``pl.zone === slotId``
    match, so a bare "OF" zone leaves the OF rows empty. Single-count
    slots like UTIL=1 stay bare ("UTIL").
    """
    _auth(client)
    me, opp, ps = _setup_minimal_rosters()
    _fake_cache(
        monkeypatch,
        {
            "roster": me,
            "opp_rosters": {"Rival": opp},
            "projections": {"projected_standings": ps, "team_sds": None},
            "ros_projections": {"hitters": [], "pitchers": []},
        },
    )
    _patch_config(monkeypatch)

    resp = client.post(
        "/api/optimize-trade-lineup",
        json={
            "opponent": "Rival",
            "send": [player_key_json(me[0])],
            "receive": [player_key_json(opp[0])],
            "my_drops": [],
            "opp_drops": [],
            "my_adds": [],
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    of_zones = sorted(z for z in body["my_slots"].values() if z.startswith("OF"))
    assert of_zones == ["OF1", "OF2", "OF3"], f"OF=3 should produce OF1/OF2/OF3, got {of_zones!r}"
    assert "OF" not in body["my_slots"].values(), "bare 'OF' should not appear when OF count > 1"
    util_zones = [z for z in body["my_slots"].values() if z.startswith("UTIL")]
    assert util_zones == ["UTIL"], f"UTIL=1 should stay bare 'UTIL', got {util_zones!r}"


def test_optimize_route_rejects_illegal_trade(client, monkeypatch):
    _auth(client)
    me, opp, ps = _setup_minimal_rosters()
    _fake_cache(
        monkeypatch,
        {
            "roster": me,
            "opp_rosters": {"Rival": opp},
            "projections": {"projected_standings": ps, "team_sds": None},
            "ros_projections": {"hitters": [], "pitchers": []},
        },
    )
    _patch_config(monkeypatch)

    # 1-for-2 trade with no compensating drop is illegal (size mismatch).
    resp = client.post(
        "/api/optimize-trade-lineup",
        json={
            "opponent": "Rival",
            "send": [player_key_json(me[0])],
            "receive": [player_key_json(opp[0]), player_key_json(opp[1])],
            "my_drops": [],
            "opp_drops": [],
            "my_adds": [],
        },
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is False
    assert body.get("reason")
