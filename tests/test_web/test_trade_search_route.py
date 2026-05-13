"""Tests for POST /api/trade-search canonical-key fields.

Task 11 of trade-builder-improvements: each candidate must include
``send_key`` and ``receive_key`` in ``name::player_type`` form so the
Compare button can link to ``/players?compare=...``.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from fantasy_baseball.web.season_app import create_app


@pytest.fixture
def client():
    """Pre-authenticated test client (whole site is behind login)."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["authenticated"] = True
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


def _seed_cache(monkeypatch):
    """Wire the route's cache reads so the early-return guards pass.

    With ``search_trades_away`` stubbed, the cache contents themselves
    don't have to be realistic — only present.
    """
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
    placeholder_player = {
        "name": "Filler",
        "player_type": "hitter",
        "positions": ["OF"],
    }
    _fake_cache(
        monkeypatch,
        {
            "standings": {
                "effective_date": "2026-04-15",
                "teams": [
                    {
                        "name": "Hart",
                        "team_key": "t.1",
                        "rank": 1,
                        "stats": dict(standings_stats),
                    }
                ],
            },
            "roster": [placeholder_player],
            "opp_rosters": {"Rival": [placeholder_player]},
            "leverage": {"Hart": {}, "Rival": {}},
            "rankings": {"x": 1},
            "projections": {},
        },
    )


def test_trade_search_response_includes_canonical_keys_away(client, monkeypatch):
    """``mode=away`` candidates expose ``send_key``/``receive_key``."""
    _seed_cache(monkeypatch)

    import fantasy_baseball.web.season_routes as routes

    monkeypatch.setattr(routes, "_load_config", lambda: _FakeCfg())

    def fake_search_trades_away(**_kwargs):
        return [
            {
                "opponent": "Rival",
                "candidates": [
                    {
                        "send": "Juan Soto",
                        "send_positions": ["OF"],
                        "send_rank": 5,
                        "send_player_type": "hitter",
                        "receive": "Aaron Judge",
                        "receive_positions": ["OF"],
                        "receive_rank": 3,
                        "receive_player_type": "hitter",
                        "hart_delta": 1.5,
                        "opp_delta": -0.5,
                        "hart_cat_deltas": {},
                        "opp_cat_deltas": {},
                    }
                ],
            }
        ]

    monkeypatch.setattr(
        "fantasy_baseball.trades.evaluate.search_trades_away",
        fake_search_trades_away,
    )

    resp = client.post(
        "/api/trade-search",
        json={"player_name": "Juan Soto", "mode": "away"},
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert isinstance(body, list)
    assert body, "expected at least one opponent group"
    cand = body[0]["candidates"][0]
    assert cand["send_key"] == "Juan Soto::hitter"
    assert cand["receive_key"] == "Aaron Judge::hitter"


def test_trade_search_response_includes_canonical_keys_for(client, monkeypatch):
    """``mode=for`` candidates also expose ``send_key``/``receive_key``."""
    _seed_cache(monkeypatch)

    import fantasy_baseball.web.season_routes as routes

    monkeypatch.setattr(routes, "_load_config", lambda: _FakeCfg())

    def fake_search_trades_for(**_kwargs):
        return [
            {
                "opponent": "Rival",
                "candidates": [
                    {
                        "send": "Spencer Strider",
                        "send_positions": ["SP"],
                        "send_rank": 4,
                        "send_player_type": "pitcher",
                        "receive": "Tarik Skubal",
                        "receive_positions": ["SP"],
                        "receive_rank": 2,
                        "receive_player_type": "pitcher",
                        "hart_delta": 2.0,
                        "opp_delta": -1.0,
                        "hart_cat_deltas": {},
                        "opp_cat_deltas": {},
                    }
                ],
            }
        ]

    monkeypatch.setattr(
        "fantasy_baseball.trades.evaluate.search_trades_for",
        fake_search_trades_for,
    )

    resp = client.post(
        "/api/trade-search",
        json={"player_name": "Tarik Skubal", "mode": "for"},
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    cand = body[0]["candidates"][0]
    assert cand["send_key"] == "Spencer Strider::pitcher"
    assert cand["receive_key"] == "Tarik Skubal::pitcher"
