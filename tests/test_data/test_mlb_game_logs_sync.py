from fantasy_baseball.data import mlb_game_logs
from fantasy_baseball.data.mlb_game_logs import (
    _merge_player_games,
    _sum_hitting,
    _sum_pitching,
)


def test_merge_upserts_by_gamepk_and_sorts_by_date():
    existing = {
        "name": "X",
        "games": [
            {
                "gamePk": 2,
                "gameNumber": 1,
                "date": "2026-04-02",
                "pa": 4,
                "ab": 4,
                "h": 1,
                "hr": 0,
                "r": 0,
                "rbi": 0,
                "sb": 0,
            }
        ],
    }
    new_rows = {
        1: {
            "gamePk": 1,
            "gameNumber": 1,
            "date": "2026-04-01",
            "pa": 3,
            "ab": 3,
            "h": 2,
            "hr": 1,
            "r": 1,
            "rbi": 2,
            "sb": 0,
        },
        2: {
            "gamePk": 2,
            "gameNumber": 1,
            "date": "2026-04-02",
            "pa": 5,
            "ab": 5,
            "h": 3,
            "hr": 0,
            "r": 1,
            "rbi": 1,
            "sb": 1,
        },  # correction overwrites gamePk 2
    }
    merged = _merge_player_games(existing, "X", new_rows)
    assert [g["gamePk"] for g in merged["games"]] == [1, 2]  # sorted by date
    assert merged["games"][1]["h"] == 3  # corrected value won


def test_merge_handles_doubleheader_same_date():
    new_rows = {
        10: {
            "gamePk": 10,
            "gameNumber": 1,
            "date": "2026-07-04",
            "pa": 4,
            "ab": 4,
            "h": 1,
            "hr": 0,
            "r": 0,
            "rbi": 0,
            "sb": 0,
        },
        11: {
            "gamePk": 11,
            "gameNumber": 2,
            "date": "2026-07-04",
            "pa": 3,
            "ab": 3,
            "h": 2,
            "hr": 1,
            "r": 1,
            "rbi": 1,
            "sb": 0,
        },
    }
    merged = _merge_player_games(None, "DH", new_rows)
    assert len(merged["games"]) == 2
    assert {g["gameNumber"] for g in merged["games"]} == {1, 2}


def test_sum_hitting():
    games = [
        {"pa": 4, "ab": 4, "h": 1, "hr": 0, "r": 0, "rbi": 0, "sb": 0},
        {"pa": 5, "ab": 5, "h": 3, "hr": 1, "r": 1, "rbi": 1, "sb": 1},
    ]
    assert _sum_hitting(games) == {"pa": 9, "ab": 9, "h": 4, "hr": 1, "r": 1, "rbi": 1, "sb": 1}


def test_sum_pitching_rounds_ip():
    games = [
        {"ip": 6.3333, "k": 8, "er": 0, "bb": 0, "h_allowed": 5, "w": 0, "sv": 0},
        {"ip": 1.0, "k": 1, "er": 1, "bb": 1, "h_allowed": 2, "w": 0, "sv": 1},
    ]
    out = _sum_pitching(games)
    assert out == {"ip": 7.3333, "k": 9, "er": 1, "bb": 1, "h_allowed": 7, "w": 0, "sv": 1}


def test_is_regular_final():
    final_reg = {"gameType": "R", "status": {"abstractGameState": "Final"}}
    live_reg = {"gameType": "R", "status": {"abstractGameState": "Live"}}
    final_spring = {"gameType": "S", "status": {"abstractGameState": "Final"}}
    assert mlb_game_logs._is_regular_final(final_reg) is True
    assert mlb_game_logs._is_regular_final(live_reg) is False
    assert mlb_game_logs._is_regular_final(final_spring) is False


def test_fetch_changed_games_flattens_dates(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self): ...
        def json(self):
            return {
                "dates": [{"games": [{"gamePk": 1}, {"gamePk": 2}]}, {"games": [{"gamePk": 3}]}]
            }

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _Resp()

    monkeypatch.setattr(mlb_game_logs.requests, "get", fake_get)
    games = mlb_game_logs._fetch_changed_games(2026, "2026-05-24T00:00:00+00:00")
    assert [g["gamePk"] for g in games] == [1, 2, 3]
    assert captured["params"] == {
        "updatedSince": "2026-05-24T00:00:00+00:00",
        "sportId": 1,
        "season": 2026,
    }
    assert captured["url"].endswith("/game/changes")


def test_fetch_positions_maps_id_to_code(monkeypatch):
    class _Resp:
        def raise_for_status(self): ...
        def json(self):
            return {
                "people": [
                    {"id": 660271, "primaryPosition": {"code": "Y"}},
                    {"id": 543037, "primaryPosition": {"code": "1"}},
                ]
            }

    monkeypatch.setattr(mlb_game_logs.requests, "get", lambda *a, **k: _Resp())
    assert mlb_game_logs._fetch_positions([660271, 543037]) == {"660271": "Y", "543037": "1"}


def test_fetch_positions_empty_short_circuits(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not call the API for an empty id list")

    monkeypatch.setattr(mlb_game_logs.requests, "get", boom)
    assert mlb_game_logs._fetch_positions([]) == {}
