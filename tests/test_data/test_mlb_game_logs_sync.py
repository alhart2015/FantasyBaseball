from datetime import UTC, datetime

from fantasy_baseball.data import mlb_game_logs, redis_store
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


# Reusable synthetic box scores (verified field names).
_OHTANI = {
    "person": {"id": 660271, "fullName": "Shohei Ohtani"},
    "stats": {
        "batting": {
            "plateAppearances": 4,
            "atBats": 3,
            "hits": 0,
            "runs": 1,
            "homeRuns": 0,
            "rbi": 0,
            "stolenBases": 0,
        },
        "pitching": {
            "inningsPitched": "6.0",
            "strikeOuts": 8,
            "earnedRuns": 0,
            "baseOnBalls": 0,
            "hits": 5,
            "wins": 0,
            "saves": 0,
            "gamesStarted": 1,
            "gamesPlayed": 1,
        },
    },
}
_BETTS_MOPUP = {
    "person": {"id": 605141, "fullName": "Mookie Betts"},
    "stats": {
        "batting": {
            "plateAppearances": 5,
            "atBats": 5,
            "hits": 2,
            "runs": 1,
            "homeRuns": 0,
            "rbi": 1,
            "stolenBases": 0,
        },
        "pitching": {
            "inningsPitched": "1.0",
            "strikeOuts": 0,
            "earnedRuns": 3,
            "baseOnBalls": 1,
            "hits": 2,
            "wins": 0,
            "saves": 0,
            "gamesStarted": 0,
            "gamesPlayed": 1,
        },
    },
}


def _final(game_pk, date, game_number=1):
    return {
        "gamePk": game_pk,
        "gameNumber": game_number,
        "officialDate": date,
        "gameType": "R",
        "status": {"abstractGameState": "Final"},
    }


def _patch_mlb(
    monkeypatch,
    *,
    season_games=None,
    changed_games=None,
    boxscores=None,
    positions=None,
    positions_raises=False,
):
    monkeypatch.setattr(mlb_game_logs, "_fetch_season_games", lambda s: season_games or [])
    monkeypatch.setattr(mlb_game_logs, "_fetch_changed_games", lambda s, since: changed_games or [])
    monkeypatch.setattr(mlb_game_logs, "_fetch_boxscore", lambda gp: (boxscores or {})[gp])

    def _pos(ids):
        if positions_raises:
            raise RuntimeError("people endpoint down")
        return {str(i): (positions or {}).get(str(i)) for i in ids}

    monkeypatch.setattr(mlb_game_logs, "_fetch_positions", _pos)


NOW = datetime(2026, 5, 24, 13, 0, tzinfo=UTC)


def test_backfill_records_two_way_and_filters_mopup(monkeypatch, fake_redis):
    _patch_mlb(
        monkeypatch,
        season_games=[_final(100, "2026-04-01")],
        boxscores={
            100: {
                "teams": {
                    "home": {"players": {"ID660271": _OHTANI, "ID605141": _BETTS_MOPUP}},
                    "away": {"players": {}},
                }
            }
        },
        positions={"660271": "Y", "605141": "6"},
    )
    mlb_game_logs.sync_game_logs(fake_redis, 2026, now_utc=NOW)

    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting")["games"]
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "pitching")["games"]
    hitters = redis_store.get_game_log_totals(fake_redis, "hitters")
    pitchers = redis_store.get_game_log_totals(fake_redis, "pitchers")
    assert hitters["660271"]["ab"] == 3 and hitters["660271"]["name"] == "Shohei Ohtani"
    assert pitchers["660271"]["k"] == 8 and pitchers["660271"]["ip"] == 6.0

    assert hitters["605141"]["h"] == 2
    assert "605141" not in pitchers
    assert redis_store.get_player_game_log(fake_redis, 2026, "605141", "pitching") is None

    assert redis_store.get_game_logs_watermark(fake_redis, 2026) == NOW.isoformat()
    assert redis_store.get_season_progress(fake_redis)["games_elapsed"] == 1


def test_incremental_correction_overwrites_by_gamepk(monkeypatch, fake_redis):
    redis_store.set_game_logs_watermark(fake_redis, 2026, "2026-05-23T13:00:00+00:00")
    redis_store.set_player_game_log(
        fake_redis,
        2026,
        "660271",
        "hitting",
        {
            "name": "Shohei Ohtani",
            "games": [
                {
                    "gamePk": 100,
                    "gameNumber": 1,
                    "date": "2026-04-01",
                    "pa": 4,
                    "ab": 3,
                    "h": 0,
                    "hr": 0,
                    "r": 1,
                    "rbi": 0,
                    "sb": 0,
                }
            ],
        },
    )
    corrected = {
        **_OHTANI,
        "stats": {**_OHTANI["stats"], "batting": {**_OHTANI["stats"]["batting"], "hits": 2}},
    }
    _patch_mlb(
        monkeypatch,
        changed_games=[_final(100, "2026-04-01")],
        boxscores={
            100: {"teams": {"home": {"players": {"ID660271": corrected}}, "away": {"players": {}}}}
        },
        positions={"660271": "Y"},
    )
    mlb_game_logs.sync_game_logs(fake_redis, 2026, now_utc=NOW)
    games = redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting")["games"]
    assert len(games) == 1 and games[0]["h"] == 2
    assert redis_store.get_game_log_totals(fake_redis, "hitters")["660271"]["h"] == 2


def test_watermark_not_advanced_when_position_unresolved(monkeypatch, fake_redis):
    redis_store.set_game_logs_watermark(fake_redis, 2026, "2026-05-23T13:00:00+00:00")
    _patch_mlb(
        monkeypatch,
        changed_games=[_final(100, "2026-04-01")],
        boxscores={
            100: {"teams": {"home": {"players": {"ID660271": _OHTANI}}, "away": {"players": {}}}}
        },
        positions_raises=True,
    )
    mlb_game_logs.sync_game_logs(fake_redis, 2026, now_utc=NOW)
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting")["games"]
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "pitching") is None
    assert redis_store.get_game_logs_watermark(fake_redis, 2026) == "2026-05-23T13:00:00+00:00"


def test_fetch_game_log_totals_preserves_public_contract(monkeypatch, fake_redis):
    monkeypatch.setattr("fantasy_baseball.data.kv_store.get_kv", lambda: fake_redis)
    _patch_mlb(
        monkeypatch,
        season_games=[_final(100, "2026-04-01")],
        boxscores={
            100: {"teams": {"home": {"players": {"ID660271": _OHTANI}}, "away": {"players": {}}}}
        },
        positions={"660271": "Y"},
    )
    hitters, pitchers, games_elapsed = mlb_game_logs.fetch_game_log_totals(2026)
    assert hitters["660271"]["ab"] == 3
    assert pitchers["660271"]["k"] == 8
    assert games_elapsed == 1


def test_watermark_not_advanced_when_boxscore_fetch_fails(monkeypatch, fake_redis):
    redis_store.set_game_logs_watermark(fake_redis, 2026, "2026-05-23T13:00:00+00:00")
    _patch_mlb(monkeypatch, changed_games=[_final(100, "2026-04-01")])

    def _boom(_game_pk):
        raise RuntimeError("boxscore 500")

    monkeypatch.setattr(mlb_game_logs, "_fetch_boxscore", _boom)
    mlb_game_logs.sync_game_logs(fake_redis, 2026, now_utc=NOW)
    # The only game's box score failed -> nothing stored, watermark unchanged so it retries.
    assert redis_store.get_player_game_log(fake_redis, 2026, "660271", "hitting") is None
    assert redis_store.get_game_logs_watermark(fake_redis, 2026) == "2026-05-23T13:00:00+00:00"
