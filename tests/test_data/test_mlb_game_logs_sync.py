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
