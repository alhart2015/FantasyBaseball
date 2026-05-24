from fantasy_baseball.data.mlb_boxscore import (
    boxscore_hitter_row,
    boxscore_pitcher_row,
    iter_boxscore_players,
    should_record_pitching,
)

# Verified field names/values from real boxscore gamePk 776213 (Ohtani, two-way).
OHTANI = {
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
BENCH = {"person": {"id": 1, "fullName": "Did Not Play"}, "stats": {"batting": {}, "pitching": {}}}


def _box(*home_players):
    players = {f"ID{p['person']['id']}": p for p in home_players}
    return {"teams": {"home": {"players": players}, "away": {"players": {}}}}


def test_iter_yields_empty_blocks_for_bench_player():
    rows = list(iter_boxscore_players(_box(OHTANI, BENCH)))
    by_id = {mlbam: (bat, pit) for mlbam, _name, bat, pit in rows}
    assert by_id["660271"][0] and by_id["660271"][1]  # Ohtani: both populated
    assert not by_id["1"][0] and not by_id["1"][1]  # bench: both empty


def test_boxscore_hitter_row():
    _id, _name, bat, _pit = next(iter(iter_boxscore_players(_box(OHTANI))))
    assert boxscore_hitter_row(bat, 776213, 1, "2025-09-23") == {
        "gamePk": 776213,
        "gameNumber": 1,
        "date": "2025-09-23",
        "pa": 4,
        "ab": 3,
        "h": 0,
        "hr": 0,
        "r": 1,
        "rbi": 0,
        "sb": 0,
    }


def test_boxscore_pitcher_row():
    _id, _name, _bat, pit = next(iter(iter_boxscore_players(_box(OHTANI))))
    assert boxscore_pitcher_row(pit, 776213, 1, "2025-09-23") == {
        "gamePk": 776213,
        "gameNumber": 1,
        "date": "2025-09-23",
        "ip": 6.0,
        "k": 8,
        "er": 0,
        "bb": 0,
        "h_allowed": 5,
        "w": 0,
        "sv": 0,
    }


def test_should_record_pitching_keeps_pitchers_and_two_way():
    assert should_record_pitching("1") is True  # pitcher
    assert should_record_pitching("Y") is True  # two-way (Ohtani)
    assert should_record_pitching("6") is False  # position player (SS)
    assert should_record_pitching(None) is False  # unknown -> not a pitcher
