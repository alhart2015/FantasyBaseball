from fantasy_baseball.analysis.game_logs import parse_hitter_game_log, parse_pitcher_game_log


def test_parse_hitter_game_log():
    raw_split = {
        "date": "2025-06-15",
        "stat": {
            "atBats": 4,
            "hits": 2,
            "homeRuns": 1,
            "runs": 1,
            "rbi": 2,
            "stolenBases": 0,
            "plateAppearances": 5,
        },
    }
    result = parse_hitter_game_log(raw_split)
    assert result["date"] == "2025-06-15"
    assert result["ab"] == 4
    assert result["h"] == 2
    assert result["hr"] == 1
    assert result["pa"] == 5


def test_parse_pitcher_game_log():
    raw_split = {
        "date": "2025-06-15",
        "stat": {
            "inningsPitched": "6.0",
            "strikeOuts": 8,
            "earnedRuns": 2,
            "baseOnBalls": 1,
            "hits": 4,
            "wins": 1,
            "losses": 0,
            "saves": 0,
            "gamesStarted": 1,
            "gamesPlayed": 1,
            "battersFaced": 23,
        },
    }
    result = parse_pitcher_game_log(raw_split)
    assert result["date"] == "2025-06-15"
    assert result["ip"] == 6.0
    assert result["k"] == 8
    assert result["er"] == 2
    assert result["gs"] == 1


def test_parse_pitcher_partial_innings():
    """6.1 IP means 6 and 1/3 innings."""
    raw_split = {
        "date": "2025-06-15",
        "stat": {
            "inningsPitched": "6.1",
            "strikeOuts": 7,
            "earnedRuns": 3,
            "baseOnBalls": 2,
            "hits": 5,
            "wins": 0,
            "losses": 1,
            "saves": 0,
            "gamesStarted": 1,
            "gamesPlayed": 1,
        },
    }
    result = parse_pitcher_game_log(raw_split)
    assert abs(result["ip"] - 6.3333) < 0.01
