from fantasy_baseball.analysis.buy_low import find_buy_low_candidates

_FLAT_LEVERAGE = {"R": 1.0, "HR": 1.0, "RBI": 1.0, "SB": 1.0, "AVG": 1.0,
                  "W": 1.0, "K": 1.0, "SV": 1.0, "ERA": 1.0, "WHIP": 1.0}


def test_hitter_below_pace_qualifies():
    """A hitter > 1 SD below projection pace across categories is a buy-low candidate."""
    players = [{
        "name": "Struggling Hitter",
        "positions": ["OF"],
        "player_type": "hitter",
        "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
        "h": 150, "ab": 540, "avg": 0.278,
    }]
    game_logs = {
        "struggling hitter": {
            "pa": 60, "ab": 54, "h": 10, "r": 4, "hr": 0, "rbi": 3, "sb": 0,
        },
    }
    leverage = _FLAT_LEVERAGE

    result = find_buy_low_candidates(players, game_logs, leverage, owner="Opponent A")
    assert len(result) == 1
    assert result[0]["name"] == "Struggling Hitter"
    assert result[0]["owner"] == "Opponent A"
    assert result[0]["avg_z"] < -1.0
    assert "stats" in result[0]


def test_hitter_on_pace_excluded():
    """A hitter near projection pace should not be a buy-low candidate."""
    players = [{
        "name": "Normal Hitter",
        "positions": ["1B"],
        "player_type": "hitter",
        "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
        "h": 150, "ab": 540, "avg": 0.278,
    }]
    game_logs = {
        "normal hitter": {
            "pa": 60, "ab": 54, "h": 15, "r": 9, "hr": 3, "rbi": 9, "sb": 1,
        },
    }
    leverage = _FLAT_LEVERAGE

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) == 0


def test_no_game_logs_excluded():
    """A player with no game logs (below sample threshold) is excluded."""
    players = [{
        "name": "No Games Player",
        "positions": ["SS"],
        "player_type": "hitter",
        "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
        "h": 150, "ab": 540, "avg": 0.278,
    }]
    game_logs = {}
    leverage = _FLAT_LEVERAGE

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) == 0


def test_sorted_most_underperforming_first():
    """Results are sorted by avg_z ascending (most negative first)."""
    players = [
        {"name": "Somewhat Bad", "positions": ["OF"], "player_type": "hitter",
         "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
         "h": 150, "ab": 540, "avg": 0.278},
        {"name": "Very Bad", "positions": ["1B"], "player_type": "hitter",
         "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
         "h": 150, "ab": 540, "avg": 0.278},
    ]
    game_logs = {
        "somewhat bad": {"pa": 60, "ab": 54, "h": 10, "r": 4, "hr": 1, "rbi": 4, "sb": 0},
        "very bad": {"pa": 60, "ab": 54, "h": 5, "r": 2, "hr": 0, "rbi": 1, "sb": 0},
    }
    leverage = _FLAT_LEVERAGE

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) >= 2
    assert result[0]["name"] == "Very Bad"
    assert result[0]["avg_z"] < result[1]["avg_z"]


def test_pitcher_below_pace_qualifies():
    """A pitcher with bad ERA and low K qualifies."""
    players = [{
        "name": "Bad Pitcher",
        "positions": ["SP"],
        "player_type": "pitcher",
        "ip": 180, "w": 12, "k": 190, "sv": 0,
        "er": 60, "bb": 50, "h_allowed": 150,
        "era": 3.00, "whip": 1.11,
    }]
    game_logs = {
        "bad pitcher": {"ip": 18.0, "k": 10, "w": 0, "sv": 0, "er": 14, "bb": 12, "h_allowed": 22},
    }
    leverage = _FLAT_LEVERAGE

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) == 1
    assert result[0]["name"] == "Bad Pitcher"
    assert result[0]["avg_z"] < -1.0


def test_below_threshold_stats_excluded_from_average():
    """Stats below sample threshold (z=0, neutral) are excluded from average, not diluted."""
    players = [{
        "name": "Small Sample Hitter",
        "positions": ["OF"],
        "player_type": "hitter",
        "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
        "h": 150, "ab": 540, "avg": 0.278,
    }]
    game_logs = {
        "small sample hitter": {"pa": 15, "ab": 13, "h": 3, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
    }
    leverage = _FLAT_LEVERAGE

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) == 1
    assert result[0]["avg_z"] < -1.0
