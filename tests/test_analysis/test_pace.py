from fantasy_baseball.analysis.pace import compute_player_pace


def test_hitter_counting_on_pace():
    """A hitter exactly on pace for all counting stats gets neutral colors."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["R"]["color_class"] == "stat-neutral"
    assert abs(result["HR"]["z_score"]) < 0.5
    assert result["HR"]["actual"] == 3
    assert result["HR"]["expected"] == 3.0
    assert result["HR"]["projection"] == 30


def test_hitter_counting_above_pace():
    """A hitter well above pace on HR gets hot color."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 6, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] == "stat-hot-2"
    assert result["HR"]["z_score"] > 1.0


def test_hitter_counting_below_pace():
    """A hitter well below pace on SB gets cold color."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 0, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["SB"]["color_class"] == "stat-cold-2"
    assert result["SB"]["z_score"] < -1.0


def test_expected_zero_shows_neutral():
    """When projected stat is 0, show neutral regardless of actual."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 0, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 2, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["SB"]["color_class"] == "stat-neutral"
    assert result["SB"]["z_score"] == 0.0


def test_pa_always_neutral():
    """PA is sample-size context, never color-coded."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["PA"]["color_class"] == "stat-neutral"
    assert result["PA"]["actual"] == 60
    assert "z_score" not in result["PA"]
