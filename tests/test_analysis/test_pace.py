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


def test_hitter_avg_above_projection():
    """Hitter batting well above projected AVG gets hot color."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 120, "r": 18, "hr": 6, "rbi": 18, "sb": 2, "h": 42, "ab": 108}
    # actual AVG = 42/108 = .389, proj .278, dev = +0.111
    # z = 0.111 / (0.103 * 0.278) = 0.111 / 0.0286 = 3.88 -> stat-hot-2
    result = compute_player_pace(actual, projected, "hitter")
    assert "AVG" in result
    assert result["AVG"]["color_class"] == "stat-hot-2"
    assert result["AVG"]["actual"] == 0.389  # rounded to 3 places


def test_hitter_avg_neutral_below_min_sample():
    """AVG with < 30 PA should be neutral regardless of value."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 20, "r": 3, "hr": 1, "rbi": 3, "sb": 0, "h": 10, "ab": 18}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["AVG"]["color_class"] == "stat-neutral"


def test_counting_neutral_below_min_sample():
    """With < 10 PA, counting stats should be neutral too."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 5, "r": 3, "hr": 2, "rbi": 3, "sb": 0, "h": 3, "ab": 5}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["AVG"]["color_class"] == "stat-neutral"


def test_pitcher_counting_and_rates():
    """Pitcher with good K rate and bad ERA."""
    projected = {"ip": 180, "w": 12, "k": 190, "sv": 0, "er": 60, "bb": 50, "h_allowed": 150,
                 "era": 3.00, "whip": 1.11}
    actual = {"ip": 18.0, "k": 22, "w": 1, "sv": 0, "er": 10, "bb": 5, "h_allowed": 16}
    result = compute_player_pace(actual, projected, "pitcher")

    assert "IP" in result
    assert result["IP"]["color_class"] == "stat-neutral"

    # K: expected = 190 * (18/180) = 19, actual 22, ratio 1.16, z = 0.16/0.139 = 1.13 -> hot-2
    assert result["K"]["color_class"] == "stat-hot-2"

    # ERA: actual = 10*9/18 = 5.00, proj 3.00, dev = +2.0
    # z = 2.0 / (0.252 * 3.00) = 2.0 / 0.756 = 2.65
    # ERA is inverse -> negate -> -2.65 -> stat-cold-2
    assert result["ERA"]["color_class"] == "stat-cold-2"
    assert result["ERA"]["z_score"] < -1.0

    # WHIP: actual = (5+16)/18 = 1.167, proj 1.11, dev = +0.057
    # z = 0.057 / (0.143 * 1.11) = 0.057 / 0.159 = 0.36
    # WHIP is inverse -> negate -> -0.36 -> neutral
    assert result["WHIP"]["color_class"] == "stat-neutral"


def test_pitcher_era_neutral_below_min_ip():
    """ERA with < 10 IP should be neutral, but counting stats with >= 5 IP should be colored."""
    projected = {"ip": 180, "w": 12, "k": 190, "sv": 0, "er": 60, "bb": 50, "h_allowed": 150,
                 "era": 3.00, "whip": 1.11}
    actual = {"ip": 5.0, "k": 15, "w": 0, "sv": 0, "er": 0, "bb": 1, "h_allowed": 3}
    result = compute_player_pace(actual, projected, "pitcher")
    assert result["ERA"]["color_class"] == "stat-neutral"
    assert result["WHIP"]["color_class"] == "stat-neutral"
    # K: ratio = 15/5.28 = 2.84, z = 1.84/0.139 = 13.2 -> stat-hot-2
    assert result["K"]["color_class"] == "stat-hot-2"


def test_intermediate_color_classes():
    """z-scores between 0.5 and 1.0 should produce stat-hot-1 / stat-cold-1."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 4, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] == "stat-hot-1"
    assert 0.5 < result["HR"]["z_score"] < 1.0


def test_middle_sample_counting_colored_rates_neutral():
    """With 10-29 PA: counting stats colored, AVG neutral."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 20, "r": 3, "hr": 5, "rbi": 3, "sb": 0, "h": 10, "ab": 18}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] != "stat-neutral"
    assert result["AVG"]["color_class"] == "stat-neutral"
