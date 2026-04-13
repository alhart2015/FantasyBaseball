import pytest

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
    """A hitter well below pace on SB gets cold color (1-2 SD = light cold)."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 0, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    # z = -1.4 -> between -1 and -2 SD = stat-cold-1
    assert result["SB"]["color_class"] == "stat-cold-1"
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

    # K: expected = 190 * (18/180) = 19, actual 22, ratio 1.16, z = 0.16/0.139 = 1.13 -> hot-1 (1-2 SD)
    assert result["K"]["color_class"] == "stat-hot-1"

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
    """z-scores between 1.0 and 2.0 should produce stat-hot-1 / stat-cold-1."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    # HR: actual=4, expected=3.0, ratio=1.33, z = 0.33/0.343 = 0.97 -> neutral (< 1.0 SD)
    actual = {"pa": 60, "r": 9, "hr": 4, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["HR"]["z_score"] < 1.0

    # With more extreme values: HR actual=6, expected=3.0, ratio=2.0, z = 1.0/0.343 = 2.92 -> hot-2
    actual2 = {"pa": 60, "r": 9, "hr": 6, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result2 = compute_player_pace(actual2, projected, "hitter")
    assert result2["HR"]["color_class"] == "stat-hot-2"
    assert result2["HR"]["z_score"] > 2.0

    # Moderate above: HR actual=5, expected=3.0, ratio=1.67, z = 0.67/0.343 = 1.94 -> hot-1 (1-2 SD)
    actual3 = {"pa": 60, "r": 9, "hr": 5, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result3 = compute_player_pace(actual3, projected, "hitter")
    assert result3["HR"]["color_class"] == "stat-hot-1"
    assert 1.0 < result3["HR"]["z_score"] < 2.0


def test_middle_sample_counting_colored_rates_neutral():
    """With 10-29 PA: counting stats colored, AVG neutral."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 20, "r": 3, "hr": 5, "rbi": 3, "sb": 0, "h": 10, "ab": 18}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] != "stat-neutral"
    assert result["AVG"]["color_class"] == "stat-neutral"


def test_small_absolute_diff_stays_neutral():
    """When actual is within 1 unit of expected, counting stats stay neutral
    regardless of z-score (e.g. 1 RBI vs 0.2 expected)."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    # 12 PA -> counting colored, but expected RBI = 90 * (12/600) = 1.8
    actual = {"pa": 12, "r": 2, "hr": 1, "rbi": 1, "sb": 1, "h": 3, "ab": 11}
    result = compute_player_pace(actual, projected, "hitter")
    # RBI: actual=1, expected=1.8, diff=0.8 < 1.0 -> neutral despite z-score
    assert result["RBI"]["color_class"] == "stat-neutral"
    # SB: actual=1, expected=0.2, diff=0.8 < 1.0 -> neutral (not bright green)
    assert result["SB"]["color_class"] == "stat-neutral"


def test_no_game_logs_shows_dashes():
    """Player with no actual stats gets 0 actuals and neutral colors."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {}  # no game logs at all
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["actual"] == 0
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["PA"]["actual"] == 0


def test_no_projection_shows_actuals_neutral():
    """Player not matched to projections — show actuals, all neutral."""
    projected = {}  # unmatched
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["actual"] == 3
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["HR"].get("z_score", 0) == 0.0


def test_significance_flags_in_pace_output():
    """Pace output includes 'significant' key per stat based on stabilization thresholds."""
    from fantasy_baseball.analysis.pace import compute_player_pace

    # Hitter with 100 PA — below HR threshold (170) but above for counting stats
    actual = {"pa": 100, "ab": 90, "h": 25, "r": 12, "hr": 5, "rbi": 15, "sb": 2}
    projected = {"pa": 600, "ab": 540, "h": 150, "r": 80, "hr": 25, "rbi": 85, "sb": 10, "avg": 0.278}
    result = compute_player_pace(actual, projected, "hitter")

    assert result["R"]["significant"] is False  # no threshold defined
    assert result["HR"]["significant"] is False  # 100 PA < 170
    assert result["RBI"]["significant"] is False  # no threshold defined
    assert result["SB"]["significant"] is False  # no threshold defined
    assert result["AVG"]["significant"] is False  # no threshold defined


def test_significance_pitcher():
    """Pitcher significance uses BF = ip*3 + h_allowed + bb."""
    from fantasy_baseball.analysis.pace import compute_player_pace

    # BF = 25*3 + 20 + 8 = 103 — above K (70) but below ERA (630) and WHIP (570)
    actual = {"ip": 25, "k": 30, "w": 2, "sv": 0, "er": 10, "bb": 8, "h_allowed": 20}
    projected = {"ip": 180, "w": 12, "k": 180, "sv": 0, "er": 60, "bb": 50, "h_allowed": 160, "era": 3.00, "whip": 1.17}
    result = compute_player_pace(actual, projected, "pitcher")

    assert result["K"]["significant"] is True   # 103 >= 70
    assert result["ERA"]["significant"] is False  # 103 < 630
    assert result["WHIP"]["significant"] is False  # 103 < 570
    assert result["W"]["significant"] is False  # no threshold defined
    assert result["SV"]["significant"] is False  # no threshold defined


def test_hitter_ros_deviation_sgp():
    """ROS deviation = (ros - preseason) / sgp_denom, positive = good."""
    preseason = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    ros = {"r": 100, "hr": 33, "rbi": 85, "sb": 12, "avg": 0.290}
    sgp = {"R": 20, "HR": 9, "RBI": 20, "SB": 8, "AVG": 0.005}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, preseason, "hitter", ros_stats=ros, sgp_denoms=sgp)

    # R: (100 - 90) / 20 = 0.5
    assert result["R"]["ros_deviation_sgp"] == pytest.approx(0.5, abs=0.01)
    # HR: (33 - 30) / 9 = 0.33
    assert result["HR"]["ros_deviation_sgp"] == pytest.approx(0.333, abs=0.01)
    # RBI: (85 - 90) / 20 = -0.25
    assert result["RBI"]["ros_deviation_sgp"] == pytest.approx(-0.25, abs=0.01)
    # SB: (12 - 10) / 8 = 0.25
    assert result["SB"]["ros_deviation_sgp"] == pytest.approx(0.25, abs=0.01)
    # AVG: (0.290 - 0.278) / 0.005 = 2.4
    assert result["AVG"]["ros_deviation_sgp"] == pytest.approx(2.4, abs=0.01)


def test_pitcher_ros_deviation_sgp():
    """Pitcher ROS deviation: ERA/WHIP are inverse (lower = positive deviation)."""
    preseason = {"ip": 180, "w": 12, "k": 190, "sv": 0, "er": 60, "bb": 50, "h_allowed": 150,
                 "era": 3.00, "whip": 1.11}
    ros = {"w": 14, "k": 200, "sv": 0, "era": 2.70, "whip": 1.05}
    sgp = {"W": 3, "K": 30, "SV": 7, "ERA": 0.15, "WHIP": 0.015}
    actual = {"ip": 18.0, "k": 22, "w": 1, "sv": 0, "er": 5, "bb": 5, "h_allowed": 16}
    result = compute_player_pace(actual, preseason, "pitcher", ros_stats=ros, sgp_denoms=sgp)

    # W: (14 - 12) / 3 = 0.667
    assert result["W"]["ros_deviation_sgp"] == pytest.approx(0.667, abs=0.01)
    # K: (200 - 190) / 30 = 0.333
    assert result["K"]["ros_deviation_sgp"] == pytest.approx(0.333, abs=0.01)
    # ERA: (2.70 - 3.00) / 0.15 = -2.0, flip sign -> +2.0 (lower ERA = good)
    assert result["ERA"]["ros_deviation_sgp"] == pytest.approx(2.0, abs=0.01)
    # WHIP: (1.05 - 1.11) / 0.015 = -4.0, flip sign -> +4.0
    assert result["WHIP"]["ros_deviation_sgp"] == pytest.approx(4.0, abs=0.01)


def test_ros_deviation_zero_when_no_ros():
    """When ros_stats or sgp_denoms are None, ros_deviation_sgp should be 0."""
    preseason = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, preseason, "hitter")
    assert result["HR"]["ros_deviation_sgp"] == 0.0
    assert result["AVG"]["ros_deviation_sgp"] == 0.0
