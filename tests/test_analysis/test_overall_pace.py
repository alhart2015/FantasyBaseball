from fantasy_baseball.analysis.pace import compute_overall_pace


def test_overall_pace_all_hot():
    """All categories above pace -> overall hot."""
    pace = {
        "R": {"z_score": 2.5},
        "HR": {"z_score": 2.5},
        "RBI": {"z_score": 2.0},
        "SB": {"z_score": 2.5},
        "AVG": {"z_score": 2.5},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 2.4
    assert result["color_class"] == "stat-hot-2"


def test_overall_pace_all_cold():
    """All categories below pace -> overall cold."""
    pace = {
        "R": {"z_score": -1.5},
        "HR": {"z_score": -2.0},
        "RBI": {"z_score": -1.8},
        "SB": {"z_score": -1.2},
        "AVG": {"z_score": -1.5},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == -1.6
    assert result["color_class"] == "stat-cold-1"


def test_overall_pace_mixed_signals():
    """Mixed hot/cold categories -> net result near neutral."""
    pace = {
        "R": {"z_score": 1.5},
        "HR": {"z_score": -1.2},
        "RBI": {"z_score": -0.8},
        "SB": {"z_score": 0.3},
        "AVG": {"z_score": 0.2},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 0.0  # (1.5 - 1.2 - 0.8 + 0.3 + 0.2) / 5 = 0.0
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_skips_pa_ip():
    """PA and IP entries (no z_score) are excluded from the average."""
    pace = {
        "PA": {"actual": 60, "color_class": "stat-neutral"},
        "R": {"z_score": 2.5},
        "HR": {"z_score": 2.5},
        "RBI": {"z_score": 2.5},
        "SB": {"z_score": 2.5},
        "AVG": {"z_score": 2.5},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 2.5
    assert result["color_class"] == "stat-hot-2"


def test_overall_pace_skips_none_z_scores():
    """Categories with z_score=None are excluded."""
    pace = {
        "R": {"z_score": 1.5},
        "HR": {"z_score": None},
        "RBI": {"z_score": 1.5},
        "SB": {"z_score": 1.5},
        "AVG": {"z_score": 1.5},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 1.5
    assert result["color_class"] == "stat-hot-1"


def test_overall_pace_empty_dict():
    """Empty pace dict -> neutral with None avg_z."""
    result = compute_overall_pace({})
    assert result["avg_z"] is None
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_none_input():
    """None pace input -> neutral with None avg_z."""
    result = compute_overall_pace(None)
    assert result["avg_z"] is None
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_pitcher():
    """Pitcher categories work the same way."""
    pace = {
        "IP": {"actual": 18, "color_class": "stat-neutral"},
        "W": {"z_score": 0.5},
        "K": {"z_score": 1.3},
        "SV": {"z_score": 0.0},
        "ERA": {"z_score": -2.5},
        "WHIP": {"z_score": -0.3},
    }
    result = compute_overall_pace(pace)
    # avg = (0.5 + 1.3 + 0.0 - 2.5 - 0.3) / 5 = -1.0 / 5 = -0.2
    assert result["avg_z"] == -0.2
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_light_hot_threshold():
    """Average z exactly at 1.0 boundary -> neutral (uses > not >=)."""
    pace = {
        "R": {"z_score": 1.0},
        "HR": {"z_score": 1.0},
        "RBI": {"z_score": 1.0},
        "SB": {"z_score": 1.0},
        "AVG": {"z_score": 1.0},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 1.0
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_just_above_threshold():
    """Average z just above 1.0 -> stat-hot-1."""
    pace = {
        "R": {"z_score": 1.1},
        "HR": {"z_score": 1.1},
        "RBI": {"z_score": 1.1},
        "SB": {"z_score": 1.1},
        "AVG": {"z_score": 1.1},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 1.1
    assert result["color_class"] == "stat-hot-1"
