from fantasy_baseball.analysis.pace import compute_overall_pace

CUTPOINTS = [3.0, 5.0, 8.0, 10.0]  # q16, q33, q66, q83


def _summary(dev):
    return {"sgp_dev": dev, "actual_sgp": dev, "expected_sgp": 0.0}


def test_bright_green_top_sixth():
    assert compute_overall_pace(_summary(11.0), CUTPOINTS)["color_class"] == "stat-hot-2"


def test_boundary_q83_is_bright_green():
    assert compute_overall_pace(_summary(10.0), CUTPOINTS)["color_class"] == "stat-hot-2"


def test_light_green():
    assert compute_overall_pace(_summary(9.0), CUTPOINTS)["color_class"] == "stat-hot-1"


def test_neutral_middle_third():
    assert compute_overall_pace(_summary(6.0), CUTPOINTS)["color_class"] == "stat-neutral"


def test_light_red():
    assert compute_overall_pace(_summary(4.0), CUTPOINTS)["color_class"] == "stat-cold-1"


def test_bright_red_bottom_sixth():
    assert compute_overall_pace(_summary(2.0), CUTPOINTS)["color_class"] == "stat-cold-2"


def test_boundary_q16_is_light_red():
    assert compute_overall_pace(_summary(3.0), CUTPOINTS)["color_class"] == "stat-cold-1"


def test_none_dev_is_neutral():
    out = compute_overall_pace(_summary(None), CUTPOINTS)
    assert out["color_class"] == "stat-neutral"
    assert out["sgp_dev"] is None


def test_missing_cutpoints_is_neutral():
    out = compute_overall_pace(_summary(11.0), None)
    assert out["color_class"] == "stat-neutral"
    assert out["sgp_dev"] == 11.0  # value preserved for the tooltip


def test_missing_summary_is_neutral():
    out = compute_overall_pace(None, CUTPOINTS)
    assert out["color_class"] == "stat-neutral"
    assert out["sgp_dev"] is None


def test_passthrough_fields():
    out = compute_overall_pace({"sgp_dev": -2.4, "actual_sgp": 5.1, "expected_sgp": 7.5}, CUTPOINTS)
    assert out["sgp_dev"] == -2.4
    assert out["actual_sgp"] == 5.1
    assert out["expected_sgp"] == 7.5
    assert out["color_class"] == "stat-cold-2"
