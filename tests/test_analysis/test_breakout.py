"""Tests for breakout.py skill/luck classifier."""
from fantasy_baseball.analysis import breakout


def test_shapes_exist():
    """Verify SkillLuckRow and LABELS are defined and importable."""
    row = breakout.SkillLuckRow(
        mlbam=1, player_type="hitter", pa=600, ip=0.0, age=27.0,
        barrel_pct=None, xslg=None, slg=None, xba=None, ba=None, babip=None,
        xwoba=None, woba=None, k_pct=None, bb_pct=None)
    assert row.player_type == "hitter"
    assert "real breakout" in breakout.LABELS


def test_line_rates_hitter():
    line = {"pa": 600, "ab": 540, "h": 162, "hr": 30, "r": 90, "rbi": 100, "sb": 20, "avg": 0.300}
    rates = breakout.line_rates(line, "hitter")
    assert abs(rates["hr"] - 30 / 600) < 1e-9
    assert abs(rates["avg"] - 0.300) < 1e-9
    assert abs(rates["sb"] - 20 / 600) < 1e-9


def test_line_rates_zero_pa_is_safe():
    rates = breakout.line_rates({"pa": 0, "hr": 0, "avg": 0.0}, "hitter")
    assert rates["hr"] == 0.0  # no ZeroDivision, no NaN
