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


def _row(**kw):
    base = dict(mlbam=1, player_type="hitter", pa=600, ip=0.0, age=27.0, barrel_pct=None,
                xslg=None, slg=None, xba=None, ba=None, babip=None, xwoba=None, woba=None,
                k_pct=None, bb_pct=None)
    base.update(kw); return breakout.SkillLuckRow(**base)


def test_barrel_backed_hr_has_higher_w_than_unbacked():
    backed = _row(barrel_pct=0.16, xslg=0.560, slg=0.560)      # xSLG confirms the power
    lucky = _row(barrel_pct=0.06, xslg=0.410, slg=0.560)       # slg >> xslg -> luck
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("hr", backed, "hitter", p) > breakout.w_for_stat("hr", lucky, "hitter", p)


def test_low_sample_shrinks_w():
    big = _row(pa=600, barrel_pct=0.16, xslg=0.560, slg=0.560)
    tiny = _row(pa=60, barrel_pct=0.16, xslg=0.560, slg=0.560)
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("hr", tiny, "hitter", p) < breakout.w_for_stat("hr", big, "hitter", p)


def test_avg_mirage_low_w_when_xba_flat_babip_high():
    mirage = _row(pa=600, ba=0.320, xba=0.255, babip=0.380)
    real = _row(pa=600, ba=0.320, xba=0.315, babip=0.300)
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("avg", mirage, "hitter", p) < breakout.w_for_stat("avg", real, "hitter", p)
