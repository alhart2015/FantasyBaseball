"""Tests for breakout.py skill/luck classifier."""

from fantasy_baseball.analysis import breakout


def test_shapes_exist():
    """Verify SkillLuckRow and LABELS are defined and importable."""
    row = breakout.SkillLuckRow(
        mlbam=1,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=27.0,
        barrel_pct=None,
        xslg=None,
        slg=None,
        xba=None,
        ba=None,
        babip=None,
        xwoba=None,
        woba=None,
        k_pct=None,
        bb_pct=None,
    )
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
    base = dict(
        mlbam=1,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=27.0,
        barrel_pct=None,
        xslg=None,
        slg=None,
        xba=None,
        ba=None,
        babip=None,
        xwoba=None,
        woba=None,
        k_pct=None,
        bb_pct=None,
    )
    base.update(kw)
    return breakout.SkillLuckRow(**base)


def test_barrel_backed_hr_has_higher_w_than_unbacked():
    backed = _row(barrel_pct=0.16, xslg=0.560, slg=0.560)  # xSLG confirms the power
    lucky = _row(barrel_pct=0.06, xslg=0.410, slg=0.560)  # slg >> xslg -> luck
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("hr", backed, "hitter", p) > breakout.w_for_stat(
        "hr", lucky, "hitter", p
    )


def test_low_sample_shrinks_w():
    big = _row(pa=600, barrel_pct=0.16, xslg=0.560, slg=0.560)
    tiny = _row(pa=60, barrel_pct=0.16, xslg=0.560, slg=0.560)
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("hr", tiny, "hitter", p) < breakout.w_for_stat(
        "hr", big, "hitter", p
    )


def test_avg_mirage_low_w_when_xba_flat_babip_high():
    mirage = _row(pa=600, ba=0.320, xba=0.255, babip=0.380)
    real = _row(pa=600, ba=0.320, xba=0.315, babip=0.300)
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("avg", mirage, "hitter", p) < breakout.w_for_stat(
        "avg", real, "hitter", p
    )


def test_adjust_line_orders_skill_above_luck_at_equal_surface():
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    surface = {"pa": 600, "ab": 540, "hr": 40, "r": 100, "rbi": 110, "sb": 10, "avg": 0.300}
    backed = breakout.SkillLuckRow(
        mlbam=1,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=26.0,
        barrel_pct=0.16,
        xslg=0.580,
        slg=0.580,
        xba=0.298,
        ba=0.300,
        babip=0.300,
        xwoba=0.380,
        woba=0.382,
        k_pct=0.20,
        bb_pct=0.10,
    )
    lucky = breakout.SkillLuckRow(
        mlbam=2,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=26.0,
        barrel_pct=0.06,
        xslg=0.410,
        slg=0.580,
        xba=0.255,
        ba=0.300,
        babip=0.380,
        xwoba=0.315,
        woba=0.382,
        k_pct=0.24,
        bb_pct=0.06,
    )
    rb = breakout.adjust_line(surface, proj, backed, "hitter")
    rl = breakout.adjust_line(surface, proj, lucky, "hitter")
    # same surface, but the barrel-backed hitter keeps more of the HR jump
    assert rb.adjusted_line["hr"] > rl.adjusted_line["hr"]
    assert rb.label == "real breakout"
    assert rl.label in ("lucky mirage", "stable")
    assert "hr" in rb.reason.lower() or "barrel" in rb.reason.lower()


def test_adjust_line_low_confidence_small_sample():
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    surface = {"pa": 80, "ab": 70, "hr": 8, "r": 15, "rbi": 16, "sb": 1, "avg": 0.330}
    row = breakout.SkillLuckRow(
        mlbam=3,
        player_type="hitter",
        pa=80,
        ip=0.0,
        age=24.0,
        barrel_pct=0.14,
        xslg=0.520,
        slg=0.560,
        xba=0.290,
        ba=0.330,
        babip=0.360,
        xwoba=0.360,
        woba=0.370,
        k_pct=0.22,
        bb_pct=0.08,
    )
    r = breakout.adjust_line(surface, proj, row, "hitter")
    assert r.confidence == "low"
    # small sample -> adjusted HR rate pulled toward the projection rate, not the hot surface
    assert r.adjusted_line["hr"] < surface["hr"]


def test_adjust_line_lucky_mirage_labeled():
    """Big HR surge, but Statcast xStats stay flat: slg >> xslg, woba >> xwoba,
    low barrel. w regresses the HR jump away almost entirely -> lucky mirage,
    not real breakout.
    """
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    surface = {"pa": 600, "ab": 540, "hr": 40, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    row = breakout.SkillLuckRow(
        mlbam=10,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=27.0,
        barrel_pct=0.05,
        xslg=0.380,
        slg=0.560,
        xba=0.250,
        ba=0.260,
        babip=0.330,
        xwoba=0.310,
        woba=0.360,
        k_pct=0.24,
        bb_pct=0.06,
    )
    r = breakout.adjust_line(surface, proj, row, "hitter")
    assert r.surface_deviation > 0
    assert abs(r.surface_deviation) > 2.0 * abs(r.believed_deviation)
    assert r.label == "lucky mirage"


def test_adjust_line_slump_labeled():
    """Big drop in HR/R/RBI/AVG, but Statcast xStats (xslg/xba/xwoba, high
    barrel) stay strong and actual results (slg/ba/woba) are suppressed by bad
    luck (low BABIP). The drop is largely unconfirmed -> slump, not real decline.
    """
    proj = {"pa": 600, "ab": 540, "hr": 35, "r": 100, "rbi": 105, "sb": 8, "avg": 0.290}
    surface = {"pa": 600, "ab": 540, "hr": 15, "r": 65, "rbi": 60, "sb": 4, "avg": 0.235}
    row = breakout.SkillLuckRow(
        mlbam=20,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=27.0,
        barrel_pct=0.17,
        xslg=0.560,
        slg=0.380,
        xba=0.300,
        ba=0.235,
        babip=0.230,
        xwoba=0.370,
        woba=0.300,
        k_pct=0.20,
        bb_pct=0.09,
    )
    r = breakout.adjust_line(surface, proj, row, "hitter")
    assert r.surface_deviation < 0
    assert abs(r.surface_deviation) > 2.0 * abs(r.believed_deviation)
    assert r.label == "slump"


def test_adjust_line_real_decline_labeled():
    proj = {"pa": 600, "ab": 540, "hr": 35, "r": 100, "rbi": 105, "sb": 8, "avg": 0.290}
    surface = {"pa": 600, "ab": 540, "hr": 15, "r": 65, "rbi": 60, "sb": 4, "avg": 0.235}
    # underlying confirms the drop is real: xSLG/xBA/xwOBA all down with the surface
    row = breakout.SkillLuckRow(
        mlbam=4,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=34.0,
        barrel_pct=0.05,
        xslg=0.360,
        slg=0.360,
        xba=0.238,
        ba=0.235,
        babip=0.270,
        xwoba=0.300,
        woba=0.302,
        k_pct=0.27,
        bb_pct=0.06,
    )
    r = breakout.adjust_line(surface, proj, row, "hitter")
    assert r.label == "real decline"
