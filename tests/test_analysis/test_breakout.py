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


def test_barrel_expected_rate_is_line_clamped_at_zero():
    # intercept + slope*brl_pa, clamped >= 0
    assert breakout.barrel_expected_rate(0.08, 0.5, 0.01) == 0.05
    assert breakout.barrel_expected_rate(0.0, 0.5, 0.01) == 0.01
    assert breakout.barrel_expected_rate(0.0, 1.0, -0.5) == 0.0  # clamp


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
    """Same surface HR jump (proj 20 -> surface 23 over 600 PA), only the
    xSLG-vs-SLG confirmation differs. LABEL_WEIGHTS["hr"]=65 puts the surface
    deviation at 65*(3/600)=0.325 -- comfortably inside the band where full
    confirmation (w~0.83) clears the 0.2 believed threshold (real breakout)
    while zero confirmation (w floors at reliability*0.5~0.42) does not.
    """
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    surface = {"pa": 600, "ab": 540, "hr": 23, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    backed = breakout.SkillLuckRow(
        mlbam=1,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=26.0,
        barrel_pct=0.16,
        xslg=0.580,
        slg=0.580,  # slg == xslg -> full HR confirmation
        xba=0.260,
        ba=0.260,
        babip=0.300,
        xwoba=0.380,
        woba=0.380,
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
        xslg=0.380,
        slg=0.560,  # slg >> xslg (gap 0.18 >= 0.15 scale) -> zero HR confirmation
        xba=0.260,
        ba=0.260,
        babip=0.380,
        xwoba=0.315,
        woba=0.380,
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
    """HR surge (proj 20 -> surface 23 over 600 PA), but Statcast xStats stay
    flat: slg >> xslg (gap 0.18 >= 0.15 scale) -> zero HR confirmation, so w
    floors at reliability*0.5~0.42. Surface deviation (0.325) clears the 0.2
    threshold; believed deviation (~0.135) does not -- a NET-no-believed-gain
    fluke, not a real breakout.
    """
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    surface = {"pa": 600, "ab": 540, "hr": 23, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
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
    assert r.surface_deviation >= 0.2
    assert abs(r.believed_deviation) < 0.2
    assert r.label == "lucky mirage"


def test_adjust_line_slump_labeled():
    """Drop in HR (20 -> 17) and AVG (.260 -> .255), but Statcast xStats
    (xslg/xba) stay strong while actual results (slg/ba) are suppressed by bad
    luck: slg/xslg gap 0.18 >= 0.15 scale and ba/xba gap 0.075 >= 0.06 scale ->
    zero confirmation on both, so w floors at reliability*0.5. Surface
    deviation (-0.525) clears the -0.2 threshold; believed deviation
    (~-0.178) does not -- the drop is largely unconfirmed -> slump, not real
    decline.
    """
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 8, "avg": 0.260}
    surface = {"pa": 600, "ab": 540, "hr": 17, "r": 80, "rbi": 80, "sb": 8, "avg": 0.255}
    row = breakout.SkillLuckRow(
        mlbam=20,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=27.0,
        barrel_pct=0.17,
        xslg=0.560,
        slg=0.380,
        xba=0.330,
        ba=0.255,
        babip=0.220,
        xwoba=0.370,
        woba=0.300,
        k_pct=0.20,
        bb_pct=0.09,
    )
    r = breakout.adjust_line(surface, proj, row, "hitter")
    assert r.surface_deviation <= -0.2
    assert abs(r.believed_deviation) < 0.2
    assert r.label == "slump"


def test_breakout_rows_surface_equals_kv_and_adjusted_regresses_luck(monkeypatch):
    import pandas as pd

    from fantasy_baseball.analysis import keeper_value

    # one lucky hitter: hot surface, flat xStats -> adjusted value below surface value
    board = pd.DataFrame(
        [
            {
                "player_id": "Lucky Guy::hitter",
                "name": "Lucky Guy",
                "player_type": "hitter",
                "positions": ["OF"],
                "fg_id": "20123",
                "pa": 600,
                "ab": 540,
                "hr": 40,
                "r": 100,
                "rbi": 110,
                "sb": 10,
                "avg": 0.320,
            }
        ]
    )
    projections = {
        "20123::hitter": {
            "pa": 600,
            "ab": 540,
            "hr": 20,
            "r": 80,
            "rbi": 80,
            "sb": 10,
            "avg": 0.260,
        }
    }
    skill_luck = {
        20123: breakout.SkillLuckRow(
            mlbam=665742,
            player_type="hitter",
            pa=600,
            ip=0.0,
            age=27.0,
            barrel_pct=0.06,
            xslg=0.410,
            slg=0.560,
            xba=0.255,
            ba=0.320,
            babip=0.385,
            xwoba=0.315,
            woba=0.380,
            k_pct=0.24,
            bb_pct=0.05,
        )
    }

    # stub keeper_value to a monotonic function of HR so the test is deterministic
    def fake_kv(pid, name, anchor, pos, ptype, zby, scale, **kw):
        return keeper_value.KeeperValueResult(
            pid, name, {2026: anchor["hr"]}, float(anchor["hr"]), [], None, None
        )

    monkeypatch.setattr(breakout, "_kv", fake_kv, raising=False)
    rows = breakout.breakout_rows(
        board,
        scale=None,
        indices={},
        skill_luck=skill_luck,
        projections=projections,
        base_year=2026,
        horizon=3,
        discount=0.8,
    )
    row = rows[0]
    assert row["surface_value"] == 40.0  # surface anchor untouched
    assert row["adjusted_value"] < row["surface_value"]  # luck regressed out
    assert row["delta"] == row["adjusted_value"] - row["surface_value"]
    assert row["label"] in breakout.LABELS
    # spec-required deviator flag + underlying numbers
    assert row["deviator"] is True  # HR 40 vs proj 20 is a big surface move
    assert abs(row["woba_xwoba_gap"] - (0.380 - 0.315)) < 1e-9
    assert row["babip"] == 0.385 and row["barrel_pct"] == 0.06


def test_breakout_rows_two_way_fgid_collision_degrades_to_no_data(monkeypatch):
    """Two-way players (e.g. Ohtani) share one FanGraphs id across their hitter
    and pitcher rows. skill_luck is keyed by bare fg_id, so a hitter row can be
    stored under the same key a pitcher board row looks up. The board row's
    player_type must be checked against the looked-up row's before use -- a
    mismatch must degrade to the no-data fallback, not borrow the wrong
    player's Statcast numbers (which would corrupt the pitcher's reliability
    and report line)."""
    import pandas as pd

    from fantasy_baseball.analysis import keeper_value

    # pitcher board row whose fg_id collides with a HITTER row in skill_luck
    board = pd.DataFrame(
        [
            {
                "player_id": "Two Way Guy::pitcher",
                "name": "Two Way Guy",
                "player_type": "pitcher",
                "positions": ["SP"],
                "fg_id": "30001",
                "ip": 150,
                "k": 180,
                "w": 12,
                "sv": 0,
                "era": 3.20,
                "whip": 1.10,
            }
        ]
    )
    projections = {
        "30001::pitcher": {
            "ip": 150,
            "k": 160,
            "w": 10,
            "sv": 0,
            "era": 3.60,
            "whip": 1.20,
        }
    }
    # mismatched row: player_type="hitter" stored under the pitcher's fg_id key
    skill_luck = {
        30001: breakout.SkillLuckRow(
            mlbam=999999,
            player_type="hitter",
            pa=600,
            ip=0.0,
            age=27.0,
            barrel_pct=0.20,
            xslg=0.600,
            slg=0.600,
            xba=0.310,
            ba=0.310,
            babip=0.340,
            xwoba=0.400,
            woba=0.400,
            k_pct=0.18,
            bb_pct=0.10,
        )
    }

    def fake_kv(pid, name, anchor, pos, ptype, zby, scale, **kw):
        return keeper_value.KeeperValueResult(
            pid, name, {2026: anchor["k"]}, float(anchor["k"]), [], None, None
        )

    monkeypatch.setattr(breakout, "_kv", fake_kv, raising=False)
    rows = breakout.breakout_rows(
        board,
        scale=None,
        indices={},
        skill_luck=skill_luck,
        projections=projections,
        base_year=2026,
        horizon=3,
        discount=0.8,
    )
    row = rows[0]
    # degraded gracefully: adjusted == surface, stable/not-a-deviator, no borrowed data
    assert row["adjusted_value"] == row["surface_value"]
    assert row["label"] == "stable"
    assert row["deviator"] is False
    assert row["reason"] == "no skill/luck data"
    assert row["woba_xwoba_gap"] is None
    assert row["babip"] is None
    assert row["barrel_pct"] is None
    assert row["k_pct"] is None
    assert row["bb_pct"] is None


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
