import math

from fantasy_baseball.analysis import breakout, hr_confirm


def _rec(**kw):
    base = dict(
        mlbam=1,
        prior_hr=0.04,
        surface_hr=0.06,
        actual_hr=0.05,
        pa=600.0,
        slg=0.520,
        xslg=0.470,
        brl_pa=0.08,
        xhr_rate=0.045,
    )
    base.update(kw)
    return base


def test_barrel_calibration_recovers_known_line():
    # y = 0.5*x + 0.01 exactly -> slope 0.5, intercept 0.01
    recs = [_rec(brl_pa=x / 100.0, surface_hr=0.5 * (x / 100.0) + 0.01) for x in range(2, 20)]
    slope, intercept = hr_confirm.fit_barrel_calibration(recs)
    assert math.isclose(slope, 0.5, rel_tol=1e-6)
    assert math.isclose(intercept, 0.01, abs_tol=1e-6)


def test_confirm_is_monotonic_in_overperformance():
    calib = (0.5, 0.01)
    # bigger surface-vs-expected gap -> lower confirm, for every candidate
    for cand in hr_confirm.CANDIDATES:
        small = hr_confirm.confirm(cand, _rec(surface_hr=0.045, slg=0.475), calib, scale=0.05)
        big = hr_confirm.confirm(cand, _rec(surface_hr=0.090, slg=0.600), calib, scale=0.05)
        assert big < small


def test_forward_hr_matches_breakout_w_for_stat_for_xslg_shipped():
    # The xslg candidate at the shipped 0.150 scale must equal w_for_stat's HR blend.
    rec = _rec()
    row = breakout.SkillLuckRow(
        mlbam=1,
        player_type="hitter",
        pa=rec["pa"],
        ip=0.0,
        age=None,
        barrel_pct=None,
        xslg=rec["xslg"],
        slg=rec["slg"],
        xba=None,
        ba=None,
        babip=None,
        xwoba=None,
        woba=None,
        k_pct=None,
        bb_pct=None,
    )
    w = breakout.w_for_stat("hr", row, "hitter", breakout.DEFAULT_WMAP)
    cv = hr_confirm.confirm("xslg", rec, (0.0, 0.0), scale=hr_confirm.SHIPPED_XSLG_SCALE)
    forward = hr_confirm.forward_hr(rec, cv)
    # forward = prior + w*(surface-prior); recover the effective w and compare.
    eff_w = (forward - rec["prior_hr"]) / (rec["surface_hr"] - rec["prior_hr"])
    assert math.isclose(eff_w, w, rel_tol=1e-9)


def test_tune_scale_picks_grid_argmax_on_fit():
    recs = [
        _rec(mlbam=i, surface_hr=0.04 + 0.001 * i, actual_hr=0.04 + 0.001 * i) for i in range(30)
    ]
    s = hr_confirm.tune_scale(recs, "xhr", (0.5, 0.01))
    assert s in hr_confirm.HRPA_SCALE_GRID


def test_verdict_rule_all_branches():
    V = hr_confirm._verdict_for
    # CI includes 0 -> no, regardless of tiers/MAE
    assert V(ci=(-0.01, 0.05), mae_delta=0.0, tier_signs=[-1, -1, -1]) == "no (CI includes 0)"
    # CI-positive but expected sign in <2 tiers -> level-confounded
    assert V(ci=(0.02, 0.08), mae_delta=0.0, tier_signs=[+1, +1, -1]) == (
        "level-confounded -- do not wire in"
    )
    # CI-positive, survives level-control, but MAE worse by > MAE_EPS -> inconsistent
    assert V(ci=(0.02, 0.08), mae_delta=0.001, tier_signs=[-1, -1, -1]) == (
        "CI-positive but MAE-inconsistent"
    )
    # CI-positive, survives level-control, MAE fine -> wire-in eligible
    assert V(ci=(0.02, 0.08), mae_delta=0.0, tier_signs=[-1, -1, -1]) == "wire-in eligible"
    # Fewer than 3 tiers (thin sample) -> inconclusive, not a false confound verdict
    assert V(ci=(0.02, 0.08), mae_delta=0.0, tier_signs=[]) == (
        "inconclusive (thin sample for level-control)"
    )
