from fantasy_baseball.analysis import breakout_backtest as bb


def test_marcel_weights_recent_years_more():
    league = {"hr": 0.03}
    # year 2023 hot (0.06), 2021 cold (0.01); recency weighting -> closer to hot
    hist = [(2023, {"hr": 0.06}), (2022, {"hr": 0.04}), (2021, {"hr": 0.01})]
    p = bb.marcel_prior(hist, league, age=27.0)
    assert 0.03 < p["hr"] < 0.06


def test_marcel_regresses_thin_history_toward_league():
    league = {"hr": 0.03}
    p = bb.marcel_prior([(2023, {"hr": 0.09})], league, age=27.0)  # one loud year
    assert p["hr"] < 0.09  # pulled toward league mean


def test_rate_mae_and_ruler():
    pred = {"hr": 0.05, "avg": 0.270}
    actual = {"hr": 0.04, "avg": 0.300}
    assert abs(bb.rate_mae(pred, actual) - (0.01 + 0.030) / 2) < 1e-9
    # ruler: higher HR rate -> higher score
    w = {"hr": 100.0, "avg": 10.0}
    assert bb.sgp_on_ruler({"hr": 0.05, "avg": 0.27}, w) > bb.sgp_on_ruler(
        {"hr": 0.02, "avg": 0.27}, w
    )


def test_ruler_penalizes_era():
    w = {"era": -1.0}  # negative weight (DEFAULT_RULER convention): lower ERA -> higher score
    assert bb.sgp_on_ruler({"era": 3.0}, w) > bb.sgp_on_ruler({"era": 5.0}, w)


def _mk_corpus():
    from fantasy_baseball.analysis import breakout

    surface = {"pa": 600, "ab": 540, "hr": 40, "r": 100, "rbi": 110, "sb": 10, "avg": 0.320}
    lucky = breakout.SkillLuckRow(
        mlbam=1,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=27.0,
        barrel_pct=0.06,
        xslg=0.41,
        slg=0.56,
        xba=0.255,
        ba=0.320,
        babip=0.385,
        xwoba=0.315,
        woba=0.380,
        k_pct=0.24,
        bb_pct=0.05,
    )
    real = breakout.SkillLuckRow(
        mlbam=2,
        player_type="hitter",
        pa=600,
        ip=0.0,
        age=26.0,
        barrel_pct=0.16,
        xslg=0.58,
        slg=0.58,
        xba=0.298,
        ba=0.300,
        babip=0.300,
        xwoba=0.382,
        woba=0.380,
        k_pct=0.20,
        bb_pct=0.10,
    )
    # next-year: lucky regresses to ~proj HR rate, real sustains
    actual_lucky = {"hr": 0.033, "avg": 0.262}
    actual_real = {"hr": 0.062, "avg": 0.298}
    hist = [(2022, {"hr": 0.033, "avg": 0.262})]
    zips_lucky = {"pa": 600, "hr": 21, "avg": 0.265}  # ZiPS already regressed the mirage
    zips_real = {"pa": 600, "hr": 37, "avg": 0.296}
    return {
        2023: {
            10: (surface, lucky, actual_lucky, hist, zips_lucky),
            20: (surface, real, actual_real, hist, zips_real),
        }
    }


def test_tune_wmap_returns_params_without_touching_report_years():
    from fantasy_baseball.analysis import breakout_backtest as bb
    from fantasy_baseball.analysis.breakout import WMapParams

    p = bb.tune_wmap(_mk_corpus(), fit_years=[2023])
    assert isinstance(p, WMapParams)


def test_run_backtest_three_estimators_and_ci():
    from fantasy_baseball.analysis import breakout_backtest as bb

    out = bb.run_backtest(_mk_corpus(), fit_years=[2023], report_years=[2023])
    # all three estimators present; skill-adjusted at least ties surface at ranking
    assert set(out["spearman"]) == {"surface", "skill_adjusted", "pure_zips"}
    assert out["spearman"]["skill_adjusted"] >= out["spearman"]["surface"]
    assert "ci_skill_vs_surface" in out and "ci_skill_vs_zips" in out
    assert "label_lift" in out and out["verdict"] in ("clears gate", "not good enough")
