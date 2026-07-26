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


def test_spearman_tie_handling_pinned():
    """Pin _spearman's tie behavior. It uses ordinal (index-broken) ranks, not
    averaged ranks; _bootstrap_diff resamples with replacement, so ties occur on
    nearly every iteration and swapping in tie-averaged ranks (e.g. scipy.rankdata)
    would shift the CI bounds and could flip the backtest verdict. This test fails
    loudly if the ranker is ever changed, so that stays a deliberate decision."""
    # No ties, perfectly monotonic -> +1.0.
    assert abs(bb._spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) - 1.0) < 1e-9
    # Tied x-values: ordinal ranking yields exactly -0.5 here; tie-averaging would
    # give ~-0.866 instead. Pinning -0.5 catches a silent swap to averaged ranks.
    assert abs(bb._spearman([5.0, 5.0, 1.0], [1.0, 2.0, 3.0]) - (-0.5)) < 1e-9


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


def _mk_hitter_row(mlbam, pa, xslg, slg, xba, ba, age=27.0):
    from fantasy_baseball.analysis import breakout

    return breakout.SkillLuckRow(
        mlbam=mlbam,
        player_type="hitter",
        pa=pa,
        ip=0.0,
        age=age,
        barrel_pct=0.10,
        xslg=xslg,
        slg=slg,
        xba=xba,
        ba=ba,
        babip=0.310,
        xwoba=0.330,
        woba=0.330,
        k_pct=0.22,
        bb_pct=0.08,
    )


def _mk_candidate_year(p_hr, q_hr, mlbam_p, mlbam_q, actual_p_hr, actual_q_hr):
    """One backtest-year's corpus data: two hitters (P, Q) sharing a common marcel-prior
    baseline (hist hr rate 0.020, avg 0.250) but far enough from it in surface HR to
    clear CANDIDATE_DEVIATION (0.15) on both sides.

    P is a small, fully-xStat-confirmed HR bump (xslg == slg); Q is a large, unconfirmed
    HR bump (xslg << slg, i.e. Statcast contradicts the surface power). Whether the
    actual next-year outcome favors P or Q (``actual_p_hr`` vs ``actual_q_hr``) decides
    which confirm_weight the grid search in tune_wmap rewards for that year. Building
    the 2021 (fit) and 2023 (report) years with opposite P/Q outcomes makes the two
    years genuinely pull tune_wmap toward *different* WMapParams when tuned in
    isolation -- which is what makes the leakage probes below non-degenerate. (A
    same-shaped-every-year corpus would let every confirm_weight tie on rank
    correlation, so a leakage bug that read the wrong year would go undetected.)
    """
    hist = [(2020, {"hr": 0.020, "avg": 0.250})]
    surface_p = {"pa": 600, "ab": 540, "hr": p_hr, "r": 90, "rbi": 90, "sb": 5, "avg": 0.270}
    surface_q = {"pa": 600, "ab": 540, "hr": q_hr, "r": 90, "rbi": 90, "sb": 5, "avg": 0.270}
    slg_p = surface_p["hr"] / surface_p["ab"] * 4  # confirmed: xslg == slg
    p_row = _mk_hitter_row(mlbam_p, 600, xslg=slg_p, slg=slg_p, xba=0.270, ba=0.270)
    q_row = _mk_hitter_row(mlbam_q, 600, xslg=0.35, slg=0.90, xba=0.230, ba=0.270)
    return {
        mlbam_p: (surface_p, p_row, {"hr": actual_p_hr, "avg": 0.270}, hist, None),
        mlbam_q: (surface_q, q_row, {"hr": actual_q_hr, "avg": 0.270}, hist, None),
    }


def _mk_distinct_years_corpus():
    """fit_years=[2021], report_years=[2023]; each year has its own 2 qualifying
    candidates (verified against CANDIDATE_DEVIATION) and its own actual-outcome shape,
    so the two years are not interchangeable for tune_wmap's grid search."""
    return {
        2021: _mk_candidate_year(28, 55, 1, 2, actual_p_hr=0.045, actual_q_hr=0.090),
        2023: _mk_candidate_year(34, 73, 101, 102, actual_p_hr=0.090, actual_q_hr=0.030),
    }


def test_tune_wmap_ignores_report_year_data():
    """tune_wmap(fit_years=[2021]) must return identical params regardless of what the
    2023 (report-year) rows contain -- proving tune_wmap never reads report-year data.
    Fails if a future edit ever folds report_years into the grid-search corpus (e.g.
    tuning over all corpus years instead of just fit_years)."""
    from fantasy_baseball.analysis import breakout_backtest as bb

    corpus = _mk_distinct_years_corpus()
    p1 = bb.tune_wmap(corpus, fit_years=[2021])

    corpus_mutated = dict(corpus)
    corpus_mutated[2023] = _mk_candidate_year(
        90, 15, 101, 102, actual_p_hr=0.005, actual_q_hr=0.400
    )
    p2 = bb.tune_wmap(corpus_mutated, fit_years=[2021])

    assert corpus[2021] is corpus_mutated[2021]  # sanity: 2021 rows are untouched
    assert p1 == p2


def test_run_backtest_uses_fit_tuned_params_not_report():
    """run_backtest(params=None) must tune on fit_years and evaluate those fixed params
    on report_years -- never re-tune on report_years. By construction (see
    _mk_candidate_year), tuning on [2021] alone picks a different confirm_weight than
    tuning on [2023] alone would, so if run_backtest's internal auto-tune ever read
    report_years instead of fit_years, the resulting skill_adjusted spearman on
    report_years would diverge from this fit-tuned, pinned-params run."""
    from fantasy_baseball.analysis import breakout_backtest as bb

    corpus = _mk_distinct_years_corpus()
    auto = bb.run_backtest(corpus, fit_years=[2021], report_years=[2023])
    pinned = bb.run_backtest(
        corpus,
        fit_years=[2021],
        report_years=[2023],
        params=bb.tune_wmap(corpus, fit_years=[2021]),
    )
    assert auto["spearman"] == pinned["spearman"]
