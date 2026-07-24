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
