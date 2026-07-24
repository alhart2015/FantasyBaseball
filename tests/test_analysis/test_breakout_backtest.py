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
