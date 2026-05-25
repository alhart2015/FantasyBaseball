import pytest

from fantasy_baseball.category_odds import CategoryOdds, category_finish_odds


def test_symmetric_teams_give_uniform_odds():
    """10 identical teams -> each equally likely in any rank slot."""
    means = [100.0] * 10
    sds = [20.0] * 10
    odds = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert odds.first_pct == pytest.approx(10.0, abs=0.5)
    assert odds.top3_pct == pytest.approx(30.0, abs=0.5)
    assert odds.clear_wins == 0
    assert odds.opponents == 9


def test_dominant_team_first_and_top3_certain():
    means = [200.0] + [100.0] * 9
    sds = [1.0] * 10
    odds = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert odds.first_pct == pytest.approx(100.0, abs=0.01)
    assert odds.top3_pct == pytest.approx(100.0, abs=0.01)
    assert odds.clear_wins == 9


def test_worst_team_first_near_zero():
    means = [50.0] + [100.0] * 9
    sds = [1.0] * 10
    odds = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert odds.first_pct == pytest.approx(0.0, abs=0.01)
    assert odds.clear_wins == 0


def test_inverse_category_lowest_is_best():
    """ERA-style: user has the lowest (best) ERA -> wins."""
    means = [3.00] + [4.00] * 9
    sds = [0.05] * 10
    odds = category_finish_odds(means, sds, 0, higher_is_better=False)
    assert odds.first_pct == pytest.approx(100.0, abs=0.01)
    assert odds.clear_wins == 9

    # Highest (worst) ERA -> never first.
    means_bad = [5.00] + [3.00] * 9
    odds_bad = category_finish_odds(means_bad, sds, 0, higher_is_better=False)
    assert odds_bad.first_pct == pytest.approx(0.0, abs=0.01)


def test_zero_sd_is_deterministic_rank():
    """No uncertainty -> first is 100% iff strictly best; top3 by exact rank."""
    means = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
    sds = [0.0] * 10
    best = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert best.first_pct == pytest.approx(100.0, abs=0.01)
    assert best.top3_pct == pytest.approx(100.0, abs=0.01)
    assert best.clear_wins == 9

    # 4th-best team (index 3, value 70): 3 teams strictly better -> not top3.
    fourth = category_finish_odds(means, sds, 3, higher_is_better=True)
    assert fourth.first_pct == pytest.approx(0.0, abs=0.01)
    assert fourth.top3_pct == pytest.approx(0.0, abs=0.01)
    assert fourth.clear_wins == 6  # clears the 6 teams strictly below it


def test_clear_wins_respects_band_overlap():
    """wins counts only opponents whose +1SD is below the user's -1SD."""
    # User 100 +/-5 -> lower bound 95. Opp A 80 +/-5 -> upper 85 (cleared).
    # Opp B 92 +/-5 -> upper 97 (overlaps, not cleared).
    means = [100.0, 80.0, 92.0]
    sds = [5.0, 5.0, 5.0]
    odds = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert odds.clear_wins == 1
    assert odds.opponents == 2


def test_returns_categoryodds_dataclass():
    odds = category_finish_odds([1.0, 2.0], [0.0, 0.0], 0, higher_is_better=True)
    assert isinstance(odds, CategoryOdds)
    assert isinstance(odds.first_pct, float)
    assert isinstance(odds.clear_wins, int)
