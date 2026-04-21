import pandas as pd
import pytest

from fantasy_baseball.sgp.var import calculate_var


def test_var_simple_hitter():
    player = pd.Series({"name": "Test Hitter", "positions": ["1B"], "total_sgp": 20.0, "player_type": "hitter"})
    replacement_levels = {"1B": 12.0, "C": 8.0, "OF": 10.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(8.0)


def test_var_multi_position_uses_most_valuable():
    player = pd.Series({"name": "Multi Pos", "positions": ["SS", "2B"], "total_sgp": 18.0, "player_type": "hitter"})
    replacement_levels = {"SS": 8.0, "2B": 12.0, "C": 8.0, "OF": 10.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(10.0)


def test_var_pitcher():
    player = pd.Series({"name": "Test Pitcher", "positions": ["SP"], "total_sgp": 15.0, "player_type": "pitcher"})
    replacement_levels = {"P": 7.0, "C": 8.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(8.0)


def test_var_below_replacement_is_negative():
    player = pd.Series({"name": "Bad Player", "positions": ["C"], "total_sgp": 5.0, "player_type": "hitter"})
    replacement_levels = {"C": 8.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(-3.0)


def test_var_assigns_best_position():
    player = pd.Series({"name": "Multi", "positions": ["1B", "OF"], "total_sgp": 20.0, "player_type": "hitter"})
    replacement_levels = {"1B": 15.0, "OF": 10.0, "C": 8.0, "P": 7.0}
    var, pos = calculate_var(player, replacement_levels, return_position=True)
    assert var == pytest.approx(10.0)
    assert pos == "OF"
