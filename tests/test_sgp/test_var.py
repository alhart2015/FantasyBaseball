import pandas as pd
import pytest

from fantasy_baseball.sgp.var import calculate_var


def test_var_simple_hitter():
    player = pd.Series(
        {"name": "Test Hitter", "positions": ["1B"], "total_sgp": 20.0, "player_type": "hitter"}
    )
    replacement_levels = {"1B": 12.0, "C": 8.0, "OF": 10.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(8.0)


def test_var_multi_position_uses_most_valuable():
    player = pd.Series(
        {"name": "Multi Pos", "positions": ["SS", "2B"], "total_sgp": 18.0, "player_type": "hitter"}
    )
    replacement_levels = {"SS": 8.0, "2B": 12.0, "C": 8.0, "OF": 10.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(10.0)


def test_var_pitcher():
    player = pd.Series(
        {"name": "Test Pitcher", "positions": ["SP"], "total_sgp": 15.0, "player_type": "pitcher"}
    )
    replacement_levels = {"P": 7.0, "C": 8.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(8.0)


def test_var_below_replacement_is_negative():
    player = pd.Series(
        {"name": "Bad Player", "positions": ["C"], "total_sgp": 5.0, "player_type": "hitter"}
    )
    replacement_levels = {"C": 8.0, "P": 7.0}
    var = calculate_var(player, replacement_levels)
    assert var == pytest.approx(-3.0)


def test_var_assigns_best_position():
    player = pd.Series(
        {"name": "Multi", "positions": ["1B", "OF"], "total_sgp": 20.0, "player_type": "hitter"}
    )
    replacement_levels = {"1B": 15.0, "OF": 10.0, "C": 8.0, "P": 7.0}
    var, pos = calculate_var(player, replacement_levels, return_position=True)
    assert var == pytest.approx(10.0)
    assert pos == "OF"


def test_var_starter_uses_sp_floor():
    """A pitcher stored as bare 'P' with starter-volume IP nets against the
    SP floor. best_position stays 'P' (the slot), not 'SP'."""
    player = pd.Series(
        {
            "name": "Ace",
            "positions": ["P"],
            "total_sgp": 15.0,
            "player_type": "pitcher",
            "ip": 190.0,
        }
    )
    levels = {"SP": 7.6, "RP": 6.3, "P": 6.65}
    var, pos = calculate_var(player, levels, return_position=True)
    assert var == pytest.approx(15.0 - 7.6)
    assert pos == "P"


def test_var_reliever_uses_rp_floor():
    """A low-IP pitcher (closer) nets against the RP floor."""
    player = pd.Series(
        {
            "name": "Closer",
            "positions": ["P"],
            "total_sgp": 12.0,
            "player_type": "pitcher",
            "ip": 65.0,
        }
    )
    levels = {"SP": 7.6, "RP": 6.3, "P": 6.65}
    var, pos = calculate_var(player, levels, return_position=True)
    assert var == pytest.approx(12.0 - 6.3)
    assert pos == "P"


def test_var_pitcher_falls_back_to_p_without_role_floors():
    """Backward compat: a demand-only levels dict (no SP/RP keys) still values
    pitchers against the unified 'P' floor."""
    player = pd.Series(
        {"name": "SP", "positions": ["P"], "total_sgp": 15.0, "player_type": "pitcher", "ip": 190.0}
    )
    levels = {"P": 7.0, "C": 8.0}
    var = calculate_var(player, levels)
    assert var == pytest.approx(8.0)


def test_var_pitcher_role_from_ip_not_position_token():
    """The board defaults unmatched pitchers to ['SP'], so role must come from
    IP, not the token -- a low-IP closer mislabeled 'SP' still nets against the
    RP floor (regression for the Emmanuel Clase misclassification)."""
    player = pd.Series(
        {
            "name": "Mislabeled Closer",
            "positions": ["SP"],
            "total_sgp": 12.0,
            "player_type": "pitcher",
            "ip": 21.0,
        }
    )
    levels = {"SP": 7.6, "RP": 6.3, "P": 6.65}
    var = calculate_var(player, levels)
    assert var == pytest.approx(12.0 - 6.3)
