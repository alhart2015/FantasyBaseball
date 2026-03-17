import pytest
import pandas as pd
from fantasy_baseball.sgp.replacement import calculate_replacement_levels


def _make_player_pool():
    hitters = []
    for i in range(15):
        hitters.append({"name": f"Catcher_{i}", "positions": ["C"], "total_sgp": 20.0 - i, "player_type": "hitter"})
    for i in range(15):
        hitters.append({"name": f"FirstBase_{i}", "positions": ["1B"], "total_sgp": 25.0 - i, "player_type": "hitter"})
    for i in range(50):
        hitters.append({"name": f"Outfielder_{i}", "positions": ["OF"], "total_sgp": 30.0 - i * 0.5, "player_type": "hitter"})
    pitchers = []
    for i in range(100):
        pitchers.append({"name": f"Pitcher_{i}", "positions": ["SP"] if i < 70 else ["RP"], "total_sgp": 25.0 - i * 0.2, "player_type": "pitcher"})
    return pd.DataFrame(hitters + pitchers)


class TestReplacementLevels:
    def test_catcher_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert levels["C"] == pytest.approx(10.0)

    def test_first_base_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert levels["1B"] == pytest.approx(15.0)

    def test_of_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert levels["OF"] == pytest.approx(10.0)

    def test_pitcher_replacement_level(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert levels["P"] == pytest.approx(7.0)

    def test_all_starter_positions_have_levels(self):
        pool = _make_player_pool()
        levels = calculate_replacement_levels(pool)
        assert "C" in levels
        assert "OF" in levels
        assert "P" in levels
