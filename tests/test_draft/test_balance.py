import pandas as pd
import pytest

from fantasy_baseball.draft.balance import CategoryBalance
from fantasy_baseball.utils.constants import Category


def _make_hitter(name, r, hr, rbi, sb, avg, ab):
    return pd.Series({
        "name": name, "player_type": "hitter",
        "r": r, "hr": hr, "rbi": rbi, "sb": sb, "avg": avg, "ab": ab, "h": int(avg * ab),
    })


def _make_pitcher(name, w, k, sv, era, whip, ip):
    return pd.Series({
        "name": name, "player_type": "pitcher",
        "w": w, "k": k, "sv": sv, "era": era, "whip": whip, "ip": ip,
        "er": era * ip / 9, "bb": int(whip * ip * 0.3), "h_allowed": int(whip * ip * 0.7),
    })


class TestCategoryBalance:
    def test_empty_roster(self):
        bal = CategoryBalance()
        totals = bal.get_totals()
        assert totals[Category.HR] == 0
        assert totals[Category.K] is None  # no pitchers → None, not 0

    def test_add_hitter(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        totals = bal.get_totals()
        assert totals[Category.HR] == 45
        assert totals[Category.R] == 110
        assert totals[Category.RBI] == 120
        assert totals[Category.SB] == 5

    def test_add_multiple_hitters_sums(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        bal.add_player(_make_hitter("Betts", 105, 28, 85, 15, .287, 540))
        totals = bal.get_totals()
        assert totals[Category.HR] == 73
        assert totals[Category.SB] == 20

    def test_avg_is_weighted(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        bal.add_player(_make_hitter("Betts", 105, 28, 85, 15, .287, 540))
        totals = bal.get_totals()
        expected = (550 * .291 + 540 * .287) / (550 + 540)
        assert totals[Category.AVG] == pytest.approx(expected, abs=0.001)

    def test_add_pitcher(self):
        bal = CategoryBalance()
        bal.add_player(_make_pitcher("Cole", 15, 240, 0, 3.15, 1.05, 200))
        totals = bal.get_totals()
        assert totals[Category.W] == 15
        assert totals[Category.K] == 240
        assert totals[Category.SV] == 0

    def test_era_whip_weighted_by_ip(self):
        bal = CategoryBalance()
        bal.add_player(_make_pitcher("Cole", 15, 240, 0, 3.00, 1.00, 200))
        bal.add_player(_make_pitcher("Clase", 4, 70, 40, 2.00, 0.90, 70))
        totals = bal.get_totals()
        total_er = 3.00 * 200 / 9 + 2.00 * 70 / 9
        expected_era = total_er * 9 / 270
        assert totals[Category.ERA] == pytest.approx(expected_era, abs=0.01)

    def test_get_warnings_flags_weak_categories(self):
        bal = CategoryBalance()
        # Add 5 low-power, no-speed hitters to pass min threshold
        for i in range(5):
            bal.add_player(_make_hitter(f"Slappy{i}", 40, 2, 30, 0, .260, 400))
        warnings = bal.get_warnings()
        assert any("SB" in w for w in warnings)
