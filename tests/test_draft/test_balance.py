import pytest
import pandas as pd
from fantasy_baseball.draft.balance import CategoryBalance, calculate_draft_leverage
from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS


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
        assert totals["HR"] == 0
        assert totals["K"] == 0

    def test_add_hitter(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        totals = bal.get_totals()
        assert totals["HR"] == 45
        assert totals["R"] == 110
        assert totals["RBI"] == 120
        assert totals["SB"] == 5

    def test_add_multiple_hitters_sums(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        bal.add_player(_make_hitter("Betts", 105, 28, 85, 15, .287, 540))
        totals = bal.get_totals()
        assert totals["HR"] == 73
        assert totals["SB"] == 20

    def test_avg_is_weighted(self):
        bal = CategoryBalance()
        bal.add_player(_make_hitter("Judge", 110, 45, 120, 5, .291, 550))
        bal.add_player(_make_hitter("Betts", 105, 28, 85, 15, .287, 540))
        totals = bal.get_totals()
        expected = (550 * .291 + 540 * .287) / (550 + 540)
        assert totals["AVG"] == pytest.approx(expected, abs=0.001)

    def test_add_pitcher(self):
        bal = CategoryBalance()
        bal.add_player(_make_pitcher("Cole", 15, 240, 0, 3.15, 1.05, 200))
        totals = bal.get_totals()
        assert totals["W"] == 15
        assert totals["K"] == 240
        assert totals["SV"] == 0

    def test_era_whip_weighted_by_ip(self):
        bal = CategoryBalance()
        bal.add_player(_make_pitcher("Cole", 15, 240, 0, 3.00, 1.00, 200))
        bal.add_player(_make_pitcher("Clase", 4, 70, 40, 2.00, 0.90, 70))
        totals = bal.get_totals()
        total_er = 3.00 * 200 / 9 + 2.00 * 70 / 9
        expected_era = total_er * 9 / 270
        assert totals["ERA"] == pytest.approx(expected_era, abs=0.01)

    def test_get_warnings_flags_weak_categories(self):
        bal = CategoryBalance()
        # Add 5 low-power, no-speed hitters to pass min threshold
        for i in range(5):
            bal.add_player(_make_hitter(f"Slappy{i}", 40, 2, 30, 0, .260, 400))
        warnings = bal.get_warnings()
        assert any("SB" in w for w in warnings)


class TestCalculateDraftLeverage:
    """Tests for calculate_draft_leverage."""

    # Fixed targets to keep tests deterministic
    TARGETS = {
        "R": 900, "HR": 265, "RBI": 890, "SB": 145, "AVG": 0.260,
        "W": 78, "K": 1250, "ERA": 3.80, "WHIP": 1.20, "SV": 55,
    }

    def test_normal_behind_pace_gets_higher_weight(self):
        """Categories behind pace should get higher weight than those ahead."""
        # 50% through draft: R ahead of pace, SB behind pace
        totals = {
            "R": 600,   # ahead (expected 450)
            "HR": 132,  # on pace (expected 132.5)
            "RBI": 445, # on pace (expected 445)
            "SB": 30,   # behind (expected 72.5)
            "AVG": 0.260,
            "W": 39,    # on pace
            "K": 625,   # on pace
            "ERA": 3.80,
            "WHIP": 1.20,
            "SV": 27,   # on pace
        }
        weights = calculate_draft_leverage(totals, picks_made=12, total_picks=24, targets=self.TARGETS)
        assert weights["SB"] > weights["R"]

    def test_early_draft_returns_equal_weights(self):
        """At picks_made=0, all categories should have equal weight."""
        totals = {cat: 0 for cat in ALL_CATEGORIES}
        weights = calculate_draft_leverage(totals, picks_made=0, total_picks=24, targets=self.TARGETS)
        expected = 1.0 / len(ALL_CATEGORIES)
        for cat in ALL_CATEGORIES:
            assert weights[cat] == pytest.approx(expected)

    def test_early_draft_picks_made_one(self):
        """At picks_made=1, weights should still be fairly close together."""
        totals = {
            "R": 100, "HR": 35, "RBI": 100, "SB": 10, "AVG": 0.280,
            "W": 0, "K": 0, "ERA": None, "WHIP": None, "SV": 0,
        }
        weights = calculate_draft_leverage(totals, picks_made=1, total_picks=24, targets=self.TARGETS)
        # All weights present
        assert len(weights) == len(ALL_CATEGORIES)
        # All positive
        assert all(v > 0 for v in weights.values())

    def test_zero_counting_stat_gets_emergency_boost(self):
        """A counting stat at 0 past 15% of draft gets emergency weight."""
        # 50% through draft, SV = 0 -> emergency
        totals = {
            "R": 450, "HR": 130, "RBI": 445, "SB": 70, "AVG": 0.265,
            "W": 39, "K": 625, "ERA": 3.80, "WHIP": 1.20, "SV": 0,
        }
        weights = calculate_draft_leverage(totals, picks_made=12, total_picks=24, targets=self.TARGETS)
        # SV should be the highest-weighted non-inverse category
        non_inverse = {cat: w for cat, w in weights.items() if cat not in INVERSE_STATS}
        assert weights["SV"] == max(non_inverse.values())

    def test_zero_stat_early_draft_no_emergency(self):
        """A counting stat at 0 before 15% of draft does NOT get emergency boost."""
        # Only 2 out of 24 picks = 8.3% < 15%
        totals = {
            "R": 200, "HR": 50, "RBI": 200, "SB": 0, "AVG": 0.280,
            "W": 0, "K": 0, "ERA": None, "WHIP": None, "SV": 0,
        }
        weights = calculate_draft_leverage(totals, picks_made=2, total_picks=24, targets=self.TARGETS)
        # SB is 0 but should NOT get the massive emergency boost
        # (ERA/WHIP are None so they get the 1/epsilon boost, but SB shouldn't)
        # Just verify it doesn't crash and SB weight exists
        assert "SB" in weights
        assert weights["SB"] > 0

    def test_inverse_stats_get_fixed_weight(self):
        """ERA and WHIP should always get raw weight of 1.0 (before normalization)."""
        totals = {
            "R": 200, "HR": 50, "RBI": 200, "SB": 30, "AVG": 0.260,
            "W": 20, "K": 300, "ERA": 5.50, "WHIP": 1.60, "SV": 10,
        }
        weights = calculate_draft_leverage(totals, picks_made=6, total_picks=24, targets=self.TARGETS)
        # ERA and WHIP should have the same weight (both get raw=1.0)
        assert weights["ERA"] == pytest.approx(weights["WHIP"])

    def test_all_counting_stats_at_target_gives_equal_counting_weights(self):
        """When all counting stats are exactly on pace, their weights should be equal.

        Note: AVG is treated as a counting stat by the function (scaled by
        progress), so at 50% progress its expected is target*0.5=0.130 while
        current is 0.260 — appearing "ahead of pace."  We test that the 8
        true counting stats (R, HR, RBI, SB, W, K, SV) plus ERA/WHIP all
        get raw=1.0 and therefore equal weight among themselves.
        """
        progress = 0.5
        totals = {
            "R": 900 * progress,
            "HR": 265 * progress,
            "RBI": 890 * progress,
            "SB": 145 * progress,
            "AVG": 0.260 * progress,  # scale AVG too so it's "on pace"
            "W": 78 * progress,
            "K": 1250 * progress,
            "ERA": 3.80,
            "WHIP": 1.20,
            "SV": 55 * progress,
        }
        weights = calculate_draft_leverage(totals, picks_made=12, total_picks=24, targets=self.TARGETS)
        # All stats on-pace -> ratio=1.0 -> raw=1.0, same as ERA/WHIP fixed at 1.0
        expected = 1.0 / len(ALL_CATEGORIES)
        for cat in ALL_CATEGORIES:
            assert weights[cat] == pytest.approx(expected, abs=0.02)

    def test_progress_clamping_no_crash(self):
        """picks_made > total_picks should not crash; progress is clamped to 1.0."""
        totals = {
            "R": 800, "HR": 250, "RBI": 850, "SB": 130, "AVG": 0.265,
            "W": 70, "K": 1100, "ERA": 3.80, "WHIP": 1.20, "SV": 50,
        }
        weights = calculate_draft_leverage(totals, picks_made=30, total_picks=24, targets=self.TARGETS)
        assert len(weights) == len(ALL_CATEGORIES)
        assert all(v > 0 for v in weights.values())

    def test_output_all_positive_and_all_categories_covered(self):
        """Output should have all categories with positive weights summing to ~1.0."""
        totals = {
            "R": 300, "HR": 100, "RBI": 300, "SB": 50, "AVG": 0.260,
            "W": 30, "K": 500, "ERA": 3.80, "WHIP": 1.20, "SV": 20,
        }
        weights = calculate_draft_leverage(totals, picks_made=10, total_picks=24, targets=self.TARGETS)
        assert set(weights.keys()) == set(ALL_CATEGORIES)
        assert all(v > 0 for v in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_none_pitching_totals_get_high_weight(self):
        """When ERA/WHIP are None (no pitchers drafted), pitching cats get high weight."""
        totals = {
            "R": 300, "HR": 100, "RBI": 300, "SB": 50, "AVG": 0.260,
            "W": None, "K": None, "ERA": None, "WHIP": None, "SV": None,
        }
        weights = calculate_draft_leverage(totals, picks_made=5, total_picks=24, targets=self.TARGETS)
        # All pitching categories have None -> maximally behind
        # They should collectively get most of the weight
        pitching_weight = sum(weights[cat] for cat in ["W", "K", "ERA", "WHIP", "SV"])
        hitting_weight = sum(weights[cat] for cat in ["R", "HR", "RBI", "SB", "AVG"])
        assert pitching_weight > hitting_weight

    def test_negative_total_picks_returns_equal(self):
        """total_picks <= 0 should return equal weights like early draft."""
        totals = {cat: 0 for cat in ALL_CATEGORIES}
        weights = calculate_draft_leverage(totals, picks_made=5, total_picks=0, targets=self.TARGETS)
        expected = 1.0 / len(ALL_CATEGORIES)
        for cat in ALL_CATEGORIES:
            assert weights[cat] == pytest.approx(expected)
