import pytest
from fantasy_baseball.lineup.delta_roto import compute_defense_comfort
from fantasy_baseball.utils.constants import ALL_CATEGORIES


SGP_DENOMS = {"R": 20, "HR": 9, "RBI": 20, "SB": 8, "AVG": 0.005,
              "W": 3, "K": 30, "ERA": 0.15, "WHIP": 0.015, "SV": 7}


def _standings(overrides=None):
    """Build a 3-team standings dict. Team A is the user."""
    base = {"R": 800, "HR": 200, "RBI": 800, "SB": 100, "AVG": 0.260,
            "W": 70, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20}
    teams = {
        "Team A": dict(base),
        "Team B": {**base, "SV": 65, "ERA": 3.20},  # better SV, better ERA
        "Team C": {**base, "SV": 40, "ERA": 3.80},  # worse SV, worse ERA
    }
    if overrides:
        for team, stats in overrides.items():
            teams[team].update(stats)
    return teams


class TestComputeDefenseComfort:
    def test_counting_stat_defense(self):
        stats = _standings()
        # Team A has 50 SV, Team C has 40 SV. Defense gap = (50-40)/7 = 1.43
        comfort = compute_defense_comfort(stats, "Team A", SGP_DENOMS)
        assert comfort["SV"] == pytest.approx(10 / 7, abs=0.01)

    def test_inverse_stat_defense(self):
        stats = _standings()
        # Team A has 3.50 ERA, Team C has 3.80 ERA (worse).
        # Defense = how far until someone worse catches you = (3.80-3.50)/0.15 = 2.0
        comfort = compute_defense_comfort(stats, "Team A", SGP_DENOMS)
        assert comfort["ERA"] == pytest.approx(0.30 / 0.15, abs=0.01)

    def test_first_place_has_infinite_attack_finite_defense(self):
        stats = _standings({"Team A": {"SV": 80}})
        # Team A is 1st in SV (80). Team B is 2nd (65). Defense = (80-65)/7 = 2.14
        comfort = compute_defense_comfort(stats, "Team A", SGP_DENOMS)
        assert comfort["SV"] == pytest.approx(15 / 7, abs=0.01)

    def test_last_place_defense_is_infinite(self):
        stats = _standings({"Team A": {"SV": 30}})
        # Team A has 30 SV, worst of all 3 teams. Nobody below to catch up.
        comfort = compute_defense_comfort(stats, "Team A", SGP_DENOMS)
        assert comfort["SV"] == float('inf')


from fantasy_baseball.lineup.delta_roto import score_swap, DeltaRotoResult


class TestScoreSwap:
    """Test the per-category scoring rules."""

    def _roto(self, team_a_pts):
        """Build minimal roto dicts for Team A."""
        before = {"Team A": {f"{c}_pts": 5.0 for c in ALL_CATEGORIES}}
        before["Team A"]["total"] = 50.0
        after = {"Team A": {f"{c}_pts": 5.0 for c in ALL_CATEGORIES}}
        after["Team A"]["total"] = 50.0
        for cat, pts in team_a_pts.items():
            after["Team A"][f"{cat}_pts"] = pts
            after["Team A"]["total"] += (pts - 5.0)
        return before, after

    def test_loss_counted_at_full_value(self):
        roto_b, roto_a = self._roto({"SV": 4.0})
        comfort_b = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a["SV"] = 3.0  # comfort improved — should NOT offset loss
        result = score_swap(roto_b, roto_a, comfort_b, comfort_a, "Team A")
        assert result.categories["SV"].score == pytest.approx(-1.0)
        assert result.total <= -1.0

    def test_gain_discounted_at_exact_tie(self):
        roto_b, roto_a = self._roto({"AVG": 7.0})
        comfort_b = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a["AVG"] = 0.0
        result = score_swap(roto_b, roto_a, comfort_b, comfort_a, "Team A")
        assert result.categories["AVG"].score == pytest.approx(1.0)

    def test_gain_at_full_credit_when_comfortable(self):
        roto_b, roto_a = self._roto({"W": 7.0})
        comfort_b = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a = {c: 2.0 for c in ALL_CATEGORIES}
        result = score_swap(roto_b, roto_a, comfort_b, comfort_a, "Team A")
        assert result.categories["W"].score == pytest.approx(2.0)

    def test_gain_partially_discounted_below_threshold(self):
        roto_b, roto_a = self._roto({"SB": 7.0})
        comfort_b = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a["SB"] = 0.5
        # discount = 0.5 + 0.5 * (0.5 / 1.0) = 0.75; score = 2.0 * 0.75 = 1.5
        result = score_swap(roto_b, roto_a, comfort_b, comfort_a, "Team A")
        assert result.categories["SB"].score == pytest.approx(1.5)

    def test_comfort_erosion_penalty(self):
        roto_b, roto_a = self._roto({})
        comfort_b = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a["R"] = 1.0  # lost 1.0 denom; penalty = 0.3 * 1.0 = 0.30
        result = score_swap(roto_b, roto_a, comfort_b, comfort_a, "Team A")
        assert result.categories["R"].score == pytest.approx(-0.30)

    def test_erosion_capped(self):
        roto_b, roto_a = self._roto({})
        comfort_b = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a["R"] = 0.0  # lost 2.0 denoms -> 0.3*2.0=0.6, capped to 0.5
        result = score_swap(roto_b, roto_a, comfort_b, comfort_a, "Team A")
        assert result.categories["R"].score == pytest.approx(-0.50)

    def test_no_double_penalty_on_loss_plus_erosion(self):
        roto_b, roto_a = self._roto({"SV": 4.0})
        comfort_b = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a["SV"] = 0.5
        result = score_swap(roto_b, roto_a, comfort_b, comfort_a, "Team A")
        assert result.categories["SV"].score == pytest.approx(-1.0)

    def test_total_is_sum_of_categories(self):
        roto_b, roto_a = self._roto({"SV": 4.0, "W": 7.0})
        comfort_b = {c: 2.0 for c in ALL_CATEGORIES}
        comfort_a = {c: 2.0 for c in ALL_CATEGORIES}
        result = score_swap(roto_b, roto_a, comfort_b, comfort_a, "Team A")
        cat_sum = sum(cd.score for cd in result.categories.values())
        assert result.total == pytest.approx(cat_sum)
