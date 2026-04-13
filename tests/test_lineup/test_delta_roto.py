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
