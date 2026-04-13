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


from fantasy_baseball.lineup.delta_roto import compute_delta_roto
from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats


def _test_hitter(name, r=70, hr=20, rbi=70, sb=5, h=130, ab=500):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        rest_of_season=HitterStats(pa=int(ab * 1.15), ab=ab, h=h,
                                   r=r, hr=hr, rbi=rbi, sb=sb),
    )


def _test_pitcher(name, w=10, k=150, sv=0, ip=180, er=60, bb=50, h_allowed=150):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=["P"],
        rest_of_season=PitcherStats(w=w, k=k, sv=sv, ip=ip, er=er,
                                    bb=bb, h_allowed=h_allowed),
    )


class TestComputeDeltaRoto:
    def test_end_to_end_swap(self):
        roster = [
            _test_hitter("Hitter A", r=80, hr=30, rbi=90, sb=10, h=150, ab=550),
            _test_pitcher("Pitcher A", w=12, k=180, sv=30, ip=60, er=20, bb=15, h_allowed=50),
        ]
        add_player = _test_hitter("Hitter B", r=90, hr=25, rbi=85, sb=15, h=155, ab=550)

        standings = [
            {"name": "My Team", "team_key": "", "rank": 0,
             "stats": {"R": 800, "HR": 200, "RBI": 800, "SB": 100, "AVG": 0.260,
                        "W": 70, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20}},
            {"name": "Rival", "team_key": "", "rank": 0,
             "stats": {"R": 810, "HR": 210, "RBI": 810, "SB": 110, "AVG": 0.265,
                        "W": 75, "K": 1250, "SV": 60, "ERA": 3.40, "WHIP": 1.15}},
        ]

        result = compute_delta_roto(
            drop_name="Hitter A",
            add_player=add_player,
            user_roster=roster,
            projected_standings=standings,
            team_name="My Team",
        )

        assert isinstance(result, DeltaRotoResult)
        assert len(result.categories) == 10
        cat_sum = sum(cd.score for cd in result.categories.values())
        assert result.total == pytest.approx(cat_sum)
        d = result.to_dict()
        assert "total" in d
        assert "categories" in d
        assert len(d["categories"]) == 10

    def test_drop_not_found_raises(self):
        roster = [_test_hitter("Hitter A")]
        add_player = _test_hitter("Hitter B")
        standings = [{"name": "My Team", "team_key": "", "rank": 0,
                      "stats": {"R": 800, "HR": 200, "RBI": 800, "SB": 100, "AVG": 0.260,
                                 "W": 70, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20}}]
        with pytest.raises(ValueError, match="not found"):
            compute_delta_roto("Nobody", add_player, roster, standings, "My Team")
