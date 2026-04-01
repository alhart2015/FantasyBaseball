import pytest
from fantasy_baseball.lineup.leverage import calculate_leverage, blend_standings


def _make_standings():
    """10 teams with standings data. User team is rank 5."""
    return [
        {"name": "Team 1", "rank": 1, "stats": {"R": 500, "HR": 150, "RBI": 480, "SB": 90, "AVG": 0.275, "W": 55, "K": 850, "SV": 55, "ERA": 3.40, "WHIP": 1.15}},
        {"name": "Team 2", "rank": 2, "stats": {"R": 490, "HR": 145, "RBI": 470, "SB": 85, "AVG": 0.272, "W": 52, "K": 830, "SV": 50, "ERA": 3.50, "WHIP": 1.18}},
        {"name": "Team 3", "rank": 3, "stats": {"R": 475, "HR": 140, "RBI": 455, "SB": 80, "AVG": 0.270, "W": 50, "K": 810, "SV": 48, "ERA": 3.60, "WHIP": 1.20}},
        {"name": "Team 4", "rank": 4, "stats": {"R": 460, "HR": 135, "RBI": 445, "SB": 75, "AVG": 0.268, "W": 48, "K": 790, "SV": 45, "ERA": 3.70, "WHIP": 1.22}},
        {"name": "User Team", "rank": 5, "stats": {"R": 450, "HR": 130, "RBI": 430, "SB": 50, "AVG": 0.265, "W": 45, "K": 770, "SV": 40, "ERA": 3.80, "WHIP": 1.25}},
        {"name": "Team 6", "rank": 6, "stats": {"R": 430, "HR": 120, "RBI": 410, "SB": 45, "AVG": 0.260, "W": 42, "K": 740, "SV": 35, "ERA": 3.95, "WHIP": 1.28}},
        {"name": "Team 7", "rank": 7, "stats": {"R": 420, "HR": 115, "RBI": 400, "SB": 40, "AVG": 0.258, "W": 40, "K": 720, "SV": 30, "ERA": 4.10, "WHIP": 1.30}},
        {"name": "Team 8", "rank": 8, "stats": {"R": 400, "HR": 105, "RBI": 380, "SB": 35, "AVG": 0.252, "W": 35, "K": 690, "SV": 25, "ERA": 4.30, "WHIP": 1.35}},
        {"name": "Team 9", "rank": 9, "stats": {"R": 380, "HR": 95, "RBI": 360, "SB": 30, "AVG": 0.248, "W": 32, "K": 660, "SV": 20, "ERA": 4.50, "WHIP": 1.40}},
        {"name": "Team 10", "rank": 10, "stats": {"R": 350, "HR": 80, "RBI": 330, "SB": 20, "AVG": 0.240, "W": 28, "K": 620, "SV": 15, "ERA": 4.80, "WHIP": 1.48}},
    ]


class TestCalculateLeverage:
    def test_returns_all_categories(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        assert "R" in leverage
        assert "HR" in leverage
        assert "ERA" in leverage
        assert len(leverage) == 10

    def test_all_weights_positive(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        for cat, weight in leverage.items():
            assert weight >= 0, f"{cat} has negative weight"

    def test_weights_sum_to_one(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_small_gap_gets_high_leverage(self):
        """SB has a tiny defensive gap (5) vs R's attack gap (10).
        The small SB defense gap dominates, so SB leverage > R."""
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team", season_progress=1.0)
        # SB defense gap (50-45=5) is smaller than R attack gap (460-450=10),
        # so SB correctly gets higher leverage from defensive pressure.
        assert leverage["SB"] > leverage["R"]

    def test_inverse_stats_correct_direction(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team")
        assert leverage["ERA"] > 0

    def test_tied_category_does_not_dominate(self):
        """When one category is nearly tied, it should get high leverage
        but NOT swamp all other categories combined."""
        standings = _make_standings()
        # Make SB nearly tied with team above (75 vs 50 → gap of 25,
        # but override to make gap tiny: 50 vs 50.01)
        standings[3]["stats"]["SB"] = 50.01  # Team 4 (above user)
        standings[4]["stats"]["SB"] = 50.00  # User
        leverage = calculate_leverage(standings, "User Team", season_progress=1.0)
        # SB should be high but not more than ~30% of total weight
        # (with 10 categories, equal would be 10% each)
        assert leverage["SB"] < 0.35, (
            f"SB leverage {leverage['SB']:.3f} is too dominant for a single category"
        )
        # Other categories should still have meaningful weight
        non_sb = sum(v for k, v in leverage.items() if k != "SB")
        assert non_sb > 0.65

    def test_early_season_leverage_near_uniform(self):
        """At season_progress=0, all categories should be equal (uniform weights)."""
        standings = _make_standings()
        leverage = calculate_leverage(standings, "User Team", season_progress=0.0)
        uniform = 1.0 / 10
        for cat, weight in leverage.items():
            assert weight == pytest.approx(uniform, abs=0.001), (
                f"{cat} = {weight:.4f}, expected ~{uniform:.4f} at season start"
            )

    def test_midseason_leverage_blended(self):
        """At season_progress=0.5, leverage is halfway between uniform and standings-based."""
        standings = _make_standings()
        full = calculate_leverage(standings, "User Team", season_progress=1.0)
        half = calculate_leverage(standings, "User Team", season_progress=0.5)
        uniform = 1.0 / 10
        for cat in full:
            expected = 0.5 * full[cat] + 0.5 * uniform
            assert half[cat] == pytest.approx(expected, abs=0.001)

    def test_last_place_team_has_leverage(self):
        standings = _make_standings()
        leverage = calculate_leverage(standings, "Team 10")
        total = sum(leverage.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestBlendStandings:
    def _make_current(self):
        return [
            {"name": "Team A", "stats": {"R": 200, "HR": 50, "RBI": 180, "SB": 30,
             "AVG": 0.260, "W": 20, "K": 300, "SV": 15, "ERA": 4.00, "WHIP": 1.30}},
            {"name": "Team B", "stats": {"R": 180, "HR": 45, "RBI": 170, "SB": 40,
             "AVG": 0.270, "W": 18, "K": 280, "SV": 12, "ERA": 3.80, "WHIP": 1.25}},
        ]

    def _make_projected(self):
        return [
            {"name": "Team A", "stats": {"R": 800, "HR": 200, "RBI": 720, "SB": 100,
             "AVG": 0.265, "W": 80, "K": 1200, "SV": 60, "ERA": 3.80, "WHIP": 1.22}},
            {"name": "Team B", "stats": {"R": 780, "HR": 210, "RBI": 700, "SB": 120,
             "AVG": 0.272, "W": 75, "K": 1150, "SV": 55, "ERA": 3.60, "WHIP": 1.20}},
        ]

    def test_progress_zero_returns_projected(self):
        blended = blend_standings(self._make_current(), self._make_projected(), 0.0)
        team_a = next(t for t in blended if t["name"] == "Team A")
        assert team_a["stats"]["R"] == pytest.approx(800)
        assert team_a["stats"]["AVG"] == pytest.approx(0.265)

    def test_progress_one_returns_current(self):
        blended = blend_standings(self._make_current(), self._make_projected(), 1.0)
        team_a = next(t for t in blended if t["name"] == "Team A")
        assert team_a["stats"]["R"] == pytest.approx(200)
        assert team_a["stats"]["AVG"] == pytest.approx(0.260)

    def test_progress_half_interpolates(self):
        blended = blend_standings(self._make_current(), self._make_projected(), 0.5)
        team_a = next(t for t in blended if t["name"] == "Team A")
        assert team_a["stats"]["R"] == pytest.approx(500)  # (200+800)/2
        assert team_a["stats"]["AVG"] == pytest.approx(0.2625)  # (0.260+0.265)/2

    def test_teams_matched_by_name(self):
        current = self._make_current()
        projected = list(reversed(self._make_projected()))  # reverse order
        blended = blend_standings(current, projected, 0.0)
        team_a = next(t for t in blended if t["name"] == "Team A")
        assert team_a["stats"]["R"] == pytest.approx(800)  # matched correctly

    def test_team_only_in_current_included_as_is(self):
        current = self._make_current() + [
            {"name": "Team C", "stats": {"R": 100, "HR": 20, "RBI": 90, "SB": 10,
             "AVG": 0.240, "W": 10, "K": 150, "SV": 5, "ERA": 4.50, "WHIP": 1.40}},
        ]
        blended = blend_standings(current, self._make_projected(), 0.5)
        team_c = next(t for t in blended if t["name"] == "Team C")
        assert team_c["stats"]["R"] == 100  # no projected match, kept as-is


class TestCalculateLeverageWithProjected:
    def _make_projected(self):
        """Projected standings where SB gaps are large but HR gaps are tiny."""
        return [
            {"name": "Team 4", "rank": 4, "stats": {"R": 780, "HR": 201, "RBI": 720, "SB": 200, "AVG": 0.268, "W": 78, "K": 1200, "SV": 72, "ERA": 3.65, "WHIP": 1.21}},
            {"name": "User Team", "rank": 5, "stats": {"R": 760, "HR": 200, "RBI": 700, "SB": 100, "AVG": 0.265, "W": 75, "K": 1180, "SV": 65, "ERA": 3.75, "WHIP": 1.24}},
            {"name": "Team 6", "rank": 6, "stats": {"R": 720, "HR": 199, "RBI": 680, "SB": 80, "AVG": 0.260, "W": 70, "K": 1150, "SV": 60, "ERA": 3.90, "WHIP": 1.27}},
        ]

    def test_projected_standings_override_uniform_ramp(self):
        """At season_progress=0 with projected standings, leverage is NOT uniform."""
        standings = _make_standings()
        projected = self._make_projected()
        leverage = calculate_leverage(
            standings, "User Team",
            season_progress=0.0, projected_standings=projected,
        )
        # Should NOT be uniform — projected gaps matter
        values = list(leverage.values())
        assert max(values) - min(values) > 0.01

    def test_projected_tiny_hr_gap_gets_high_leverage(self):
        """HR gap is 1 in projected standings → high HR leverage."""
        standings = _make_standings()
        projected = self._make_projected()
        leverage = calculate_leverage(
            standings, "User Team",
            season_progress=0.0, projected_standings=projected,
        )
        # HR gaps are tiny (201 vs 200 vs 199) so HR should be high leverage
        # SB gaps are huge (200 vs 100 vs 80) so SB should be low
        assert leverage["HR"] > leverage["SB"]

    def test_no_projected_preserves_existing_behavior(self):
        """Without projected_standings, behavior is unchanged (uniform ramp)."""
        standings = _make_standings()
        leverage_old = calculate_leverage(
            standings, "User Team", season_progress=0.0,
        )
        leverage_new = calculate_leverage(
            standings, "User Team", season_progress=0.0, projected_standings=None,
        )
        for cat in leverage_old:
            assert leverage_old[cat] == pytest.approx(leverage_new[cat])
