from datetime import date

import pytest
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.models.standings import CategoryStats, StandingsEntry, StandingsSnapshot


def _list_to_snapshot(standings_list: list[dict]) -> StandingsSnapshot:
    """Convert a test standings list[dict] to StandingsSnapshot."""
    return StandingsSnapshot(
        effective_date=date.min,
        entries=[
            StandingsEntry(
                team_name=t["name"],
                team_key=t.get("team_key", ""),
                rank=t.get("rank", 0),
                stats=CategoryStats.from_dict(t.get("stats", {})),
            )
            for t in standings_list
        ],
    )


def _make_standings():
    """10 teams with standings data. User team is rank 5 overall.

    Per-category ranks for User Team:
      R: 5th (450, gap above=10, gap below=20)
      HR: 5th (130, gap above=5, gap below=10)
      RBI: 5th (430, gap above=15, gap below=20)
      SB: 8th (50, gap above=25, gap below=10)  <- bad at SB
      AVG: 5th (.265, gap above=.003, gap below=.005)
      W: 5th (45, gap above=3, gap below=3)
      K: 5th (770, gap above=20, gap below=30)
      SV: 5th (40, gap above=5, gap below=5)
      ERA: 5th (3.80, gap above=.10, gap below=.15)
      WHIP: 5th (1.25, gap above=.03, gap below=.03)
    """
    return _list_to_snapshot([
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
    ])


class TestCalculateLeverage:
    def test_returns_all_categories(self):
        leverage = calculate_leverage(_make_standings(), "User Team")
        assert len(leverage) == 10
        for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]:
            assert cat in leverage

    def test_all_weights_positive(self):
        leverage = calculate_leverage(_make_standings(), "User Team")
        for cat, weight in leverage.items():
            assert weight > 0, f"{cat} has non-positive weight"

    def test_weights_sum_to_one(self):
        leverage = calculate_leverage(_make_standings(), "User Team")
        assert sum(leverage.values()) == pytest.approx(1.0, abs=0.001)

    def test_uses_per_category_neighbors_not_overall_rank(self):
        """The critical test: per-category ranking must be used.

        Create a scenario where overall-rank neighbors give wrong results
        but per-category neighbors give correct results.

        User is 1st overall. The 2nd-place overall team has identical AVG
        but very different HR. With the OLD bug (overall-rank neighbors),
        AVG would get extreme leverage from the near-tie with 2nd overall.
        With the fix (per-category neighbors), the AVG neighbor is a
        different team with a real gap.
        """
        standings = _list_to_snapshot([
            {"name": "User", "rank": 1, "stats": {
                "R": 100, "HR": 50, "RBI": 100, "SB": 40, "AVG": 0.270,
                "W": 20, "K": 200, "SV": 10, "ERA": 3.50, "WHIP": 1.20,
            }},
            {"name": "Team 2", "rank": 2, "stats": {
                "R": 95, "HR": 30, "RBI": 95, "SB": 38, "AVG": 0.270,  # same AVG!
                "W": 18, "K": 190, "SV": 9, "ERA": 3.60, "WHIP": 1.22,
            }},
            {"name": "Team 3", "rank": 3, "stats": {
                "R": 90, "HR": 45, "RBI": 90, "SB": 35, "AVG": 0.260,  # AVG neighbor below
                "W": 16, "K": 180, "SV": 8, "ERA": 3.70, "WHIP": 1.25,
            }},
        ])
        leverage = calculate_leverage(standings, "User", season_progress=1.0)
        # Per-category: User is 1st in AVG (tied with Team 2, gap=0.010 to Team 3)
        # Per-category: User is 1st in HR (gap=5 to Team 3 who has 45)
        # HR gap (5) is much larger than AVG gap (0.010), so AVG > HR
        # But under the old bug, AVG would compare against Team 2 (gap=0) -> infinite leverage
        # With per-category, AVG defense gap is 0.010 which is still high but not infinite
        # The key assertion: HR should NOT be near-zero (the old bug made non-tied cats negligible)
        assert leverage["HR"] > 0.02, (
            f"HR leverage ({leverage['HR']:.4f}) should be meaningful, "
            f"not suppressed by a coincidental AVG tie with the overall-rank neighbor"
        )
        # AVG should be high (tied at 1st, tiny defense gap of 0.010)
        # but HR should still be real, not negligible
        assert leverage["HR"] / leverage["AVG"] > 0.05, (
            f"HR/AVG ratio ({leverage['HR']/leverage['AVG']:.3f}) too small — "
            f"HR is being suppressed by AVG dominance"
        )

    def test_first_place_large_cushion_gets_low_leverage(self):
        """1st in a category with a big cushion = low leverage (no point to gain,
        hard to lose). This was the user's actual scenario with SB."""
        import dataclasses
        base = _make_standings()
        # Make User Team 1st in SB by a mile
        entries = [
            dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=200))
            if e.team_name == "User Team" else e
            for e in base.entries
        ]
        standings = StandingsSnapshot(effective_date=base.effective_date, entries=entries)
        # Next best is Team 1 with 90 — gap of 110
        leverage = calculate_leverage(standings, "User Team", season_progress=1.0)
        # SB should have the lowest or near-lowest leverage
        min_lev = min(leverage.values())
        assert leverage["SB"] < min_lev * 1.5, (
            f"SB leverage ({leverage['SB']:.4f}) should be near minimum "
            f"when user is 1st with a huge cushion"
        )

    def test_last_place_large_deficit_gets_low_leverage(self):
        """Last in a category with a huge deficit = low leverage (very hard
        to gain a point, nothing below to lose)."""
        import dataclasses
        base = _make_standings()
        # Make User Team dead last in SB by a mile
        entries = [
            dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=1))
            if e.team_name == "User Team" else e
            for e in base.entries
        ]
        standings = StandingsSnapshot(effective_date=base.effective_date, entries=entries)
        # Next worst is Team 10 with 20 — gap of 19
        leverage = calculate_leverage(standings, "User Team", season_progress=1.0)
        min_lev = min(leverage.values())
        assert leverage["SB"] < min_lev * 1.5, (
            f"SB leverage ({leverage['SB']:.4f}) should be near minimum "
            f"when user is last with a huge deficit"
        )

    def test_packed_cluster_gets_high_leverage(self):
        """A category where many teams are packed within 1 SGP denom should
        get high leverage — gaining or losing 1 denom of production swings
        multiple roto points."""
        import dataclasses
        base = _make_standings()
        # Pack 5 teams within 1 SB denom (8) of user: user at 50,
        # others at 51, 52, 53, 54, 55 — all within 8 SB.
        # Gaining 8 SB passes all 5; losing 8 could drop through others below.
        entries = []
        for e in base.entries:
            if e.team_name == "Team 4":
                e = dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=55))
            elif e.team_name == "Team 3":
                e = dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=54))
            elif e.team_name == "Team 2":
                e = dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=53))
            elif e.team_name == "Team 1":
                e = dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=52))
            elif e.team_name == "Team 6":
                e = dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=51))
            entries.append(e)
        standings = StandingsSnapshot(effective_date=base.effective_date, entries=entries)
        leverage = calculate_leverage(standings, "User Team", season_progress=1.0)
        assert leverage["SB"] == max(leverage.values()) or leverage["SB"] > 0.15, (
            f"SB leverage ({leverage['SB']:.4f}) should be high when 5 teams "
            f"are packed within 1 denom"
        )

    def test_uniform_sgp_gaps_give_equal_leverage(self):
        """When all per-category gaps are exactly 1 SGP denominator,
        leverage should be uniform (each category is equally close to
        gaining/losing a point)."""
        from fantasy_baseball.sgp.player_value import get_sgp_denominators
        denoms = get_sgp_denominators()
        standings = _list_to_snapshot([
            {"name": "Above", "rank": 1, "stats": {
                "R": 100 + denoms["R"], "HR": 100 + denoms["HR"],
                "RBI": 100 + denoms["RBI"], "SB": 100 + denoms["SB"],
                "AVG": 0.270 + denoms["AVG"],
                "W": 100 + denoms["W"], "K": 100 + denoms["K"],
                "SV": 100 + denoms["SV"],
                "ERA": 3.50 - denoms["ERA"], "WHIP": 1.20 - denoms["WHIP"],
            }},
            {"name": "User", "rank": 2, "stats": {
                "R": 100, "HR": 100, "RBI": 100, "SB": 100, "AVG": 0.270,
                "W": 100, "K": 100, "SV": 100, "ERA": 3.50, "WHIP": 1.20,
            }},
            {"name": "Below", "rank": 3, "stats": {
                "R": 100 - denoms["R"], "HR": 100 - denoms["HR"],
                "RBI": 100 - denoms["RBI"], "SB": 100 - denoms["SB"],
                "AVG": 0.270 - denoms["AVG"],
                "W": 100 - denoms["W"], "K": 100 - denoms["K"],
                "SV": 100 - denoms["SV"],
                "ERA": 3.50 + denoms["ERA"], "WHIP": 1.20 + denoms["WHIP"],
            }},
        ])
        leverage = calculate_leverage(standings, "User", season_progress=1.0)
        expected = 1.0 / 10
        for cat, weight in leverage.items():
            assert weight == pytest.approx(expected, abs=0.005), (
                f"{cat} = {weight:.4f}, expected ~{expected:.4f} with uniform SGP gaps"
            )

    def test_tied_category_does_not_dominate(self):
        """A nearly-tied category should get high leverage but not swamp
        all others (outlier capping)."""
        import dataclasses
        base = _make_standings()
        entries = []
        for e in base.entries:
            if e.team_name == "Team 4":
                e = dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=50.01))
            elif e.team_name == "User Team":
                e = dataclasses.replace(e, stats=dataclasses.replace(e.stats, sb=50.00))
            entries.append(e)
        standings = StandingsSnapshot(effective_date=base.effective_date, entries=entries)
        leverage = calculate_leverage(standings, "User Team", season_progress=1.0)
        assert leverage["SB"] < 0.35, (
            f"SB leverage {leverage['SB']:.3f} too dominant for a single tied category"
        )
        non_sb = sum(v for k, v in leverage.items() if k != "SB")
        assert non_sb > 0.65

    def test_early_season_leverage_near_uniform(self):
        """At season_progress=0 without projections, all categories equal."""
        leverage = calculate_leverage(_make_standings(), "User Team", season_progress=0.0)
        uniform = 1.0 / 10
        for cat, weight in leverage.items():
            assert weight == pytest.approx(uniform, abs=0.001), (
                f"{cat} = {weight:.4f}, expected ~{uniform:.4f} at season start"
            )

    def test_midseason_leverage_blended(self):
        """At season_progress=0.5 without projections, halfway between
        uniform and standings-based."""
        full = calculate_leverage(_make_standings(), "User Team", season_progress=1.0)
        half = calculate_leverage(_make_standings(), "User Team", season_progress=0.5)
        uniform = 1.0 / 10
        for cat in full:
            expected = 0.5 * full[cat] + 0.5 * uniform
            assert half[cat] == pytest.approx(expected, abs=0.001)

    def test_first_place_team_has_leverage(self):
        leverage = calculate_leverage(_make_standings(), "Team 1")
        assert sum(leverage.values()) == pytest.approx(1.0, abs=0.01)

    def test_last_place_team_has_leverage(self):
        leverage = calculate_leverage(_make_standings(), "Team 10")
        assert sum(leverage.values()) == pytest.approx(1.0, abs=0.01)

    def test_unknown_team_returns_uniform(self):
        leverage = calculate_leverage(_make_standings(), "Nonexistent Team")
        uniform = 1.0 / 10
        for weight in leverage.values():
            assert weight == pytest.approx(uniform, abs=0.001)

    def test_rate_stat_gap_not_inflated_vs_counting(self):
        """A 0.001 AVG gap and a 1-run R gap represent similar difficulty
        to close (~1 hit). With SGP normalization, they should produce
        similar leverage, not 1000x different."""
        standings = _list_to_snapshot([
            {"name": "Above", "rank": 1, "stats": {
                "R": 101, "HR": 100, "RBI": 100, "SB": 100, "AVG": 0.271,
                "W": 100, "K": 100, "SV": 100, "ERA": 3.50, "WHIP": 1.20,
            }},
            {"name": "User", "rank": 2, "stats": {
                "R": 100, "HR": 100, "RBI": 100, "SB": 100, "AVG": 0.270,
                "W": 100, "K": 100, "SV": 100, "ERA": 3.50, "WHIP": 1.20,
            }},
            {"name": "Below", "rank": 3, "stats": {
                "R": 99, "HR": 100, "RBI": 100, "SB": 100, "AVG": 0.269,
                "W": 100, "K": 100, "SV": 100, "ERA": 3.50, "WHIP": 1.20,
            }},
        ])
        leverage = calculate_leverage(standings, "User", season_progress=1.0)
        # R gap is 1 run, AVG gap is 0.001. Without normalization, AVG
        # would dominate by ~1000x. With SGP normalization, they should
        # be within an order of magnitude of each other.
        ratio = leverage["AVG"] / leverage["R"] if leverage["R"] > 0 else float("inf")
        assert ratio < 10, (
            f"AVG/R leverage ratio is {ratio:.1f} — AVG still dominates "
            f"despite similar SGP-normalized gaps"
        )

    def test_real_world_scenario_first_place_sb_leader(self):
        """Regression test for the actual bug: user is 1st overall and
        1st in SB with a large cushion, but 5th in R with a tight gap.
        R should have higher leverage than SB."""
        standings = _list_to_snapshot([
            {"name": "User", "rank": 1, "stats": {
                "R": 71, "HR": 20, "RBI": 75, "SB": 200,
                "AVG": 0.253, "W": 7, "K": 114, "SV": 6,
                "ERA": 3.16, "WHIP": 0.99,
            }},
            {"name": "Team 2", "rank": 2, "stats": {
                "R": 72, "HR": 18, "RBI": 77, "SB": 180,
                "AVG": 0.253, "W": 6, "K": 109, "SV": 6,
                "ERA": 3.14, "WHIP": 0.99,
            }},
            {"name": "Team 3", "rank": 3, "stats": {
                "R": 70, "HR": 22, "RBI": 71, "SB": 170,
                "AVG": 0.249, "W": 8, "K": 119, "SV": 5,
                "ERA": 3.18, "WHIP": 1.01,
            }},
            {"name": "Team 4", "rank": 4, "stats": {
                "R": 68, "HR": 15, "RBI": 65, "SB": 160,
                "AVG": 0.245, "W": 5, "K": 100, "SV": 7,
                "ERA": 3.50, "WHIP": 1.10,
            }},
        ])
        leverage = calculate_leverage(standings, "User", season_progress=1.0)
        # SB has a 20-steal cushion to 2nd place → low leverage
        # R has a 1-run gap to the team above in R → higher leverage
        assert leverage["R"] > leverage["SB"], (
            f"R ({leverage['R']:.4f}) should beat SB ({leverage['SB']:.4f}): "
            f"R gap is 1 run, SB cushion is 20 steals"
        )


    def test_cliff_risk_multiple_roto_points_at_stake(self):
        """Regression test for the cliff-risk bug (TODO: wSGP underweights
        cliff risk in packed categories).

        User is 1st in SB at 100, teams 2-6 are packed at 93-97.
        Losing one SB denom (8) drops to 92, falling behind all 5.
        SB should get high leverage: 5 roto points at risk, not 1.

        With the old single-neighbor approach, SB leverage was low because
        the gap to 2nd (3 SB) was smaller than other counting gaps. The
        marginal roto approach correctly identifies the cliff.
        """
        standings = _list_to_snapshot([
            {"name": "User", "rank": 1, "stats": {
                "R": 100, "HR": 30, "RBI": 100, "SB": 100,
                "AVG": 0.270, "W": 15, "K": 200, "SV": 10,
                "ERA": 3.50, "WHIP": 1.20,
            }},
            # Teams 2-6: packed in SB just below user (93-97)
            {"name": "Team 2", "rank": 2, "stats": {
                "R": 95, "HR": 28, "RBI": 95, "SB": 97,
                "AVG": 0.265, "W": 14, "K": 190, "SV": 9,
                "ERA": 3.60, "WHIP": 1.22,
            }},
            {"name": "Team 3", "rank": 3, "stats": {
                "R": 90, "HR": 26, "RBI": 90, "SB": 96,
                "AVG": 0.260, "W": 13, "K": 180, "SV": 8,
                "ERA": 3.70, "WHIP": 1.25,
            }},
            {"name": "Team 4", "rank": 4, "stats": {
                "R": 85, "HR": 24, "RBI": 85, "SB": 95,
                "AVG": 0.255, "W": 12, "K": 170, "SV": 7,
                "ERA": 3.80, "WHIP": 1.28,
            }},
            {"name": "Team 5", "rank": 5, "stats": {
                "R": 80, "HR": 22, "RBI": 80, "SB": 94,
                "AVG": 0.250, "W": 11, "K": 160, "SV": 6,
                "ERA": 3.90, "WHIP": 1.30,
            }},
            {"name": "Team 6", "rank": 6, "stats": {
                "R": 75, "HR": 20, "RBI": 75, "SB": 93,
                "AVG": 0.245, "W": 10, "K": 150, "SV": 5,
                "ERA": 4.00, "WHIP": 1.32,
            }},
        ])
        leverage = calculate_leverage(standings, "User", season_progress=1.0)
        # SB: user at 100, pack at 93-97. Losing 8 SB → 92, falling behind
        # all 5 teams. 5 roto points of defensive risk.
        assert leverage["SB"] == max(leverage.values()), (
            f"SB leverage ({leverage['SB']:.4f}) should be highest when "
            f"losing 1 denom risks dropping through a 5-team pack"
        )


class TestCalculateLeverageWithProjected:
    def _make_projected(self):
        """Projected standings where SB gaps are large but HR gaps are tiny."""
        return _list_to_snapshot([
            {"name": "Team 4", "rank": 4, "stats": {"R": 780, "HR": 201, "RBI": 720, "SB": 200, "AVG": 0.268, "W": 78, "K": 1200, "SV": 72, "ERA": 3.65, "WHIP": 1.21}},
            {"name": "User Team", "rank": 5, "stats": {"R": 760, "HR": 200, "RBI": 700, "SB": 100, "AVG": 0.265, "W": 75, "K": 1180, "SV": 65, "ERA": 3.75, "WHIP": 1.24}},
            {"name": "Team 6", "rank": 6, "stats": {"R": 720, "HR": 199, "RBI": 680, "SB": 80, "AVG": 0.260, "W": 70, "K": 1150, "SV": 60, "ERA": 3.90, "WHIP": 1.27}},
        ])

    def test_projected_early_season_near_uniform(self):
        """At season_progress=0.0, leverage is uniform even with projected
        standings — projections have wide error bars early on."""
        standings = _make_standings()
        projected = self._make_projected()
        leverage = calculate_leverage(
            standings, "User Team",
            season_progress=0.0, projected_standings=projected,
        )
        uniform = 1.0 / 10
        for cat, weight in leverage.items():
            assert weight == pytest.approx(uniform, abs=0.001), (
                f"{cat} = {weight:.4f}, expected ~{uniform:.4f} at season start"
            )

    def test_projected_midseason_blended(self):
        """At season_progress=0.5, leverage is halfway between projected-
        standings-derived and uniform."""
        standings = _make_standings()
        projected = self._make_projected()
        full = calculate_leverage(
            standings, "User Team",
            season_progress=1.0, projected_standings=projected,
        )
        half = calculate_leverage(
            standings, "User Team",
            season_progress=0.5, projected_standings=projected,
        )
        uniform = 1.0 / 10
        for cat in full:
            expected = 0.5 * full[cat] + 0.5 * uniform
            assert half[cat] == pytest.approx(expected, abs=0.001)

    def test_projected_full_season_tiny_hr_gap_gets_high_leverage(self):
        """HR gap is 1 in projected → high leverage. SB gap is 100 → low."""
        standings = _make_standings()
        projected = self._make_projected()
        leverage = calculate_leverage(
            standings, "User Team",
            season_progress=1.0, projected_standings=projected,
        )
        assert leverage["HR"] > leverage["SB"]

    def test_no_projected_preserves_existing_behavior(self):
        standings = _make_standings()
        leverage_old = calculate_leverage(standings, "User Team", season_progress=0.0)
        leverage_new = calculate_leverage(standings, "User Team", season_progress=0.0, projected_standings=None)
        for cat in leverage_old:
            assert leverage_old[cat] == pytest.approx(leverage_new[cat])
