import pytest
from fantasy_baseball.analysis.spoe import (
    ALL_COMPONENTS,
    components_to_roto_stats,
    prorate_spoe,
)
from fantasy_baseball.scoring import score_roto


def _make_components(r, hr, rbi, sb, h, ab, w, k, sv, ip, er, bb, ha):
    return {
        "r": r, "hr": hr, "rbi": rbi, "sb": sb, "h": h, "ab": ab,
        "w": w, "k": k, "sv": sv, "ip": ip, "er": er, "bb": bb, "h_allowed": ha,
    }


class TestProrateSpoe:
    def test_full_week_returns_unchanged_spoe(self):
        """7 days played = full week, no proration needed."""
        current = {
            "Team A": _make_components(8, 3, 7, 1, 20, 70, 2, 20, 1, 14, 5, 4, 12),
            "Team B": _make_components(6, 2, 5, 2, 15, 65, 1, 15, 2, 12, 6, 5, 14),
        }
        previous = {
            "Team A": {c: 0.0 for c in ALL_COMPONENTS},
            "Team B": {c: 0.0 for c in ALL_COMPONENTS},
        }
        actual_stats = {
            "Team A": {"R": 30, "HR": 8, "RBI": 25, "SB": 3, "AVG": .280,
                        "W": 3, "K": 40, "SV": 2, "ERA": 3.20, "WHIP": 1.10},
            "Team B": {"R": 25, "HR": 6, "RBI": 20, "SB": 5, "AVG": .260,
                        "W": 2, "K": 35, "SV": 4, "ERA": 3.80, "WHIP": 1.25},
        }
        results = prorate_spoe(current, previous, actual_stats, days_played=7)
        for r in results:
            if r["category"] != "total":
                proj_stats = components_to_roto_stats(current[r["team"]])
                assert r["projected_stat"] == pytest.approx(
                    proj_stats[r["category"]], abs=0.01
                )

    def test_partial_week_scales_current_contribution(self):
        """3 days played = current week scaled to 3/7."""
        current = {
            "Team A": _make_components(8, 3, 7, 1, 20, 70, 2, 20, 1, 14, 5, 4, 12),
            "Team B": _make_components(6, 2, 5, 2, 15, 65, 1, 15, 2, 12, 6, 5, 14),
        }
        previous = {
            "Team A": {c: 0.0 for c in ALL_COMPONENTS},
            "Team B": {c: 0.0 for c in ALL_COMPONENTS},
        }
        actual_stats = {
            "Team A": {"R": 30, "HR": 8, "RBI": 25, "SB": 3, "AVG": .280,
                        "W": 3, "K": 40, "SV": 2, "ERA": 3.20, "WHIP": 1.10},
            "Team B": {"R": 25, "HR": 6, "RBI": 20, "SB": 5, "AVG": .260,
                        "W": 2, "K": 35, "SV": 4, "ERA": 3.80, "WHIP": 1.25},
        }
        results = prorate_spoe(current, previous, actual_stats, days_played=3)
        team_a_r = next(r for r in results
                        if r["team"] == "Team A" and r["category"] == "R")
        assert team_a_r["projected_stat"] == pytest.approx(8.0 * 3 / 7, abs=0.01)

    def test_with_prior_weeks_accumulated(self):
        """Previous weeks' components are preserved; only current week scales."""
        previous = {
            "Team A": _make_components(4, 1, 3, 0, 10, 35, 1, 10, 0, 7, 2, 2, 6),
            "Team B": _make_components(3, 1, 2, 1, 8, 30, 0, 8, 1, 6, 3, 2, 7),
        }
        current = {
            "Team A": _make_components(12, 4, 10, 1, 30, 105, 3, 30, 1, 21, 7, 6, 18),
            "Team B": _make_components(9, 3, 7, 3, 23, 95, 1, 23, 3, 18, 9, 7, 21),
        }
        actual_stats = {
            "Team A": {"R": 30, "HR": 8, "RBI": 25, "SB": 3, "AVG": .280,
                        "W": 3, "K": 40, "SV": 2, "ERA": 3.20, "WHIP": 1.10},
            "Team B": {"R": 25, "HR": 6, "RBI": 20, "SB": 5, "AVG": .260,
                        "W": 2, "K": 35, "SV": 4, "ERA": 3.80, "WHIP": 1.25},
        }
        results = prorate_spoe(current, previous, actual_stats, days_played=3)
        team_a_r = next(r for r in results
                        if r["team"] == "Team A" and r["category"] == "R")
        assert team_a_r["projected_stat"] == pytest.approx(4 + 8 * 3 / 7, abs=0.01)

    def test_spoe_is_actual_minus_projected_pts(self):
        """SPOE = actual roto points - projected roto points."""
        current = {
            "Team A": _make_components(8, 3, 7, 1, 20, 70, 2, 20, 1, 14, 5, 4, 12),
            "Team B": _make_components(6, 2, 5, 2, 15, 65, 1, 15, 2, 12, 6, 5, 14),
        }
        previous = {
            "Team A": {c: 0.0 for c in ALL_COMPONENTS},
            "Team B": {c: 0.0 for c in ALL_COMPONENTS},
        }
        actual_stats = {
            "Team A": {"R": 30, "HR": 8, "RBI": 25, "SB": 3, "AVG": .280,
                        "W": 3, "K": 40, "SV": 2, "ERA": 3.20, "WHIP": 1.10},
            "Team B": {"R": 25, "HR": 6, "RBI": 20, "SB": 5, "AVG": .260,
                        "W": 2, "K": 35, "SV": 4, "ERA": 3.80, "WHIP": 1.25},
        }
        results = prorate_spoe(current, previous, actual_stats, days_played=3)
        for r in results:
            if r["category"] != "total":
                assert r["spoe"] == pytest.approx(
                    r["actual_pts"] - r["projected_pts"]
                ), f"SPOE mismatch for {r['team']} {r['category']}"

    def test_total_spoe_is_sum_of_categories(self):
        """Total SPOE row sums per-category SPOE."""
        current = {
            "Team A": _make_components(8, 3, 7, 1, 20, 70, 2, 20, 1, 14, 5, 4, 12),
            "Team B": _make_components(6, 2, 5, 2, 15, 65, 1, 15, 2, 12, 6, 5, 14),
        }
        previous = {
            "Team A": {c: 0.0 for c in ALL_COMPONENTS},
            "Team B": {c: 0.0 for c in ALL_COMPONENTS},
        }
        actual_stats = {
            "Team A": {"R": 30, "HR": 8, "RBI": 25, "SB": 3, "AVG": .280,
                        "W": 3, "K": 40, "SV": 2, "ERA": 3.20, "WHIP": 1.10},
            "Team B": {"R": 25, "HR": 6, "RBI": 20, "SB": 5, "AVG": .260,
                        "W": 2, "K": 35, "SV": 4, "ERA": 3.80, "WHIP": 1.25},
        }
        results = prorate_spoe(current, previous, actual_stats, days_played=3)
        for team in ("Team A", "Team B"):
            team_results = [r for r in results if r["team"] == team]
            cat_spoes = [r["spoe"] for r in team_results if r["category"] != "total"]
            total_row = next(r for r in team_results if r["category"] == "total")
            assert total_row["spoe"] == pytest.approx(sum(cat_spoes))

    def test_zero_days_played_uses_previous_only(self):
        """0 days played = current week contributes nothing."""
        previous = {
            "Team A": _make_components(4, 1, 3, 0, 10, 35, 1, 10, 0, 7, 2, 2, 6),
            "Team B": _make_components(3, 1, 2, 1, 8, 30, 0, 8, 1, 6, 3, 2, 7),
        }
        current = {
            "Team A": _make_components(12, 4, 10, 1, 30, 105, 3, 30, 1, 21, 7, 6, 18),
            "Team B": _make_components(9, 3, 7, 3, 23, 95, 1, 23, 3, 18, 9, 7, 21),
        }
        actual_stats = {
            "Team A": {"R": 30, "HR": 8, "RBI": 25, "SB": 3, "AVG": .280,
                        "W": 3, "K": 40, "SV": 2, "ERA": 3.20, "WHIP": 1.10},
            "Team B": {"R": 25, "HR": 6, "RBI": 20, "SB": 5, "AVG": .260,
                        "W": 2, "K": 35, "SV": 4, "ERA": 3.80, "WHIP": 1.25},
        }
        results = prorate_spoe(current, previous, actual_stats, days_played=0)
        team_a_r = next(r for r in results
                        if r["team"] == "Team A" and r["category"] == "R")
        assert team_a_r["projected_stat"] == pytest.approx(4.0, abs=0.01)
