"""Tests for Monte Carlo simulation functions (ROS extensions)."""

from unittest.mock import patch

import numpy as np
import pytest

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.simulation import (
    _GENERIC_HITTER_REPL,
    _apply_variance,
    _playing_time_scales,
    _projected_volume,
    _replacement_line,
    run_ros_monte_carlo,
    simulate_remaining_season,
)
from fantasy_baseball.utils.constants import REPLACEMENT_BY_POSITION


def _make_hitter(name, r=80, hr=25, rbi=80, sb=10, h=150, ab=550, positions=None):
    """Create a minimal hitter dict for testing."""
    d = {
        "name": name,
        "player_type": "hitter",
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "h": h,
        "ab": ab,
    }
    if positions is not None:
        d["positions"] = positions
    return d


def _make_pitcher(name, w=10, k=150, sv=0, ip=180, er=70, bb=50, h_allowed=150):
    """Create a minimal pitcher dict for testing."""
    return {
        "name": name,
        "player_type": "pitcher",
        "w": w,
        "k": k,
        "sv": sv,
        "ip": ip,
        "er": er,
        "bb": bb,
        "h_allowed": h_allowed,
    }


def _make_closer(name, w=3, k=60, sv=30, ip=65, er=20, bb=20, h_allowed=55):
    """Create a minimal closer dict for testing."""
    return _make_pitcher(name, w=w, k=k, sv=sv, ip=ip, er=er, bb=bb, h_allowed=h_allowed)


def _build_two_team_rosters():
    """Build a 2-team roster dict with enough players for h_slots=3, p_slots=2."""
    return {
        "Team A": [
            _make_hitter("H1", r=90, hr=30, rbi=100, sb=15, h=160, ab=550),
            _make_hitter("H2", r=70, hr=20, rbi=70, sb=8, h=140, ab=520),
            _make_hitter("H3", r=60, hr=15, rbi=55, sb=5, h=130, ab=500),
            _make_pitcher("P1", w=12, k=180, sv=0, ip=190, er=65, bb=45, h_allowed=160),
            _make_closer("C1", w=3, k=70, sv=35, ip=65, er=18, bb=18, h_allowed=50),
        ],
        "Team B": [
            _make_hitter("H4", r=85, hr=28, rbi=90, sb=12, h=155, ab=540),
            _make_hitter("H5", r=65, hr=18, rbi=65, sb=20, h=135, ab=510),
            _make_hitter("H6", r=55, hr=12, rbi=50, sb=3, h=125, ab=490),
            _make_pitcher("P2", w=10, k=160, sv=0, ip=175, er=70, bb=50, h_allowed=155),
            _make_closer("C2", w=2, k=55, sv=28, ip=60, er=22, bb=22, h_allowed=55),
        ],
    }


def _build_actual_standings():
    """Build actual YTD standings for 2 teams (midseason-like values)."""
    return {
        "Team A": {
            "R": 350,
            "HR": 100,
            "RBI": 340,
            "SB": 40,
            "AVG": 0.265,
            "W": 35,
            "K": 600,
            "SV": 25,
            "ERA": 3.80,
            "WHIP": 1.20,
        },
        "Team B": {
            "R": 320,
            "HR": 90,
            "RBI": 310,
            "SB": 50,
            "AVG": 0.255,
            "W": 30,
            "K": 550,
            "SV": 20,
            "ERA": 4.10,
            "WHIP": 1.28,
        },
    }


# ---------------------------------------------------------------------------
# Task 5 tests: simulate_remaining_season
# ---------------------------------------------------------------------------


class TestSimulateRemainingSeason:
    """Tests for simulate_remaining_season()."""

    def test_blends_actuals(self):
        """Counting stats should be greater than actuals (actual + simulated)
        and rate stats should be in valid ranges."""
        rosters = _build_two_team_rosters()
        actuals = _build_actual_standings()
        rng = np.random.default_rng(42)

        team_stats, injuries = simulate_remaining_season(
            actuals,
            rosters,
            fraction_remaining=0.5,
            rng=rng,
            h_slots=3,
            p_slots=2,
        )

        for team in ["Team A", "Team B"]:
            stats = team_stats[team]
            act = actuals[team]

            # Counting stats: final should be >= actual (sim adds non-negative)
            for cat in ["R", "HR", "RBI", "SB", "W", "K", "SV"]:
                assert stats[cat] >= act[cat], (
                    f"{team} {cat}: final {stats[cat]} < actual {act[cat]}"
                )

            # AVG should be in a valid range
            assert 0.150 < stats["AVG"] < 0.350, f"{team} AVG out of range: {stats['AVG']}"

            # ERA should be in a valid range
            assert 1.0 < stats["ERA"] < 8.0, f"{team} ERA out of range: {stats['ERA']}"

            # WHIP should be in a valid range
            assert 0.8 < stats["WHIP"] < 2.0, f"{team} WHIP out of range: {stats['WHIP']}"

        # Both teams should be in injuries dict
        assert "Team A" in injuries
        assert "Team B" in injuries

    def test_zero_remaining(self):
        """When fraction_remaining=0, result must equal actuals exactly."""
        rosters = _build_two_team_rosters()
        actuals = _build_actual_standings()

        rng = np.random.default_rng(99)
        team_stats, injuries = simulate_remaining_season(
            actuals,
            rosters,
            fraction_remaining=0.0,
            rng=rng,
            h_slots=3,
            p_slots=2,
        )

        for team in ["Team A", "Team B"]:
            act = actuals[team]
            result = team_stats[team]
            for cat in ["R", "HR", "RBI", "SB", "W", "K", "SV"]:
                assert result[cat] == pytest.approx(act[cat]), (
                    f"{team} {cat}: expected {act[cat]} at fraction_remaining=0, got {result[cat]}"
                )
            assert injuries[team] == []


class TestYtdPlayingTime:
    """The YTD blend weight must use real accumulated AB/IP when available,
    not a league-typical full-season constant scaled by elapsed fraction.

    Bug: `_TYPICAL_TEAM_IP = 1450` is too high for a 9-pitcher league, so the
    actual-vs-remaining blend over-weighted YTD pace against the projection's
    regression. Real PA/IP ride along on the standings (Yahoo `extras`)."""

    def test_uses_real_values_when_present(self):
        from fantasy_baseball.simulation import _ytd_playing_time

        ab, ip = _ytd_playing_time({"AB": 2000.0, "IP": 400.0}, fraction_elapsed=0.3)
        assert ab == pytest.approx(2000.0)
        assert ip == pytest.approx(400.0)

    def test_falls_back_to_typical_constants_when_absent(self):
        from fantasy_baseball.simulation import (
            _TYPICAL_TEAM_AB,
            _TYPICAL_TEAM_IP,
            _ytd_playing_time,
        )

        ab, ip = _ytd_playing_time({}, fraction_elapsed=0.3)
        assert ab == pytest.approx(_TYPICAL_TEAM_AB * 0.3)
        assert ip == pytest.approx(_TYPICAL_TEAM_IP * 0.3)

    def test_real_ip_flows_through_and_changes_era_blend(self):
        """Real IP that flips the YTD-vs-sim clamp must change the blended ERA."""
        rosters = _build_two_team_rosters()
        base = _build_actual_standings()

        constant_out, _ = simulate_remaining_season(
            base,
            rosters,
            fraction_remaining=0.5,
            rng=np.random.default_rng(7),
            h_slots=3,
            p_slots=2,
        )
        # Real accumulated IP far below the 1450*0.5 constant estimate -> YTD
        # no longer dominates, so the blend leans on the simulated remainder.
        with_real = {t: {**s, "IP": 100.0, "AB": 400.0} for t, s in base.items()}
        real_out, _ = simulate_remaining_season(
            with_real,
            rosters,
            fraction_remaining=0.5,
            rng=np.random.default_rng(7),
            h_slots=3,
            p_slots=2,
        )
        assert constant_out["Team A"]["ERA"] != pytest.approx(real_out["Team A"]["ERA"])


class TestProjectedVolume:
    """Curve-lookup volume: PA for hitters (AB/0.90 fallback), IP for pitchers."""

    def test_hitter_uses_pa_when_present(self):
        assert _projected_volume({"pa": 600, "ab": 540}, is_hitter=True) == 600

    def test_hitter_falls_back_to_ab_over_per_pa(self):
        assert _projected_volume({"ab": 540}, is_hitter=True) == pytest.approx(540 / 0.90)

    def test_pitcher_uses_ip(self):
        assert _projected_volume({"ip": 180}, is_hitter=False) == 180

    def test_nan_coerced_to_zero(self):
        # NaN PA/AB/IP (pandas-sourced dicts) must not slip through as NaN.
        assert _projected_volume({"ab": float("nan")}, is_hitter=True) == 0.0
        assert _projected_volume({"ip": float("nan")}, is_hitter=False) == 0.0


class TestPlayingTimeScales:
    """The two-sided, volume-scaled playing-time multiplier."""

    def test_full_season_returns_curve_mean_when_no_spread(self):
        players = [_make_hitter("A"), _make_hitter("B")]
        with patch("fantasy_baseball.simulation.playing_time_params", return_value=(0.8, 0.0)):
            scales = _playing_time_scales(players, PlayerType.HITTER, np.random.default_rng(1), 1.0)
        assert scales == pytest.approx([0.8, 0.8])

    def test_partial_season_damps_haircut(self):
        players = [_make_hitter("A")]
        with patch("fantasy_baseball.simulation.playing_time_params", return_value=(0.8, 0.0)):
            scales = _playing_time_scales(players, PlayerType.HITTER, np.random.default_rng(1), 0.5)
        # Only the remaining half is at risk: eff_mean = 1 - (1 - 0.8) * 0.5 = 0.9
        assert scales == pytest.approx([0.9])

    def test_two_sided_and_clipped(self):
        players = [_make_hitter(f"H{i}") for i in range(4000)]
        with patch("fantasy_baseball.simulation.playing_time_params", return_value=(1.0, 0.3)):
            scales = _playing_time_scales(players, PlayerType.HITTER, np.random.default_rng(0), 1.0)
        assert (scales > 1.0).any()  # a player can beat his projection (old model could not)
        assert (scales < 1.0).any()
        assert scales.min() >= 0.0
        assert scales.max() <= 2.0
        assert scales.mean() == pytest.approx(1.0, abs=0.02)

    def test_full_time_hitter_ceiling_is_realistic(self):
        # New empirical model: a full-time hitter roster (real curve + shape, no
        # patch) never spikes near the old flat 2.0 clip, and the realized
        # distribution is left-skewed (injuries reach further than over-play).
        players = [_make_hitter(f"H{i}") for i in range(3000)]  # ab=550 -> ~611 PA band
        scales = _playing_time_scales(players, PlayerType.HITTER, np.random.default_rng(0), 1.0)
        assert scales.max() < 1.3  # was ~1.7 under the symmetric-Normal-clip model
        assert scales.min() >= 0.0
        assert (1.0 - scales.min()) > (scales.max() - 1.0)  # left-skew


class TestReplacementLineRouting:
    """Position-aware replacement: the injured slot is filled per the player's spot."""

    def test_catcher_only_uses_catcher_line(self):
        p = {"player_type": PlayerType.HITTER, "positions": ["C"]}
        assert _replacement_line(p, is_hitter=True) == REPLACEMENT_BY_POSITION["C"]

    def test_multi_position_uses_best_available_by_sgp(self):
        # Eligible at weak C and strong OF -> the higher-SGP OF replacement.
        p = {"player_type": PlayerType.HITTER, "positions": ["C", "OF"]}
        assert _replacement_line(p, is_hitter=True) == REPLACEMENT_BY_POSITION["OF"]

    def test_util_only_uses_neutral_generic_line_not_speed_maxed(self):
        # UTIL/DH-only (no C/1B/2B/3B/SS/OF) must not get the SB-rich best line --
        # a power bat in a UTIL slot floors at the neutral mean, not 15 SB.
        p = {"player_type": PlayerType.HITTER, "positions": ["UTIL"]}
        line = _replacement_line(p, is_hitter=True)
        assert line == _GENERIC_HITTER_REPL
        assert line["sb"] < REPLACEMENT_BY_POSITION["OF"]["sb"]  # not speed-maximized

    def test_missing_positions_uses_generic_line(self):
        p = {"player_type": PlayerType.HITTER}
        assert _replacement_line(p, is_hitter=True) == _GENERIC_HITTER_REPL

    def test_sp_eligible_pitcher_routes_sp_despite_low_ip(self):
        # An injured/returning starter projected <100 IP must still floor at the SP
        # line (0 SV), NOT the RP line (8 SV) -- no phantom saves.
        p = {"player_type": PlayerType.PITCHER, "positions": ["SP"], "ip": 70}
        assert _replacement_line(p, is_hitter=False) == REPLACEMENT_BY_POSITION["SP"]

    def test_rp_eligible_pitcher_routes_rp(self):
        p = {"player_type": PlayerType.PITCHER, "positions": ["RP"], "ip": 70}
        assert _replacement_line(p, is_hitter=False) == REPLACEMENT_BY_POSITION["RP"]

    def test_swingman_prefers_sp_line(self):
        p = {"player_type": PlayerType.PITCHER, "positions": ["SP", "RP"], "ip": 70}
        assert _replacement_line(p, is_hitter=False) == REPLACEMENT_BY_POSITION["SP"]

    def test_starter_without_sp_rp_eligibility_uses_ip(self):
        # Bare dict / generic "P" eligibility -> classify by projected IP.
        assert (
            _replacement_line({"player_type": PlayerType.PITCHER, "ip": 180}, is_hitter=False)
            == REPLACEMENT_BY_POSITION["SP"]
        )
        assert (
            _replacement_line(
                {"player_type": PlayerType.PITCHER, "positions": ["P"], "ip": 60}, is_hitter=False
            )
            == REPLACEMENT_BY_POSITION["RP"]
        )

    def test_backfill_is_position_aware_for_speed(self):
        # Same player + same scale: a hurt SS (15-SB line) backfills more SB than a
        # hurt 1B (6-SB line) -- the position shape the flat line erased.
        out: list = []
        with patch("fantasy_baseball.simulation.playing_time_params", return_value=(0.5, 0.0)):
            ss = _apply_variance(
                [_make_hitter("S", positions=["SS"])],
                PlayerType.HITTER,
                np.random.default_rng(1),
                out,
            )
            fb = _apply_variance(
                [_make_hitter("F", positions=["1B"])],
                PlayerType.HITTER,
                np.random.default_rng(1),
                out,
            )
        assert ss[0]["sb"] > fb[0]["sb"]


class TestApplyVariancePlayingTime:
    """Injury-report logging and replacement backfill under the new model."""

    def test_logs_notable_playing_time_loss(self):
        out: list = []
        with patch("fantasy_baseball.simulation.playing_time_params", return_value=(0.5, 0.0)):
            _apply_variance([_make_hitter("X")], PlayerType.HITTER, np.random.default_rng(1), out)
        assert out == [("X", pytest.approx(0.5))]

    def test_does_not_log_minor_loss(self):
        out: list = []
        with patch("fantasy_baseball.simulation.playing_time_params", return_value=(0.95, 0.0)):
            _apply_variance([_make_hitter("X")], PlayerType.HITTER, np.random.default_rng(1), out)
        assert out == []  # frac_missed 0.05 < _NOTABLE_PT_LOSS (0.15)

    def test_replacement_backfill_keeps_stats_positive(self):
        out: list = []
        with patch("fantasy_baseball.simulation.playing_time_params", return_value=(0.3, 0.0)):
            adj = _apply_variance(
                [_make_hitter("X", r=80)], PlayerType.HITTER, np.random.default_rng(1), out
            )
        assert adj[0]["r"] > 0  # replacement fills the missed 0.7 fraction


# ---------------------------------------------------------------------------
# Task 6 tests: run_ros_monte_carlo
# ---------------------------------------------------------------------------


class TestRunRosMonteCarlo:
    """Tests for run_ros_monte_carlo()."""

    def test_returns_expected_format(self):
        """Verify return has team_results and category_risk with expected keys."""
        rosters = _build_two_team_rosters()
        actuals = _build_actual_standings()

        result = run_ros_monte_carlo(
            team_rosters=rosters,
            actual_standings=actuals,
            fraction_remaining=0.5,
            h_slots=3,
            p_slots=2,
            user_team_name="Team A",
            n_iterations=100,
            seed=42,
        )

        # Top-level keys
        assert "team_results" in result
        assert "category_risk" in result

        # team_results should have both teams
        tr = result["team_results"]
        assert "Team A" in tr
        assert "Team B" in tr

        # Each team result should have the expected keys
        expected_team_keys = {"median_pts", "p10", "p90", "first_pct", "top3_pct"}
        for team in ["Team A", "Team B"]:
            assert set(tr[team].keys()) == expected_team_keys, (
                f"{team} keys: {set(tr[team].keys())} != {expected_team_keys}"
            )
            # Sanity: median_pts should be positive (roto points)
            assert tr[team]["median_pts"] > 0
            # first_pct and top3_pct are percentages 0-100
            assert 0 <= tr[team]["first_pct"] <= 100
            assert 0 <= tr[team]["top3_pct"] <= 100

        # With only 2 teams, first_pct should sum to ~100
        total_first = sum(tr[t]["first_pct"] for t in tr)
        assert abs(total_first - 100.0) < 0.1, f"first_pct sum: {total_first}"

        # category_risk should have all 10 categories
        cr = result["category_risk"]
        expected_cats = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "ERA", "WHIP", "SV"]
        for cat in expected_cats:
            assert cat in cr, f"Missing category: {cat}"
            expected_cat_keys = {"median_pts", "p10", "p90", "top3_pct", "bot3_pct"}
            assert set(cr[cat].keys()) == expected_cat_keys, (
                f"{cat} keys: {set(cr[cat].keys())} != {expected_cat_keys}"
            )


# ---------------------------------------------------------------------------
# Regression: run_ros_monte_carlo must flatten full-season projections
# ---------------------------------------------------------------------------


class TestRosMonteCarloUsesFullSeason:
    """run_ros_monte_carlo must operate on full-season (ROS+YTD) stats so the
    YTD-blending math in simulate_remaining_season is well-formed."""

    def test_player_input_uses_full_season_projection(self):
        """Team A wins on full-season R only when MC honors full_season_projection."""
        from fantasy_baseball.models.player import (
            HitterStats,
            PitcherStats,
            Player,
            PlayerType,
        )

        def hitter(name, ros_r, fs_r):
            return Player(
                name=name,
                player_type=PlayerType.HITTER,
                positions=[],
                team="",
                rest_of_season=HitterStats(
                    pa=350, ab=315, h=82, r=ros_r, hr=8, rbi=40, sb=4, avg=0.260
                ),
                full_season_projection=HitterStats(
                    pa=650, ab=585, h=152, r=fs_r, hr=15, rbi=75, sb=8, avg=0.260
                ),
            )

        def pitcher(name):
            return Player(
                name=name,
                player_type=PlayerType.PITCHER,
                positions=[],
                team="",
                rest_of_season=PitcherStats(
                    ip=100, w=6, k=95, sv=0, er=38, bb=28, h_allowed=85, era=3.42, whip=1.13
                ),
                full_season_projection=PitcherStats(
                    ip=180, w=11, k=170, sv=0, er=68, bb=50, h_allowed=155, era=3.40, whip=1.14
                ),
            )

        # A's R is back-loaded into YTD (full-season 440 vs 360 actual), B's is
        # spread evenly (380 full-season vs 20 actual). A wins R only if MC
        # ranks on full-season; ROS-only collapses both to ~max(actual, ROS).
        rosters = {
            "Team A": [hitter(f"A{i}", ros_r=20, fs_r=110) for i in range(4)]
            + [pitcher(f"AP{i}") for i in range(3)],
            "Team B": [hitter(f"B{i}", ros_r=90, fs_r=95) for i in range(4)]
            + [pitcher(f"BP{i}") for i in range(3)],
        }
        common_actuals = {
            "HR": 30,
            "RBI": 140,
            "SB": 15,
            "AVG": 0.260,
            "W": 15,
            "K": 240,
            "SV": 0,
            "ERA": 3.40,
            "WHIP": 1.14,
        }
        actuals = {
            "Team A": {"R": 360, **common_actuals},
            "Team B": {"R": 20, **common_actuals},
        }

        result = run_ros_monte_carlo(
            team_rosters=rosters,
            actual_standings=actuals,
            fraction_remaining=0.5,
            h_slots=3,
            p_slots=2,
            user_team_name="Team A",
            n_iterations=200,
            seed=42,
        )

        tr = result["team_results"]
        assert tr["Team A"]["first_pct"] > tr["Team B"]["first_pct"], (
            f"A:{tr['Team A']['first_pct']}% vs B:{tr['Team B']['first_pct']}%"
        )

    def test_dict_input_with_nested_full_season(self):
        """Dict inputs with nested full_season_projection get flattened."""
        from fantasy_baseball.simulation import _flatten_full_season

        p = {
            "name": "X",
            "player_type": "hitter",
            "r": 20,  # ROS-flat
            "full_season_projection": {
                "r": 100,
                "hr": 25,
                "rbi": 80,
                "sb": 5,
                "h": 150,
                "ab": 550,
                "pa": 600,
                "avg": 0.272,
            },
        }
        flat = _flatten_full_season(p)
        assert flat["r"] == 100, "full_season_projection should overlay ROS"
        assert flat["hr"] == 25

    def test_dict_input_without_full_season_passes_through(self):
        """Legacy dicts with only flat top-level stats are preserved."""
        from fantasy_baseball.simulation import _flatten_full_season

        p = {"name": "X", "player_type": "hitter", "r": 80, "hr": 25}
        flat = _flatten_full_season(p)
        assert flat["r"] == 80
        assert flat["hr"] == 25
